"""
Spatial embedding utilities for geo-experiments.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import euclidean_distances
from typing import Optional, Tuple
import torch


def add_spectral_spatial_embedding(
    geo_features: pd.DataFrame, 
    spatial_cols: list = ['xy1', 'xy2'],
    spatial_emb_dim: int = 2, 
    spatial_neighbors: int = 8, 
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Add spectral spatial embedding columns to geo_features.
    
    This function creates a k-nearest neighbor graph based on spatial coordinates
    and uses spectral embedding to create lower-dimensional spatial features.
    
    Args:
        geo_features: DataFrame with geo-level features including spatial coordinates
        spatial_cols: List of column names for spatial coordinates (default: ['xy1', 'xy2'])
        spatial_emb_dim: Dimension of spatial embedding (default: 2)
        spatial_neighbors: Number of neighbors for k-NN graph (default: 8)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with added spatial embedding columns named 'spatial_emb_{i}'
        
    Raises:
        ValueError: If spatial_cols are not found in geo_features
    """
    # Validate inputs
    missing_cols = set(spatial_cols) - set(geo_features.columns)
    if missing_cols:
        raise ValueError(f"Missing spatial columns: {missing_cols}")
    
    if len(geo_features) < spatial_neighbors + 1:
        raise ValueError(f"Need at least {spatial_neighbors + 1} geos for {spatial_neighbors} neighbors")
    
    # Extract spatial coordinates
    X = geo_features[spatial_cols].values
    
    # Build k-NN graph
    knn_graph = kneighbors_graph(
        X, 
        n_neighbors=spatial_neighbors, 
        include_self=True,
        mode='connectivity'
    )
    
    # Symmetrize adjacency matrix
    A = knn_graph.toarray()
    A = 0.5 * (A + A.T)
    
    # Apply spectral embedding
    spec_emb = SpectralEmbedding(
        n_components=spatial_emb_dim,
        affinity='precomputed',
        random_state=seed or 42
    )
    
    spatial_embedding = spec_emb.fit_transform(A)
    
    # Add embedding columns to DataFrame
    geo_features = geo_features.copy()
    spatial_emb_cols = [f'spatial_emb_{i}' for i in range(spatial_emb_dim)]
    
    for i, col in enumerate(spatial_emb_cols):
        geo_features[col] = spatial_embedding[:, i]
    
    return geo_features


def standardize_features(features: np.ndarray) -> np.ndarray:
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        features: Array of features to standardize
        
    Returns:
        Standardized features
    """
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)


def build_spatial_adjacency_matrix(
    geo_features: pd.DataFrame,
    spatial_cols: list = ['xy1', 'xy2'],
    connection_method: str = 'knn',
    k_neighbors: int = 8,
    distance_threshold: Optional[float] = None,
    include_self_loops: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build spatial adjacency matrix and edge indices for STGCN.
    
    This function creates a graph representation of geographic relationships
    between geos based on their spatial coordinates. The graph can be used
    in Spatio-Temporal Graph Convolutional Networks (STGCN) to model
    spatial dependencies and spillover effects.
    
    Args:
        geo_features: DataFrame with geo-level features including spatial coordinates
        spatial_cols: List of column names for spatial coordinates (default: ['xy1', 'xy2'])
        connection_method: Method to determine connections ('knn', 'threshold', 'hybrid')
            - 'knn': Connect each geo to k nearest neighbors
            - 'threshold': Connect geos within distance threshold
            - 'hybrid': Combine knn and threshold methods
        k_neighbors: Number of nearest neighbors for knn method (default: 8)
        distance_threshold: Maximum distance for threshold method (auto-computed if None)
        include_self_loops: Whether to include self-connections (default: False)
        
    Returns:
        Tuple containing:
        - edge_index: Tensor of shape [2, num_edges] with source and target node indices
        - edge_weight: Tensor of shape [num_edges] with edge weights (inverse distance)
        
    Raises:
        ValueError: If spatial_cols are not found in geo_features or invalid method
    """
    # Validate inputs
    missing_cols = set(spatial_cols) - set(geo_features.columns)
    if missing_cols:
        raise ValueError(f"Missing spatial columns: {missing_cols}")
    
    if connection_method not in ['knn', 'threshold', 'hybrid']:
        raise ValueError(f"Invalid connection_method: {connection_method}")
    
    n_geos = len(geo_features)
    if n_geos < 2:
        raise ValueError("Need at least 2 geos to build adjacency matrix")
    
    # Extract spatial coordinates and compute pairwise distances
    coords = geo_features[spatial_cols].values
    distances = euclidean_distances(coords)
    
    # Initialize adjacency matrix
    adjacency = np.zeros((n_geos, n_geos), dtype=np.float32)
    
    if connection_method in ['knn', 'hybrid']:
        # K-nearest neighbors approach
        k = min(k_neighbors, n_geos - 1)  # Ensure k doesn't exceed available neighbors
        
        for i in range(n_geos):
            # Find k nearest neighbors (excluding self)
            neighbor_distances = distances[i].copy()
            neighbor_distances[i] = np.inf  # Exclude self
            nearest_indices = np.argsort(neighbor_distances)[:k]
            
            # Add edges to nearest neighbors
            for j in nearest_indices:
                adjacency[i, j] = 1.0
    
    if connection_method in ['threshold', 'hybrid']:
        # Distance threshold approach
        if distance_threshold is None:
            # Auto-compute threshold as median of non-zero distances
            non_zero_distances = distances[distances > 0]
            distance_threshold = np.median(non_zero_distances)
        
        # Connect geos within threshold distance
        within_threshold = (distances <= distance_threshold) & (distances > 0)
        adjacency[within_threshold] = 1.0
    
    # Make adjacency matrix symmetric (undirected graph)
    adjacency = np.maximum(adjacency, adjacency.T)
    
    # Add self-loops if requested
    if include_self_loops:
        np.fill_diagonal(adjacency, 1.0)
    
    # Convert to edge list format for PyTorch Geometric
    edge_indices = np.nonzero(adjacency)
    edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
    
    # Compute edge weights as inverse distances (closer geos have higher weights)
    edge_weights = []
    for i, j in zip(edge_indices[0], edge_indices[1]):
        if i == j and include_self_loops:
            # Self-loop weight
            edge_weights.append(1.0)
        else:
            # Inverse distance weight with small epsilon to avoid division by zero
            dist = distances[i, j]
            weight = 1.0 / (dist + 1e-6)
            edge_weights.append(weight)
    
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    return edge_index, edge_weight


def normalize_adjacency_matrix(adjacency: torch.Tensor, epsilon: float = 1e-8, max_norm: float = 1e4) -> torch.Tensor:
    """
    Normalize adjacency matrix with numerical stability protections.
    
    Applies symmetric normalization: D^(-1/2) * A * D^(-1/2)
    where D is the degree matrix and A is the adjacency matrix.
    
    Args:
        adjacency: Adjacency matrix tensor of shape [num_nodes, num_nodes]
        epsilon: Small value to prevent division by zero (default: 1e-8)
        max_norm: Maximum allowed norm value to prevent explosion (default: 1e4)
        
    Returns:
        Normalized adjacency matrix with numerical stability guarantees
    """
    # Compute degree matrix
    degree = torch.sum(adjacency, dim=1)  # [num_nodes]
    
    # Add epsilon to prevent division by zero and clamp to prevent explosion
    degree_with_eps = degree + epsilon
    deg_inv_sqrt = torch.pow(degree_with_eps, -0.5)
    deg_inv_sqrt = torch.clamp(deg_inv_sqrt, max=max_norm)
    
    # Handle isolated nodes (degree = 0) by setting their norm to 0
    isolated_mask = degree < epsilon
    deg_inv_sqrt[isolated_mask] = 0.0
    
    # Create diagonal degree matrix
    deg_matrix = torch.diag(deg_inv_sqrt)
    
    # Apply symmetric normalization: D^(-1/2) * A * D^(-1/2)
    normalized_adj = torch.mm(torch.mm(deg_matrix, adjacency), deg_matrix)
    
    # Additional numerical stability checks
    normalized_adj = torch.clamp(normalized_adj, min=-max_norm, max=max_norm)
    
    # Replace any NaN or Inf values with 0
    normalized_adj = torch.where(
        torch.isfinite(normalized_adj), 
        normalized_adj, 
        torch.zeros_like(normalized_adj)
    )
    
    return normalized_adj


def prepare_stgcn_data(
    panel_data: pd.DataFrame,
    geo_features: pd.DataFrame,
    feature_cols: list = ['sales', 'spend'],
    window_size: int = 10
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Prepare panel data for STGCN input format.
    
    Converts long-format panel data into the tensor format required by
    Spatio-Temporal Graph Convolutional Networks: [num_nodes, num_timesteps, num_features].
    
    Args:
        panel_data: Long-format panel data with columns ['geo', 'date', ...features]
        geo_features: DataFrame with geo information to determine node ordering
        feature_cols: List of feature column names to include (default: ['sales', 'spend'])
        window_size: Size of temporal windows for sequence modeling (default: 10)
        
    Returns:
        Tuple containing:
        - data_tensor: Tensor of shape [num_nodes, num_timesteps, num_features]
        - geo_to_idx: Array mapping geo names to node indices
        
    Raises:
        ValueError: If required columns are missing or data is inconsistent
    """
    # Validate inputs
    required_cols = ['geo', 'date'] + feature_cols
    missing_cols = set(required_cols) - set(panel_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in panel_data: {missing_cols}")
    
    # Create geo to index mapping based on geo_features order
    unique_geos = geo_features['geo'].values
    geo_to_idx = {geo: idx for idx, geo in enumerate(unique_geos)}
    
    # Filter panel data to only include geos in geo_features
    panel_filtered = panel_data[panel_data['geo'].isin(unique_geos)].copy()
    
    if len(panel_filtered) == 0:
        raise ValueError("No matching geos found between panel_data and geo_features")
    
    # Sort by geo and date
    panel_filtered = panel_filtered.sort_values(['geo', 'date'])
    
    # Pivot to wide format: geos as rows, dates as columns, features as additional dimension
    data_list = []
    
    for feature in feature_cols:
        # Pivot each feature separately
        feature_pivot = panel_filtered.pivot(index='geo', columns='date', values=feature)
        
        # Reorder rows to match geo_features order
        feature_pivot = feature_pivot.reindex(unique_geos)
        
        # Handle missing values with forward fill then backward fill
        feature_pivot = feature_pivot.ffill(axis=1)
        feature_pivot = feature_pivot.bfill(axis=1)
        feature_pivot = feature_pivot.fillna(0)  # Final fallback
        
        data_list.append(feature_pivot.values)
    
    # Stack features to create [num_nodes, num_timesteps, num_features] tensor
    data_array = np.stack(data_list, axis=-1)
    data_tensor = torch.tensor(data_array, dtype=torch.float32)
    
    return data_tensor, np.array(unique_geos)