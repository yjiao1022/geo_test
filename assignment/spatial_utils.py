"""
Spatial embedding utilities for geo-experiments.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from typing import Optional


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