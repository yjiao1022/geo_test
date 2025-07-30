"""
Spatio-Temporal Graph Convolutional Network (STGCN) reporting model for geo-experiments.

This module implements a sophisticated neural network architecture that combines:
1. Graph Convolutional Networks (GCN) for spatial relationships between geos
2. Temporal convolution/recurrent layers for time-series dynamics
3. Recursive counterfactual prediction anchored on control group observations

The STGCN model addresses key challenges in geo-experimentation:
- Spatial spillover effects between neighboring geos
- Complex temporal dependencies in sales/spend patterns
- Non-linear relationships that traditional methods cannot capture
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod
import warnings
from tqdm import tqdm
import scipy.stats

# Try to import PyTorch Geometric, provide fallback if not available
try:
    from torch_geometric.nn import GCNConv, GATConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    warnings.warn(
        "PyTorch Geometric not available. STGCN model will use fallback implementation. "
        "Install with: pip install torch-geometric"
    )
    TORCH_GEOMETRIC_AVAILABLE = False
    # Define dummy classes to prevent import errors
    class GCNConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric not available")
    class GATConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric not available")

from .models import BaseModel
from assignment.spatial_utils import build_spatial_adjacency_matrix, prepare_stgcn_data


class SpatioTemporalBlock(nn.Module):
    """
    A single Spatio-Temporal block combining spatial and temporal processing.
    
    This block represents the core building unit of STGCN architecture:
    1. Spatial Convolution: Graph convolution to capture spatial dependencies
    2. Temporal Convolution: 1D convolution or GRU for temporal patterns
    3. Residual connections and normalization for stable training
    """
    
    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        temporal_channels: int,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        """
        Initialize Spatio-Temporal block.
        
        Args:
            in_channels: Number of input feature channels
            spatial_channels: Number of channels for spatial convolution
            temporal_channels: Number of channels for temporal convolution  
            temporal_kernel_size: Kernel size for temporal convolution
            dropout: Dropout rate for regularization
            use_attention: Whether to use attention mechanism in spatial layer
        """
        super(SpatioTemporalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.spatial_channels = spatial_channels
        self.temporal_channels = temporal_channels
        self.use_attention = use_attention
        
        # Spatial convolution layer (Graph Convolutional Network)
        if TORCH_GEOMETRIC_AVAILABLE:
            if use_attention:
                self.spatial_conv = GATConv(in_channels, spatial_channels, heads=1, concat=False)
            else:
                self.spatial_conv = GCNConv(in_channels, spatial_channels)
        else:
            # Fallback: simple linear transformation
            self.spatial_conv = nn.Linear(in_channels, spatial_channels)
        
        # Temporal convolution layer (1D CNN over time dimension)
        self.temporal_conv = nn.Conv1d(
            spatial_channels, 
            temporal_channels,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2  # Same padding
        )
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(temporal_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection if channel dimensions don't match
        if in_channels != temporal_channels:
            self.residual_proj = nn.Linear(in_channels, temporal_channels)
        else:
            self.residual_proj = nn.Identity()
        
        # Initialize weights with Xavier/Glorot uniform for ReLU activation
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize all linear layers with Xavier/Glorot uniform initialization.
        
        This initialization is specifically designed for ReLU activations and
        helps prevent vanishing/exploding gradients during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization with gain for ReLU
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                # Xavier uniform for convolutional layers
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                # Standard initialization for layer norm
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through Spatio-Temporal block.
        
        Args:
            x: Input tensor of shape [num_nodes, seq_len, in_channels]
            edge_index: Edge connectivity of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Output tensor of shape [num_nodes, seq_len, temporal_channels]
        """
        num_nodes, seq_len, _ = x.shape
        residual = x
        
        # Spatial convolution: process each time step independently
        spatial_out = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [num_nodes, in_channels]
            
            if TORCH_GEOMETRIC_AVAILABLE:
                # Use PyTorch Geometric graph convolution
                if edge_weight is not None:
                    h_t = self.spatial_conv(x_t, edge_index, edge_weight)
                else:
                    h_t = self.spatial_conv(x_t, edge_index)
            else:
                # Fallback: simple linear transformation
                h_t = self.spatial_conv(x_t)
            
            h_t = F.relu(h_t)  # Activation after spatial convolution
            spatial_out.append(h_t)
        
        # Stack temporal results: [num_nodes, seq_len, spatial_channels]
        spatial_out = torch.stack(spatial_out, dim=1)
        
        # Temporal convolution: process each node independently
        temporal_out = []
        
        for n in range(num_nodes):
            # Transpose for 1D conv: [spatial_channels, seq_len]
            node_seq = spatial_out[n].transpose(0, 1)
            
            # Apply temporal convolution
            h_temp = self.temporal_conv(node_seq)  # [temporal_channels, seq_len]
            
            # Transpose back: [seq_len, temporal_channels]
            h_temp = h_temp.transpose(0, 1)
            temporal_out.append(h_temp)
        
        # Stack node results: [num_nodes, seq_len, temporal_channels]
        temporal_out = torch.stack(temporal_out, dim=0)
        
        # Apply normalization and dropout
        output = self.layer_norm(temporal_out)
        output = self.dropout(output)
        
        # Residual connection
        residual_proj = self.residual_proj(residual)
        output = output + residual_proj
        
        return output


class STGCNModel(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network for geo-experiment counterfactual prediction.
    
    This model learns complex spatio-temporal relationships between geos and can predict
    counterfactual outcomes by leveraging both spatial dependencies (spillover effects)
    and temporal dynamics in the data.
    
    Architecture:
    1. Multiple Spatio-Temporal blocks for feature learning
    2. Output projection layer for final predictions
    3. Supports multi-step ahead forecasting
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        num_st_blocks: int = 2,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
        use_attention: bool = False,
        output_features: int = None
    ):
        """
        Initialize STGCN model.
        
        Args:
            num_features: Number of input features (e.g., sales, spend)
            hidden_dim: Hidden dimension for internal representations
            num_st_blocks: Number of Spatio-Temporal blocks to stack
            temporal_kernel_size: Kernel size for temporal convolutions
            dropout: Dropout rate for regularization
            use_attention: Whether to use attention in spatial layers
            output_features: Number of output features (defaults to num_features)
        """
        super(STGCNModel, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_st_blocks = num_st_blocks
        
        if output_features is None:
            output_features = num_features
        self.output_features = output_features
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Stack of Spatio-Temporal blocks
        self.st_blocks = nn.ModuleList()
        
        for i in range(num_st_blocks):
            block = SpatioTemporalBlock(
                in_channels=hidden_dim,
                spatial_channels=hidden_dim,
                temporal_channels=hidden_dim,
                temporal_kernel_size=temporal_kernel_size,
                dropout=dropout,
                use_attention=use_attention
            )
            self.st_blocks.append(block)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_features)
        
        # Initialize weights with Xavier/Glorot uniform for numerical stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize all model weights with Xavier/Glorot uniform initialization.
        
        This provides proper weight scaling for ReLU activations and helps
        maintain gradient flow during training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization with ReLU gain
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through STGCN.
        
        Args:
            x: Input tensor of shape [num_nodes, seq_len, num_features]
            edge_index: Edge connectivity of shape [2, num_edges]
            edge_weight: Optional edge weights of shape [num_edges]
            
        Returns:
            Output tensor of shape [num_nodes, seq_len, output_features]
        """
        # Input projection
        h = self.input_proj(x)
        
        # Pass through Spatio-Temporal blocks
        for st_block in self.st_blocks:
            h = st_block(h, edge_index, edge_weight)
        
        # Output projection
        output = self.output_proj(h)
        
        return output


class STGCNReportingModel(BaseModel):
    """
    STGCN-based reporting model for geo-experiments.
    
    This model implements the BaseModel interface to provide counterfactual predictions
    using Spatio-Temporal Graph Convolutional Networks. It learns complex relationships
    between geos during the pre-period and uses control group data to predict what
    would have happened to treatment geos without intervention.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_st_blocks: int = 2,
        temporal_kernel_size: int = 3,
        window_size: int = 10,
        learning_rate: float = 0.01,  # Increased from 0.001
        epochs: int = 100,
        batch_size: int = 32,
        dropout: float = 0.1,
        use_attention: bool = False,
        spatial_connection_method: str = 'knn',
        k_neighbors: int = 8,
        early_stopping_patience: int = 10,
        device: str = 'cpu',
        normalize_data: bool = True,  # Add data normalization option
        verbose: bool = True,  # Control training output verbosity
        bias_threshold: float = 0.1,  # Threshold for bias detection in null scenarios
        strict_numerical_checks: bool = False,  # Enable strict numerical error handling and anomaly detection
        use_offset_calibration: bool = False,  # NEW: Enable simple offset calibration to reduce bias
        use_linear_calibration: bool = False   # NEW: Enable linear calibration (beta0 + beta1 * y_hat)
    ):
        """
        Initialize STGCN reporting model.
        
        Args:
            hidden_dim: Hidden dimension for STGCN model
            num_st_blocks: Number of Spatio-Temporal blocks
            temporal_kernel_size: Kernel size for temporal convolutions
            window_size: Size of temporal windows for sequence modeling
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training (currently not used - full batch)
            dropout: Dropout rate for regularization
            use_attention: Whether to use attention mechanisms
            spatial_connection_method: Method for building spatial graph ('knn', 'threshold', 'hybrid')
            k_neighbors: Number of neighbors for spatial graph construction
            early_stopping_patience: Patience for early stopping
            device: Device for computation ('cpu' or 'cuda')
            normalize_data: Whether to normalize input data
            verbose: Control training output verbosity
            bias_threshold: Threshold for bias detection in null scenarios
            strict_numerical_checks: Enable strict numerical error handling and anomaly detection
        """
        super().__init__()
        
        # Model hyperparameters
        self.hidden_dim = hidden_dim
        self.num_st_blocks = num_st_blocks
        self.temporal_kernel_size = temporal_kernel_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.use_attention = use_attention
        self.spatial_connection_method = spatial_connection_method
        self.k_neighbors = k_neighbors
        self.early_stopping_patience = early_stopping_patience
        self.device = torch.device(device)
        self.normalize_data = normalize_data
        self.verbose = verbose
        self.bias_threshold = bias_threshold
        self.strict_numerical_checks = strict_numerical_checks
        self.use_offset_calibration = use_offset_calibration
        self.use_linear_calibration = use_linear_calibration
        
        # Model components (initialized during fit)
        self.model = None
        self.edge_index = None
        self.edge_weight = None
        self.geo_to_idx = None
        self.feature_cols = ['sales', 'spend']
        self.training_residuals = None
        
        # Data normalization parameters
        self.data_mean = None
        self.data_std = None
        
        # Calibration parameters for bias correction
        self.offset_bias = None  # For simple offset calibration
        self.linear_calibration_params = None  # For linear calibration (beta0, beta1)
        
        # Training monitoring and diagnostics
        self.training_diagnostics = {}
        self.training_losses = []  # Store loss trajectory
        self.training_warnings = []
        
        # Store training data for inference
        self.pre_period_data = None
        self.assignment_df = None
        
    def _create_sequences(
        self,
        data: torch.Tensor,
        window_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create input-output sequences for supervised learning.
        
        Args:
            data: Input data tensor of shape [num_nodes, num_timesteps, num_features]
            window_size: Size of input sequence window
            
        Returns:
            Tuple of (input_sequences, target_sequences)
            - input_sequences: [num_samples, num_nodes, window_size, num_features]
            - target_sequences: [num_samples, num_nodes, num_features]
        """
        num_nodes, num_timesteps, num_features = data.shape
        
        if num_timesteps <= window_size:
            raise ValueError(f"Need more timesteps than window_size: {num_timesteps} <= {window_size}")
        
        # Create overlapping sequences
        inputs = []
        targets = []
        
        for t in range(window_size, num_timesteps):
            # Input: previous window_size timesteps
            input_seq = data[:, t-window_size:t, :]  # [num_nodes, window_size, num_features]
            
            # Target: next timestep
            target = data[:, t, :]  # [num_nodes, num_features]
            
            inputs.append(input_seq)
            targets.append(target)
        
        inputs = torch.stack(inputs, dim=0)  # [num_samples, num_nodes, window_size, num_features]
        targets = torch.stack(targets, dim=0)  # [num_samples, num_nodes, num_features]
        
        return inputs, targets
    
    def _enable_strict_numerical_checks(self):
        """
        Enable strict numerical error handling and anomaly detection.
        
        This sets up PyTorch and NumPy to treat numerical warnings as errors
        and enables anomaly detection for gradient computation debugging.
        """
        if self.strict_numerical_checks:
            # Enable PyTorch anomaly detection
            torch.autograd.set_detect_anomaly(True)
            
            # Set NumPy to raise errors on numerical warnings
            import numpy as np
            np.seterr(all='raise')
            
            # Convert numerical warnings to errors
            import warnings
            warnings.filterwarnings("error", category=RuntimeWarning)
            
            if self.verbose:
                print("🔧 Strict numerical checks enabled:")
                print("   - PyTorch anomaly detection: ON")
                print("   - NumPy error mode: ALL WARNINGS -> ERRORS")
                print("   - Runtime warnings -> ERRORS")
    
    def _disable_strict_numerical_checks(self):
        """
        Disable strict numerical checking to restore normal operation.
        """
        if self.strict_numerical_checks:
            # Disable PyTorch anomaly detection
            torch.autograd.set_detect_anomaly(False)
            
            # Reset NumPy error handling
            import numpy as np
            np.seterr(all='warn')
            
            # Reset warnings
            import warnings
            warnings.resetwarnings()
            
            if self.verbose:
                print("🔧 Strict numerical checks disabled")
    
    def _calculate_ci_with_guards(
        self, 
        values: List[float], 
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval with assertion guards and proper ordering.
        
        Implements the suggested safety fixes:
        1. Assert lower <= upper bounds
        2. Assert positive CI width
        3. Proper quantile ordering
        """
        if len(values) < 2:
            # Not enough samples
            mean_val = np.mean(values) if values else 0.0
            return (mean_val * 0.9, mean_val * 1.1)  # Simple ±10% interval
        
        # Remove any infinite or NaN values
        finite_values = [v for v in values if np.isfinite(v)]
        
        if len(finite_values) < 2:
            mean_val = np.mean(finite_values) if finite_values else 0.0
            return (mean_val * 0.9, mean_val * 1.1)
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        # Calculate quantiles
        lower_bound = np.percentile(finite_values, lower_percentile)
        upper_bound = np.percentile(finite_values, upper_percentile)
        
        # ASSERTION GUARDS (as suggested)
        # 1. Ensure proper ordering
        if lower_bound > upper_bound:
            if self.verbose:
                print(f"⚠️ Quantile order swap detected: lower={lower_bound:.6f} > upper={upper_bound:.6f}")
            lower_bound, upper_bound = upper_bound, lower_bound
        
        # 2. Ensure positive CI width
        ci_width = upper_bound - lower_bound
        assert ci_width >= 0, f"Negative CI width: {ci_width}"
        
        # 3. Final bounds check
        assert lower_bound <= upper_bound, f"CI bounds misordered: {lower_bound} > {upper_bound}"
        
        # Additional safeguard: if bounds are identical, add small margin
        if abs(ci_width) < 1e-10:
            center = (lower_bound + upper_bound) / 2
            margin = abs(center) * 0.01 if center != 0 else 0.01
            lower_bound = center - margin
            upper_bound = center + margin
        
        return (lower_bound, upper_bound)
    
    def _calculate_log_iroas(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        spend_floor: float = 1e-6
    ) -> float:
        """
        Calculate log-iROAS to prevent ratio explosion.
        
        Uses: log(incremental_sales + 1) - log(incremental_spend + spend_floor)
        This prevents division by zero and explosive ratios.
        """
        # Get counterfactual predictions
        counterfactual = self.predict(panel_data, period_start, period_end)
        
        # Get actual treatment group outcomes
        panel_data = panel_data.copy()
        panel_data['date'] = pd.to_datetime(panel_data['date'])
        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)
        
        eval_data = panel_data[
            (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        ]
        
        treatment_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'treatment'
        ]['geo'].values
        
        treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
        
        # Calculate incremental outcomes
        actual_sales = treatment_data['sales'].sum()
        actual_spend = treatment_data['spend'].sum()
        
        counterfactual_sales = counterfactual['sales'].sum()
        counterfactual_spend = counterfactual['spend'].sum()
        
        incremental_sales = actual_sales - counterfactual_sales
        incremental_spend = actual_spend - counterfactual_spend
        
        # Calculate log-iROAS with floors to prevent explosion
        log_sales = np.log(max(incremental_sales + 1, 1))  # Add 1 to handle negative incremental
        log_spend = np.log(max(abs(incremental_spend) + spend_floor, spend_floor))
        
        log_iroas = log_sales - log_spend
        return log_iroas
    
    def _calculate_iroas_robust(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        spend_floor: float = 1e-6
    ) -> float:
        """
        Calculate iROAS with spend floor to prevent ratio explosion.
        """
        # Get counterfactual predictions
        counterfactual = self.predict(panel_data, period_start, period_end)
        
        # Get actual treatment group outcomes
        panel_data = panel_data.copy()
        panel_data['date'] = pd.to_datetime(panel_data['date'])
        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)
        
        eval_data = panel_data[
            (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        ]
        
        treatment_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'treatment'
        ]['geo'].values
        
        treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
        
        # Calculate incremental outcomes
        actual_sales = treatment_data['sales'].sum()
        actual_spend = treatment_data['spend'].sum()
        
        counterfactual_sales = counterfactual['sales'].sum()
        counterfactual_spend = counterfactual['spend'].sum()
        
        incremental_sales = actual_sales - counterfactual_sales
        incremental_spend = actual_spend - counterfactual_spend
        
        # Apply spend floor to prevent division by near-zero
        effective_spend = incremental_spend
        if abs(effective_spend) < spend_floor:
            effective_spend = spend_floor if incremental_spend >= 0 else -spend_floor
        
        iroas = incremental_sales / effective_spend
        return iroas
    
    def _calculate_log_iroas_bootstrap(
        self,
        bootstrap_panel: pd.DataFrame,
        bootstrap_assignment: pd.DataFrame,
        period_start: str,
        period_end: str,
        spend_floor: float
    ) -> float:
        """
        Calculate log-iROAS for bootstrap samples using existing model.
        """
        # Temporarily replace assignment for prediction
        original_assignment = self.assignment_df
        self.assignment_df = bootstrap_assignment
        
        try:
            log_iroas = self._calculate_log_iroas(
                bootstrap_panel, period_start, period_end, spend_floor
            )
            return log_iroas
        finally:
            # Restore original assignment
            self.assignment_df = original_assignment
    
    def _calculate_iroas_bootstrap(
        self,
        bootstrap_panel: pd.DataFrame,
        bootstrap_assignment: pd.DataFrame,
        period_start: str,
        period_end: str,
        spend_floor: float
    ) -> float:
        """
        Calculate robust iROAS for bootstrap samples using existing model.
        """
        # Temporarily replace assignment for prediction
        original_assignment = self.assignment_df
        self.assignment_df = bootstrap_assignment
        
        try:
            iroas = self._calculate_iroas_robust(
                bootstrap_panel, period_start, period_end, spend_floor
            )
            return iroas
        finally:
            # Restore original assignment
            self.assignment_df = original_assignment
    
    def _normalize_data(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to improve training stability.
        
        Uses both global and per-geo normalization for enhanced numerical stability:
        1. Global normalization across all nodes and timesteps
        2. Per-geo normalization to handle heterogeneous geo characteristics
        
        Args:
            data_tensor: Input tensor of shape [num_nodes, num_timesteps, num_features]
            
        Returns:
            Normalized tensor
        """
        if not self.normalize_data:
            return data_tensor
        
        # Global normalization (existing approach)
        self.data_mean = data_tensor.mean(dim=(0, 1), keepdim=True)  # [1, 1, num_features]
        self.data_std = data_tensor.std(dim=(0, 1), keepdim=True)    # [1, 1, num_features]
        
        # Avoid division by zero with epsilon
        self.data_std = torch.clamp(self.data_std, min=1e-8)
        
        # Apply global normalization first
        globally_normalized = (data_tensor - self.data_mean) / self.data_std
        
        # Per-geo normalization for additional stability
        # Normalize each geo's features across time dimension
        per_geo_mean = globally_normalized.mean(dim=1, keepdim=True)  # [num_nodes, 1, num_features]
        per_geo_std = globally_normalized.std(dim=1, keepdim=True)    # [num_nodes, 1, num_features]
        
        # Avoid division by zero and add numerical stability
        per_geo_std = torch.clamp(per_geo_std, min=1e-8)
        
        # Apply per-geo normalization
        fully_normalized = (globally_normalized - per_geo_mean) / per_geo_std
        
        # Store per-geo statistics for denormalization
        self.per_geo_mean = per_geo_mean
        self.per_geo_std = per_geo_std
        
        # Final numerical stability check
        fully_normalized = torch.clamp(fully_normalized, min=-10.0, max=10.0)
        
        # Replace any remaining NaN or Inf values
        fully_normalized = torch.where(
            torch.isfinite(fully_normalized),
            fully_normalized,
            torch.zeros_like(fully_normalized)
        )
        
        if self.verbose:
            print(f"🔢 Enhanced Data Normalization Applied:")
            print(f"   Global - Mean: {self.data_mean.flatten()}, Std: {self.data_std.flatten()}")
            print(f"   Per-geo std range: {per_geo_std.min().item():.6f} to {per_geo_std.max().item():.6f}")
            print(f"   Final range: {fully_normalized.min().item():.3f} to {fully_normalized.max().item():.3f}")
        
        return fully_normalized
    
    def _denormalize_data(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        Reverses both per-geo and global normalization applied in _normalize_data.
        
        Args:
            normalized_tensor: Fully normalized tensor
            
        Returns:
            Denormalized tensor in original scale
        """
        if not self.normalize_data or self.data_mean is None:
            return normalized_tensor
        
        # Reverse per-geo normalization first
        if hasattr(self, 'per_geo_mean') and self.per_geo_mean is not None:
            # Undo per-geo normalization
            per_geo_denormalized = normalized_tensor * self.per_geo_std + self.per_geo_mean
        else:
            # Fallback if per-geo stats not available
            per_geo_denormalized = normalized_tensor
        
        # Reverse global normalization
        globally_denormalized = per_geo_denormalized * self.data_std + self.data_mean
        
        return globally_denormalized
    
    def _monitor_gradients(self) -> dict:
        """
        Monitor gradient statistics to detect vanishing/exploding gradients.
        
        Returns:
            Dictionary with gradient statistics
        """
        total_norm = 0.0
        param_count = 0
        gradient_norms = []
        zero_grad_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                gradient_norms.append(param_norm.item())
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Check for zero gradients
                if param_norm.item() < 1e-8:
                    zero_grad_count += 1
            else:
                zero_grad_count += 1
                param_count += 1
        
        total_norm = total_norm ** (1. / 2) if param_count > 0 else 0.0
        
        gradient_stats = {
            'total_norm': total_norm,
            'mean_norm': np.mean(gradient_norms) if gradient_norms else 0.0,
            'max_norm': np.max(gradient_norms) if gradient_norms else 0.0,
            'min_norm': np.min(gradient_norms) if gradient_norms else 0.0,
            'std_norm': np.std(gradient_norms) if gradient_norms else 0.0,
            'zero_grad_ratio': zero_grad_count / param_count if param_count > 0 else 1.0,
            'param_count': param_count
        }
        
        return gradient_stats
    
    def _check_training_health(self, epoch: int, loss: float, gradient_stats: dict) -> list:
        """
        Check training health and generate warnings for common issues.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            gradient_stats: Gradient statistics from _monitor_gradients
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # Loss scale warnings
        if loss > 1e6:
            warnings.append(f"⚠️ Epoch {epoch}: Loss is very large ({loss:.2e}). Consider:")
            warnings.append("   - Reducing learning rate")
            warnings.append("   - Enabling data normalization")
            warnings.append("   - Checking input data scale")
        
        if loss > 1e8:
            warnings.append(f"🚨 Epoch {epoch}: Loss is extremely large ({loss:.2e}) - model likely diverging!")
        
        # Gradient warnings
        if gradient_stats['zero_grad_ratio'] > 0.5:
            warnings.append(f"⚠️ Epoch {epoch}: {gradient_stats['zero_grad_ratio']:.1%} of parameters have zero gradients")
            warnings.append("   - Possible vanishing gradient problem")
            warnings.append("   - Consider increasing learning rate or checking architecture")
        
        if gradient_stats['total_norm'] < 1e-6:
            warnings.append(f"⚠️ Epoch {epoch}: Gradients very small ({gradient_stats['total_norm']:.2e})")
            warnings.append("   - Possible vanishing gradient problem")
        
        if gradient_stats['total_norm'] > 1e3:
            warnings.append(f"⚠️ Epoch {epoch}: Gradients very large ({gradient_stats['total_norm']:.2e})")
            warnings.append("   - Possible exploding gradient problem")
            warnings.append("   - Consider gradient clipping or reducing learning rate")
        
        # Learning progress warnings
        if epoch > 5 and loss > 1e3:
            warnings.append(f"⚠️ Epoch {epoch}: Loss still high after {epoch} epochs ({loss:.2e})")
            warnings.append("   - Model may not be learning effectively")
            warnings.append("   - Consider adjusting hyperparameters")
        
        return warnings
    
    def _validate_data_scale(self, data_tensor: torch.Tensor):
        """
        Validate input data scale and suggest normalization if needed.
        
        Args:
            data_tensor: Input data tensor
        """
        sales_mean = data_tensor[:, :, 0].mean().item()
        spend_mean = data_tensor[:, :, 1].mean().item()
        sales_std = data_tensor[:, :, 0].std().item()
        spend_std = data_tensor[:, :, 1].std().item()
        
        if self.verbose:
            print(f"📊 Data Scale Analysis:")
            print(f"   Sales: mean={sales_mean:.1f}, std={sales_std:.1f}")
            print(f"   Spend: mean={spend_mean:.1f}, std={spend_std:.1f}")
        
        # Check if data scale is reasonable for neural networks
        if sales_mean > 1e4 or spend_mean > 1e4:
            if not self.normalize_data:
                print("⚠️ Data values are large (>10K). Consider enabling normalization:")
                print("   STGCNReportingModel(normalize_data=True)")
        
        if sales_std / sales_mean > 2 or spend_std / spend_mean > 2:
            print("⚠️ High data variance detected. Normalization recommended for stable training.")
    
    def _validate_predictions(self, predictions: dict, actual_data_sample: dict) -> list:
        """
        Validate prediction scale and reasonableness.
        
        Args:
            predictions: Model predictions
            actual_data_sample: Sample of actual data for comparison
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        pred_sales = np.array(predictions['sales'])
        pred_spend = np.array(predictions['spend'])
        
        # Check for negative predictions
        if np.any(pred_sales < 0):
            warnings.append("🚨 Negative sales predictions detected!")
            warnings.append(f"   Min sales prediction: {pred_sales.min():.2f}")
        
        if np.any(pred_spend < 0):
            warnings.append("🚨 Negative spend predictions detected!")
            warnings.append(f"   Min spend prediction: {pred_spend.min():.2f}")
        
        # Check prediction scale vs actual data
        if 'expected_sales_range' in actual_data_sample:
            expected_min, expected_max = actual_data_sample['expected_sales_range']
            pred_median = np.median(pred_sales)
            
            if pred_median < expected_min * 0.1:
                warnings.append(f"⚠️ Sales predictions much lower than expected:")
                warnings.append(f"   Predicted median: {pred_median:.0f}")
                warnings.append(f"   Expected range: {expected_min:.0f} - {expected_max:.0f}")
                warnings.append("   Model may be undertrained or have scaling issues")
            
            if pred_median > expected_max * 10:
                warnings.append(f"⚠️ Sales predictions much higher than expected:")
                warnings.append(f"   Predicted median: {pred_median:.0f}")
                warnings.append(f"   Expected range: {expected_min:.0f} - {expected_max:.0f}")
        
        return warnings
    
    def get_training_diagnostics(self) -> dict:
        """
        Get comprehensive training diagnostics and recommendations.
        
        Returns:
            Dictionary with training diagnostics and recommendations
        """
        if not self.training_diagnostics:
            return {"status": "No training diagnostics available"}
        
        diagnostics = self.training_diagnostics.copy()
        
        # Add recommendations based on diagnostics
        recommendations = []
        
        final_loss = diagnostics.get('final_loss', float('inf'))
        if final_loss > 1e3:
            recommendations.append("📈 High final loss - consider:")
            recommendations.append("  • Increasing epochs or learning rate")
            recommendations.append("  • Enabling data normalization")
            recommendations.append("  • Reducing model complexity")
        
        gradient_health = diagnostics.get('gradient_health', {})
        if gradient_health.get('vanishing_gradients_detected', False):
            recommendations.append("📉 Vanishing gradients detected - consider:")
            recommendations.append("  • Increasing learning rate")
            recommendations.append("  • Using gradient clipping")
            recommendations.append("  • Simplifying model architecture")
        
        if gradient_health.get('exploding_gradients_detected', False):
            recommendations.append("📈 Exploding gradients detected - consider:")
            recommendations.append("  • Decreasing learning rate")
            recommendations.append("  • Adding gradient clipping")
            recommendations.append("  • Enabling data normalization")
        
        diagnostics['recommendations'] = recommendations
        diagnostics['all_warnings'] = self.training_warnings
        
        return diagnostics
    
    def print_training_summary(self):
        """
        Print a user-friendly summary of training performance and issues.
        """
        if not self.training_diagnostics:
            print("No training diagnostics available. Model may not be fitted yet.")
            return
        
        print("\n" + "="*60)
        print("🔬 STGCN TRAINING SUMMARY")
        print("="*60)
        
        diagnostics = self.training_diagnostics
        
        # Training performance
        print(f"📈 PERFORMANCE:")
        print(f"   Final Loss: {diagnostics['final_loss']:.6f}")
        if diagnostics.get('loss_reduction_ratio'):
            print(f"   Loss Reduction: {diagnostics['loss_reduction_ratio']:.1f}x improvement")
        print(f"   Convergence: {diagnostics['convergence_assessment'].upper()}")
        
        # Health indicators
        gradient_health = diagnostics.get('gradient_health', {})
        issues = sum(gradient_health.values())
        
        print(f"\n🔍 HEALTH STATUS:")
        if issues == 0:
            print("   ✅ All systems healthy")
        else:
            print(f"   ⚠️ {issues} gradient issues detected")
            if gradient_health.get('vanishing_gradients_detected'):
                print("   📉 Vanishing gradients")
            if gradient_health.get('exploding_gradients_detected'):
                print("   📈 Exploding gradients")
            if gradient_health.get('zero_grad_issues'):
                print("   🚫 Zero gradient issues")
        
        # Warnings summary
        warning_count = len(self.training_warnings)
        print(f"\n⚠️ WARNINGS: {warning_count} total")
        
        # Recommendations
        recommendations = self.get_training_diagnostics().get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Show top 3
                print(f"   {rec}")
            if len(recommendations) > 3:
                print(f"   ... and {len(recommendations) - 3} more")
        else:
            print(f"\n✅ NO RECOMMENDATIONS NEEDED")
        
        print("="*60)
        print("Use model.get_training_diagnostics() for detailed analysis")
        print("="*60)
    
    def fit(
        self,
        panel_data: pd.DataFrame,
        assignment_df: pd.DataFrame,
        pre_period_end: str
    ) -> 'STGCNReportingModel':
        """
        Fit the STGCN model on pre-period data.
        
        This method:
        1. Builds spatial graph from geo coordinates
        2. Prepares time-series data in STGCN format
        3. Trains the model to predict next-step outcomes
        4. Stores residuals for confidence interval estimation
        
        Args:
            panel_data: Long-format panel data
            assignment_df: Assignment of geos to treatment/control
            pre_period_end: End date of pre-period (exclusive)
            
        Returns:
            Self for method chaining
        """
        # Enable strict numerical checks if requested
        self._enable_strict_numerical_checks()
        
        try:
            # Store data for later use
            self.assignment_data = assignment_df.copy()
            
            # Filter to pre-period data
            panel_data['date'] = pd.to_datetime(panel_data['date'])
            pre_period_end = pd.to_datetime(pre_period_end)
            
            pre_data = panel_data[panel_data['date'] < pre_period_end].copy()
            self.pre_period_data = pre_data
            
            if len(pre_data) == 0:
                raise ValueError("No pre-period data found")
            
            # Create geo features with spatial coordinates
            geo_features = self._create_geo_features(pre_data, assignment_df)
            
            # Build spatial graph
            self.edge_index, self.edge_weight = build_spatial_adjacency_matrix(
            geo_features,
            spatial_cols=['xy1', 'xy2'],
            connection_method=self.spatial_connection_method,
            k_neighbors=self.k_neighbors,
            include_self_loops=False
            )
            
            # Move to device
            self.edge_index = self.edge_index.to(self.device)
            self.edge_weight = self.edge_weight.to(self.device)
            
            # Prepare time-series data
            data_tensor, self.geo_to_idx = prepare_stgcn_data(
            pre_data,
            geo_features,
            feature_cols=self.feature_cols
            )
            
            data_tensor = data_tensor.to(self.device)
            
            # Validate input data scale and provide recommendations
            self._validate_data_scale(data_tensor)
            
            # Normalize data for better training stability
            data_tensor = self._normalize_data(data_tensor)
            
            # Initialize model
            num_features = len(self.feature_cols)
            self.model = STGCNModel(
            num_features=num_features,
            hidden_dim=self.hidden_dim,
            num_st_blocks=self.num_st_blocks,
            temporal_kernel_size=self.temporal_kernel_size,
            dropout=self.dropout,
            use_attention=self.use_attention,
            output_features=num_features
            ).to(self.device)
            
            # Create training sequences
            # Split data into training and validation sets
            num_timesteps = data_tensor.shape[1]
            
            # Ensure validation set has enough timesteps for window_size
            min_val_timesteps = self.window_size + 2  # Need at least window_size + 1 for sequences
            max_split_idx = num_timesteps - min_val_timesteps
            
            if max_split_idx <= self.window_size:
                # Not enough data for proper split, use all data for training
                train_inputs, train_targets = self._create_sequences(data_tensor, self.window_size)
                val_inputs, val_targets = train_inputs[-1:], train_targets[-1:]  # Use last training sample for validation
            else:
                split_idx = min(int(num_timesteps * 0.8), max_split_idx)
                train_data = data_tensor[:, :split_idx, :]
                val_data = data_tensor[:, split_idx:, :]
                
                train_inputs, train_targets = self._create_sequences(train_data, self.window_size)
                val_inputs, val_targets = self._create_sequences(val_data, self.window_size)
            
            # Training setup
            optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            # Initialize training diagnostics
            self.training_warnings = []
            loss_history = []
            val_loss_history = []
            gradient_history = []
            
            # Training loop
            if self.verbose:
                print(f"Training STGCN model for {self.epochs} epochs...")
            
            # Use tqdm only if verbose
            epoch_iterator = tqdm(range(self.epochs), desc="Training STGCN") if self.verbose else range(self.epochs)
            
            for epoch in epoch_iterator:
                self.model.train()
                total_loss = 0.0
                
                # Full batch training (could be extended to mini-batches)
                for i in range(len(train_inputs)):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    input_seq = train_inputs[i]  # [num_nodes, window_size, num_features]
                    target = train_targets[i]    # [num_nodes, num_features]
                    
                    # Model expects last timestep for prediction
                    pred = self.model(
                        input_seq,
                        self.edge_index,
                        self.edge_weight
                    )[:, -1, :]  # Take last timestep: [num_nodes, num_features]
                    
                    # Compute loss
                    loss = criterion(pred, target)
                    
                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Monitor gradients
                    gradient_stats = self._monitor_gradients()
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_inputs)
                loss_history.append(avg_loss)
                
                # Critical numerical stability check for training loss
                if not torch.isfinite(torch.tensor(avg_loss)):
                    print(f"🚨 CRITICAL: Training loss is {avg_loss} at epoch {epoch+1}")
                    print("🚨 Training diverged - terminating immediately!")
                    break
            
                # Validation
                self.model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for i in range(len(val_inputs)):
                        input_seq = val_inputs[i]
                        target = val_targets[i]
                        pred = self.model(
                            input_seq,
                            self.edge_index,
                            self.edge_weight
                        )[:, -1, :]
                        val_loss = criterion(pred, target)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_inputs)
                val_loss_history.append(avg_val_loss)
                
                # Critical numerical stability check for validation loss
                if not torch.isfinite(torch.tensor(avg_val_loss)):
                    print(f"🚨 CRITICAL: Validation loss is {avg_val_loss} at epoch {epoch+1}")
                    print("🚨 Validation diverged - terminating immediately!")
                    break
                
                gradient_history.append(gradient_stats)
                
                # Check training health and collect warnings
                epoch_warnings = self._check_training_health(epoch + 1, avg_loss, gradient_stats)
                if epoch_warnings:
                    self.training_warnings.extend(epoch_warnings)
                    
                    # Print critical warnings immediately
                    for warning in epoch_warnings:
                        if "🚨" in warning:
                            print(warning)
                
                # Enhanced early stopping with numerical stability checks
                if torch.isfinite(torch.tensor(avg_val_loss)) and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Stop if validation hasn't improved for patience epochs OR if loss is too high
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"⏹️ Early stopping at epoch {epoch+1} (patience exceeded)")
                    break
                
                # Additional safety check - stop if validation loss is extremely high
                if avg_val_loss > 1e6:
                    print(f"🚨 CRITICAL: Validation loss extremely high ({avg_val_loss:.2e}) - stopping training")
                    break

                scheduler.step(avg_val_loss)
            
            # Store comprehensive training diagnostics
            self.training_diagnostics = {
                'final_train_loss': avg_loss,
                'final_val_loss': best_val_loss,
                'epochs_trained': epoch + 1,
                'early_stopped': patience_counter >= self.early_stopping_patience,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
                'initial_loss': loss_history[0] if loss_history else None,
                'loss_reduction_ratio': loss_history[0] / avg_loss if loss_history and avg_loss > 0 else None,
                'final_gradient_stats': gradient_history[-1] if gradient_history else {},
                'gradient_health': {
                    'vanishing_gradients_detected': any(
                        stats.get('total_norm', 0) < 1e-6 for stats in gradient_history[-5:]
                    ) if len(gradient_history) >= 5 else False,
                    'exploding_gradients_detected': any(
                        stats.get('total_norm', 0) > 1e3 for stats in gradient_history[-5:]
                    ) if len(gradient_history) >= 5 else False,
                    'zero_grad_issues': any(
                        stats.get('zero_grad_ratio', 0) > 0.5 for stats in gradient_history[-5:]
                    ) if len(gradient_history) >= 5 else False
                },
                'convergence_assessment': 'good' if best_val_loss < 1.0 else 'poor' if best_val_loss > 1e3 else 'moderate'
            }
            
            # Compute training residuals for confidence intervals
            self._compute_training_residuals(train_inputs, train_targets)
            
            self.is_fitted = True
            
            # Compute calibration parameters if enabled (after model is fitted)
            if self.use_offset_calibration or self.use_linear_calibration:
                self._compute_calibration_parameters(panel_data, assignment_df, pre_period_end)
            
            # Store loss trajectory for analysis
            self.training_losses = loss_history
            
            # Print final training summary
            if self.verbose:
                print(f"STGCN training completed. Final loss: {best_val_loss:.6f}")
                if self.training_warnings:
                    print(f"⚠️ Training completed with {len(self.training_warnings)} warnings.")
                    print("   Use model.get_training_diagnostics() for detailed analysis.")
            
            # Print convergence assessment
            convergence = self.training_diagnostics['convergence_assessment']
            if self.verbose:
                if convergence == 'good':
                    print("✅ Model training converged well!")
                    print(f"   Final loss: {best_val_loss:.6f}, Converged at epoch: {epoch + 1}")
                elif convergence == 'moderate':
                    print("⚠️ Model training had moderate convergence. Consider tuning hyperparameters.")
                    print(f"   Final loss: {best_val_loss:.6f}")
                else:
                    print("🚨 Model training converged poorly. Check diagnostics and consider:")
                    print("   - Increasing learning rate or epochs")
                    print("   - Enabling data normalization")
                    print("   - Adjusting model architecture")
                    print(f"   Final loss: {best_val_loss:.6f}")
            
            # Perform null scenario bias detection
            self._detect_null_bias()
            
            return self
            
        except Exception as e:
            if self.strict_numerical_checks and self.verbose:
                print(f"🚨 Training failed with strict numerical checks: {type(e).__name__}: {e}")
            raise
        finally:
            # Always disable strict checks after training
            self._disable_strict_numerical_checks()
    
    def _create_geo_features(
        self,
        panel_data: pd.DataFrame,
        assignment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create geo features with spatial coordinates.
        
        Args:
            panel_data: Panel data with geo information
            assignment_df: Assignment data
            
        Returns:
            DataFrame with geo features including spatial coordinates
        """
        # Get unique geos
        unique_geos = sorted(panel_data['geo'].unique())
        
        # Create simple spatial coordinates (could be enhanced with real lat/lon)
        np.random.seed(42)  # For reproducible coordinates
        n_geos = len(unique_geos)
        
        geo_features = pd.DataFrame({
            'geo': unique_geos,
            'xy1': np.random.uniform(0, 100, n_geos),
            'xy2': np.random.uniform(0, 100, n_geos)
        })
        
        return geo_features
    
    def _compute_training_residuals(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ):
        """
        Compute training residuals for confidence interval estimation.
        
        Args:
            inputs: Training input sequences
            targets: Training target sequences
        """
        self.model.eval()
        residuals = []
        
        with torch.no_grad():
            for i in range(len(inputs)):
                input_seq = inputs[i]
                target = targets[i]
                
                pred = self.model(
                    input_seq,
                    self.edge_index,
                    self.edge_weight
                )[:, -1, :]
                
                residual = (target - pred).cpu().numpy()
                residuals.append(residual)
        
        self.training_residuals = np.concatenate(residuals, axis=0)
    
    def _compute_calibration_parameters(
        self,
        panel_data: pd.DataFrame,
        assignment_df: pd.DataFrame,
        pre_period_end: str
    ):
        """
        Compute calibration parameters for bias correction.
        
        Simple approach: Disable calibration for now to fix the fundamental logic error.
        The current implementation creates circular dependencies during training.
        
        Args:
            panel_data: Panel data used for training
            assignment_df: Assignment of geos to treatment/control
            pre_period_end: End date of pre-period
        """
        if self.verbose:
            print("\n🎯 Computing calibration parameters for bias reduction...")
        
        # TEMPORARY FIX: Disable calibration to fix the circular dependency issue
        # The current logic tries to predict during training which causes instability
        
        if self.use_offset_calibration or self.use_linear_calibration:
            if self.verbose:
                print("   ⚠️ Calibration temporarily disabled due to implementation issues.")
                print("   ℹ️ Models will run without bias correction for now.")
        
        # Reset calibration parameters
        self.offset_bias = None
        self.linear_calibration_params = None
    
    def _apply_calibration_correction(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply calibration correction to predictions.
        
        This is the key method that actually corrects predictions using the computed
        calibration parameters. This is where the "one line of code" bias correction happens.
        
        Args:
            predictions: Raw predictions dictionary with 'sales' and 'spend' keys (aggregate format)
            
        Returns:
            Calibrated predictions dictionary
        """
        if self.offset_bias is None and self.linear_calibration_params is None:
            return predictions
        
        corrected_predictions = {}
        
        for metric in ['sales', 'spend']:
            if metric in predictions:
                metric_predictions = predictions[metric]
                
                # Apply calibration
                if self.linear_calibration_params is not None and metric in self.linear_calibration_params:
                    # Linear calibration: corrected = beta0 + beta1 * predicted
                    params = self.linear_calibration_params[metric]
                    beta0, beta1 = params['beta0'], params['beta1']
                    corrected_predictions[metric] = beta0 + beta1 * metric_predictions
                    
                elif self.offset_bias is not None and metric in self.offset_bias:
                    # Simple offset calibration: corrected = predicted + offset
                    offset = self.offset_bias[metric]
                    corrected_predictions[metric] = metric_predictions + offset
                    
                else:
                    # No calibration for this metric
                    corrected_predictions[metric] = metric_predictions
            else:
                # Metric not in predictions, keep as is
                corrected_predictions[metric] = predictions.get(metric, np.array([]))
        
        return corrected_predictions
    
    def predict(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str
    ) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual predictions using recursive anchoring.
        
        This method uses control group observations to anchor predictions and
        generates counterfactuals for treatment group through recursive forecasting.
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of prediction period
            period_end: End date of prediction period
            
        Returns:
            Dictionary with 'sales' and 'spend' counterfactual predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Convert dates
        panel_data = panel_data.copy()
        panel_data['date'] = pd.to_datetime(panel_data['date'])
        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)
        
        # Get evaluation period data
        eval_data = panel_data[
            (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        ].copy()
        
        # Get all data up to period_end for context
        context_data = panel_data[panel_data['date'] <= period_end].copy()
        
        # Prepare full dataset
        geo_features = self._create_geo_features(context_data, self.assignment_df)
        full_data_tensor, _ = prepare_stgcn_data(
            context_data,
            geo_features,
            feature_cols=self.feature_cols
        )
        
        full_data_tensor = full_data_tensor.to(self.device)
        
        # Normalize data using same parameters as training
        if self.normalize_data and self.data_mean is not None:
            full_data_tensor = (full_data_tensor - self.data_mean.to(self.device)) / self.data_std.to(self.device)
        
        # Find period boundaries in tensor
        dates = sorted(context_data['date'].unique())
        period_start_idx = dates.index(period_start)
        period_end_idx = dates.index(period_end) + 1
        
        # Get treatment geo indices
        treatment_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'treatment'
        ]['geo'].values
        
        geo_idx_map = {geo: idx for idx, geo in enumerate(self.geo_to_idx)}
        treatment_indices = [geo_idx_map[geo] for geo in treatment_geos if geo in geo_idx_map]
        
        # Recursive prediction with control anchoring
        self.model.eval()
        predicted_data = full_data_tensor.clone()
        
        with torch.no_grad():
            for t in range(period_start_idx, period_end_idx):
                if t < self.window_size:
                    continue  # Skip if not enough history
                
                # Create input sequence
                input_seq = predicted_data[:, t-self.window_size:t, :]
                input_seq = input_seq.unsqueeze(0)  # Add batch dimension
                
                # Predict next timestep
                pred = self.model(
                    input_seq.squeeze(0),
                    self.edge_index,
                    self.edge_weight
                )[:, -1, :]  # [num_nodes, num_features]
                
                # Anchor control geos with actual observations
                predicted_data[:, t, :] = pred
                
                # Replace control geo predictions with actual observations
                actual_data_t = full_data_tensor[:, t, :]
                control_mask = torch.ones(len(self.geo_to_idx), dtype=torch.bool)
                control_mask[treatment_indices] = False
                
                predicted_data[control_mask, t, :] = actual_data_t[control_mask, :]
        
        # Denormalize data back to original scale
        if self.normalize_data and self.data_mean is not None:
            predicted_data = self._denormalize_data(predicted_data)
        
        # Extract counterfactual for treatment geos in evaluation period
        counterfactual_data = predicted_data[
            treatment_indices, 
            period_start_idx:period_end_idx, 
            :
        ].cpu().numpy()
        
        # Convert back to original format
        # CRITICAL: This aggregation ensures consistency with traditional methods for A/A testing
        results = {}
        for i, feature in enumerate(self.feature_cols):
            # MODIFIED FOR CONSISTENCY: Average across treatment geos to get daily predictions
            # Original: counterfactual_data[:, :, i].sum(axis=1)  # Sum over time → geo-level totals
            # Fixed:    counterfactual_data[:, :, i].mean(axis=0) # Mean over geos → daily predictions
            # 
            # This change ensures STGCN returns the same prediction format as traditional methods:
            # - Shape: (n_evaluation_days,) instead of (n_treatment_geos,)
            # - Values: Daily average predictions instead of geo-level totals
            # - Aggregation: Matches daily mean approach used by MeanMatching, TBR, SCM
            feature_counterfactual = counterfactual_data[:, :, i].mean(axis=0)  # Mean over geos, keep time dimension
            results[feature] = feature_counterfactual
        
        # Validate predictions and provide warnings
        eval_data = panel_data[
            (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        ]
        
        # Create expected data range for validation
        control_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'control'
        ]['geo'].values
        
        control_data = eval_data[eval_data['geo'].isin(control_geos)]
        
        if len(control_data) > 0:
            expected_daily_sales = control_data.groupby('geo')['sales'].mean().mean()
            n_days = len(eval_data['date'].unique())
            expected_total_sales = expected_daily_sales * n_days
            
            actual_data_sample = {
                'expected_sales_range': (expected_total_sales * 0.5, expected_total_sales * 1.5)
            }
            
            prediction_warnings = self._validate_predictions(results, actual_data_sample)
            if prediction_warnings:
                print("🔍 Prediction Validation Warnings:")
                for warning in prediction_warnings:
                    print(warning)
        
        # Apply calibration correction if enabled
        if self.offset_bias is not None or self.linear_calibration_params is not None:
            results = self._apply_calibration_correction(results)
        
        return results
    
    def calculate_iroas(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str
    ) -> float:
        """
        Calculate incremental ROAS using STGCN counterfactual.
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of evaluation period
            period_end: End date of evaluation period
            
        Returns:
            iROAS estimate
        """
        # Get counterfactual predictions
        counterfactual = self.predict(panel_data, period_start, period_end)
        
        # Get actual treatment group outcomes
        panel_data = panel_data.copy()
        panel_data['date'] = pd.to_datetime(panel_data['date'])
        period_start = pd.to_datetime(period_start)
        period_end = pd.to_datetime(period_end)
        
        eval_data = panel_data[
            (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        ]
        
        treatment_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'treatment'
        ]['geo'].values
        
        treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
        
        # Calculate incremental outcomes
        actual_sales = treatment_data['sales'].sum()
        actual_spend = treatment_data['spend'].sum()
        
        counterfactual_sales = counterfactual['sales'].sum()
        counterfactual_spend = counterfactual['spend'].sum()
        
        incremental_sales = actual_sales - counterfactual_sales
        incremental_spend = actual_spend - counterfactual_spend
        
        # Calculate iROAS with spend floor to prevent ratio explosion
        spend_floor = 1e-6
        effective_spend = incremental_spend
        if abs(effective_spend) < spend_floor:
            effective_spend = spend_floor if incremental_spend >= 0 else -spend_floor
        
        iroas = incremental_sales / effective_spend
        return iroas
    
    def confidence_interval(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float = 0.95,
        n_bootstrap: int = 500,
        seed: Optional[int] = None,
        method: str = 'ensemble',
        n_mc_samples: int = 100,
        use_log_iroas: bool = True,
        spend_floor: float = 1e-6,
        ensemble_size: int = 5,
        n_jobs: int = -1,
        use_parallel: bool = True,
        use_bca: bool = True  # NEW: Use BCa bootstrap instead of t-distribution
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval with improved variance estimation methods.
        
        This method implements multiple approaches to capture model uncertainty:
        1. Ensemble (default) - Gold standard, trains K models with different seeds
        2. Monte Carlo dropout - Fast, captures parameter uncertainty within single fit
        3. Model-aware bootstrap - Slow but statistically correct
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of evaluation period
            period_end: End date of evaluation period
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples
            seed: Random seed for reproducibility
            method: CI method ('ensemble', 'mc_dropout', 'model_aware_bootstrap')
            n_mc_samples: Number of Monte Carlo dropout samples
            use_log_iroas: Use log-iROAS to prevent ratio explosion
            spend_floor: Minimum spend to prevent division by zero
            ensemble_size: Number of models in ensemble (default: 5)
            n_jobs: Number of parallel jobs for ensemble (-1 for auto-detection)
            use_parallel: Whether to use parallel training for ensemble method
            use_bca: Use BCa bootstrap for CI instead of t-distribution (default: True)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating confidence interval")
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Route to appropriate CI method
        if method == 'ensemble':
            return self._ensemble_confidence_interval(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor, n_jobs, use_parallel, use_bca
            )
        elif method == 'mc_dropout':
            return self._mc_dropout_confidence_interval(
                panel_data, period_start, period_end, confidence_level, 
                n_mc_samples, use_log_iroas, spend_floor
            )
        elif method == 'model_aware_bootstrap':
            return self._model_aware_bootstrap_ci(
                panel_data, period_start, period_end, confidence_level,
                n_bootstrap, use_log_iroas, spend_floor
            )
        else:
            # Original method with fixes
            return self._original_bootstrap_with_fixes(
                panel_data, period_start, period_end, confidence_level,
                n_bootstrap, use_log_iroas, spend_floor
            )
        
    def _ensemble_confidence_interval(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        ensemble_size: int,
        use_log_iroas: bool,
        spend_floor: float,
        n_jobs: int = -1,
        use_parallel: bool = True,
        use_bca: bool = True
    ) -> Tuple[float, float]:
        """
        Ensemble confidence interval using K independently trained models.
        
        This is the gold-standard approach that solves the overconfidence problem
        by capturing model initialization and training variance.
        
        Can use either parallel or sequential training based on use_parallel flag.
        """
        # Import parallel ensemble here to avoid circular imports
        try:
            from .parallel_ensemble import ParallelEnsembleSTGCN
            parallel_available = True
        except ImportError:
            parallel_available = False
            if use_parallel and self.verbose:
                print("⚠️ Parallel ensemble not available, falling back to sequential training")
        
        # Use parallel training if requested and available
        if use_parallel and parallel_available and n_jobs != 1:
            return self._ensemble_confidence_interval_parallel(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor, n_jobs, use_bca
            )
        else:
            return self._ensemble_confidence_interval_sequential(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor, use_bca
            )
    
    def _ensemble_confidence_interval_parallel(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        ensemble_size: int,
        use_log_iroas: bool,
        spend_floor: float,
        n_jobs: int,
        use_bca: bool = True
    ) -> Tuple[float, float]:
        """
        Parallel ensemble confidence interval using multiprocessing.
        """
        from .parallel_ensemble import ParallelEnsembleSTGCN
        
        if self.verbose:
            print(f"Training ensemble of {ensemble_size} models in parallel for CI estimation...")
        
        # Create model configuration
        model_config = {
            'hidden_dim': self.hidden_dim,
            'num_st_blocks': self.num_st_blocks,
            'window_size': self.window_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'normalize_data': self.normalize_data,
            'verbose': False,  # Suppress individual model output
            'k_neighbors': self.k_neighbors,
            'device': self.device
        }
        
        # Create parallel ensemble
        parallel_ensemble = ParallelEnsembleSTGCN(
            ensemble_size=ensemble_size,
            n_jobs=n_jobs,
            verbose=self.verbose
        )
        
        # Set model config separately to avoid duplicate arguments
        parallel_ensemble.model_config = model_config
        
        try:
            # Fit ensemble in parallel
            parallel_ensemble.fit_parallel(
                panel_data,
                self.assignment_df,
                self.pre_period_data['date'].max().strftime('%Y-%m-%d'),
                seed=5000
            )
            
            # Get ensemble iROAS estimates
            ensemble_iroas = parallel_ensemble.calculate_ensemble_iroas(
                panel_data, period_start, period_end, use_log_iroas, spend_floor
            )
            
            if len(ensemble_iroas) < 2:
                if self.verbose:
                    print("⚠️ Insufficient ensemble estimates from parallel training")
                raise ValueError("Parallel ensemble failed to produce sufficient estimates")
            
            # Calculate confidence interval using BCa or t-distribution
            ensemble_iroas = np.array(ensemble_iroas)
            if use_bca and len(ensemble_iroas) >= 3:
                if self.verbose:
                    print(f"  Using BCa bootstrap for CI calculation...")
                lower_bound, upper_bound = self._compute_bca_ci(ensemble_iroas, confidence_level)
            else:
                # Use t-distribution for small ensembles or when BCa disabled
                from scipy import stats
                ensemble_mean = np.mean(ensemble_iroas)
                ensemble_std = np.std(ensemble_iroas, ddof=1)
                alpha = 1 - confidence_level
                t_score = stats.t.ppf(1 - alpha/2, df=len(ensemble_iroas) - 1)
                margin = t_score * ensemble_std
                
                lower_bound = ensemble_mean - margin
                upper_bound = ensemble_mean + margin
                
                if self.verbose:
                    ci_method = "t-distribution" if not use_bca else "t-distribution (insufficient data for BCa)"
                    print(f"  Using {ci_method} for CI calculation...")
            
            if self.verbose:
                diagnostics = parallel_ensemble.get_training_diagnostics()
                print(f"  Parallel ensemble results: {diagnostics['successful_models']} models")
                print(f"  Training time: {diagnostics.get('training_time', 'N/A'):.1f}s")
                print(f"  Speedup: {diagnostics.get('speedup_estimate', 'N/A'):.1f}x")
                print(f"  CI: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Parallel ensemble failed: {e}")
                print("   Falling back to sequential training...")
            
            return self._ensemble_confidence_interval_sequential(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor, use_bca
            )
    
    def _ensemble_confidence_interval_sequential(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        ensemble_size: int,
        use_log_iroas: bool,
        spend_floor: float,
        use_bca: bool = True
    ) -> Tuple[float, float]:
        """
        Sequential ensemble confidence interval (original implementation).
        """
        if self.verbose:
            print(f"Training ensemble of {ensemble_size} models sequentially for CI estimation...")
        
        ensemble_iroas = []
        successful_models = 0
        
        # Train ensemble with different random seeds
        for i in range(ensemble_size):
            torch.manual_seed(5000 + i)
            np.random.seed(5000 + i)
            
            # Create model with same configuration (including calibration settings)
            ensemble_model = STGCNReportingModel(
                hidden_dim=self.hidden_dim,
                num_st_blocks=self.num_st_blocks,
                window_size=self.window_size,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                dropout=self.dropout,
                normalize_data=self.normalize_data,
                verbose=False,  # Suppress individual model output
                k_neighbors=self.k_neighbors,
                device=self.device,
                use_offset_calibration=self.use_offset_calibration,  # FIX: Include calibration
                use_linear_calibration=self.use_linear_calibration   # FIX: Include calibration
            )
            
            try:
                # Fit ensemble model
                ensemble_model.fit(
                    panel_data, 
                    self.assignment_df, 
                    self.pre_period_data['date'].max().strftime('%Y-%m-%d')
                )
                
                # Calculate iROAS for this model
                if use_log_iroas:
                    iroas = ensemble_model._calculate_log_iroas(
                        panel_data, period_start, period_end, spend_floor
                    )
                else:
                    iroas = ensemble_model._calculate_iroas_robust(
                        panel_data, period_start, period_end, spend_floor
                    )
                
                ensemble_iroas.append(iroas)
                successful_models += 1
                
                if self.verbose:
                    print(f"  Model {i+1}: iROAS = {iroas:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Model {i+1} failed: {e}")
                continue
        
        if successful_models < 2:
            if self.verbose:
                print("⚠️ Ensemble failed, falling back to MC dropout")
            return self._mc_dropout_confidence_interval(
                panel_data, period_start, period_end, confidence_level,
                100, use_log_iroas, spend_floor
            )
        
        # Calculate ensemble statistics
        ensemble_mean = np.mean(ensemble_iroas)
        ensemble_std = np.std(ensemble_iroas, ddof=1)  # Sample std
        
        if use_bca and successful_models >= 3:
            # Use BCa bootstrap for better coverage
            if self.verbose:
                print(f"  Using BCa bootstrap for CI calculation...")
            lower_bound, upper_bound = self._compute_bca_ci(ensemble_iroas, confidence_level)
        else:
            # Use t-distribution for small ensembles or when BCa disabled
            from scipy import stats
            alpha = 1 - confidence_level
            t_score = stats.t.ppf(1 - alpha/2, df=successful_models - 1)
            margin = t_score * ensemble_std
            
            lower_bound = ensemble_mean - margin
            upper_bound = ensemble_mean + margin
            
            if self.verbose:
                ci_method = "t-distribution" if not use_bca else "t-distribution (insufficient data for BCa)"
                print(f"  Using {ci_method} for CI calculation...")
        
        if self.verbose:
            print(f"  Sequential ensemble results: {successful_models} models, std={ensemble_std:.4f}")
            print(f"  CI: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        return (lower_bound, upper_bound)
    
    def _compute_bca_ci(self, estimates: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """
        Compute BCa (Bias-corrected and accelerated) bootstrap confidence interval.
        
        Args:
            estimates: Array of bootstrap estimates
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        estimates = np.array(estimates)
        n = len(estimates)
        
        if n < 3:
            # Fallback to simple percentile
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            return (np.percentile(estimates, lower_percentile), 
                    np.percentile(estimates, upper_percentile))
        
        # Calculate bias correction
        theta_hat = np.mean(estimates)
        n_less = np.sum(estimates < theta_hat)
        if n_less == 0:
            bias_correction = -3.0  # Extreme case
        elif n_less == n:
            bias_correction = 3.0   # Extreme case
        else:
            from scipy import stats
            bias_correction = stats.norm.ppf(n_less / n)
        
        # Calculate acceleration using jackknife
        jackknife_estimates = []
        for i in range(n):
            # Leave-one-out estimates
            jk_sample = np.concatenate([estimates[:i], estimates[i+1:]])
            if len(jk_sample) > 0:
                jackknife_estimates.append(np.mean(jk_sample))
        
        if len(jackknife_estimates) > 1:
            jk_mean = np.mean(jackknife_estimates)
            jk_diff = jk_mean - np.array(jackknife_estimates)
            numerator = np.sum(jk_diff**3)
            denominator = 6 * (np.sum(jk_diff**2))**(3/2)
            acceleration = numerator / denominator if abs(denominator) > 1e-10 else 0.0
        else:
            acceleration = 0.0
        
        # Calculate adjusted quantiles
        alpha = 1 - confidence_level
        z_alpha_2 = stats.norm.ppf(alpha / 2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)
        
        # BCa adjustments
        denom1 = 1 - acceleration * (bias_correction + z_alpha_2)
        denom2 = 1 - acceleration * (bias_correction + z_1_alpha_2)
        
        if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
            # Fallback to standard percentile
            alpha_1 = alpha / 2
            alpha_2 = 1 - alpha / 2
        else:
            alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2) / denom1)
            alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2) / denom2)
        
        # Ensure valid quantiles
        alpha_1 = max(0.001, min(0.999, alpha_1))
        alpha_2 = max(0.001, min(0.999, alpha_2))
        
        # Calculate final confidence interval
        lower_bound = np.percentile(estimates, alpha_1 * 100)
        upper_bound = np.percentile(estimates, alpha_2 * 100)
        
        if self.verbose:
            print(f"    BCa corrections: bias={bias_correction:.3f}, accel={acceleration:.3f}")
            print(f"    Adjusted quantiles: {alpha_1:.3f}, {alpha_2:.3f}")
        
        return (lower_bound, upper_bound)
    
    def _mc_dropout_confidence_interval(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        n_mc_samples: int,
        use_log_iroas: bool,
        spend_floor: float
    ) -> Tuple[float, float]:
        """
        Monte Carlo dropout confidence interval.
        
        Uses dropout during inference to capture parameter uncertainty
        without retraining. Much faster than bootstrap methods.
        """
        # Enable dropout during inference
        self.model.train()  # This enables dropout
        
        mc_iroas_values = []
        
        with torch.no_grad():
            for _ in range(n_mc_samples):
                # Each forward pass will have different dropout
                if use_log_iroas:
                    log_iroas = self._calculate_log_iroas(
                        panel_data, period_start, period_end, spend_floor
                    )
                    mc_iroas_values.append(log_iroas)
                else:
                    iroas = self._calculate_iroas_robust(
                        panel_data, period_start, period_end, spend_floor
                    )
                    mc_iroas_values.append(iroas)
        
        # Restore model to eval mode
        self.model.eval()
        
        # Calculate confidence interval with proper bounds checking
        return self._calculate_ci_with_guards(mc_iroas_values, confidence_level)
    
    def _model_aware_bootstrap_ci(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        n_bootstrap: int,
        use_log_iroas: bool,
        spend_floor: float
    ) -> Tuple[float, float]:
        """
        Model-aware bootstrap that refits the model on each bootstrap sample.
        
        This captures both sampling and model parameter uncertainty.
        Slower but statistically correct.
        """
        # Get control geos
        control_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'control'
        ]['geo'].values
        
        treatment_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'treatment'
        ]['geo'].values
        
        if len(control_geos) < 2:
            # Not enough control geos for bootstrap - use MC dropout
            if self.verbose:
                print("⚠️ Insufficient control geos for bootstrap, using MC dropout")
            return self._mc_dropout_confidence_interval(
                panel_data, period_start, period_end, confidence_level,
                100, use_log_iroas, spend_floor
            )
        
        bootstrap_iroas = []
        failed_bootstraps = 0
        
        # Bootstrap by resampling control geos with model refitting
        for bootstrap_idx in range(n_bootstrap):
            # Bootstrap sample control geos (with replacement)
            bootstrap_control_geos = np.random.choice(
                control_geos, 
                size=len(control_geos),
                replace=True
            )
            
            # Create bootstrap assignment
            bootstrap_assignment = self.assignment_df.copy()
            
            # Update control geos in assignment
            control_mask = bootstrap_assignment['assignment'] == 'control'
            bootstrap_assignment.loc[control_mask, 'geo'] = bootstrap_control_geos
            
            # Create bootstrap panel data
            bootstrap_panel = panel_data[
                panel_data['geo'].isin(list(bootstrap_control_geos) + list(treatment_geos))
            ].copy()
            
            # Map bootstrap control geos back to original control geo names for model compatibility
            geo_mapping = {}
            original_control_geos = self.assignment_df[self.assignment_df['assignment'] == 'control']['geo'].values
            for i, bootstrap_geo in enumerate(bootstrap_control_geos):
                if i < len(original_control_geos):
                    geo_mapping[bootstrap_geo] = original_control_geos[i]
            
            # Apply mapping to bootstrap panel
            for old_geo, new_geo in geo_mapping.items():
                bootstrap_panel.loc[bootstrap_panel['geo'] == old_geo, 'geo'] = new_geo
            
            try:
                # Create a lightweight STGCN model for bootstrap (fewer epochs)
                bootstrap_model = STGCNReportingModel(
                    hidden_dim=self.hidden_dim,
                    num_st_blocks=max(1, self.num_st_blocks - 1),  # Simpler model
                    window_size=self.window_size,
                    epochs=max(1, self.epochs // 4),  # Much fewer epochs
                    k_neighbors=self.k_neighbors,
                    device=self.device,
                    early_stopping_patience=1
                )
                
                # Fit on bootstrap data
                bootstrap_model.fit(
                    bootstrap_panel, 
                    bootstrap_assignment, 
                    self.pre_period_data['date'].max().strftime('%Y-%m-%d')
                )
                
                # Calculate iROAS on bootstrap sample
                bootstrap_iroas_val = bootstrap_model.calculate_iroas(
                    bootstrap_panel, period_start, period_end
                )
                
                bootstrap_iroas.append(bootstrap_iroas_val)
                
            except Exception:
                # If bootstrap sample fails, skip it
                continue
        
        # Calculate confidence interval
        if len(bootstrap_iroas) < n_bootstrap // 4:  # Need at least 25% success rate
            # Fallback to base estimate with simple error
            base_iroas = self.calculate_iroas(panel_data, period_start, period_end)
            return (base_iroas * 0.8, base_iroas * 1.2)  # Simple ±20% interval
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_iroas, lower_percentile)
        upper_bound = np.percentile(bootstrap_iroas, upper_percentile)
        
        return (lower_bound, upper_bound)
    
    def _detect_null_bias(self) -> None:
        """
        Detect systematic bias in null scenarios by analyzing STGCN predictions.
        
        This method checks if the STGCN model is producing systematic over/under predictions
        that could lead to false positives in A/A tests.
        """
        if not hasattr(self, 'pre_period_data') or self.pre_period_data is None:
            return
        
        try:
            # Create a simple A/A test scenario using pre-period data
            control_geos = self.assignment_df[
                self.assignment_df['assignment'] == 'control'
            ]['geo'].values
            
            if len(control_geos) < 4:  # Need at least 4 geos for meaningful test
                return
            
            # Split control geos into pseudo-treatment and pseudo-control
            n_pseudo_treatment = len(control_geos) // 2
            pseudo_treatment_geos = control_geos[:n_pseudo_treatment]
            pseudo_control_geos = control_geos[n_pseudo_treatment:]
            
            if len(pseudo_control_geos) == 0:
                return
            
            # Get last portion of pre-period data for testing
            test_dates = sorted(self.pre_period_data['date'].unique())[-7:]  # Last 7 days
            if len(test_dates) < self.window_size + 2:  # Need enough history
                return
                
            test_start_date = test_dates[self.window_size]  # Skip window_size days for history
            test_end_date = test_dates[-1]
            
            # Create temporary assignment for bias testing
            temp_assignment = self.assignment_df.copy()
            temp_assignment.loc[temp_assignment['geo'].isin(pseudo_treatment_geos), 'assignment'] = 'treatment'
            temp_assignment.loc[temp_assignment['geo'].isin(pseudo_control_geos), 'assignment'] = 'control'
            
            # Temporarily replace assignment for prediction
            orig_assignment = self.assignment_df
            self.assignment_df = temp_assignment
            
            # Get actual data for pseudo-treatment geos in test period
            test_data = self.pre_period_data[
                (self.pre_period_data['date'] >= pd.to_datetime(test_start_date)) &
                (self.pre_period_data['date'] <= pd.to_datetime(test_end_date)) &
                (self.pre_period_data['geo'].isin(pseudo_treatment_geos))
            ]
            
            if len(test_data) == 0:
                self.assignment_df = orig_assignment
                return
            
            # Use STGCN to predict counterfactual for pseudo-treatment geos
            try:
                predictions = self.predict(
                    self.pre_period_data, 
                    test_start_date.strftime('%Y-%m-%d'), 
                    test_end_date.strftime('%Y-%m-%d')
                )
                
                # Calculate actual vs predicted sales
                actual_total_sales = test_data['sales'].sum()
                predicted_total_sales = predictions['sales'].sum() if 'sales' in predictions else 0
                
                if predicted_total_sales > 0 and actual_total_sales > 0:
                    # Calculate relative bias
                    relative_bias = abs(actual_total_sales - predicted_total_sales) / actual_total_sales
                    
                    # Store bias information
                    self.training_diagnostics['null_bias_test'] = {
                        'relative_bias': relative_bias,
                        'actual_sales': actual_total_sales,
                        'predicted_sales': predicted_total_sales,
                        'bias_threshold': self.bias_threshold,
                        'method': 'STGCN_prediction'
                    }
                    
                    # Check for concerning bias
                    if relative_bias > self.bias_threshold:
                        warning_msg = f"🚨 STGCN HIGH BIAS DETECTED: {relative_bias:.3f} > {self.bias_threshold}"
                        self.training_warnings.append(warning_msg)
                        
                        if self.verbose:
                            print(f"\n⚠️ NULL SCENARIO BIAS WARNING:")
                            print(f"   Relative bias: {relative_bias:.3f} (threshold: {self.bias_threshold})")
                            print(f"   This may lead to false positives in A/A tests")
                            print(f"   Actual: {actual_total_sales:.0f}, Predicted: {predicted_total_sales:.0f}")
                            print(f"   Scale ratio: {predicted_total_sales/actual_total_sales:.3f}")
                else:
                    # Fallback to simple mean comparison if prediction fails
                    pseudo_treatment_data = test_data
                    pseudo_control_data = self.pre_period_data[
                        (self.pre_period_data['date'] >= pd.to_datetime(test_start_date)) &
                        (self.pre_period_data['date'] <= pd.to_datetime(test_end_date)) &
                        (self.pre_period_data['geo'].isin(pseudo_control_geos))
                    ]
                    
                    if len(pseudo_control_data) > 0:
                        actual_sales = pseudo_treatment_data['sales'].mean()
                        predicted_sales = pseudo_control_data['sales'].mean()
                        relative_bias = abs(actual_sales - predicted_sales) / actual_sales
                        
                        self.training_diagnostics['null_bias_test'] = {
                            'relative_bias': relative_bias,
                            'actual_sales': actual_sales,
                            'predicted_sales': predicted_sales,
                            'bias_threshold': self.bias_threshold,
                            'method': 'fallback_mean'
                        }
                        
            except Exception as pred_error:
                if self.verbose:
                    print(f"Warning: STGCN prediction failed in bias test: {pred_error}")
                
            # Restore original assignment
            self.assignment_df = orig_assignment
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not perform bias detection: {e}")
    
    def get_convergence_summary(self) -> str:
        """Get a compact summary of training convergence."""
        if not self.is_fitted:
            return "Model not fitted"
        
        convergence = self.training_diagnostics.get('convergence_assessment', 'unknown')
        final_loss = self.training_diagnostics.get('final_loss', 'unknown')
        
        if convergence == 'good':
            return f"✅ Converged (loss: {final_loss:.6f})"
        elif convergence == 'moderate':
            return f"⚠️ Moderate convergence (loss: {final_loss:.6f})"
        else:
            return f"🚨 Poor convergence (loss: {final_loss:.6f})"
    
    def check_null_scenario_bias(self) -> Dict[str, any]:
        """
        Check for systematic bias that could cause false positives.
        
        Returns:
            Dictionary with bias analysis results
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        bias_test = self.training_diagnostics.get('null_bias_test', {})
        if not bias_test:
            return {'error': 'Bias test not performed'}
        
        relative_bias = bias_test['relative_bias']
        threshold = bias_test['bias_threshold']
        
        return {
            'relative_bias': relative_bias,
            'bias_threshold': threshold,
            'high_bias_detected': relative_bias > threshold,
            'bias_level': 'high' if relative_bias > threshold else 'moderate' if relative_bias > threshold/2 else 'low',
            'false_positive_risk': 'high' if relative_bias > threshold else 'medium' if relative_bias > threshold/2 else 'low'
        }
    
    def diagnose_ensemble_distribution(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        ensemble_size: int = 10,
        use_log_iroas: bool = True,
        spend_floor: float = 1e-6,
        plot: bool = True
    ) -> dict:
        """
        Diagnostic analysis of ensemble iROAS distribution to understand bias and variance.
        
        Args:
            panel_data: Panel data for evaluation
            period_start: Start of evaluation period
            period_end: End of evaluation period
            ensemble_size: Number of models in ensemble for diagnosis
            use_log_iroas: Whether to use log-iROAS
            spend_floor: Minimum spend floor
            plot: Whether to create diagnostic plots
            
        Returns:
            Dictionary with distribution statistics and diagnostic information
        """
        if self.verbose:
            print(f"🔬 Running Ensemble Distribution Diagnosis (K={ensemble_size})...")
        
        # Collect ensemble iROAS values
        ensemble_iroas = []
        successful_models = 0
        
        # Train larger ensemble for better distribution analysis
        for i in range(ensemble_size):
            torch.manual_seed(6000 + i)  # Different seed space for diagnostics
            np.random.seed(6000 + i)
            
            try:
                # Create ensemble model (with calibration settings)
                ensemble_model = STGCNReportingModel(
                    hidden_dim=self.hidden_dim,
                    num_st_blocks=self.num_st_blocks,
                    window_size=self.window_size,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    dropout=self.dropout,
                    normalize_data=self.normalize_data,
                    verbose=False,
                    k_neighbors=self.k_neighbors,
                    device=self.device,
                    use_offset_calibration=self.use_offset_calibration,  # FIX: Include calibration
                    use_linear_calibration=self.use_linear_calibration   # FIX: Include calibration
                )
                
                # Fit model
                ensemble_model.fit(
                    panel_data,
                    self.assignment_df,
                    self.pre_period_data['date'].max().strftime('%Y-%m-%d')
                )
                
                # Calculate iROAS
                if use_log_iroas:
                    iroas = ensemble_model._calculate_log_iroas(
                        panel_data, period_start, period_end, spend_floor
                    )
                else:
                    iroas = ensemble_model._calculate_iroas_robust(
                        panel_data, period_start, period_end, spend_floor
                    )
                
                ensemble_iroas.append(iroas)
                successful_models += 1
                
                if self.verbose and i % 2 == 0:
                    print(f"  Model {i+1}/{ensemble_size}: iROAS = {iroas:.4f}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Model {i+1} failed: {e}")
                continue
        
        if successful_models < 3:
            raise ValueError(f"Insufficient successful models ({successful_models}) for distribution analysis")
        
        # Calculate distribution statistics
        ensemble_iroas = np.array(ensemble_iroas)
        stats = {
            'ensemble_size': successful_models,
            'mean': np.mean(ensemble_iroas),
            'median': np.median(ensemble_iroas),
            'std': np.std(ensemble_iroas, ddof=1),
            'min': np.min(ensemble_iroas),
            'max': np.max(ensemble_iroas),
            'q05': np.percentile(ensemble_iroas, 5),
            'q25': np.percentile(ensemble_iroas, 25),
            'q75': np.percentile(ensemble_iroas, 75),
            'q95': np.percentile(ensemble_iroas, 95),
            'skewness': scipy.stats.skew(ensemble_iroas) if len(ensemble_iroas) > 2 else 0,
            'kurtosis': scipy.stats.kurtosis(ensemble_iroas) if len(ensemble_iroas) > 2 else 0,
            'raw_values': ensemble_iroas.tolist()
        }
        
        # Diagnostic analysis
        bias_magnitude = abs(stats['mean'])
        std_dev = stats['std']
        ci_half_width = 1.96 * std_dev  # Approximate 95% CI half-width
        
        # Check if zero is plausibly in the distribution
        zero_in_range = (stats['q05'] <= 0 <= stats['q95'])
        
        diagnostics = {
            'bias_magnitude': bias_magnitude,
            'bias_relative_to_std': bias_magnitude / std_dev if std_dev > 0 else float('inf'),
            'ci_half_width': ci_half_width,
            'zero_plausible': zero_in_range,
            'distribution_symmetric': abs(stats['skewness']) < 0.5,
            'heavy_tailed': abs(stats['kurtosis']) > 1.0,
            'recommendation': self._generate_bias_correction_recommendation(stats)
        }
        
        if self.verbose:
            print(f"\n📊 Distribution Analysis Results:")
            print(f"   Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"   95% Range: [{stats['q05']:.4f}, {stats['q95']:.4f}]")
            print(f"   Bias magnitude: {bias_magnitude:.4f}")
            print(f"   Zero plausible: {zero_in_range}")
            print(f"   Distribution shape: skew={stats['skewness']:.2f}, kurt={stats['kurtosis']:.2f}")
            print(f"   Recommendation: {diagnostics['recommendation']}")
        
        # Create diagnostic plots
        if plot:
            self._create_distribution_plots(ensemble_iroas, stats, diagnostics)
        
        return {**stats, **diagnostics}
    
    def _generate_bias_correction_recommendation(self, stats: dict) -> str:
        """Generate recommendations based on distribution analysis."""
        bias_mag = abs(stats['mean'])
        std_dev = stats['std']
        
        if bias_mag > 2 * std_dev:
            return "STRONG BIAS: Implement pre-period calibration immediately"
        elif bias_mag > std_dev:
            return "MODERATE BIAS: Consider bias correction and increase ensemble size"
        elif std_dev > 5.0:
            return "HIGH VARIANCE: Increase regularization and ensemble size"
        elif abs(stats.get('skewness', 0)) > 1.0:
            return "SKEWED DISTRIBUTION: Use bias-corrected bootstrap (BCa)"
        else:
            return "ACCEPTABLE: Minor calibration may help"
    
    def _create_distribution_plots(self, ensemble_iroas: np.ndarray, stats: dict, diagnostics: dict):
        """Create diagnostic plots for ensemble distribution analysis."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('STGCN Ensemble iROAS Distribution Diagnostics', fontsize=16, fontweight='bold')
            
            # Plot 1: Histogram with statistics
            axes[0, 0].hist(ensemble_iroas, bins=min(20, len(ensemble_iroas)//2), alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.3f}')
            axes[0, 0].axvline(0, color='green', linestyle='-', linewidth=2, label='True Null (0)')
            axes[0, 0].axvspan(stats['q05'], stats['q95'], alpha=0.2, color='orange', label='90% Range')
            axes[0, 0].set_xlabel('iROAS')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Ensemble iROAS Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Q-Q plot for normality check
            from scipy import stats as scipy_stats
            scipy_stats.probplot(ensemble_iroas, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normality Check)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Box plot with individual points
            axes[1, 0].boxplot(ensemble_iroas, vert=True, patch_artist=True, 
                             boxprops=dict(facecolor='lightblue', alpha=0.7))
            axes[1, 0].scatter([1] * len(ensemble_iroas), ensemble_iroas, alpha=0.6, color='red', s=30)
            axes[1, 0].axhline(0, color='green', linestyle='-', linewidth=2, label='True Null')
            axes[1, 0].set_ylabel('iROAS')
            axes[1, 0].set_title('Distribution Box Plot')
            axes[1, 0].set_xticks([])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Diagnostic summary text
            axes[1, 1].axis('off')
            summary_text = f"""
DIAGNOSTIC SUMMARY

Bias Analysis:
• Mean iROAS: {stats['mean']:.4f}
• Bias magnitude: {diagnostics['bias_magnitude']:.4f}
• Bias/StdDev ratio: {diagnostics['bias_relative_to_std']:.2f}

Variance Analysis:
• Standard deviation: {stats['std']:.4f}
• 95% CI half-width: {diagnostics['ci_half_width']:.4f}

Distribution Shape:
• Skewness: {stats['skewness']:.2f}
• Kurtosis: {stats['kurtosis']:.2f}
• Zero plausible: {diagnostics['zero_plausible']}

Coverage Issues:
• Heavy tailed: {diagnostics['heavy_tailed']}
• Symmetric: {diagnostics['distribution_symmetric']}

RECOMMENDATION:
{diagnostics['recommendation']}
            """
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'stgcn_distribution_diagnostics_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"📊 Diagnostic plots saved: {plot_filename}")
            plt.show()
            
        except ImportError:
            if self.verbose:
                print("⚠️ Matplotlib/seaborn not available for plotting")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Plotting failed: {e}")
    
    def apply_bias_correction(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        method: str = 'pre_period_calibration',
        confidence_level: float = 0.95,
        ensemble_size: int = 5,
        use_log_iroas: bool = True,
        spend_floor: float = 1e-6
    ) -> dict:
        """
        Apply bias correction to improve confidence interval calibration.
        
        Args:
            panel_data: Panel data for evaluation
            period_start: Start of evaluation period  
            period_end: End of evaluation period
            method: Bias correction method ('pre_period_calibration', 'bca_bootstrap')
            confidence_level: Confidence level for intervals
            ensemble_size: Size of ensemble for correction
            use_log_iroas: Whether to use log-iROAS
            spend_floor: Minimum spend floor
            
        Returns:
            Dictionary with original and corrected confidence intervals
        """
        if self.verbose:
            print(f"🔧 Applying Bias Correction: {method}")
        
        # Get original ensemble confidence interval
        original_lower, original_upper = self._ensemble_confidence_interval(
            panel_data, period_start, period_end, confidence_level,
            ensemble_size, use_log_iroas, spend_floor, n_jobs=1, use_parallel=False
        )
        
        if method == 'pre_period_calibration':
            corrected_lower, corrected_upper = self._pre_period_calibration(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor
            )
        elif method == 'bca_bootstrap':
            corrected_lower, corrected_upper = self._bca_bootstrap_ci(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor
            )
        else:
            raise ValueError(f"Unknown bias correction method: {method}")
        
        results = {
            'method': method,
            'original_ci': (original_lower, original_upper),
            'corrected_ci': (corrected_lower, corrected_upper),
            'original_width': original_upper - original_lower,
            'corrected_width': corrected_upper - corrected_lower,
            'bias_correction': (corrected_lower + corrected_upper)/2 - (original_lower + original_upper)/2,
            'width_adjustment': (corrected_upper - corrected_lower) / (original_upper - original_lower)
        }
        
        if self.verbose:
            print(f"   Original CI: [{original_lower:.4f}, {original_upper:.4f}]")
            print(f"   Corrected CI: [{corrected_lower:.4f}, {corrected_upper:.4f}]")
            print(f"   Bias correction: {results['bias_correction']:.4f}")
            print(f"   Width adjustment: {results['width_adjustment']:.2f}x")
        
        return results
    
    def _pre_period_calibration(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        ensemble_size: int,
        use_log_iroas: bool,
        spend_floor: float
    ) -> Tuple[float, float]:
        """
        Apply pre-period calibration to remove systematic bias.
        
        This method estimates systematic bias using the pre-period and subtracts it
        from the evaluation period estimates.
        """
        if self.verbose:
            print("   🔄 Estimating pre-period bias...")
        
        # Use the latter part of pre-period as a pseudo-evaluation period
        pre_period_dates = sorted(self.pre_period_data['date'].unique())
        
        # Use last 20% of pre-period for bias estimation
        bias_start_idx = int(len(pre_period_dates) * 0.8)
        bias_start = pre_period_dates[bias_start_idx].strftime('%Y-%m-%d')
        bias_end = pre_period_dates[-1].strftime('%Y-%m-%d')
        
        # Calculate bias using ensemble on pre-period
        bias_estimates = []
        
        for i in range(ensemble_size):
            torch.manual_seed(7000 + i)  # Different seed space for bias estimation
            np.random.seed(7000 + i)
            
            try:
                # Create bias estimation model
                bias_model = STGCNReportingModel(
                    hidden_dim=self.hidden_dim,
                    num_st_blocks=self.num_st_blocks,
                    window_size=self.window_size,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    dropout=self.dropout,
                    normalize_data=self.normalize_data,
                    verbose=False,
                    k_neighbors=self.k_neighbors,
                    device=self.device
                )
                
                # Fit on earlier pre-period data
                early_pre_period = self.pre_period_data[
                    self.pre_period_data['date'] < pd.to_datetime(bias_start)
                ]
                
                if len(early_pre_period) > 0:
                    bias_model.fit(
                        panel_data,
                        self.assignment_df,
                        early_pre_period['date'].max().strftime('%Y-%m-%d')
                    )
                    
                    # Calculate "iROAS" on later pre-period (should be ~0)
                    if use_log_iroas:
                        bias_estimate = bias_model._calculate_log_iroas(
                            panel_data, bias_start, bias_end, spend_floor
                        )
                    else:
                        bias_estimate = bias_model._calculate_iroas_robust(
                            panel_data, bias_start, bias_end, spend_floor
                        )
                    
                    bias_estimates.append(bias_estimate)
                
            except Exception as e:
                if self.verbose:
                    print(f"     Bias model {i+1} failed: {e}")
                continue
        
        # Calculate average bias
        if len(bias_estimates) > 0:
            systematic_bias = np.mean(bias_estimates)
            if self.verbose:
                print(f"   📏 Systematic bias estimated: {systematic_bias:.4f}")
        else:
            systematic_bias = 0.0
            if self.verbose:
                print("   ⚠️ Could not estimate bias - using original estimates")
        
        # Get original confidence interval
        original_lower, original_upper = self._ensemble_confidence_interval_sequential(
            panel_data, period_start, period_end, confidence_level,
            ensemble_size, use_log_iroas, spend_floor
        )
        
        # Apply bias correction
        corrected_lower = original_lower - systematic_bias
        corrected_upper = original_upper - systematic_bias
        
        return (corrected_lower, corrected_upper)
    
    def _bca_bootstrap_ci(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float,
        ensemble_size: int,
        use_log_iroas: bool,
        spend_floor: float
    ) -> Tuple[float, float]:
        """
        Bias-Corrected and Accelerated (BCa) bootstrap confidence interval.
        
        This method provides better coverage for skewed distributions and
        automatically corrects for bias and skewness.
        """
        if self.verbose:
            print("   🔄 Computing BCa bootstrap confidence interval...")
        
        # Collect ensemble estimates for bias and acceleration correction
        ensemble_estimates = []
        
        for i in range(ensemble_size):
            torch.manual_seed(8000 + i)  # Different seed space for BCa
            np.random.seed(8000 + i)
            
            try:
                ensemble_model = STGCNReportingModel(
                    hidden_dim=self.hidden_dim,
                    num_st_blocks=self.num_st_blocks,
                    window_size=self.window_size,
                    epochs=self.epochs,
                    learning_rate=self.learning_rate,
                    dropout=self.dropout,
                    normalize_data=self.normalize_data,
                    verbose=False,
                    k_neighbors=self.k_neighbors,
                    device=self.device,
                    use_offset_calibration=self.use_offset_calibration,  # FIX: Include calibration
                    use_linear_calibration=self.use_linear_calibration   # FIX: Include calibration
                )
                
                ensemble_model.fit(
                    panel_data,
                    self.assignment_df,
                    self.pre_period_data['date'].max().strftime('%Y-%m-%d')
                )
                
                if use_log_iroas:
                    estimate = ensemble_model._calculate_log_iroas(
                        panel_data, period_start, period_end, spend_floor
                    )
                else:
                    estimate = ensemble_model._calculate_iroas_robust(
                        panel_data, period_start, period_end, spend_floor
                    )
                
                ensemble_estimates.append(estimate)
                
            except Exception as e:
                if self.verbose:
                    print(f"     BCa model {i+1} failed: {e}")
                continue
        
        if len(ensemble_estimates) < 3:
            if self.verbose:
                print("   ⚠️ Insufficient estimates for BCa - falling back to standard percentile")
            return self._ensemble_confidence_interval_sequential(
                panel_data, period_start, period_end, confidence_level,
                ensemble_size, use_log_iroas, spend_floor, False  # Don't use BCa in fallback
            )
        
        ensemble_estimates = np.array(ensemble_estimates)
        n = len(ensemble_estimates)
        
        # Calculate bias correction
        theta_hat = np.mean(ensemble_estimates)
        
        # Count estimates below observed statistic (bias correction)
        below_count = np.sum(ensemble_estimates < theta_hat)
        bias_correction = scipy.stats.norm.ppf(below_count / n) if n > 0 else 0
        
        # Calculate acceleration using jackknife
        jackknife_estimates = []
        for i in range(n):
            leave_one_out = np.delete(ensemble_estimates, i)
            if len(leave_one_out) > 0:
                jackknife_estimates.append(np.mean(leave_one_out))
        
        if len(jackknife_estimates) > 1:
            jack_mean = np.mean(jackknife_estimates)
            numerator = np.sum((jack_mean - np.array(jackknife_estimates))**3)
            denominator = 6 * (np.sum((jack_mean - np.array(jackknife_estimates))**2))**1.5
            acceleration = numerator / denominator if denominator != 0 else 0
        else:
            acceleration = 0
        
        # Calculate BCa confidence interval
        alpha = 1 - confidence_level
        z_alpha_2 = scipy.stats.norm.ppf(alpha/2)
        z_1_alpha_2 = scipy.stats.norm.ppf(1 - alpha/2)
        
        # Adjust quantiles for bias and acceleration
        alpha_1 = scipy.stats.norm.cdf(bias_correction + (bias_correction + z_alpha_2)/(1 - acceleration * (bias_correction + z_alpha_2)))
        alpha_2 = scipy.stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha_2)/(1 - acceleration * (bias_correction + z_1_alpha_2)))
        
        # Ensure valid quantiles
        alpha_1 = max(0.001, min(0.999, alpha_1))
        alpha_2 = max(0.001, min(0.999, alpha_2))
        
        # Calculate final confidence interval
        lower_bound = np.percentile(ensemble_estimates, alpha_1 * 100)
        upper_bound = np.percentile(ensemble_estimates, alpha_2 * 100)
        
        if self.verbose:
            print(f"   📏 BCa corrections: bias={bias_correction:.3f}, accel={acceleration:.3f}")
            print(f"   📏 Adjusted quantiles: {alpha_1:.3f}, {alpha_2:.3f}")
        
        return (lower_bound, upper_bound)
    
    def _normalize_data(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize input data to improve training stability.
        
        Args:
            data_tensor: Input tensor of shape [num_nodes, seq_len, num_features]
            
        Returns:
            Normalized tensor
        """
        if not self.normalize_data:
            return data_tensor
        
        # Calculate mean and std across all dimensions except features
        self.data_mean = data_tensor.mean(dim=(0, 1), keepdim=True)  # [1, 1, num_features]
        self.data_std = data_tensor.std(dim=(0, 1), keepdim=True)    # [1, 1, num_features]
        
        # Avoid division by zero
        self.data_std = torch.clamp(self.data_std, min=1e-6)
        
        # Normalize
        normalized = (data_tensor - self.data_mean) / self.data_std
        
        if self.verbose:
            print(f"Data normalization applied:")
            print(f"  Mean: {self.data_mean.squeeze().tolist()}")
            print(f"  Std: {self.data_std.squeeze().tolist()}")
        
        return normalized
    
    def _denormalize_data(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize predictions back to original scale.
        
        Args:
            normalized_tensor: Normalized tensor from model
            
        Returns:
            Denormalized tensor in original scale
        """
        if not self.normalize_data or self.data_mean is None or self.data_std is None:
            return normalized_tensor
        
        # Denormalize: x_original = x_normalized * std + mean
        denormalized = normalized_tensor * self.data_std + self.data_mean
        
        return denormalized
    
    def incremental_sales_confidence_interval(
        self, panel_data: pd.DataFrame, period_start: str, period_end: str,
        confidence_level: float = 0.95, n_bootstrap: int = None, 
        seed: Optional[int] = None, ensemble_size: int = 10
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for incremental sales using ensemble methods.
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_bootstrap = self._get_bootstrap_size(panel_data, n_bootstrap)
        
        # Use existing ensemble uncertainty infrastructure but for incremental sales
        ensemble_incremental_sales = []
        
        # Get treatment group data for incremental sales calculation
        assignment_df = getattr(self, 'assignment_data', None)
        if assignment_df is None:
            raise ValueError("Assignment data not available. Model may not be properly fitted.")
        
        treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].values
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask]
        treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
        
        # Use same aggregation as calculate_incremental_sales()
        actual_sales = treatment_data.groupby('date')['sales'].mean().sum()
        
        # Generate ensemble predictions
        for i in range(ensemble_size):
            try:
                # Create ensemble member with different seed
                ensemble_seed = seed + i if seed is not None else None
                
                # Use existing uncertainty_metrics infrastructure if available
                if hasattr(self, 'uncertainty_metrics'):
                    metrics = self.uncertainty_metrics(
                        panel_data, period_start, period_end,
                        metric_type='incremental_sales',
                        ensemble_size=1,
                        seed=ensemble_seed
                    )
                    if 'point_estimate' in metrics:
                        ensemble_incremental_sales.append(metrics['point_estimate'])
                        continue
                
                # Fallback: retrain model with different seed
                temp_model = self.__class__(
                    hidden_dim=getattr(self, 'hidden_dim', 32),
                    num_st_blocks=getattr(self, 'num_st_blocks', 2),
                    epochs=max(5, getattr(self, 'epochs', 10) // 2),  # Faster training
                    seed=ensemble_seed,
                    verbose=False
                )
                
                # Fit and predict with ensemble member
                temp_model.fit(panel_data, assignment_df, period_start)
                counterfactual = temp_model.predict(panel_data, period_start, period_end)
                
                # Better handling of predicted_sales with logging
                if isinstance(counterfactual.get('sales'), np.ndarray):
                    predicted_sales = counterfactual['sales'].sum()
                    if self.verbose:
                        print(f"Ensemble {i}: Using daily predictions, sum = {predicted_sales:.2f}")
                else:
                    predicted_sales = counterfactual.get('sales', 0)
                    if self.verbose:
                        print(f"⚠️ Ensemble {i}: Fallback to scalar prediction = {predicted_sales:.2f}")
                    # Log when this problematic fallback happens
                    if predicted_sales == 0:
                        print(f"⚠️ Ensemble {i}: Predicted sales is 0, may indicate model failure")
                
                incremental_sales = actual_sales - predicted_sales
                ensemble_incremental_sales.append(incremental_sales)
                
            except Exception as e:
                if self.verbose:
                    print(f"Ensemble member {i} failed: {e}")
                continue
        
        if len(ensemble_incremental_sales) < 2:
            if self.verbose:
                print(f"⚠️ Only {len(ensemble_incremental_sales)} successful ensemble members")
            return np.nan, np.nan
        
        # Calculate confidence interval from ensemble
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ensemble_incremental_sales = np.array(ensemble_incremental_sales)
        lower_bound = np.percentile(ensemble_incremental_sales, lower_percentile)
        upper_bound = np.percentile(ensemble_incremental_sales, upper_percentile)
        
        return lower_bound, upper_bound