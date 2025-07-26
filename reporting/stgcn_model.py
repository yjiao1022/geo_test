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
        bias_threshold: float = 0.1  # Threshold for bias detection in null scenarios
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
            verbose: Whether to print training diagnostics and progress
            bias_threshold: Threshold for detecting systematic bias in null scenarios
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
    
    def _normalize_data(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize data to improve training stability.
        
        Args:
            data_tensor: Input tensor of shape [num_nodes, num_timesteps, num_features]
            
        Returns:
            Normalized tensor
        """
        if not self.normalize_data:
            return data_tensor
        
        # Compute mean and std across all nodes and timesteps for each feature
        # Shape: [num_features]
        self.data_mean = data_tensor.mean(dim=(0, 1), keepdim=True)  # [1, 1, num_features]
        self.data_std = data_tensor.std(dim=(0, 1), keepdim=True)    # [1, 1, num_features]
        
        # Avoid division by zero
        self.data_std = torch.clamp(self.data_std, min=1e-6)
        
        # Normalize
        normalized = (data_tensor - self.data_mean) / self.data_std
        
        return normalized
    
    def _denormalize_data(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Denormalize data back to original scale.
        
        Args:
            normalized_tensor: Normalized tensor
            
        Returns:
            Denormalized tensor
        """
        if not self.normalize_data or self.data_mean is None:
            return normalized_tensor
        
        return normalized_tensor * self.data_std + self.data_mean
    
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
        # Store data for later use
        self.assignment_df = assignment_df.copy()
        
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
            
            gradient_history.append(gradient_stats)
            
            # Check training health and collect warnings
            epoch_warnings = self._check_training_health(epoch + 1, avg_loss, gradient_stats)
            if epoch_warnings:
                self.training_warnings.extend(epoch_warnings)
                
                # Print critical warnings immediately
                for warning in epoch_warnings:
                    if "🚨" in warning:
                        print(warning)
            
            # Early stopping based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
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
        results = {}
        for i, feature in enumerate(self.feature_cols):
            # Average across treatment geos and sum across time
            feature_counterfactual = counterfactual_data[:, :, i].sum(axis=1)  # Sum over time
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
        
        # Calculate iROAS
        if abs(incremental_spend) < 1e-6:
            return 0.0
        
        iroas = incremental_sales / incremental_spend
        return iroas
    
    def confidence_interval(
        self,
        panel_data: pd.DataFrame,
        period_start: str,
        period_end: str,
        confidence_level: float = 0.95,
        n_bootstrap: int = 500,
        seed: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap resampling of control geos.
        
        This method follows the same approach as traditional models by resampling
        control geos and re-fitting the model to estimate uncertainty.
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of evaluation period
            period_end: End date of evaluation period
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating confidence interval")
        
        if seed is not None:
            np.random.seed(seed)
        
        # Get control geos
        control_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'control'
        ]['geo'].values
        
        treatment_geos = self.assignment_df[
            self.assignment_df['assignment'] == 'treatment'
        ]['geo'].values
        
        if len(control_geos) < 2:
            # Not enough control geos for bootstrap - fallback to simple interval
            base_iroas = self.calculate_iroas(panel_data, period_start, period_end)
            # Use a simple heuristic based on data variance
            panel_data_eval = panel_data[
                (pd.to_datetime(panel_data['date']) >= pd.to_datetime(period_start)) &
                (pd.to_datetime(panel_data['date']) <= pd.to_datetime(period_end))
            ]
            
            treatment_data = panel_data_eval[panel_data_eval['geo'].isin(treatment_geos)]
            sales_std = treatment_data['sales'].std()
            spend_std = treatment_data['spend'].std()
            
            # Simple error propagation estimate
            error_estimate = np.sqrt((sales_std / treatment_data['spend'].mean())**2 + 
                                   (spend_std * treatment_data['sales'].mean() / treatment_data['spend'].mean()**2)**2)
            
            margin = 1.96 * error_estimate  # 95% CI
            return (base_iroas - margin, base_iroas + margin)
        
        bootstrap_iroas = []
        
        # Bootstrap by resampling control geos
        for _ in range(n_bootstrap):
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