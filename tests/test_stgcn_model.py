"""
Tests for STGCN reporting models.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

from reporting.stgcn_model import STGCNReportingModel, STGCNModel, SpatioTemporalBlock
from assignment.spatial_utils import build_spatial_adjacency_matrix, prepare_stgcn_data


class TestSpatioTemporalBlock:
    """Test the core Spatio-Temporal block component."""
    
    def test_st_block_forward(self):
        """Test forward pass through Spatio-Temporal block."""
        # Create test data
        num_nodes, seq_len, in_channels = 5, 10, 3
        spatial_channels, temporal_channels = 8, 16
        
        # Create block
        block = SpatioTemporalBlock(
            in_channels=in_channels,
            spatial_channels=spatial_channels,
            temporal_channels=temporal_channels,
            temporal_kernel_size=3,
            dropout=0.1
        )
        
        # Create test input
        x = torch.randn(num_nodes, seq_len, in_channels)
        
        # Create simple edge index (each node connects to next)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        edge_weight = torch.ones(4, dtype=torch.float32)
        
        # Forward pass
        output = block(x, edge_index, edge_weight)
        
        # Check output shape
        assert output.shape == (num_nodes, seq_len, temporal_channels)
        
        # Check that output is not same as input (model is learning something)
        assert not torch.allclose(output.mean(), x.mean(), atol=1e-3)


class TestSTGCNModel:
    """Test the complete STGCN model."""
    
    def test_stgcn_model_forward(self):
        """Test forward pass through complete STGCN model."""
        num_nodes, seq_len, num_features = 8, 15, 2
        hidden_dim = 32
        
        # Create model
        model = STGCNModel(
            num_features=num_features,
            hidden_dim=hidden_dim,
            num_st_blocks=2,
            temporal_kernel_size=3,
            dropout=0.1
        )
        
        # Create test input
        x = torch.randn(num_nodes, seq_len, num_features)
        
        # Create test graph (k-NN style connectivity)
        edge_list = []
        edge_weights = []
        for i in range(num_nodes):
            for j in range(max(0, i-2), min(num_nodes, i+3)):  # Connect to nearby nodes
                if i != j:
                    edge_list.append([i, j])
                    edge_weights.append(1.0 / (abs(i-j) + 1))  # Inverse distance weight
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        
        # Forward pass
        output = model(x, edge_index, edge_weight)
        
        # Check output shape
        assert output.shape == (num_nodes, seq_len, num_features)
        
        # Check that model parameters have gradients after backward pass
        loss = output.mean()
        loss.backward()
        
        param_count = 0
        grad_count = 0
        for param in model.parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
        
        assert grad_count > 0, "Model should have gradients"
        assert grad_count == param_count, "All parameters should have gradients"


class TestSTGCNReportingModel:
    """Test the STGCN reporting model integration."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create geo features
        n_geos = 10
        n_days = 60
        
        geos = [f'geo_{i:03d}' for i in range(n_geos)]
        dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
        
        # Create panel data
        panel_data = []
        np.random.seed(42)
        
        for geo in geos:
            base_sales = np.random.normal(10000, 2000)
            base_spend = np.random.normal(5000, 1000)
            
            for i, date in enumerate(dates):
                # Add some temporal patterns
                seasonal = 1000 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
                trend = 50 * i  # Linear trend
                noise = np.random.normal(0, 500)
                
                sales = base_sales + seasonal + trend + noise
                spend = base_spend + np.random.normal(0, 200)
                
                panel_data.append({
                    'geo': geo,
                    'date': date,
                    'sales': max(sales, 1000),  # Ensure positive
                    'spend': max(spend, 100)  # Ensure positive
                })
        
        panel_df = pd.DataFrame(panel_data)
        
        # Create assignment
        assignment_df = pd.DataFrame({
            'geo': geos,
            'assignment': ['treatment' if i < n_geos//2 else 'control' for i in range(n_geos)]
        })
        
        return panel_df, assignment_df
    
    def test_stgcn_model_initialization(self):
        """Test STGCN model initialization."""
        model = STGCNReportingModel(
            hidden_dim=32,
            num_st_blocks=1,
            epochs=5,
            device='cpu'
        )
        
        assert model.hidden_dim == 32
        assert model.num_st_blocks == 1
        assert model.epochs == 5
        assert not model.is_fitted
    
    def test_stgcn_model_fit_predict(self, sample_data):
        """Test STGCN model fitting and prediction."""
        panel_data, assignment_df = sample_data
        
        # Create model with minimal parameters for fast testing
        model = STGCNReportingModel(
            hidden_dim=16,
            num_st_blocks=1,
            window_size=5,
            epochs=3,  # Very few epochs for testing
            k_neighbors=3,
            device='cpu',
            early_stopping_patience=2
        )
        
        # Fit model on first 40 days
        pre_period_end = '2024-02-10'
        model.fit(panel_data, assignment_df, pre_period_end)
        
        assert model.is_fitted
        assert model.model is not None
        assert model.edge_index is not None
        assert model.geo_to_idx is not None
        
        # Test prediction
        prediction = model.predict(panel_data, '2024-02-11', '2024-02-15')
        
        assert 'sales' in prediction
        assert 'spend' in prediction
        assert len(prediction['sales']) == len(assignment_df[assignment_df['assignment'] == 'treatment'])
        
        # Test iROAS calculation
        iroas = model.calculate_iroas(panel_data, '2024-02-11', '2024-02-15')
        assert isinstance(iroas, float)
        
        # Test confidence interval (with few bootstrap samples for speed)
        ci_lower, ci_upper = model.confidence_interval(
            panel_data, '2024-02-11', '2024-02-15',
            n_bootstrap=5  # Very few for testing
        )
        
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower <= ci_upper
    
    def test_stgcn_model_edge_cases(self, sample_data):
        """Test STGCN model edge cases and error handling."""
        panel_data, assignment_df = sample_data
        
        model = STGCNReportingModel(
            hidden_dim=8,
            epochs=1,
            device='cpu'
        )
        
        # Test prediction before fitting
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(panel_data, '2024-02-11', '2024-02-15')
        
        # Test with insufficient pre-period data
        with pytest.raises(ValueError):
            model.fit(panel_data, assignment_df, '2024-01-02')  # Too early
    
    def test_graph_construction_utilities(self, sample_data):
        """Test the graph construction utilities."""
        panel_data, assignment_df = sample_data
        
        # Create geo features with spatial coordinates
        n_geos = len(assignment_df)
        np.random.seed(42)
        
        geo_features = pd.DataFrame({
            'geo': assignment_df['geo'].values,
            'xy1': np.random.uniform(0, 100, n_geos),
            'xy2': np.random.uniform(0, 100, n_geos)
        })
        
        # Test adjacency matrix construction
        edge_index, edge_weight = build_spatial_adjacency_matrix(
            geo_features,
            spatial_cols=['xy1', 'xy2'],
            connection_method='knn',
            k_neighbors=3
        )
        
        assert edge_index.shape[0] == 2  # [2, num_edges]
        assert edge_index.shape[1] == edge_weight.shape[0]  # Same number of edges
        assert edge_weight.min() > 0  # All weights should be positive
        
        # Test data preparation
        data_tensor, geo_to_idx = prepare_stgcn_data(
            panel_data,
            geo_features,
            feature_cols=['sales', 'spend']
        )
        
        assert data_tensor.shape[0] == n_geos  # num_nodes
        assert data_tensor.shape[2] == 2  # num_features
        assert len(geo_to_idx) == n_geos
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stgcn_model_cuda(self, sample_data):
        """Test STGCN model on CUDA device if available."""
        panel_data, assignment_df = sample_data
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=2,
            device='cuda'
        )
        
        # Should work without errors on CUDA
        model.fit(panel_data, assignment_df, '2024-02-10')
        prediction = model.predict(panel_data, '2024-02-11', '2024-02-15')
        
        assert 'sales' in prediction
        assert 'spend' in prediction


class TestSTGCNIntegration:
    """Test STGCN integration with existing framework."""
    
    def test_stgcn_import(self):
        """Test that STGCN can be imported from reporting module."""
        from reporting import STGCNReportingModel
        assert STGCNReportingModel is not None
    
    def test_stgcn_basemodel_interface(self):
        """Test that STGCN implements BaseModel interface correctly."""
        from reporting.models import BaseModel
        from reporting.stgcn_model import STGCNReportingModel
        
        model = STGCNReportingModel()
        assert isinstance(model, BaseModel)
        
        # Check that required methods exist
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'calculate_iroas')
        assert hasattr(model, 'confidence_interval')
        
        # Check that methods are callable
        assert callable(model.fit)
        assert callable(model.predict)
        assert callable(model.calculate_iroas)
        assert callable(model.confidence_interval)


if __name__ == "__main__":
    pytest.main([__file__])