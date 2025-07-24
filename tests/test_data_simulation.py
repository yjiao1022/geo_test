"""
Tests for data simulation module.
"""

import pytest
import pandas as pd
import numpy as np

from data_simulation.generators import SimpleNullGenerator, DataConfig


class TestSimpleNullGenerator:
    """Test cases for SimpleNullGenerator."""
    
    def test_basic_generation(self):
        """Test basic data generation with default parameters."""
        config = DataConfig(n_geos=10, n_days=30, seed=42)
        generator = SimpleNullGenerator(config)
        
        panel_data, geo_features = generator.generate()
        
        # Check dimensions
        assert len(panel_data) == 10 * 30  # n_geos * n_days
        assert len(geo_features) == 10
        
        # Check columns
        assert set(panel_data.columns) >= {'geo', 'date', 'sales', 'spend', 'covariate'}
        assert set(geo_features.columns) >= {'geo', 'base_sales', 'base_spend', 'covariate'}
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        config = DataConfig(n_geos=5, n_days=10, seed=123)
        
        generator1 = SimpleNullGenerator(config)
        panel1, features1 = generator1.generate()
        
        generator2 = SimpleNullGenerator(config)
        panel2, features2 = generator2.generate()
        
        # Should be identical with same seed
        pd.testing.assert_frame_equal(panel1, panel2)
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_non_negative_values(self):
        """Test that sales and spend are non-negative."""
        config = DataConfig(n_geos=20, n_days=50, seed=42)
        generator = SimpleNullGenerator(config)
        
        panel_data, _ = generator.generate()
        
        assert (panel_data['sales'] >= 0).all()
        assert (panel_data['spend'] >= 0).all()
    
    def test_data_validation(self):
        """Test data validation method."""
        config = DataConfig(n_geos=5, n_days=10, seed=42)
        generator = SimpleNullGenerator(config)
        
        panel_data, geo_features = generator.generate()
        
        # Should pass validation
        assert generator.validate_data(panel_data, geo_features) is True
        
        # Should fail with wrong dimensions
        bad_panel = panel_data.iloc[:10]  # Wrong number of rows
        assert generator.validate_data(bad_panel, geo_features) is False
    
    def test_custom_parameters(self):
        """Test generation with custom parameters."""
        config = DataConfig(
            n_geos=8,
            n_days=20,
            base_sales_mean=5000,
            base_spend_mean=2000,
            seed=999
        )
        generator = SimpleNullGenerator(config)
        
        panel_data, geo_features = generator.generate()
        
        # Check dimensions match config
        assert len(panel_data) == 8 * 20
        assert len(geo_features) == 8
        
        # Check that base values are roughly around the specified means
        assert geo_features['base_sales'].mean() == pytest.approx(5000, rel=0.5)
        assert geo_features['base_spend'].mean() == pytest.approx(2000, rel=0.5)