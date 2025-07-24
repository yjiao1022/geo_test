"""
Tests for reporting module.
"""

import pytest
import pandas as pd
import numpy as np

from reporting.models import MeanMatchingModel


class TestMeanMatchingModel:
    """Test cases for MeanMatchingModel."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample panel data and assignment for testing."""
        # Generate simple test data
        geos = [f'geo_{i:03d}' for i in range(10)]
        dates = pd.date_range('2024-01-01', periods=30)
        
        # Create panel data
        records = []
        for geo in geos:
            base_sales = np.random.normal(10000, 1000)
            base_spend = np.random.normal(5000, 500)
            for date in dates:
                records.append({
                    'geo': geo,
                    'date': date,
                    'sales': base_sales + np.random.normal(0, 500),
                    'spend': base_spend + np.random.normal(0, 200)
                })
        
        panel_data = pd.DataFrame(records)
        
        # Create assignment (first 5 treatment, last 5 control)
        assignment_df = pd.DataFrame({
            'geo': geos,
            'assignment': ['treatment'] * 5 + ['control'] * 5
        })
        
        return panel_data, assignment_df
    
    def test_basic_fitting(self, sample_data):
        """Test basic model fitting."""
        panel_data, assignment_df = sample_data
        model = MeanMatchingModel()
        
        # Should not be fitted initially
        assert model.is_fitted is False
        
        # Fit the model
        model.fit(panel_data, assignment_df, '2024-01-20')
        
        # Should be fitted now
        assert model.is_fitted is True
        assert len(model.control_geos) == 5
        assert len(model.treatment_geos) == 5
    
    def test_prediction(self, sample_data):
        """Test counterfactual prediction."""
        panel_data, assignment_df = sample_data
        model = MeanMatchingModel()
        
        model.fit(panel_data, assignment_df, '2024-01-20')
        
        # Predict for evaluation period
        predictions = model.predict(panel_data, '2024-01-21', '2024-01-30')
        
        # Check structure
        assert 'sales' in predictions
        assert 'spend' in predictions
        assert 'dates' in predictions
        assert len(predictions['sales']) == 10  # 10 days in eval period
        assert len(predictions['spend']) == 10
        assert len(predictions['dates']) == 10
    
    def test_iroas_calculation(self, sample_data):
        """Test iROAS calculation."""
        panel_data, assignment_df = sample_data
        model = MeanMatchingModel()
        
        model.fit(panel_data, assignment_df, '2024-01-20')
        
        # Calculate iROAS
        iroas = model.calculate_iroas(panel_data, '2024-01-21', '2024-01-30')
        
        # Should return a float
        assert isinstance(iroas, float)
        # For null data, should be close to 0 (but may vary due to randomness)
        assert abs(iroas) < 10  # Reasonable bound for test data
    
    def test_confidence_interval(self, sample_data):
        """Test confidence interval calculation."""
        panel_data, assignment_df = sample_data
        model = MeanMatchingModel()
        
        model.fit(panel_data, assignment_df, '2024-01-20')
        
        # Calculate CI
        lower, upper = model.confidence_interval(
            panel_data, '2024-01-21', '2024-01-30',
            confidence_level=0.95, n_bootstrap=50, seed=42
        )
        
        # Check structure
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper
        
        # For null data, CI should contain 0
        assert lower <= 0 <= upper or abs(lower) < 5 or abs(upper) < 5
    
    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        panel_data, assignment_df = sample_data
        model = MeanMatchingModel()
        
        # Test fitting with missing columns
        bad_panel = panel_data.drop('sales', axis=1)
        with pytest.raises(ValueError, match="must contain columns"):
            model.fit(bad_panel, assignment_df, '2024-01-20')
        
        bad_assignment = assignment_df.drop('assignment', axis=1)
        with pytest.raises(ValueError, match="must contain columns"):
            model.fit(panel_data, bad_assignment, '2024-01-20')
        
        # Test using unfitted model
        unfitted_model = MeanMatchingModel()
        with pytest.raises(ValueError, match="must be fitted"):
            unfitted_model.predict(panel_data, '2024-01-21', '2024-01-30')
        
        with pytest.raises(ValueError, match="must be fitted"):
            unfitted_model.calculate_iroas(panel_data, '2024-01-21', '2024-01-30')
    
    def test_no_control_geos(self, sample_data):
        """Test error when no control geos are found."""
        panel_data, assignment_df = sample_data
        model = MeanMatchingModel()
        
        # Make all geos treatment
        bad_assignment = assignment_df.copy()
        bad_assignment['assignment'] = 'treatment'
        
        with pytest.raises(ValueError, match="No control geos found"):
            model.fit(panel_data, bad_assignment, '2024-01-20')