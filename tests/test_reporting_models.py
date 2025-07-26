"""
Tests for new reporting models (GBR, TBR, SCM).
"""

import pytest
import pandas as pd
import numpy as np

from reporting.models import GBRModel, TBRModel, SyntheticControlModel


@pytest.fixture
def sample_data_for_regression():
    """Sample panel data and assignment for regression model testing."""
    geos = [f'geo_{i:03d}' for i in range(20)]
    dates = pd.date_range('2024-01-01', periods=40)
    
    records = []
    np.random.seed(42)
    for i, geo in enumerate(geos):
        # Create a base level for each geo
        base_sales = 10000 + i * 100
        base_spend = 5000 + i * 50
        
        # Add a slight trend to sales
        sales_trend = np.linspace(0, 500, len(dates))
        
        for j, date in enumerate(dates):
            # Treatment group gets a lift in the post-period
            is_post_period = date >= pd.to_datetime('2024-01-31')
            is_treatment = i < 10 # First 10 geos are treatment
            
            sales = base_sales + sales_trend[j] + np.random.normal(0, 300)
            spend = base_spend + np.random.normal(0, 100)
            
            if is_treatment and is_post_period:
                spend += 200 # Incremental spend
                sales += spend * 2.5 # iROAS of 2.5

            records.append({
                'geo': geo,
                'date': date,
                'sales': sales,
                'spend': spend
            })
    
    panel_data = pd.DataFrame(records)
    
    assignment_df = pd.DataFrame({
        'geo': geos,
        'assignment': ['treatment'] * 10 + ['control'] * 10
    })
    
    return panel_data, assignment_df


class TestGBRModel:
    """Test cases for GBRModel."""

    def test_fitting_and_iroas(self, sample_data_for_regression):
        """Test basic model fitting and iROAS calculation."""
        panel_data, assignment_df = sample_data_for_regression
        model = GBRModel(alpha=0.0)
        
        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')
        
        assert model.is_fitted
        iroas = model.calculate_iroas(panel_data, '2024-01-31', '2024-02-09')
        
        assert isinstance(iroas, float)
        # With perfect data, this would be 2.5. With noise, it should be close.
        assert iroas == pytest.approx(2.5, rel=0.5)

    def test_confidence_interval(self, sample_data_for_regression):
        """Test confidence interval calculation."""
        panel_data, assignment_df = sample_data_for_regression
        model = GBRModel(alpha=0.0)
        
        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')
        
        lower, upper = model.confidence_interval(
            panel_data, '2024-01-31', '2024-02-09',
            n_bootstrap=50, seed=42
        )
        
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper
        # The true iROAS of 2.5 should be within the CI
        assert lower < 2.5 < upper


class TestTBRModel:
    """Test cases for TBRModel."""

    def test_fitting_and_iroas(self, sample_data_for_regression):
        """Test basic model fitting and iROAS calculation."""
        panel_data, assignment_df = sample_data_for_regression
        model = TBRModel()
        
        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')
        
        assert model.is_fitted
        assert 'pre_period_diff_mean' in model.model_params
        
        iroas = model.calculate_iroas(panel_data, '2024-01-31', '2024-02-09')
        
        assert isinstance(iroas, float)
        assert iroas == pytest.approx(2.5, rel=0.5)

    def test_confidence_interval(self, sample_data_for_regression):
        """Test confidence interval calculation."""
        panel_data, assignment_df = sample_data_for_regression
        model = TBRModel()
        
        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')
        
        lower, upper = model.confidence_interval(
            panel_data, '2024-01-31', '2024-02-09',
            confidence_level=0.9, seed=42
        )
        
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper
        assert lower < 2.5 < upper

class TestSyntheticControlModel:
    """Test cases for SyntheticControlModel."""

    def test_fitting_and_prediction(self, sample_data_for_regression):
        """Test model fitting and counterfactual prediction."""
        panel_data, assignment_df = sample_data_for_regression
        model = SyntheticControlModel()

        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')

        assert model.is_fitted
        assert 'weights' in model.model_params
        assert len(model.model_params['weights']) == 10 # Number of control geos
        assert np.isclose(np.sum(model.model_params['weights']), 1.0)

        predictions = model.predict(panel_data, '2024-01-31', '2024-02-09')
        assert 'sales' in predictions
        assert len(predictions['sales']) == 10 # 10 days in eval period

    def test_iroas_calculation(self, sample_data_for_regression):
        """Test iROAS calculation."""
        panel_data, assignment_df = sample_data_for_regression
        model = SyntheticControlModel()

        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')
        iroas = model.calculate_iroas(panel_data, '2024-01-31', '2024-02-09')

        assert isinstance(iroas, float)
        assert iroas == pytest.approx(2.5, rel=0.5)

    def test_confidence_interval(self, sample_data_for_regression):
        """Test confidence interval calculation."""
        panel_data, assignment_df = sample_data_for_regression
        model = SyntheticControlModel()

        model.fit(panel_data, assignment_df, pre_period_end='2024-01-31')
        lower, upper = model.confidence_interval(
            panel_data, '2024-01-31', '2024-02-09',
            n_bootstrap=10, seed=42 # Lower bootstrap for speed
        )

        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper
        assert lower < 2.5 < upper