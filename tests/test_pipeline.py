"""
Tests for pipeline module.
"""

import pytest
import pandas as pd
import numpy as np

from pipeline.config import ExperimentConfig
from pipeline.runner import ExperimentRunner


class TestExperimentConfig:
    """Test cases for ExperimentConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExperimentConfig()
        
        assert config.n_geos == 50
        assert config.n_days == 90
        assert config.n_simulations == 100
        assert config.seed is None
    
    def test_config_update(self):
        """Test configuration update method."""
        config = ExperimentConfig()
        
        updated_config = config.update(n_geos=100, seed=42)
        
        # Original should be unchanged
        assert config.n_geos == 50
        assert config.seed is None
        
        # Updated should have new values
        assert updated_config.n_geos == 100
        assert updated_config.seed == 42
        assert updated_config.n_days == 90  # Unchanged value
    
    def test_config_conversion(self):
        """Test conversion to other config types."""
        config = ExperimentConfig(n_geos=20, n_days=60, seed=123)
        
        # Test conversion to DataConfig
        data_config = config.to_data_config()
        assert data_config.n_geos == 20
        assert data_config.n_days == 60
        assert data_config.seed == 123
        
        # Test conversion to EvaluationConfig
        eval_config = config.to_evaluation_config()
        assert eval_config.pre_period_days == config.pre_period_days
        assert eval_config.seed == 123


class TestExperimentRunner:
    """Test cases for ExperimentRunner."""
    
    def test_initialization(self):
        """Test runner initialization."""
        config = ExperimentConfig(n_geos=10, n_days=30)
        runner = ExperimentRunner(config)
        
        assert runner.config.n_geos == 10
        assert runner.config.n_days == 30
        assert 'Random' in runner.assignment_methods
        assert 'MeanMatching' in runner.reporting_methods
    
    def test_single_experiment(self):
        """Test running a single experiment."""
        config = ExperimentConfig(
            n_geos=10, 
            n_days=30, 
            pre_period_days=20,
            eval_period_days=10,
            seed=42
        )
        runner = ExperimentRunner(config)
        
        # Run without plots to avoid display issues in tests
        results = runner.run_single_experiment(show_plots=False)
        
        # Check results structure
        assert 'panel_data' in results
        assert 'geo_features' in results
        assert 'assignment_df' in results
        assert 'iroas_estimate' in results
        assert 'iroas_ci' in results
        
        # Check data dimensions
        assert len(results['panel_data']) == 10 * 30
        assert len(results['geo_features']) == 10
        assert len(results['assignment_df']) == 10
        
        # Check estimate types
        assert isinstance(results['iroas_estimate'], float)
        assert len(results['iroas_ci']) == 2
        assert isinstance(results['significant'], bool)
    
    def test_method_addition(self):
        """Test adding custom methods."""
        from assignment.methods import RandomAssignment
        from reporting.models import MeanMatchingModel
        
        runner = ExperimentRunner()
        
        # Add custom methods
        runner.add_assignment_method('CustomRandom', RandomAssignment())
        runner.add_reporting_method('CustomMean', MeanMatchingModel())
        
        assert 'CustomRandom' in runner.assignment_methods
        assert 'CustomMean' in runner.reporting_methods
    
    def test_quick_run(self):
        """Test quick run method."""
        config = ExperimentConfig(
            n_geos=8,
            n_days=20,
            pre_period_days=15,
            eval_period_days=5,
            n_simulations=5,  # Very small for testing
            n_bootstrap=20,
            seed=42
        )
        runner = ExperimentRunner(config)
        
        # Run without plots and verbose output
        results = runner.quick_run(show_plots=False, verbose=False)
        
        # Check structure
        assert 'single_experiment' in results
        assert 'detailed_results' in results
        assert 'summary_results' in results
        
        # Check types
        assert isinstance(results['detailed_results'], pd.DataFrame)
        assert isinstance(results['summary_results'], pd.DataFrame)
        
        # Check that we have some results
        assert len(results['detailed_results']) > 0
        assert len(results['summary_results']) > 0
    
    @pytest.mark.slow
    def test_full_evaluation(self):
        """Test full evaluation (marked as slow)."""
        config = ExperimentConfig(
            n_geos=5,
            n_days=20,
            n_simulations=3,  # Very small for testing
            n_bootstrap=10,
            seed=42
        )
        runner = ExperimentRunner(config)
        
        detailed_results, summary_results = runner.run_full_evaluation(verbose=False)
        
        # Check results structure
        assert isinstance(detailed_results, pd.DataFrame)
        assert isinstance(summary_results, pd.DataFrame)
        
        # Check columns
        expected_detailed_cols = {
            'simulation_id', 'assignment_method', 'reporting_method',
            'iroas_estimate', 'iroas_lower', 'iroas_upper', 'significant'
        }
        assert expected_detailed_cols.issubset(detailed_results.columns)
        
        expected_summary_cols = {
            'assignment_method', 'reporting_method', 'mean_iroas',
            'false_positive_rate', 'coverage_rate'
        }
        assert expected_summary_cols.issubset(summary_results.columns)
        
        # Check dimensions
        assert len(detailed_results) == 3  # n_simulations
        assert len(summary_results) == 1   # 1 method combination