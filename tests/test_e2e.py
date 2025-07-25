"""
End-to-end tests for the geo-experimentation framework.

These tests run the complete pipeline from data generation through 
assignment, modeling, and evaluation to ensure all components work together.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from pipeline.config import ExperimentConfig
from pipeline.runner import ExperimentRunner
from assignment.methods import RandomAssignment, KMeansEmbeddingAssignment
from reporting.models import MeanMatchingModel


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_basic_pipeline_flow(self):
        """Test complete pipeline from data generation to results."""
        # Create minimal configuration for fast testing
        config = ExperimentConfig(
            n_geos=10,
            n_days=30,
            pre_period_days=20,
            eval_period_days=10,
            n_simulations=2,
            n_bootstrap=5,
            seed=42
        )
        
        runner = ExperimentRunner(config)
        
        # Test single experiment
        results = runner.run_single_experiment(show_plots=False)
        
        # Verify all expected outputs are present
        expected_keys = {
            'panel_data', 'geo_features', 'assignment_df', 
            'iroas_estimate', 'iroas_ci', 'significant'
        }
        assert set(results.keys()).issuperset(expected_keys)
        
        # Check data integrity
        assert len(results['panel_data']) == 10 * 30  # n_geos * n_days
        assert len(results['geo_features']) == 10     # n_geos
        assert len(results['assignment_df']) == 10    # n_geos
        
        # Check assignment balance
        treatment_count = (results['assignment_df']['assignment'] == 'treatment').sum()
        control_count = (results['assignment_df']['assignment'] == 'control').sum()
        assert treatment_count > 0 and control_count > 0  # Both groups have geos
        
        # Check result types
        assert isinstance(results['iroas_estimate'], float)
        assert isinstance(results['iroas_ci'], tuple)
        assert len(results['iroas_ci']) == 2
        assert isinstance(results['significant'], bool)
    
    def test_multiple_assignment_methods(self):
        """Test pipeline with different assignment methods."""
        config = ExperimentConfig(
            n_geos=12,
            n_days=25,
            pre_period_days=15,
            eval_period_days=10,
            seed=123
        )
        
        runner = ExperimentRunner(config)
        
        # Add multiple assignment methods
        runner.add_assignment_method("random", RandomAssignment())
        runner.add_assignment_method("kmeans", KMeansEmbeddingAssignment(n_clusters=3))
        
        # Test that both methods work
        for method_name in ["random", "kmeans"]:
            # Create a new runner with just one method
            single_runner = ExperimentRunner(config)
            if method_name == "random":
                single_runner.add_assignment_method("random", RandomAssignment())
            else:
                single_runner.add_assignment_method("kmeans", KMeansEmbeddingAssignment(n_clusters=3))
            
            results = single_runner.run_single_experiment(show_plots=False)
            
            # Verify assignment worked
            assert len(results['assignment_df']) == 12
            assert set(results['assignment_df']['assignment'].unique()) == {'treatment', 'control'}
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation with multiple simulations."""
        config = ExperimentConfig(
            n_geos=8,
            n_days=20,
            pre_period_days=12,
            eval_period_days=8,
            n_simulations=3,
            n_bootstrap=5,
            seed=456
        )
        
        runner = ExperimentRunner(config)
        
        # Run full evaluation
        detailed_results, summary_results = runner.run_full_evaluation(verbose=False)
        
        # Check results structure
        assert isinstance(detailed_results, pd.DataFrame)
        assert isinstance(summary_results, pd.DataFrame)
        
        # Check detailed results
        expected_detailed_cols = {
            'simulation_id', 'assignment_method', 'reporting_method',
            'iroas_estimate', 'iroas_lower', 'iroas_upper', 'significant'
        }
        assert expected_detailed_cols.issubset(detailed_results.columns)
        assert len(detailed_results) == 3  # n_simulations
        
        # Check summary results
        expected_summary_cols = {
            'assignment_method', 'reporting_method', 'mean_iroas',
            'false_positive_rate', 'coverage_rate'
        }
        assert expected_summary_cols.issubset(summary_results.columns)
        assert len(summary_results) == 1  # 1 method combination
        
        # Verify false positive rate is reasonable (should be around 0.05 for null data)
        fpr = summary_results['false_positive_rate'].iloc[0]
        assert 0 <= fpr <= 1, "False positive rate should be between 0 and 1"
    
    def test_csv_output_integration(self):
        """Test CSV export in full pipeline."""
        config = ExperimentConfig(
            n_geos=6,
            n_days=15,
            pre_period_days=10,
            eval_period_days=5,
            n_simulations=2,
            n_bootstrap=3,
            seed=789
        )
        
        runner = ExperimentRunner(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run with CSV export
            detailed_results, summary_results = runner.run_full_evaluation(
                verbose=False,
                save_csv=True,
                output_dir=temp_dir
            )
            
            # Check files were created
            files = os.listdir(temp_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            assert len(csv_files) >= 2, "Should create at least detailed and summary CSV files"
            
            # Verify file contents
            detailed_csv = [f for f in csv_files if 'detailed_results' in f][0]
            summary_csv = [f for f in csv_files if 'evaluation_summary' in f][0]
            
            detailed_loaded = pd.read_csv(os.path.join(temp_dir, detailed_csv))
            summary_loaded = pd.read_csv(os.path.join(temp_dir, summary_csv))
            
            # Check data integrity
            assert len(detailed_loaded) == 2  # n_simulations
            assert len(summary_loaded) == 1   # 1 method combination
            
            # Check key columns exist
            assert 'iroas_estimate' in detailed_loaded.columns
            assert 'false_positive_rate' in summary_loaded.columns
    
    def test_spatial_utilities_integration(self):
        """Test spatial utilities work in the full pipeline."""
        from assignment.spatial_utils import add_spectral_spatial_embedding
        from data_simulation.generators import SimpleNullGenerator
        
        # Generate test data with spatial coordinates
        from data_simulation.generators import SimpleNullGenerator
        
        config = ExperimentConfig(n_geos=15, n_days=20, seed=999)
        data_config = config.to_data_config()
        generator = SimpleNullGenerator(data_config)
        
        panel_data, geo_features = generator.generate()
        
        # Add fake spatial coordinates for testing (since SimpleNullGenerator doesn't include them)
        np.random.seed(999)
        geo_features['xy1'] = np.random.uniform(0, 100, len(geo_features))
        geo_features['xy2'] = np.random.uniform(0, 100, len(geo_features))
        
        # Add spatial embedding
        geo_features_with_spatial = add_spectral_spatial_embedding(
            geo_features, 
            spatial_cols=['xy1', 'xy2'],
            spatial_emb_dim=2,
            spatial_neighbors=5,
            seed=999
        )
        
        # Check spatial columns were added
        spatial_cols = [col for col in geo_features_with_spatial.columns if 'spatial_emb_' in col]
        assert len(spatial_cols) == 2  # spatial_emb_0, spatial_emb_1
        
        # Use in assignment method
        kmeans_method = KMeansEmbeddingAssignment(
            feature_cols=['base_sales', 'base_spend'] + spatial_cols,
            n_clusters=4
        )
        
        assignment_df = kmeans_method.assign(geo_features_with_spatial, seed=999)
        
        # Verify assignment worked with spatial features
        assert len(assignment_df) == 15
        assert set(assignment_df['assignment'].unique()) == {'treatment', 'control'}
        assert 'cluster' in assignment_df.columns
    
    @pytest.mark.slow
    def test_comprehensive_pipeline(self):
        """Comprehensive test of full pipeline with all features."""
        config = ExperimentConfig(
            n_geos=20,
            n_days=40,
            pre_period_days=25,
            eval_period_days=15,
            n_simulations=5,
            n_bootstrap=10,
            seed=2024
        )
        
        runner = ExperimentRunner(config)
        
        # Add multiple methods
        runner.add_assignment_method("random", RandomAssignment())
        runner.add_assignment_method("kmeans", KMeansEmbeddingAssignment(n_clusters=5))
        runner.add_reporting_method("mean_matching", MeanMatchingModel())
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run full evaluation with all outputs
            detailed_results, summary_results = runner.run_full_evaluation(
                verbose=False,
                save_csv=True,
                save_plots=True,
                output_dir=temp_dir
            )
            
            # Check results dimensions
            # Runner starts with default methods, plus we added 2 assignment and 1 reporting method
            # So we expect multiple combinations
            assert len(summary_results) >= 2, f"Expected at least 2 combinations, got {len(summary_results)}"
            assert len(detailed_results) >= 10, f"Expected at least 10 detailed results, got {len(detailed_results)}"
            
            # Check all files were created
            files = os.listdir(temp_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            png_files = [f for f in files if f.endswith('.png')]
            
            assert len(csv_files) >= 2, "Should have detailed and summary CSV files"
            assert len(png_files) >= 1, "Should have at least one plot file"
            
            # Verify method comparison worked
            methods_in_summary = set(summary_results['assignment_method'].unique())
            assert 'random' in methods_in_summary
            assert 'kmeans' in methods_in_summary
            
            # Check false positive rates are reasonable
            fprs = summary_results['false_positive_rate']
            assert all(0 <= fpr <= 1 for fpr in fprs), "All FPRs should be between 0 and 1"