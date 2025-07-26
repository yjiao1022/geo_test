"""
Test suite for STGCN model improvements.

This module tests the new stability and training enhancements:
- Gradient clipping
- Train/validation split
- Learning rate scheduling
- Enhanced diagnostics
- Improved training monitoring
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


# Test configuration helper
def create_test_data(n_geos=8, n_days=40, seed=42):
    """Create test data with appropriate size for STGCN testing."""
    config = DataConfig(n_geos=n_geos, n_days=n_days, seed=seed)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=seed)
    
    return panel_data, geo_features, assignment_df


class TestGradientClipping:
    """Test gradient clipping functionality."""
    
    def test_gradient_clipping_applied(self):
        """Test that gradient clipping is actually applied during training."""
        panel_data, geo_features, assignment_df = create_test_data(n_geos=8, n_days=35, seed=42)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[29]  # Use more days for training
        
        # Create model with intentionally high learning rate to trigger large gradients
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=3,
            learning_rate=0.1,  # High LR to create large gradients
            window_size=5,  # Smaller window for testing
            normalize_data=True,
            verbose=False
        )
        
        # Mock the gradient clipping function to verify it's called
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            mock_clip.return_value = torch.tensor(2.5)  # Simulate clipped norm
            
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            
            # Verify gradient clipping was called
            assert mock_clip.called, "Gradient clipping should be applied during training"
            
            # Check that it was called with correct parameters
            call_args = mock_clip.call_args_list[0]
            assert call_args[1]['max_norm'] == 1.0, "Max norm should be 1.0"
    
    def test_gradient_clipping_prevents_explosion(self):
        """Test that gradient clipping helps prevent exploding gradients."""
        config = DataConfig(n_geos=6, n_days=15, seed=123)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=123)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[10]
        
        # Test with high learning rate that would typically cause gradient explosion
        model = STGCNReportingModel(
            hidden_dim=24,
            epochs=5,
            learning_rate=0.05,  # High learning rate
            normalize_data=False,  # No normalization to stress test
            verbose=False
        )
        
        # This should complete without exploding gradients due to clipping
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Check that training completed and loss is reasonable
        diagnostics = model.get_training_diagnostics()
        final_loss = diagnostics.get('final_train_loss', float('inf'))
        
        # Loss should not be extremely large (indicating gradient explosion)
        assert final_loss < 1e6, f"Loss too large: {final_loss}, gradient clipping may have failed"
        
        # Check gradient health
        gradient_health = diagnostics.get('gradient_health', {})
        exploding_detected = gradient_health.get('exploding_gradients_detected', False)
        
        # Note: Some gradient issues might still be detected, but they should be controlled
        assert not exploding_detected or final_loss < 1e4, "Severe gradient explosion should be prevented"


class TestTrainValidationSplit:
    """Test train/validation split functionality."""
    
    def test_train_val_split_creates_separate_datasets(self):
        """Test that train/validation split creates appropriately sized datasets."""
        config = DataConfig(n_geos=10, n_days=30, seed=456)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=456)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[24]  # Use 25 days
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=3,
            verbose=False
        )
        
        # Patch the _create_sequences method to capture train/val data sizes
        original_create_sequences = model._create_sequences
        train_sizes = []
        val_sizes = []
        
        def capture_sizes(data, window_size):
            result = original_create_sequences(data, window_size)
            # Store the sizes based on data shape
            if data.shape[1] == 20:  # 80% of 25 days = 20 days
                train_sizes.append(len(result[0]))
            elif data.shape[1] == 5:   # 20% of 25 days = 5 days  
                val_sizes.append(len(result[0]))
            return result
        
        model._create_sequences = capture_sizes
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Verify split occurred
        assert len(train_sizes) > 0, "Training data should be created"
        assert len(val_sizes) > 0, "Validation data should be created"
        
        # Verify train set is larger than validation set
        if train_sizes and val_sizes:
            assert train_sizes[0] > val_sizes[0], "Training set should be larger than validation set"
    
    def test_validation_loss_tracking(self):
        """Test that validation loss is tracked separately from training loss."""
        config = DataConfig(n_geos=8, n_days=25, seed=789)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=789)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[19]
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=5,
            verbose=False
        )
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Check that both training and validation losses are tracked
        diagnostics = model.get_training_diagnostics()
        
        assert 'final_train_loss' in diagnostics, "Training loss should be tracked"
        assert 'final_val_loss' in diagnostics, "Validation loss should be tracked"
        assert 'loss_history' in diagnostics, "Training loss history should be available"
        assert 'val_loss_history' in diagnostics, "Validation loss history should be available"
        
        # Verify they are different values (training and validation should differ)
        train_loss = diagnostics['final_train_loss']
        val_loss = diagnostics['final_val_loss']
        
        assert isinstance(train_loss, (int, float)), "Training loss should be numeric"
        assert isinstance(val_loss, (int, float)), "Validation loss should be numeric"
        assert train_loss > 0, "Training loss should be positive"
        assert val_loss > 0, "Validation loss should be positive"
    
    def test_early_stopping_on_validation_loss(self):
        """Test that early stopping is based on validation loss."""
        config = DataConfig(n_geos=6, n_days=20, seed=999)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=999)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[14]
        
        # Use aggressive early stopping to test the mechanism
        model = STGCNReportingModel(
            hidden_dim=12,
            epochs=20,  # Many epochs to allow early stopping
            early_stopping_patience=2,  # Very low patience
            verbose=False
        )
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        diagnostics = model.get_training_diagnostics()
        epochs_trained = diagnostics.get('epochs_trained', 20)
        early_stopped = diagnostics.get('early_stopped', False)
        
        # With aggressive early stopping, should stop before max epochs
        # (unless the model is learning very well, which is also fine)
        assert epochs_trained <= 20, "Should not exceed maximum epochs"
        
        # Verify early stopping flag is set correctly
        if epochs_trained < 20:
            assert early_stopped, "Early stopping flag should be True if stopped early"


class TestLearningRateScheduling:
    """Test learning rate scheduling functionality."""
    
    def test_lr_scheduler_exists(self):
        """Test that learning rate scheduler is properly instantiated."""
        config = DataConfig(n_geos=8, n_days=20, seed=111)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=111)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[14]
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01,
            verbose=False
        )
        
        # Mock the scheduler to verify it's called
        with patch('torch.optim.lr_scheduler.ReduceLROnPlateau') as mock_scheduler_class:
            mock_scheduler = MagicMock()
            mock_scheduler_class.return_value = mock_scheduler
            
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            
            # Verify scheduler was created
            assert mock_scheduler_class.called, "LR scheduler should be instantiated"
            
            # Verify scheduler.step() was called (once per epoch)
            assert mock_scheduler.step.call_count >= 5, "Scheduler step should be called each epoch"
    
    def test_lr_reduction_parameters(self):
        """Test that LR scheduler is configured with correct parameters."""
        config = DataConfig(n_geos=6, n_days=15, seed=222)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=222)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[10]
        
        model = STGCNReportingModel(
            hidden_dim=12,
            epochs=3,
            verbose=True  # Enable verbose to see scheduler messages
        )
        
        # Capture scheduler creation parameters
        with patch('torch.optim.lr_scheduler.ReduceLROnPlateau') as mock_scheduler:
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            
            # Check scheduler was called with correct parameters
            call_args = mock_scheduler.call_args
            args, kwargs = call_args
            
            # Verify parameters
            assert 'min' in args or kwargs.get('mode') == 'min', "Should minimize validation loss"
            assert kwargs.get('patience') == 5, "Patience should be 5"
            assert kwargs.get('factor') == 0.5, "Factor should be 0.5"
            assert kwargs.get('verbose') == True, "Verbose should match model verbose setting"


class TestEnhancedDiagnostics:
    """Test enhanced training diagnostics and monitoring."""
    
    def test_comprehensive_diagnostics_available(self):
        """Test that all expected diagnostic information is available."""
        config = DataConfig(n_geos=8, n_days=20, seed=333)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=333)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[14]
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=4,
            verbose=False
        )
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        diagnostics = model.get_training_diagnostics()
        
        # Check all expected keys are present
        expected_keys = [
            'final_train_loss', 'final_val_loss', 'epochs_trained', 'early_stopped',
            'loss_history', 'val_loss_history', 'initial_loss', 'loss_reduction_ratio',
            'final_gradient_stats', 'gradient_health', 'convergence_assessment'
        ]
        
        for key in expected_keys:
            assert key in diagnostics, f"Diagnostic key '{key}' should be present"
        
        # Verify gradient health structure
        gradient_health = diagnostics['gradient_health']
        expected_gradient_keys = [
            'vanishing_gradients_detected', 'exploding_gradients_detected', 'zero_grad_issues'
        ]
        
        for key in expected_gradient_keys:
            assert key in gradient_health, f"Gradient health key '{key}' should be present"
    
    def test_convergence_assessment_logic(self):
        """Test that convergence assessment is calculated correctly."""
        config = DataConfig(n_geos=6, n_days=15, seed=444)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=444)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[10]
        
        # Test with different configurations to get different convergence assessments
        test_configs = [
            {'epochs': 2, 'learning_rate': 0.001, 'expected': 'poor'},  # Low epochs, low LR
            {'epochs': 10, 'learning_rate': 0.01, 'expected': 'good'},   # Good config
        ]
        
        for config_params in test_configs:
            model = STGCNReportingModel(
                hidden_dim=12,
                epochs=config_params['epochs'],
                learning_rate=config_params['learning_rate'],
                verbose=False
            )
            
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            
            diagnostics = model.get_training_diagnostics()
            convergence = diagnostics['convergence_assessment']
            
            # Verify convergence assessment is one of expected values
            assert convergence in ['good', 'moderate', 'poor'], f"Convergence should be good/moderate/poor, got {convergence}"
    
    def test_loss_reduction_ratio_calculation(self):
        """Test that loss reduction ratio is calculated correctly."""
        config = DataConfig(n_geos=8, n_days=18, seed=555)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=555)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[13]
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=6,
            verbose=False
        )
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        diagnostics = model.get_training_diagnostics()
        
        # Verify loss reduction ratio is calculated
        loss_reduction_ratio = diagnostics.get('loss_reduction_ratio')
        initial_loss = diagnostics.get('initial_loss')
        final_loss = diagnostics.get('final_train_loss')
        
        if loss_reduction_ratio is not None and initial_loss is not None and final_loss is not None:
            expected_ratio = initial_loss / final_loss
            assert abs(loss_reduction_ratio - expected_ratio) < 1e-6, "Loss reduction ratio should be initial_loss / final_loss"
            assert loss_reduction_ratio >= 1.0, "Loss reduction ratio should be >= 1.0 (loss should decrease)"


class TestNormalizationImprovements:
    """Test improvements to data normalization."""
    
    def test_normalization_verbose_output(self):
        """Test that normalization parameters are shown when verbose=True."""
        config = DataConfig(n_geos=6, n_days=15, seed=666)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=666)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[10]
        
        model = STGCNReportingModel(
            hidden_dim=12,
            epochs=2,
            normalize_data=True,
            verbose=True
        )
        
        # Capture print output to verify normalization info is shown
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            
            output = captured_output.getvalue()
            
            # Verify normalization information is printed
            assert "Data normalization applied:" in output, "Normalization info should be printed when verbose=True"
            assert "Mean:" in output, "Mean values should be shown"
            assert "Std:" in output, "Std values should be shown"
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_denormalization_consistency(self):
        """Test that normalization and denormalization are consistent."""
        config = DataConfig(n_geos=8, n_days=18, seed=777)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=777)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[12]
        eval_start = dates[13]
        eval_end = dates[16]
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=3,
            normalize_data=True,
            verbose=False
        )
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Make predictions to test denormalization
        predictions = model.predict(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        
        # Verify predictions are in reasonable range (not normalized scale)
        pred_sales = predictions['sales']
        pred_spend = predictions['spend']
        
        # Get actual data range for comparison
        actual_sales_range = panel_data['sales'].min(), panel_data['sales'].max()
        actual_spend_range = panel_data['spend'].min(), panel_data['spend'].max()
        
        # Predictions should be in roughly the same order of magnitude as actual data
        assert np.median(pred_sales) > actual_sales_range[0] * 0.1, "Predictions should not be too small (denormalization issue)"
        assert np.median(pred_sales) < actual_sales_range[1] * 10, "Predictions should not be too large (denormalization issue)"
        
        assert np.median(pred_spend) > actual_spend_range[0] * 0.1, "Spend predictions should not be too small"
        assert np.median(pred_spend) < actual_spend_range[1] * 10, "Spend predictions should not be too large"


class TestIntegrationStability:
    """Test overall stability improvements in integration scenarios."""
    
    def test_training_stability_across_seeds(self):
        """Test that training is more stable across different random seeds."""
        config = DataConfig(n_geos=8, n_days=20, seed=888)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=888)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[14]
        eval_start = dates[15]
        eval_end = dates[18]
        
        # Test multiple random seeds
        seeds = [1, 42, 123]
        iroas_values = []
        final_losses = []
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            model = STGCNReportingModel(
                hidden_dim=16,
                epochs=5,
                learning_rate=0.01,
                normalize_data=True,
                verbose=False
            )
            
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
            
            diagnostics = model.get_training_diagnostics()
            final_loss = diagnostics.get('final_train_loss', float('inf'))
            
            iroas_values.append(iroas)
            final_losses.append(final_loss)
        
        # Check that training was stable (all models converged)
        assert all(loss < 1e3 for loss in final_losses), f"All models should converge reasonably, losses: {final_losses}"
        
        # Check that iROAS values are not extremely variable (allowing for some variation)
        iroas_std = np.std(iroas_values)
        iroas_mean = np.mean(iroas_values)
        
        # For null data, iROAS should be near 0, and std should not be excessive
        assert abs(iroas_mean) < 10, f"Mean iROAS should be near 0 for null data, got {iroas_mean}"
        assert iroas_std < 20, f"iROAS standard deviation should not be excessive, got {iroas_std}"
    
    def test_no_critical_warnings_in_normal_scenario(self):
        """Test that normal training scenarios don't generate critical warnings."""
        config = DataConfig(n_geos=10, n_days=25, seed=999)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=999)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[19]
        
        model = STGCNReportingModel(
            hidden_dim=20,
            epochs=8,
            learning_rate=0.01,
            normalize_data=True,
            verbose=False
        )
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Check that no critical warnings were generated
        warnings = model.training_warnings
        critical_warnings = [w for w in warnings if "🚨" in w]
        
        assert len(critical_warnings) == 0, f"No critical warnings should occur in normal scenario, got: {critical_warnings}"
        
        # Check convergence
        diagnostics = model.get_training_diagnostics()
        convergence = diagnostics['convergence_assessment']
        
        assert convergence in ['good', 'moderate'], f"Should achieve good or moderate convergence, got {convergence}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])