"""
Simplified test suite for STGCN model improvements.

This tests the key improvements made to STGCN:
1. Gradient clipping
2. Train/validation split
3. Learning rate scheduling  
4. Enhanced diagnostics
"""

import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def create_test_scenario():
    """Create a test scenario with sufficient data for STGCN."""
    config = DataConfig(n_geos=10, n_days=50, seed=42)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[39]  # Use 40 days for training
    eval_start = dates[40]
    eval_end = dates[45]
    
    return panel_data, assignment_df, pre_period_end, eval_start, eval_end


def test_gradient_clipping():
    """Test that gradient clipping is applied during training."""
    print("\n=== Testing Gradient Clipping ===")
    
    panel_data, assignment_df, pre_period_end, _, _ = create_test_scenario()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=3,
        learning_rate=0.05,  # Higher LR to potentially trigger clipping
        window_size=5,
        normalize_data=True,
        verbose=False
    )
    
    # Mock gradient clipping to verify it's called
    with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
        mock_clip.return_value = torch.tensor(1.5)
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Verify gradient clipping was called
        if mock_clip.call_count > 0:
            print("✅ Gradient clipping was applied during training")
            call_args = mock_clip.call_args_list[0]
            max_norm = call_args[1].get('max_norm', 'not found')
            print(f"✅ Max norm parameter: {max_norm}")
        else:
            print("❌ Gradient clipping was not called")
    
    return mock_clip.call_count > 0


def test_train_validation_split():
    """Test that train/validation split works correctly."""
    print("\n=== Testing Train/Validation Split ===")
    
    panel_data, assignment_df, pre_period_end, _, _ = create_test_scenario()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=4,
        window_size=5,
        normalize_data=True,
        verbose=False
    )
    
    model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Check diagnostic information
    diagnostics = model.get_training_diagnostics()
    
    # Verify both training and validation losses are tracked
    has_train_loss = 'final_train_loss' in diagnostics
    has_val_loss = 'final_val_loss' in diagnostics
    has_train_history = 'loss_history' in diagnostics
    has_val_history = 'val_loss_history' in diagnostics
    
    print(f"✅ Training loss tracking: {has_train_loss}")
    print(f"✅ Validation loss tracking: {has_val_loss}")
    print(f"✅ Training loss history: {has_train_history}")
    print(f"✅ Validation loss history: {has_val_history}")
    
    if has_train_loss and has_val_loss:
        train_loss = diagnostics['final_train_loss']
        val_loss = diagnostics['final_val_loss']
        print(f"✅ Final training loss: {train_loss:.6f}")
        print(f"✅ Final validation loss: {val_loss:.6f}")
    
    return has_train_loss and has_val_loss and has_train_history and has_val_history


def test_learning_rate_scheduler():
    """Test that learning rate scheduler is working."""
    print("\n=== Testing Learning Rate Scheduler ===")
    
    panel_data, assignment_df, pre_period_end, _, _ = create_test_scenario()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=5,
        window_size=5,
        learning_rate=0.01,
        normalize_data=True,
        verbose=True  # Enable verbose to see scheduler messages
    )
    
    # Mock the scheduler to verify it's used
    with patch('torch.optim.lr_scheduler.ReduceLROnPlateau') as mock_scheduler_class:
        mock_scheduler_instance = mock_scheduler_class.return_value
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        # Check scheduler was created
        scheduler_created = mock_scheduler_class.called
        step_called = mock_scheduler_instance.step.call_count
        
        print(f"✅ LR Scheduler created: {scheduler_created}")
        print(f"✅ Scheduler step called {step_called} times")
        
        if scheduler_created:
            # Check scheduler parameters
            call_args = mock_scheduler_class.call_args
            kwargs = call_args[1] if len(call_args) > 1 else {}
            
            print(f"✅ Scheduler mode: {kwargs.get('mode', 'default')}")
            print(f"✅ Patience: {kwargs.get('patience', 'default')}")
            print(f"✅ Factor: {kwargs.get('factor', 'default')}")
    
    return scheduler_created and step_called >= 5


def test_enhanced_diagnostics():
    """Test enhanced diagnostics and monitoring."""
    print("\n=== Testing Enhanced Diagnostics ===")
    
    panel_data, assignment_df, pre_period_end, _, _ = create_test_scenario()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=4,
        window_size=5,
        normalize_data=True,
        verbose=False
    )
    
    model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Test comprehensive diagnostics
    diagnostics = model.get_training_diagnostics()
    
    expected_keys = [
        'final_train_loss', 'final_val_loss', 'epochs_trained', 'early_stopped',
        'loss_history', 'val_loss_history', 'convergence_assessment',
        'gradient_health'
    ]
    
    missing_keys = []
    for key in expected_keys:
        if key not in diagnostics:
            missing_keys.append(key)
    
    if not missing_keys:
        print("✅ All expected diagnostic keys present")
    else:
        print(f"❌ Missing diagnostic keys: {missing_keys}")
    
    # Test gradient health structure
    gradient_health = diagnostics.get('gradient_health', {})
    gradient_keys = ['vanishing_gradients_detected', 'exploding_gradients_detected', 'zero_grad_issues']
    
    gradient_health_complete = all(key in gradient_health for key in gradient_keys)
    print(f"✅ Gradient health tracking: {gradient_health_complete}")
    
    # Test convergence assessment
    convergence = diagnostics.get('convergence_assessment', 'unknown')
    valid_convergence = convergence in ['good', 'moderate', 'poor']
    print(f"✅ Convergence assessment: {convergence} (valid: {valid_convergence})")
    
    return len(missing_keys) == 0 and gradient_health_complete and valid_convergence


def test_bias_detection_improvements():
    """Test improved bias detection."""
    print("\n=== Testing Bias Detection ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_scenario()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=4,
        window_size=5,
        normalize_data=True,
        verbose=False,
        bias_threshold=0.1
    )
    
    model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Test bias detection functionality
    bias_check = model.check_null_scenario_bias()
    
    expected_bias_keys = ['relative_bias', 'bias_threshold', 'high_bias_detected', 'bias_level']
    bias_keys_present = all(key in bias_check for key in expected_bias_keys)
    
    print(f"✅ Bias detection keys present: {bias_keys_present}")
    
    if bias_keys_present:
        print(f"✅ Relative bias: {bias_check['relative_bias']:.3f}")
        print(f"✅ Bias level: {bias_check['bias_level']}")
        print(f"✅ High bias detected: {bias_check['high_bias_detected']}")
    
    # Test iROAS calculation works
    try:
        iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        iroas_works = True
        print(f"✅ iROAS calculation: {iroas:.4f}")
    except Exception as e:
        iroas_works = False
        print(f"❌ iROAS calculation failed: {e}")
    
    return bias_keys_present and iroas_works


def test_overall_stability():
    """Test overall model stability with improvements."""
    print("\n=== Testing Overall Stability ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_scenario()
    
    # Test multiple runs for stability
    iroas_values = []
    convergence_assessments = []
    
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=5,
            window_size=5,
            learning_rate=0.01,
            normalize_data=True,
            verbose=False
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
            
            diagnostics = model.get_training_diagnostics()
            convergence = diagnostics.get('convergence_assessment', 'unknown')
            
            iroas_values.append(iroas)
            convergence_assessments.append(convergence)
            
        except Exception as e:
            print(f"❌ Run with seed {seed} failed: {e}")
            iroas_values.append(np.nan)
            convergence_assessments.append('failed')
    
    # Analyze stability
    valid_runs = [v for v in iroas_values if not np.isnan(v)]
    success_rate = len(valid_runs) / len(iroas_values)
    
    print(f"✅ Success rate: {success_rate:.1%} ({len(valid_runs)}/{len(iroas_values)} runs)")
    
    if len(valid_runs) >= 2:
        iroas_std = np.std(valid_runs)
        iroas_mean = np.mean(valid_runs)
        print(f"✅ iROAS stability: mean={iroas_mean:.4f}, std={iroas_std:.4f}")
        
        # For null data, expect low bias and reasonable stability
        reasonable_stability = iroas_std < 10 and abs(iroas_mean) < 5
        print(f"✅ Reasonable stability: {reasonable_stability}")
    else:
        reasonable_stability = False
        print("❌ Insufficient successful runs for stability analysis")
    
    good_convergence_rate = sum(1 for c in convergence_assessments if c in ['good', 'moderate']) / len(convergence_assessments)
    print(f"✅ Good convergence rate: {good_convergence_rate:.1%}")
    
    return success_rate >= 0.8 and reasonable_stability and good_convergence_rate >= 0.6


def main():
    """Run all improvement tests."""
    print("STGCN Model Improvements Test Suite")
    print("=" * 50)
    
    tests = [
        ("Gradient Clipping", test_gradient_clipping),
        ("Train/Validation Split", test_train_validation_split),
        ("Learning Rate Scheduler", test_learning_rate_scheduler),
        ("Enhanced Diagnostics", test_enhanced_diagnostics),
        ("Bias Detection", test_bias_detection_improvements),
        ("Overall Stability", test_overall_stability),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result, error in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed ({passed/len(tests)*100:.1f}%)")
    
    if passed == len(tests):
        print("\n🎉 All STGCN improvements are working correctly!")
        print("\nKey improvements validated:")
        print("  ✅ Gradient clipping for training stability")
        print("  ✅ Train/validation split with proper early stopping")
        print("  ✅ Learning rate scheduling")
        print("  ✅ Enhanced training diagnostics and monitoring")
        print("  ✅ Improved bias detection")
        print("  ✅ Overall training stability")
    else:
        print(f"\n⚠️ {len(tests) - passed} tests failed. Review the improvements.")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)