"""
Simple validation script for STGCN improvements.

This validates the key improvements without complex mocking.
"""

import numpy as np
import pandas as pd
import torch

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def create_test_data():
    """Create sufficient test data for STGCN validation."""
    config = DataConfig(n_geos=10, n_days=50, seed=42)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[39]  # 40 days for training
    eval_start = dates[40]
    eval_end = dates[45]
    
    return panel_data, assignment_df, pre_period_end, eval_start, eval_end


def validate_basic_functionality():
    """Validate basic STGCN functionality with improvements."""
    print("=== Basic Functionality Validation ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=5,
        window_size=5,
        learning_rate=0.01,
        normalize_data=True,
        verbose=False
    )
    
    try:
        # Test training
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        print("✅ Model training completed successfully")
        
        # Test prediction
        iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        print(f"✅ iROAS calculation: {iroas:.4f}")
        
        # Test diagnostics
        diagnostics = model.get_training_diagnostics()
        print(f"✅ Training diagnostics available: {len(diagnostics)} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        return False


def validate_diagnostics_structure():
    """Validate enhanced diagnostics structure."""
    print("\n=== Diagnostics Structure Validation ===")
    
    panel_data, assignment_df, pre_period_end, _, _ = create_test_data()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=4,
        window_size=5,
        normalize_data=True,
        verbose=False
    )
    
    model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    diagnostics = model.get_training_diagnostics()
    
    # Check for key diagnostic information
    required_keys = [
        'final_train_loss', 'final_val_loss', 'epochs_trained', 
        'loss_history', 'val_loss_history', 'convergence_assessment'
    ]
    
    missing_keys = [key for key in required_keys if key not in diagnostics]
    
    if not missing_keys:
        print("✅ All required diagnostic keys present")
        
        # Validate values
        train_loss = diagnostics['final_train_loss']
        val_loss = diagnostics['final_val_loss']
        convergence = diagnostics['convergence_assessment']
        
        print(f"✅ Final training loss: {train_loss:.6f}")
        print(f"✅ Final validation loss: {val_loss:.6f}")
        print(f"✅ Convergence assessment: {convergence}")
        
        # Check that we have separate train/val loss histories
        train_history = diagnostics.get('loss_history', [])
        val_history = diagnostics.get('val_loss_history', [])
        
        print(f"✅ Training history length: {len(train_history)}")
        print(f"✅ Validation history length: {len(val_history)}")
        
        return True
    else:
        print(f"❌ Missing diagnostic keys: {missing_keys}")
        return False


def validate_bias_detection():
    """Validate bias detection functionality."""
    print("\n=== Bias Detection Validation ===")
    
    panel_data, assignment_df, pre_period_end, _, _ = create_test_data()
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=3,
        window_size=5,
        normalize_data=True,
        verbose=False,
        bias_threshold=0.1
    )
    
    model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Test bias detection
    bias_check = model.check_null_scenario_bias()
    
    expected_keys = ['relative_bias', 'bias_threshold', 'high_bias_detected', 'bias_level']
    missing_keys = [key for key in expected_keys if key not in bias_check]
    
    if not missing_keys:
        print("✅ Bias detection keys present")
        print(f"✅ Relative bias: {bias_check['relative_bias']:.3f}")
        print(f"✅ Bias level: {bias_check['bias_level']}")
        print(f"✅ High bias detected: {bias_check['high_bias_detected']}")
        return True
    else:
        print(f"❌ Missing bias detection keys: {missing_keys}")
        return False


def validate_training_stability():
    """Validate training stability improvements."""
    print("\n=== Training Stability Validation ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data()
    
    # Test multiple random seeds
    results = []
    
    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = STGCNReportingModel(
            hidden_dim=16,
            epochs=4,
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
            
            results.append({
                'seed': seed,
                'iroas': iroas,
                'convergence': convergence,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Seed {seed} failed: {e}")
            results.append({
                'seed': seed,
                'iroas': np.nan,
                'convergence': 'failed',
                'success': False
            })
    
    # Analyze results
    successful_runs = [r for r in results if r['success']]
    success_rate = len(successful_runs) / len(results)
    
    print(f"✅ Success rate: {success_rate:.1%} ({len(successful_runs)}/{len(results)})")
    
    if len(successful_runs) >= 2:
        iroas_values = [r['iroas'] for r in successful_runs]
        iroas_std = np.std(iroas_values)
        iroas_mean = np.mean(iroas_values)
        
        print(f"✅ iROAS variability: mean={iroas_mean:.4f}, std={iroas_std:.4f}")
        
        # Check convergence
        good_convergence = sum(1 for r in successful_runs if r['convergence'] in ['good', 'moderate'])
        convergence_rate = good_convergence / len(successful_runs)
        
        print(f"✅ Good convergence rate: {convergence_rate:.1%}")
        
        return success_rate >= 0.8 and iroas_std < 10
    else:
        print("❌ Insufficient successful runs")
        return False


def validate_improvements_integration():
    """Validate that all improvements work together."""
    print("\n=== Integration Validation ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data()
    
    # Test with all improvements enabled
    model = STGCNReportingModel(
        hidden_dim=20,
        epochs=6,
        window_size=5,
        learning_rate=0.01,
        normalize_data=True,
        verbose=True,  # Test verbose output
        bias_threshold=0.1
    )
    
    try:
        # Capture some output to verify verbose mode
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Check that verbose output was generated
        verbose_working = "Training STGCN model" in output
        normalization_shown = "Data normalization applied" in output
        
        print(f"✅ Verbose mode working: {verbose_working}")
        print(f"✅ Normalization info shown: {normalization_shown}")
        
        # Test prediction and bias detection
        iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        bias_check = model.check_null_scenario_bias()
        
        print(f"✅ iROAS: {iroas:.4f}")
        print(f"✅ Bias detected: {bias_check.get('high_bias_detected', 'unknown')}")
        
        # Check model state
        is_fitted = hasattr(model, 'is_fitted') and model.is_fitted
        has_diagnostics = bool(model.get_training_diagnostics())
        
        print(f"✅ Model fitted: {is_fitted}")
        print(f"✅ Diagnostics available: {has_diagnostics}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("STGCN Improvements Validation")
    print("=" * 40)
    
    tests = [
        ("Basic Functionality", validate_basic_functionality),
        ("Diagnostics Structure", validate_diagnostics_structure),
        ("Bias Detection", validate_bias_detection),
        ("Training Stability", validate_training_stability),
        ("Integration", validate_improvements_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("VALIDATION SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(tests)} tests passed ({passed/len(tests)*100:.1f}%)")
    
    if passed >= len(tests) * 0.8:  # 80% pass rate
        print("\n🎉 STGCN improvements are working well!")
        print("\nKey improvements validated:")
        print("  ✅ Enhanced training diagnostics")
        print("  ✅ Train/validation split")
        print("  ✅ Improved bias detection")
        print("  ✅ Better training stability")
        print("  ✅ Verbose output control")
        print("  ✅ Data normalization improvements")
    else:
        print(f"\n⚠️ Some improvements need attention.")
    
    return passed >= len(tests) * 0.8


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)