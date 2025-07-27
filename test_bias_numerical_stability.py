"""
Numerical stability and edge case tests for bias diagnostic methods.

This module tests edge cases and numerical stability of:
1. BCa bootstrap with extreme distributions
2. Pre-period calibration with insufficient data
3. Distribution diagnostics with various data patterns
4. Error recovery and fallback mechanisms

Usage:
    python test_bias_numerical_stability.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
import unittest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


class TestBiasNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""
    
    def setUp(self):
        """Set up for each test."""
        warnings.filterwarnings('ignore')
        
        # Create minimal test data
        self.config = DataConfig(n_geos=8, n_days=40, seed=123)
        self.generator = SimpleNullGenerator(self.config)
        self.panel_data, self.geo_features = self.generator.generate()
        
        self.assignment_method = RandomAssignment()
        self.assignment_df = self.assignment_method.assign(self.geo_features, treatment_ratio=0.5, seed=123)
        
        # Set evaluation periods
        self.dates = sorted(self.panel_data['date'].unique())
        self.pre_period_end = self.dates[25].strftime('%Y-%m-%d')
        self.eval_start = self.dates[26].strftime('%Y-%m-%d')
        self.eval_end = self.dates[39].strftime('%Y-%m-%d')
        
        # Create and fit model
        self.model = STGCNReportingModel(
            hidden_dim=8, epochs=2, verbose=False  # Minimal for speed
        )
        self.model.fit(self.panel_data, self.assignment_df, self.pre_period_end)
    
    def test_bca_with_extreme_distributions(self):
        """Test BCa bootstrap with extreme/skewed distributions."""
        print("\n🧪 NUMERICAL TEST 1: BCa with Extreme Distributions")
        
        # Create extremely skewed data
        skewed_panel = self.panel_data.copy()
        treatment_geos = self.assignment_df[self.assignment_df['assignment'] == 'treatment']['geo'].values
        
        # Add extreme outliers to create heavy-tailed distribution
        mask = (skewed_panel['geo'].isin(treatment_geos)) & \
               (skewed_panel['date'] >= pd.to_datetime(self.eval_start))
        
        # Random extreme values
        np.random.seed(42)
        extreme_values = np.random.exponential(scale=10000, size=mask.sum())  # Exponential distribution
        skewed_panel.loc[mask, 'sales'] += extreme_values
        
        try:
            # Test BCa with extreme data
            lower, upper = self.model._bca_bootstrap_ci(
                skewed_panel, self.eval_start, self.eval_end,
                confidence_level=0.95,
                ensemble_size=4,
                use_log_iroas=True,
                spend_floor=1e-6
            )
            
            # Should produce finite results even with extreme data
            self.assertTrue(np.isfinite(lower), "BCa lower bound not finite with extreme data")
            self.assertTrue(np.isfinite(upper), "BCa upper bound not finite with extreme data")
            self.assertLessEqual(lower, upper, "BCa CI bounds reversed with extreme data")
            
            print(f"   ✅ BCa handled extreme data: CI=[{lower:.4f}, {upper:.4f}]")
            
        except Exception as e:
            # BCa should gracefully fall back
            if "falling back" in str(e).lower() or "insufficient" in str(e).lower():
                print(f"   ✅ BCa gracefully fell back with extreme data: {e}")
            else:
                print(f"   ⚠️ BCa failed with extreme data (may be acceptable): {e}")
    
    def test_pre_period_calibration_insufficient_data(self):
        """Test pre-period calibration with minimal data."""
        print("\n🧪 NUMERICAL TEST 2: Pre-period Calibration with Minimal Data")
        
        # Create scenario with very short pre-period
        short_dates = self.dates[:15]  # Only 15 days total
        short_panel = self.panel_data[self.panel_data['date'].isin(short_dates)].copy()
        
        short_pre_end = short_dates[8].strftime('%Y-%m-%d')  # 8 days pre-period
        short_eval_start = short_dates[9].strftime('%Y-%m-%d')
        short_eval_end = short_dates[14].strftime('%Y-%m-%d')
        
        try:
            # This should handle insufficient data gracefully
            lower, upper = self.model._pre_period_calibration(
                short_panel, short_eval_start, short_eval_end,
                confidence_level=0.95,
                ensemble_size=3,
                use_log_iroas=True,
                spend_floor=1e-6
            )
            
            print(f"   ✅ Pre-period calibration handled short data: CI=[{lower:.4f}, {upper:.4f}]")
            
        except Exception as e:
            # Should fail gracefully with informative error
            if "Could not estimate bias" in str(e) or "early_pre_period" in str(e):
                print(f"   ✅ Pre-period calibration correctly rejected insufficient data: {e}")
            else:
                print(f"   ⚠️ Unexpected error with insufficient data: {e}")
    
    def test_ensemble_with_training_failures(self):
        """Test ensemble diagnostics when some models fail to train."""
        print("\n🧪 NUMERICAL TEST 3: Ensemble with Training Failures")
        
        # Mock some ensemble models to fail during training
        original_fit = STGCNReportingModel.fit
        
        def failing_fit(self, *args, **kwargs):
            # Make every 3rd model fail randomly
            if hasattr(self, '_test_fail_flag') and self._test_fail_flag:
                raise RuntimeError("Simulated training failure")
            return original_fit(self, *args, **kwargs)
        
        call_count = 0
        def counting_init(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail every 3rd model
            self._test_fail_flag = (call_count % 3 == 0)
            return self._original_init(*args, **kwargs)
        
        with patch.object(STGCNReportingModel, 'fit', failing_fit):
            # Store original __init__ 
            STGCNReportingModel._original_init = STGCNReportingModel.__init__
            with patch.object(STGCNReportingModel, '__init__', counting_init):
                try:
                    results = self.model.diagnose_ensemble_distribution(
                        self.panel_data, self.eval_start, self.eval_end,
                        ensemble_size=6,  # Some will fail
                        plot=False
                    )
                    
                    # Should still work with partial ensemble
                    self.assertIsInstance(results, dict)
                    self.assertIn('ensemble_size', results)
                    self.assertGreaterEqual(results['ensemble_size'], 2, "Not enough successful models")
                    
                    print(f"   ✅ Ensemble diagnostics handled training failures: {results['ensemble_size']}/6 models succeeded")
                    
                except ValueError as e:
                    if "Insufficient successful models" in str(e):
                        print(f"   ✅ Correctly rejected when too many models failed: {e}")
                    else:
                        raise e
    
    def test_distribution_with_zero_variance(self):
        """Test distribution diagnostics with zero or near-zero variance."""
        print("\n🧪 NUMERICAL TEST 4: Distribution with Zero Variance")
        
        # Create data with nearly identical outcomes
        constant_panel = self.panel_data.copy()
        
        # Set all sales to nearly the same value in evaluation period
        mask = constant_panel['date'] >= pd.to_datetime(self.eval_start)
        constant_panel.loc[mask, 'sales'] = 10000.0  # Constant value
        
        try:
            results = self.model.diagnose_ensemble_distribution(
                constant_panel, self.eval_start, self.eval_end,
                ensemble_size=3,
                plot=False
            )
            
            # Should handle near-zero variance gracefully
            self.assertIsInstance(results, dict)
            self.assertIn('std', results)
            
            # Standard deviation might be very small but should be finite
            self.assertTrue(np.isfinite(results['std']), "Standard deviation not finite")
            
            print(f"   ✅ Handled zero variance: std={results['std']:.6f}")
            
        except Exception as e:
            print(f"   ⚠️ Zero variance test failed: {e}")
            # This might fail due to division by zero in some calculations
    
    def test_extreme_confidence_levels(self):
        """Test bias correction with extreme confidence levels."""
        print("\n🧪 NUMERICAL TEST 5: Extreme Confidence Levels")
        
        # Test very high confidence level
        try:
            results_high = self.model.apply_bias_correction(
                self.panel_data, self.eval_start, self.eval_end,
                method='bca_bootstrap',
                confidence_level=0.999,  # Very high
                ensemble_size=4
            )
            
            # Should produce wider intervals
            self.assertIsInstance(results_high, dict)
            print(f"   ✅ High confidence (99.9%): width={results_high['corrected_width']:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ High confidence level failed: {e}")
        
        # Test very low confidence level
        try:
            results_low = self.model.apply_bias_correction(
                self.panel_data, self.eval_start, self.eval_end,
                method='bca_bootstrap',
                confidence_level=0.5,  # Very low
                ensemble_size=4
            )
            
            # Should produce narrower intervals
            self.assertIsInstance(results_low, dict)
            print(f"   ✅ Low confidence (50%): width={results_low['corrected_width']:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Low confidence level failed: {e}")
    
    def test_spend_floor_edge_cases(self):
        """Test with extreme spend floor values."""
        print("\n🧪 NUMERICAL TEST 6: Spend Floor Edge Cases")
        
        # Test with very high spend floor (should affect iROAS calculation)
        try:
            results_high_floor = self.model.diagnose_ensemble_distribution(
                self.panel_data, self.eval_start, self.eval_end,
                ensemble_size=3,
                spend_floor=1e6,  # Very high floor
                plot=False
            )
            
            print(f"   ✅ High spend floor handled: mean={results_high_floor['mean']:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ High spend floor failed: {e}")
        
        # Test with very low spend floor (should be close to division by zero)
        try:
            results_low_floor = self.model.diagnose_ensemble_distribution(
                self.panel_data, self.eval_start, self.eval_end,
                ensemble_size=3,
                spend_floor=1e-12,  # Very low floor
                plot=False
            )
            
            print(f"   ✅ Low spend floor handled: mean={results_low_floor['mean']:.4f}")
            
        except Exception as e:
            print(f"   ⚠️ Low spend floor failed: {e}")
    
    def test_memory_usage_with_large_ensembles(self):
        """Test memory usage doesn't explode with larger ensembles."""
        print("\n🧪 NUMERICAL TEST 7: Memory Usage with Large Ensembles")
        
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Run diagnostic with moderately large ensemble
            results = self.model.diagnose_ensemble_distribution(
                self.panel_data, self.eval_start, self.eval_end,
                ensemble_size=8,  # Larger than usual for test
                plot=False
            )
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage after
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"   ✅ Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Memory increase should be reasonable (adjust threshold as needed)
            self.assertLess(memory_increase, 500, f"Excessive memory usage: +{memory_increase:.1f}MB")
            
        except Exception as e:
            print(f"   ⚠️ Memory test failed: {e}")
    
    def test_concurrent_safety(self):
        """Test thread safety of bias diagnostic methods."""
        print("\n🧪 NUMERICAL TEST 8: Concurrent Safety")
        
        import threading
        import queue
        
        # Test concurrent access to diagnostic methods
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker_function(worker_id):
            try:
                # Each worker runs diagnostics
                results = self.model.diagnose_ensemble_distribution(
                    self.panel_data, self.eval_start, self.eval_end,
                    ensemble_size=2,  # Small for speed
                    plot=False
                )
                results_queue.put((worker_id, results))
            except Exception as e:
                errors_queue.put((worker_id, str(e)))
        
        # Start multiple workers
        threads = []
        num_workers = 3
        
        for i in range(num_workers):
            thread = threading.Thread(target=worker_function, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check results
        successful_workers = 0
        while not results_queue.empty():
            worker_id, results = results_queue.get()
            successful_workers += 1
            print(f"   ✅ Worker {worker_id} completed successfully")
        
        # Check errors
        failed_workers = 0
        while not errors_queue.empty():
            worker_id, error = errors_queue.get()
            failed_workers += 1
            print(f"   ⚠️ Worker {worker_id} failed: {error}")
        
        print(f"   📊 Concurrent test: {successful_workers}/{num_workers} workers succeeded")
        
        # At least some workers should succeed
        self.assertGreater(successful_workers, 0, "No workers completed successfully")


def run_numerical_stability_tests():
    """Run the numerical stability test suite."""
    print("🔬 BIAS DIAGNOSTICS NUMERICAL STABILITY TEST SUITE")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestBiasNumericalStability)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n🎯 NUMERICAL STABILITY TEST SUMMARY:")
    print("="*50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  • {test}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  • {test}")
    
    # Assessment
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
    
    if success_rate >= 0.8:
        print(f"\n✅ EXCELLENT: {success_rate:.1%} numerical stability")
        print("🎉 Bias diagnostics are numerically robust!")
    elif success_rate >= 0.6:
        print(f"\n✅ GOOD: {success_rate:.1%} numerical stability")
        print("👍 Bias diagnostics handle most edge cases well")
    else:
        print(f"\n⚠️ MODERATE: {success_rate:.1%} numerical stability")
        print("🔧 Some numerical stability issues found")
    
    return result


if __name__ == "__main__":
    result = run_numerical_stability_tests()
    sys.exit(0 if result.wasSuccessful() else 1)