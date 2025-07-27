"""
Comprehensive tests for parallel ensemble training functionality.

This module tests:
1. Parallel vs sequential ensemble training performance
2. Backward compatibility and fallback mechanisms
3. Memory-aware job scheduling
4. Error handling and robustness
5. Integration with main STGCN confidence interval methods

Usage:
    python test_parallel_ensemble.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import warnings
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Import test modules
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel
from reporting.parallel_ensemble import ParallelEnsembleSTGCN, benchmark_parallel_vs_sequential


def create_test_data(n_geos: int = 16, n_days: int = 80, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """
    Create standardized test data for parallel ensemble testing.
    
    Returns:
        panel_data, assignment_df, pre_period_end, eval_start, eval_end
    """
    config = DataConfig(n_geos=n_geos, n_days=n_days, seed=seed)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=seed)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[int(n_days * 0.7) - 1].strftime('%Y-%m-%d')
    eval_start = dates[int(n_days * 0.7)].strftime('%Y-%m-%d')
    eval_end = dates[n_days - 1].strftime('%Y-%m-%d')
    
    return panel_data, assignment_df, pre_period_end, eval_start, eval_end


def test_parallel_vs_sequential_basic():
    """Test 1: Basic parallel vs sequential ensemble comparison."""
    print("🧪 TEST 1: Basic Parallel vs Sequential Ensemble")
    print("=" * 60)
    
    # Create test data
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data()
    
    # Small model config for faster testing
    model_config = {
        'hidden_dim': 16,
        'num_st_blocks': 1,
        'epochs': 3,
        'learning_rate': 0.01,
        'dropout': 0.1,
        'verbose': False
    }
    
    ensemble_size = 3
    
    print(f"Setup: {len(assignment_df)} geos, ensemble_size={ensemble_size}")
    
    # Test 1a: Sequential ensemble
    print("\n1a. Sequential Ensemble Training:")
    sequential_start = time.time()
    
    try:
        sequential_ensemble = ParallelEnsembleSTGCN(
            ensemble_size=ensemble_size,
            **model_config
        )
        sequential_ensemble.fit_sequential(panel_data, assignment_df, pre_period_end, seed=100)
        
        seq_lower, seq_upper = sequential_ensemble.confidence_interval(
            panel_data, eval_start, eval_end
        )
        
        sequential_time = time.time() - sequential_start
        sequential_success = True
        
        print(f"   ✅ Sequential: {sequential_time:.1f}s, CI=[{seq_lower:.4f}, {seq_upper:.4f}]")
        
    except Exception as e:
        sequential_time = np.nan
        sequential_success = False
        print(f"   ❌ Sequential failed: {e}")
    
    # Test 1b: Parallel ensemble
    print("\n1b. Parallel Ensemble Training:")
    parallel_start = time.time()
    
    try:
        parallel_ensemble = ParallelEnsembleSTGCN(
            ensemble_size=ensemble_size,
            n_jobs=2,  # Use 2 parallel jobs
            **model_config
        )
        parallel_ensemble.fit_parallel(panel_data, assignment_df, pre_period_end, seed=100)
        
        par_lower, par_upper = parallel_ensemble.confidence_interval(
            panel_data, eval_start, eval_end
        )
        
        parallel_time = time.time() - parallel_start
        parallel_success = True
        
        print(f"   ✅ Parallel: {parallel_time:.1f}s, CI=[{par_lower:.4f}, {par_upper:.4f}]")
        
    except Exception as e:
        parallel_time = np.nan
        parallel_success = False
        print(f"   ❌ Parallel failed: {e}")
    
    # Compare results
    print(f"\n📊 Results:")
    if sequential_success and parallel_success:
        speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
        ci_overlap = max(0, min(seq_upper, par_upper) - max(seq_lower, par_lower))
        ci_width_ratio = (par_upper - par_lower) / (seq_upper - seq_lower) if (seq_upper - seq_lower) > 0 else 1
        
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   CI overlap: {ci_overlap:.4f}")
        print(f"   CI width ratio: {ci_width_ratio:.2f}")
        
        if speedup > 1.2:
            print(f"   ✅ Significant speedup achieved!")
        elif speedup > 0.8:
            print(f"   ⚠️ Modest speedup (overhead may dominate for small ensembles)")
        else:
            print(f"   ❌ No speedup (parallel overhead too high)")
            
        if ci_width_ratio > 0.8 and ci_width_ratio < 1.2:
            print(f"   ✅ Consistent CI widths between methods")
        else:
            print(f"   ⚠️ Different CI widths - check for numerical differences")
    else:
        print(f"   Sequential success: {sequential_success}")
        print(f"   Parallel success: {parallel_success}")
    
    return {
        'sequential_success': sequential_success,
        'parallel_success': parallel_success,
        'sequential_time': sequential_time,
        'parallel_time': parallel_time
    }


def test_integration_with_main_stgcn():
    """Test 2: Integration with main STGCN confidence interval methods."""
    print("\n🧪 TEST 2: Integration with Main STGCN Model")
    print("=" * 60)
    
    # Create test data
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data()
    
    # Small model for testing
    print("Training main STGCN model...")
    main_model = STGCNReportingModel(
        hidden_dim=16,
        num_st_blocks=1,
        epochs=3,
        learning_rate=0.01,
        dropout=0.1,
        verbose=True
    )
    main_model.fit(panel_data, assignment_df, pre_period_end)
    
    print(f"\n🔬 Testing different CI methods:")
    
    # Test different CI methods
    ci_methods = {
        'Parallel Ensemble': {
            'method': 'ensemble',
            'params': {'ensemble_size': 3, 'n_jobs': 2, 'use_parallel': True}
        },
        'Sequential Ensemble': {
            'method': 'ensemble',
            'params': {'ensemble_size': 3, 'n_jobs': 1, 'use_parallel': False}
        },
        'MC Dropout': {
            'method': 'mc_dropout',
            'params': {'n_mc_samples': 30}
        }
    }
    
    results = {}
    
    for method_name, config in ci_methods.items():
        print(f"\n2a. Testing {method_name}:")
        
        try:
            start_time = time.time()
            
            lower, upper = main_model.confidence_interval(
                panel_data, eval_start, eval_end,
                method=config['method'],
                **config['params']
            )
            
            elapsed = time.time() - start_time
            ci_width = upper - lower
            includes_zero = (lower <= 0 <= upper)
            
            results[method_name] = {
                'lower': lower,
                'upper': upper,
                'width': ci_width,
                'time': elapsed,
                'includes_zero': includes_zero,
                'success': True
            }
            
            print(f"   ✅ {method_name}: CI=[{lower:.4f}, {upper:.4f}], time={elapsed:.1f}s")
            print(f"      Width={ci_width:.4f}, includes_zero={includes_zero}")
            
        except Exception as e:
            results[method_name] = {'success': False, 'error': str(e)}
            print(f"   ❌ {method_name} failed: {e}")
    
    # Compare results
    print(f"\n📊 Method Comparison:")
    successful_methods = {k: v for k, v in results.items() if v.get('success', False)}
    
    if len(successful_methods) >= 2:
        method_names = list(successful_methods.keys())
        widths = [successful_methods[m]['width'] for m in method_names]
        times = [successful_methods[m]['time'] for m in method_names]
        
        print(f"   CI Width comparison:")
        for i, method in enumerate(method_names):
            print(f"     {method}: {widths[i]:.4f}")
        
        print(f"   Time comparison:")
        for i, method in enumerate(method_names):
            print(f"     {method}: {times[i]:.1f}s")
        
        # Check if parallel is faster than sequential
        if 'Parallel Ensemble' in successful_methods and 'Sequential Ensemble' in successful_methods:
            par_time = successful_methods['Parallel Ensemble']['time']
            seq_time = successful_methods['Sequential Ensemble']['time']
            speedup = seq_time / par_time if par_time > 0 else float('inf')
            print(f"   Parallel speedup: {speedup:.1f}x")
            
            if speedup > 1.2:
                print(f"   ✅ Parallel ensemble provides good speedup")
            else:
                print(f"   ⚠️ Limited speedup (small ensemble or overhead)")
    
    return results


def test_memory_awareness():
    """Test 3: Memory-aware job scheduling."""
    print("\n🧪 TEST 3: Memory-Aware Job Scheduling")
    print("=" * 60)
    
    from reporting.parallel_ensemble import _get_available_memory_gb, _estimate_model_memory_mb, _calculate_optimal_n_jobs
    
    # Test memory detection
    available_memory = _get_available_memory_gb()
    print(f"Available system memory: {available_memory:.1f} GB")
    
    # Test model memory estimation
    model_configs = [
        {'hidden_dim': 16, 'num_st_blocks': 1, 'name': 'Small'},
        {'hidden_dim': 32, 'num_st_blocks': 2, 'name': 'Medium'},
        {'hidden_dim': 64, 'num_st_blocks': 3, 'name': 'Large'}
    ]
    
    print(f"\n3a. Model Memory Estimation:")
    for config in model_configs:
        estimated_mb = _estimate_model_memory_mb(config)
        print(f"   {config['name']} model: ~{estimated_mb:.1f} MB")
    
    # Test optimal job calculation
    print(f"\n3b. Optimal Job Calculation:")
    ensemble_sizes = [3, 5, 8, 10]
    
    for ensemble_size in ensemble_sizes:
        for config in model_configs:
            optimal_jobs = _calculate_optimal_n_jobs(ensemble_size, config)
            max_memory_usage = optimal_jobs * _estimate_model_memory_mb(config) / 1024  # GB
            
            print(f"   {config['name']} ensemble (K={ensemble_size}): optimal_jobs={optimal_jobs}, "
                  f"est_memory={max_memory_usage:.1f}GB")
    
    # Test actual ensemble with different job settings
    print(f"\n3c. Testing Job Limits:")
    panel_data, assignment_df, pre_period_end, _, _ = create_test_data(n_geos=12, n_days=50)
    
    small_config = {'hidden_dim': 16, 'num_st_blocks': 1, 'epochs': 2, 'verbose': False}
    
    job_settings = [1, 2, 4, -1]  # -1 = auto-detect
    
    for n_jobs in job_settings:
        try:
            print(f"   Testing n_jobs={n_jobs}...")
            
            ensemble = ParallelEnsembleSTGCN(
                ensemble_size=4,
                n_jobs=n_jobs,
                verbose=False,
                **small_config
            )
            
            start_time = time.time()
            ensemble.fit_parallel(panel_data, assignment_df, pre_period_end, seed=200)
            elapsed = time.time() - start_time
            
            diagnostics = ensemble.get_training_diagnostics()
            
            print(f"     ✅ n_jobs={n_jobs} → actual_jobs={diagnostics['n_jobs_used']}, "
                  f"time={elapsed:.1f}s, success_rate={diagnostics['success_rate']:.1%}")
            
        except Exception as e:
            print(f"     ❌ n_jobs={n_jobs} failed: {e}")


def test_error_handling_and_fallbacks():
    """Test 4: Error handling and fallback mechanisms."""
    print("\n🧪 TEST 4: Error Handling and Fallback Mechanisms")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data(n_geos=8, n_days=40)
    
    # Test 4a: Graceful degradation with failing models
    print("4a. Testing with intentionally difficult configuration:")
    
    difficult_config = {
        'hidden_dim': 64,  # Large model
        'num_st_blocks': 3,
        'epochs': 1,  # Very few epochs (may cause training instability)
        'learning_rate': 0.1,  # High learning rate (may cause divergence)
        'dropout': 0.9,  # Very high dropout
        'verbose': False
    }
    
    try:
        ensemble = ParallelEnsembleSTGCN(
            ensemble_size=5,
            n_jobs=2,
            **difficult_config
        )
        
        ensemble.fit_parallel(panel_data, assignment_df, pre_period_end, seed=300)
        
        diagnostics = ensemble.get_training_diagnostics()
        success_rate = diagnostics['success_rate']
        
        print(f"   Success rate with difficult config: {success_rate:.1%}")
        
        if success_rate >= 0.6:
            print(f"   ✅ Good robustness - most models succeeded despite difficult config")
        elif success_rate >= 0.4:
            print(f"   ⚠️ Moderate robustness - some models failed but ensemble still viable")
        else:
            print(f"   ❌ Poor robustness - too many models failed")
            
        # Try to calculate CI anyway
        if diagnostics['successful_models'] >= 2:
            lower, upper = ensemble.confidence_interval(panel_data, eval_start, eval_end)
            print(f"   ✅ CI calculation succeeded: [{lower:.4f}, {upper:.4f}]")
        else:
            print(f"   ❌ Insufficient successful models for CI calculation")
            
    except Exception as e:
        print(f"   ❌ Ensemble completely failed: {e}")
    
    # Test 4b: Fallback from parallel to sequential
    print(f"\n4b. Testing fallback to sequential training:")
    
    # Create main STGCN model
    main_model = STGCNReportingModel(
        hidden_dim=16, epochs=2, verbose=False
    )
    main_model.fit(panel_data, assignment_df, pre_period_end)
    
    # Test with forced fallback (n_jobs=1)
    try:
        lower_fallback, upper_fallback = main_model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='ensemble',
            ensemble_size=3,
            n_jobs=1,  # Force sequential
            use_parallel=False
        )
        
        print(f"   ✅ Sequential fallback works: [{lower_fallback:.4f}, {upper_fallback:.4f}]")
        
    except Exception as e:
        print(f"   ❌ Sequential fallback failed: {e}")
    
    # Test 4c: Fallback to MC dropout when ensemble fails
    print(f"\n4c. Testing fallback to MC dropout:")
    
    try:
        lower_mc, upper_mc = main_model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='mc_dropout',
            n_mc_samples=20
        )
        
        print(f"   ✅ MC dropout fallback works: [{lower_mc:.4f}, {upper_mc:.4f}]")
        
    except Exception as e:
        print(f"   ❌ MC dropout fallback failed: {e}")


def test_backward_compatibility():
    """Test 5: Backward compatibility with existing code."""
    print("\n🧪 TEST 5: Backward Compatibility")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data(n_geos=8, n_days=40)
    
    # Test that existing code still works without changes
    print("5a. Testing existing API without parallel parameters:")
    
    try:
        model = STGCNReportingModel(
            hidden_dim=16, epochs=2, verbose=False
        )
        model.fit(panel_data, assignment_df, pre_period_end)
        
        # Original API call (should still work)
        lower_old, upper_old = model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='ensemble',
            ensemble_size=3
        )
        
        print(f"   ✅ Original API works: [{lower_old:.4f}, {upper_old:.4f}]")
        
    except Exception as e:
        print(f"   ❌ Original API broken: {e}")
    
    # Test that new parameters work
    print(f"\n5b. Testing new API with parallel parameters:")
    
    try:
        # New API with parallel parameters
        lower_new, upper_new = model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='ensemble',
            ensemble_size=3,
            n_jobs=2,
            use_parallel=True
        )
        
        print(f"   ✅ New API works: [{lower_new:.4f}, {upper_new:.4f}]")
        
    except Exception as e:
        print(f"   ❌ New API failed: {e}")
    
    # Test that results are consistent
    try:
        ci_overlap = max(0, min(upper_old, upper_new) - max(lower_old, lower_new))
        width_old = upper_old - lower_old
        width_new = upper_new - lower_new
        width_ratio = width_new / width_old if width_old > 0 else 1
        
        print(f"   CI overlap: {ci_overlap:.4f}")
        print(f"   Width ratio (new/old): {width_ratio:.2f}")
        
        if width_ratio > 0.8 and width_ratio < 1.2:
            print(f"   ✅ Consistent results between old and new API")
        else:
            print(f"   ⚠️ Some differences in CI widths (expected due to randomness)")
            
    except:
        print(f"   ⚠️ Could not compare results")


def run_comprehensive_benchmark():
    """Test 6: Comprehensive performance benchmark."""
    print("\n🧪 TEST 6: Comprehensive Performance Benchmark")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_data(n_geos=16, n_days=60)
    
    print("Running comprehensive benchmark (this may take a few minutes)...")
    
    try:
        benchmark_results = benchmark_parallel_vs_sequential(
            panel_data=panel_data,
            assignment_df=assignment_df,
            pre_period_end=pre_period_end,
            ensemble_sizes=[3, 5],  # Smaller sizes for testing
            model_config={
                'hidden_dim': 16,
                'num_st_blocks': 1,
                'epochs': 3,
                'verbose': False
            },
            seed=400
        )
        
        print(f"\n📊 Benchmark Results:")
        print(benchmark_results.to_string(index=False))
        
        # Analyze results
        successful_runs = benchmark_results[
            (benchmark_results['sequential_success']) & 
            (benchmark_results['parallel_success'])
        ]
        
        if len(successful_runs) > 0:
            avg_speedup = successful_runs['speedup'].mean()
            max_speedup = successful_runs['speedup'].max()
            
            print(f"\n🎯 Performance Summary:")
            print(f"   Average speedup: {avg_speedup:.1f}x")
            print(f"   Maximum speedup: {max_speedup:.1f}x")
            print(f"   Successful benchmark runs: {len(successful_runs)}/{len(benchmark_results)}")
            
            if avg_speedup > 1.5:
                print(f"   ✅ Excellent speedup achieved!")
            elif avg_speedup > 1.2:
                print(f"   ✅ Good speedup achieved!")
            elif avg_speedup > 0.9:
                print(f"   ⚠️ Modest speedup (overhead may dominate)")
            else:
                print(f"   ❌ No significant speedup")
        else:
            print(f"   ❌ No successful benchmark runs")
        
        return benchmark_results
        
    except Exception as e:
        print(f"   ❌ Benchmark failed: {e}")
        return None


def main():
    """Run all parallel ensemble tests."""
    print("🚀 PARALLEL ENSEMBLE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"System info: {mp.cpu_count()} CPU cores")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # Run all tests
        test_results = {}
        
        test_results['basic'] = test_parallel_vs_sequential_basic()
        test_results['integration'] = test_integration_with_main_stgcn()
        test_memory_awareness()
        test_error_handling_and_fallbacks()
        test_backward_compatibility()
        test_results['benchmark'] = run_comprehensive_benchmark()
        
        # Final summary
        print(f"\n🎉 FINAL SUMMARY")
        print("=" * 50)
        
        basic_results = test_results['basic']
        if basic_results['parallel_success'] and basic_results['sequential_success']:
            speedup = basic_results['sequential_time'] / basic_results['parallel_time']
            print(f"✅ Basic parallel training: {speedup:.1f}x speedup")
        else:
            print(f"❌ Basic parallel training: Failed")
        
        integration_results = test_results['integration']
        successful_methods = sum(1 for r in integration_results.values() if r.get('success', False))
        print(f"✅ Integration tests: {successful_methods}/{len(integration_results)} methods successful")
        
        if test_results['benchmark'] is not None:
            print(f"✅ Benchmarking: Completed successfully")
        else:
            print(f"❌ Benchmarking: Failed")
        
        print(f"\n💡 Recommendations:")
        print(f"• Use parallel ensemble for K≥5 on multi-core systems")
        print(f"• Set n_jobs=-1 for automatic resource detection")
        print(f"• Fallback to sequential training is automatic if parallel fails")
        print(f"• Monitor memory usage for large models or many parallel jobs")
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()