#!/usr/bin/env python3
"""
Test STGCN confidence interval bug directly.

This creates a simple A/A test scenario to debug the CI calculation
without running the full evaluation pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
import time
import numpy as np
import pandas as pd

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel

def test_stgcn_ci_directly():
    """Test STGCN CI calculation on simple A/A scenario."""
    print("🔧 TESTING STGCN CONFIDENCE INTERVAL BUG")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    # 1. Create simple A/A test data
    print("1. Creating A/A test data...")
    config = DataConfig(n_geos=20, n_days=120, seed=42)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[99].strftime('%Y-%m-%d')  # 100 days pre-period
    eval_start = dates[100].strftime('%Y-%m-%d')
    eval_end = dates[119].strftime('%Y-%m-%d')      # 20 days eval
    
    print(f"   Data: {len(geo_features)} geos, {len(dates)} days")
    print(f"   Treatment/Control: {(assignment_df['assignment'] == 'treatment').sum()}/{(assignment_df['assignment'] == 'control').sum()}")
    print(f"   Pre-period: 100 days, Eval period: 20 days")
    
    # 2. Train STGCN model
    print(f"\n2. Training STGCN model...")
    
    model = STGCNReportingModel(
        hidden_dim=32,
        epochs=10,
        learning_rate=0.01,
        dropout=0.1,
        verbose=True,
        use_offset_calibration=False  # Disabled as we determined
    )
    
    start_time = time.time()
    model.fit(panel_data, assignment_df, pre_period_end)
    fit_time = time.time() - start_time
    
    # Calculate iROAS 
    iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
    
    print(f"   ✅ Model trained successfully:")
    print(f"      Training time: {fit_time:.1f}s")
    print(f"      iROAS: {iroas:.4f} (should be ~0 for A/A test)")
    
    # 3. Test different CI methods systematically
    print(f"\n3. Testing different confidence interval methods...")
    
    ci_results = {}
    
    # Method 1: MC Dropout (baseline)
    print(f"\n   🎲 Testing MC Dropout CI...")
    try:
        start_time = time.time()
        mc_lower, mc_upper = model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='mc_dropout',
            n_mc_samples=50
        )
        mc_time = time.time() - start_time
        mc_width = mc_upper - mc_lower
        mc_includes_zero = (mc_lower <= 0 <= mc_upper)
        
        ci_results['MC_Dropout'] = {
            'lower': mc_lower, 'upper': mc_upper, 'width': mc_width,
            'time': mc_time, 'includes_zero': mc_includes_zero, 'success': True
        }
        
        print(f"      CI: [{mc_lower:.4f}, {mc_upper:.4f}] (width: {mc_width:.4f})")
        print(f"      Time: {mc_time:.1f}s, Includes zero: {mc_includes_zero}")
        
    except Exception as e:
        print(f"      ❌ MC Dropout failed: {e}")
        ci_results['MC_Dropout'] = {'success': False, 'error': str(e)}
    
    # Method 2: Ensemble (the problematic one)
    print(f"\n   🤖 Testing Ensemble CI (the problematic method)...")
    try:
        start_time = time.time()
        ens_lower, ens_upper = model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='ensemble',
            ensemble_size=3,  # Small for faster testing
            use_parallel=True,
            use_bca=False  # Use simple t-distribution first
        )
        ens_time = time.time() - start_time
        ens_width = ens_upper - ens_lower
        ens_includes_zero = (ens_lower <= 0 <= ens_upper)
        
        ci_results['Ensemble'] = {
            'lower': ens_lower, 'upper': ens_upper, 'width': ens_width,
            'time': ens_time, 'includes_zero': ens_includes_zero, 'success': True
        }
        
        print(f"      CI: [{ens_lower:.4f}, {ens_upper:.4f}] (width: {ens_width:.4f})")
        print(f"      Time: {ens_time:.1f}s, Includes zero: {ens_includes_zero}")
        
        # Check if this matches the playground results
        if abs(ens_width - 0.0435) < 0.01:
            print(f"      🚨 CONFIRMED BUG: Width matches playground problem (0.0435)")
        
    except Exception as e:
        print(f"      ❌ Ensemble failed: {e}")
        ci_results['Ensemble'] = {'success': False, 'error': str(e)}
    
    # Method 3: Original bootstrap (control)
    print(f"\n   📊 Testing Original Bootstrap CI...")
    try:
        start_time = time.time()
        orig_lower, orig_upper = model.confidence_interval(
            panel_data, eval_start, eval_end,
            method='original',  # Should fall back to original method
            n_bootstrap=50
        )
        orig_time = time.time() - start_time
        orig_width = orig_upper - orig_lower
        orig_includes_zero = (orig_lower <= 0 <= orig_upper)
        
        ci_results['Original'] = {
            'lower': orig_lower, 'upper': orig_upper, 'width': orig_width,
            'time': orig_time, 'includes_zero': orig_includes_zero, 'success': True
        }
        
        print(f"      CI: [{orig_lower:.4f}, {orig_upper:.4f}] (width: {orig_width:.4f})")
        print(f"      Time: {orig_time:.1f}s, Includes zero: {orig_includes_zero}")
        
    except Exception as e:
        print(f"      ❌ Original failed: {e}")
        ci_results['Original'] = {'success': False, 'error': str(e)}
    
    # 4. Analyze results and identify the bug
    print(f"\n4. CONFIDENCE INTERVAL BUG ANALYSIS:")
    print("="*50)
    
    successful_methods = {k: v for k, v in ci_results.items() if v.get('success', False)}
    
    if len(successful_methods) >= 2:
        # Compare CI widths
        print("Method comparison:")
        for method, result in successful_methods.items():
            status = "✅ GOOD" if result['width'] > 1.0 else "❌ TOO NARROW"
            coverage = "✅ COVERS" if result['includes_zero'] else "❌ MISSES"
            print(f"  {method:12}: width={result['width']:.4f} {status}, coverage={coverage}")
        
        # Identify the problem
        ensemble_result = successful_methods.get('Ensemble')
        if ensemble_result and ensemble_result['width'] < 0.1:
            print(f"\n🚨 ENSEMBLE CI BUG CONFIRMED:")
            print(f"   Ensemble CI width: {ensemble_result['width']:.4f} (extremely narrow)")
            print(f"   This explains 100% FPR - all intervals are too narrow!")
            
            print(f"\n🔍 LIKELY CAUSES:")
            print("   1. Ensemble variance calculation is wrong")
            print("   2. BCa bootstrap implementation has bugs") 
            print("   3. Ensemble model differences are too small")
            print("   4. t-distribution scaling is incorrect")
            
        else:
            print(f"\n✅ No obvious CI width problem detected")
    
    else:
        print("❌ Not enough successful CI methods to compare")
    
    # 5. Test the ensemble training directly for debugging
    print(f"\n5. DEBUGGING ENSEMBLE TRAINING DIRECTLY:")
    print("="*50)
    
    try:
        # Get ensemble iROAS values directly to see variance
        print("Training mini ensemble for variance inspection...")
        
        ensemble_iroas = []
        for i in range(3):
            # Train individual ensemble model
            import torch
            torch.manual_seed(5000 + i)
            np.random.seed(5000 + i)
            
            ens_model = STGCNReportingModel(
                hidden_dim=32,
                epochs=10,
                learning_rate=0.01,
                dropout=0.1,
                verbose=False
            )
            ens_model.fit(panel_data, assignment_df, pre_period_end)
            ens_iroas = ens_model.calculate_iroas(panel_data, eval_start, eval_end)
            ensemble_iroas.append(ens_iroas)
            
            print(f"   Ensemble model {i+1}: iROAS = {ens_iroas:.6f}")
        
        ensemble_mean = np.mean(ensemble_iroas)
        ensemble_std = np.std(ensemble_iroas, ddof=1)
        
        print(f"\n   Ensemble statistics:")
        print(f"      Mean: {ensemble_mean:.6f}")
        print(f"      Std:  {ensemble_std:.6f}")
        print(f"      Variance: {ensemble_std**2:.8f}")
        
        if ensemble_std < 0.01:
            print(f"   🚨 EXTREMELY LOW ENSEMBLE VARIANCE!")
            print(f"      This explains narrow CIs - models are too similar")
        elif ensemble_std < 0.1:
            print(f"   ⚠️ Low ensemble variance")
        else:
            print(f"   ✅ Reasonable ensemble variance")
            
    except Exception as e:
        print(f"   ❌ Ensemble debug failed: {e}")
    
    return ci_results

def simulate_multiple_aa_tests(n_tests=5):
    """Run multiple A/A tests to estimate true FPR."""
    print(f"\n6. SIMULATING {n_tests} A/A TESTS FOR FPR ESTIMATION:")
    print("="*60)
    
    false_positives = 0
    results = []
    
    for test_idx in range(n_tests):
        print(f"\nA/A Test {test_idx + 1}/{n_tests}:")
        
        # Create different data for each test
        config = DataConfig(n_geos=20, n_days=120, seed=42 + test_idx)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42 + test_idx)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[99].strftime('%Y-%m-%d')
        eval_start = dates[100].strftime('%Y-%m-%d')
        eval_end = dates[119].strftime('%Y-%m-%d')
        
        # Train model
        model = STGCNReportingModel(
            hidden_dim=32,
            epochs=8,
            learning_rate=0.01,
            dropout=0.1,
            verbose=False
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end)
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            
            # Test ensemble CI
            ci_lower, ci_upper = model.confidence_interval(
                panel_data, eval_start, eval_end,
                method='ensemble',
                ensemble_size=3,
                use_parallel=True,
                use_bca=False
            )
            
            ci_width = ci_upper - ci_lower
            includes_zero = (ci_lower <= 0 <= ci_upper)
            is_significant = not includes_zero  # False positive if significant
            
            if is_significant:
                false_positives += 1
            
            results.append({
                'test': test_idx + 1,
                'iroas': iroas,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_width,
                'significant': is_significant
            })
            
            status = "❌ FALSE POSITIVE" if is_significant else "✅ Correct"
            print(f"   iROAS: {iroas:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}] {status}")
            
        except Exception as e:
            print(f"   ❌ Test {test_idx + 1} failed: {e}")
    
    fpr = false_positives / n_tests
    print(f"\n📊 FALSE POSITIVE RATE ANALYSIS:")
    print(f"   False positives: {false_positives}/{n_tests}")
    print(f"   Estimated FPR: {fpr:.1%} (target: ~5%)")
    
    if fpr > 0.5:
        print(f"   🚨 EXTREMELY HIGH FPR - CI method is severely broken!")
    elif fpr > 0.2:
        print(f"   ⚠️ High FPR - CI method needs fixing")
    elif fpr < 0.15:
        print(f"   ✅ Reasonable FPR")
    
    return results

if __name__ == "__main__":
    print("🎯 STGCN CONFIDENCE INTERVAL BUG DIAGNOSIS")
    print("="*70)
    
    # Test 1: Single comprehensive CI test
    ci_results = test_stgcn_ci_directly()
    
    # Test 2: Multiple A/A tests for FPR
    fpr_results = simulate_multiple_aa_tests(n_tests=5)
    
    print(f"\n💡 SUMMARY AND NEXT STEPS:")
    print("="*50)
    print("Based on this diagnosis:")
    print("• If ensemble CI width < 0.1: Fix ensemble variance calculation")
    print("• If FPR > 50%: Fix CI method implementation") 
    print("• If both methods fail: Check basic iROAS calculation")
    print("• Focus on the method with smallest CI width for debugging")