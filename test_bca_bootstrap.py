#!/usr/bin/env python3
"""
Test script for BCa bootstrap confidence intervals.

This script tests the new BCa bootstrap feature that should:
1. Improve coverage from 80% to 90%+ compared to t-distribution
2. Better handle skewed ensemble distributions
3. Provide more accurate confidence intervals

Usage:
    python test_bca_bootstrap.py
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

def test_bca_bootstrap():
    """Test BCa bootstrap vs t-distribution CI."""
    print("🔧 TESTING BCa BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    # 1. Create A/A test data
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
    print(f"   Pre-period: 100 days, Eval period: 20 days")
    
    # 2. Test model with offset calibration
    print(f"\n2. Training STGCN with offset calibration...")
    
    model = STGCNReportingModel(
        hidden_dim=32,
        epochs=10,
        learning_rate=0.01,
        dropout=0.1,
        use_offset_calibration=True,  # Use bias correction
        verbose=True
    )
    
    start_time = time.time()
    model.fit(panel_data, assignment_df, pre_period_end)
    fit_time = time.time() - start_time
    
    # Calculate iROAS 
    iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
    
    print(f"   ✅ Model trained successfully:")
    print(f"      Training time: {fit_time:.1f}s")
    print(f"      iROAS: {iroas:.4f} (should be ~0 for A/A test)")
    
    # 3. Test t-distribution CI (old method)
    print(f"\n3. Testing t-distribution confidence intervals...")
    
    start_time = time.time()
    ci_lower_t, ci_upper_t = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='ensemble',
        ensemble_size=5,
        use_parallel=True,
        use_bca=False  # DISABLE BCa
    )
    ci_time_t = time.time() - start_time
    
    ci_width_t = ci_upper_t - ci_lower_t
    includes_zero_t = (ci_lower_t <= 0 <= ci_upper_t)
    
    print(f"   📊 t-distribution CI: [{ci_lower_t:.4f}, {ci_upper_t:.4f}]")
    print(f"      Width: {ci_width_t:.4f}, Includes zero: {includes_zero_t}")
    print(f"      Computation time: {ci_time_t:.1f}s")
    
    # 4. Test BCa bootstrap CI (new method)
    print(f"\n4. Testing BCa bootstrap confidence intervals...")
    
    start_time = time.time()
    ci_lower_bca, ci_upper_bca = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='ensemble',
        ensemble_size=5,
        use_parallel=True,
        use_bca=True  # ENABLE BCa
    )
    ci_time_bca = time.time() - start_time
    
    ci_width_bca = ci_upper_bca - ci_lower_bca
    includes_zero_bca = (ci_lower_bca <= 0 <= ci_upper_bca)
    
    print(f"   📊 BCa bootstrap CI: [{ci_lower_bca:.4f}, {ci_upper_bca:.4f}]")
    print(f"      Width: {ci_width_bca:.4f}, Includes zero: {includes_zero_bca}")
    print(f"      Computation time: {ci_time_bca:.1f}s")
    
    # 5. Compare methods
    print(f"\n5. COMPARISON ANALYSIS:")
    print("="*50)
    
    width_change = (ci_width_bca - ci_width_t) / ci_width_t if ci_width_t > 0 else 0
    time_change = (ci_time_bca - ci_time_t) / ci_time_t if ci_time_t > 0 else 0
    
    print(f"t-distribution CI:    [{ci_lower_t:.4f}, {ci_upper_t:.4f}]")
    print(f"BCa bootstrap CI:     [{ci_lower_bca:.4f}, {ci_upper_bca:.4f}]")
    print(f"")
    print(f"Width change:         {width_change:+.1%} (BCa vs t-dist)")
    print(f"Time overhead:        {time_change:+.1%}")
    print(f"")
    print(f"Coverage (A/A test):")
    print(f"  t-distribution:     {'✅' if includes_zero_t else '❌'} (includes zero: {includes_zero_t})")
    print(f"  BCa bootstrap:      {'✅' if includes_zero_bca else '❌'} (includes zero: {includes_zero_bca})")
    
    # 6. Test with larger ensemble for better BCa performance
    print(f"\n6. Testing BCa with larger ensemble (8 models)...")
    
    start_time = time.time()
    ci_lower_large, ci_upper_large = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='ensemble',
        ensemble_size=8,  # Larger ensemble
        use_parallel=True,
        use_bca=True
    )
    ci_time_large = time.time() - start_time
    
    ci_width_large = ci_upper_large - ci_lower_large
    includes_zero_large = (ci_lower_large <= 0 <= ci_upper_large)
    
    print(f"   📊 BCa CI (8 models): [{ci_lower_large:.4f}, {ci_upper_large:.4f}]")
    print(f"      Width: {ci_width_large:.4f}, Includes zero: {includes_zero_large}")
    print(f"      Computation time: {ci_time_large:.1f}s")
    
    # 7. Summary and assessment
    print(f"\n🎯 BCa BOOTSTRAP TEST SUMMARY:")
    print("="*50)
    
    # Assess BCa improvements
    coverage_improved = includes_zero_bca and not includes_zero_t
    if includes_zero_bca and includes_zero_t:
        coverage_status = "✅ Both methods provide good coverage"
    elif includes_zero_bca and not includes_zero_t:
        coverage_status = "✅ BCa improved coverage (t-dist failed)"
    elif not includes_zero_bca and includes_zero_t:
        coverage_status = "⚠️ BCa coverage worse than t-dist"
    else:
        coverage_status = "❌ Both methods have poor coverage"
    
    print(f"Coverage assessment: {coverage_status}")
    
    # Assess computational efficiency
    if time_change < 0.5:  # Less than 50% overhead
        efficiency_status = "✅ BCa overhead acceptable"
    elif time_change < 1.0:  # Less than 100% overhead
        efficiency_status = "⚠️ BCa has moderate overhead"
    else:
        efficiency_status = "❌ BCa overhead too high"
    
    print(f"Efficiency assessment: {efficiency_status}")
    
    # Overall assessment
    if includes_zero_bca and time_change < 1.0:
        overall_status = "✅ BCa BOOTSTRAP SUCCESSFUL!"
        next_step = "Step 3: Re-evaluate FPR & coverage"
    elif includes_zero_bca:
        overall_status = "⚠️ BCa works but is slow"
        next_step = "Optimize BCa performance"
    else:
        overall_status = "❌ BCa needs more work"
        next_step = "Try Jackknife+ or larger ensembles"
    
    print(f"\n{overall_status}")
    print(f"💡 NEXT STEP: {next_step}")
    print(f"⏱️ Total test time: {(fit_time + ci_time_t + ci_time_bca + ci_time_large):.1f}s")
    
    return {
        'bca_coverage': includes_zero_bca,
        't_coverage': includes_zero_t,
        'bca_width': ci_width_bca,
        't_width': ci_width_t,
        'time_overhead': time_change,
        'success': includes_zero_bca and time_change < 1.0
    }

if __name__ == "__main__":
    results = test_bca_bootstrap()
    
    print(f"\n🔍 BCa BOOTSTRAP USAGE:")
    print("="*50)
    print("The BCa bootstrap feature is now available:")
    print("• use_bca=True    # Enable BCa bootstrap (default)")
    print("• use_bca=False   # Use t-distribution (legacy)")
    print("")
    print("Usage in confidence_interval:")
    print("ci_lower, ci_upper = model.confidence_interval(")
    print("    panel_data, start, end, use_bca=True)")
    
    exit_code = 0 if results['success'] else 1
    sys.exit(exit_code)