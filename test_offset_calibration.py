#!/usr/bin/env python3
"""
Test script for STGCN offset calibration bias reduction.

This script tests the new offset calibration feature that should:
1. Reduce bias from ~6.69 to ~0.5 (80-90% reduction)
2. Improve coverage from 80% to 90%+
3. Reduce FPR from 20% to closer to 5%

Usage:
    python test_offset_calibration.py
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

def test_offset_calibration():
    """Test offset calibration on A/A simulation."""
    print("🔧 TESTING STGCN OFFSET CALIBRATION")
    print("="*60)
    
    warnings.filterwarnings('ignore')
    
    # 1. Create A/A test data (same as original problematic case)
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
    
    # 2. Test WITHOUT calibration (baseline)
    print(f"\n2. Testing STGCN WITHOUT calibration...")
    
    model_no_cal = STGCNReportingModel(
        hidden_dim=32,
        epochs=10,
        learning_rate=0.01,
        dropout=0.1,
        use_offset_calibration=False,  # DISABLED
        verbose=True
    )
    
    start_time = time.time()
    model_no_cal.fit(panel_data, assignment_df, pre_period_end)
    fit_time_no_cal = time.time() - start_time
    
    # Calculate iROAS without calibration
    iroas_no_cal = model_no_cal.calculate_iroas(panel_data, eval_start, eval_end)
    
    print(f"   ❌ Without calibration:")
    print(f"      iROAS bias: {iroas_no_cal:.4f} (should be ~0)")
    print(f"      Training time: {fit_time_no_cal:.1f}s")
    
    # 3. Test WITH calibration (new feature)
    print(f"\n3. Testing STGCN WITH offset calibration...")
    
    model_with_cal = STGCNReportingModel(
        hidden_dim=32,
        epochs=10,
        learning_rate=0.01,
        dropout=0.1,
        use_offset_calibration=True,   # ENABLED!
        verbose=True
    )
    
    start_time = time.time()
    model_with_cal.fit(panel_data, assignment_df, pre_period_end)
    fit_time_with_cal = time.time() - start_time
    
    # Calculate iROAS with calibration
    iroas_with_cal = model_with_cal.calculate_iroas(panel_data, eval_start, eval_end)
    
    print(f"   ✅ With offset calibration:")
    print(f"      iROAS bias: {iroas_with_cal:.4f} (should be ~0)")
    print(f"      Training time: {fit_time_with_cal:.1f}s")
    
    # 4. Compare results
    print(f"\n4. CALIBRATION IMPACT ANALYSIS:")
    print("="*50)
    
    bias_reduction = abs(iroas_no_cal) - abs(iroas_with_cal)
    bias_reduction_pct = bias_reduction / abs(iroas_no_cal) if abs(iroas_no_cal) > 0 else 0
    
    print(f"Original bias:           {iroas_no_cal:.4f}")
    print(f"Calibrated bias:         {iroas_with_cal:.4f}")
    print(f"Absolute bias reduction: {bias_reduction:.4f}")
    print(f"Relative bias reduction: {bias_reduction_pct:.1%}")
    
    # Expected vs actual
    print(f"\n📊 TARGET vs ACTUAL:")
    target_bias_reduction = 0.8  # Expect 80%+ reduction
    target_final_bias = 0.5      # Expect final bias ≤ 0.5
    
    if bias_reduction_pct >= target_bias_reduction:
        print(f"✅ Bias reduction: {bias_reduction_pct:.1%} ≥ {target_bias_reduction:.1%} (EXCELLENT)")
    elif bias_reduction_pct >= 0.6:
        print(f"✅ Bias reduction: {bias_reduction_pct:.1%} ≥ 60% (GOOD)")
    else:
        print(f"⚠️ Bias reduction: {bias_reduction_pct:.1%} < 60% (NEEDS WORK)")
    
    if abs(iroas_with_cal) <= target_final_bias:
        print(f"✅ Final bias: {abs(iroas_with_cal):.4f} ≤ {target_final_bias} (EXCELLENT)")
    elif abs(iroas_with_cal) <= 1.0:
        print(f"✅ Final bias: {abs(iroas_with_cal):.4f} ≤ 1.0 (GOOD)")
    else:
        print(f"⚠️ Final bias: {abs(iroas_with_cal):.4f} > 1.0 (NEEDS MORE WORK)")
    
    # 5. Test confidence intervals (quick check)
    print(f"\n5. Quick CI test with calibration...")
    
    try:
        start_time = time.time()
        ci_lower, ci_upper = model_with_cal.confidence_interval(
            panel_data, eval_start, eval_end,
            method='ensemble',
            ensemble_size=3,  # Small for speed
            use_parallel=True
        )
        ci_time = time.time() - start_time
        
        ci_width = ci_upper - ci_lower
        includes_zero = (ci_lower <= 0 <= ci_upper)
        
        print(f"   ✅ Calibrated CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"      Width: {ci_width:.4f}, Includes zero: {includes_zero}")
        print(f"      CI computation time: {ci_time:.1f}s")
        
        if includes_zero:
            print(f"      ✅ CI correctly includes zero (good coverage)")
        else:
            print(f"      ⚠️ CI does not include zero (may indicate remaining bias)")
    
    except Exception as e:
        print(f"   ❌ CI test failed: {e}")
    
    # 6. Summary and next steps
    print(f"\n🎯 OFFSET CALIBRATION TEST SUMMARY:")
    print("="*50)
    
    if bias_reduction_pct >= 0.7 and abs(iroas_with_cal) <= 1.0:
        print("✅ OFFSET CALIBRATION SUCCESSFUL!")
        print("   Ready for Step 2: Replace percentile CI with BCa bootstrap")
        next_step = "Step 2: BCa bootstrap"
    elif bias_reduction_pct >= 0.4:
        print("⚠️ PARTIAL SUCCESS - Need additional bias correction")
        print("   Consider: linear calibration or Huber loss")
        next_step = "Linear calibration + BCa"
    else:
        print("❌ OFFSET CALIBRATION INSUFFICIENT")
        print("   Need: Huber loss + log-space prediction")
        next_step = "Advanced bias correction"
    
    print(f"\n💡 NEXT STEP: {next_step}")
    print(f"\n⏱️ Total test time: {(fit_time_no_cal + fit_time_with_cal):.1f}s")
    
    return {
        'bias_reduction_pct': bias_reduction_pct,
        'final_bias': abs(iroas_with_cal),
        'original_bias': abs(iroas_no_cal),
        'success': bias_reduction_pct >= 0.6 and abs(iroas_with_cal) <= 1.0
    }

if __name__ == "__main__":
    results = test_offset_calibration()
    
    print(f"\n🔍 CALIBRATION PARAMETERS INSPECTION:")
    print("="*50)
    print("The calibration feature is now available with these options:")
    print("• use_offset_calibration=True   # Simple offset correction")
    print("• use_linear_calibration=True   # Beta0 + beta1 * y_hat correction")
    print("")
    print("Usage in playground:")
    print("model = STGCNReportingModel(use_offset_calibration=True)")
    
    exit_code = 0 if results['success'] else 1
    sys.exit(exit_code)