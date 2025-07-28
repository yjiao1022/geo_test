#!/usr/bin/env python3
"""
Test to verify if baseline methods work correctly when using original iROAS calculation 
instead of component metrics, to match the behavior from commit baafb1f.
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/yangjiao/Documents/Projects/geo_test')

from data_simulation.generators import IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel, SyntheticControlModel

def test_original_iroas_calculation():
    """Test baseline methods using original iROAS calculation (like baafb1f)."""
    
    print("=" * 80)
    print("TESTING ORIGINAL iROAS CALCULATION (like commit baafb1f)")
    print("=" * 80)
    
    # Setup
    data_config = DataConfig(
        n_geos=24,
        n_days=180,
        base_sales_mean=1000,
        base_sales_std=0,  # Identical baselines
        daily_sales_noise=100,
        seed=42
    )
    
    generator = IdenticalGeoGenerator(data_config)
    panel_data, geo_features = generator.generate()
    
    # Assignment
    assignment = RandomAssignment().assign(geo_features, 0.5, seed=42)
    
    # Time periods
    dates = pd.to_datetime(panel_data['date'].unique())
    pre_period_end = dates[119]  # 120 pre-period days
    eval_period_start = dates[120]
    eval_period_end = dates[179]  # 60 eval-period days
    
    pre_period_end_str = pre_period_end.strftime('%Y-%m-%d')
    eval_period_start_str = eval_period_start.strftime('%Y-%m-%d')
    eval_period_end_str = eval_period_end.strftime('%Y-%m-%d')
    
    # Test baseline methods using original approach
    methods = {
        'MeanMatching': MeanMatchingModel(),
        'TBR': TBRModel(),
        'SCM': SyntheticControlModel()
    }
    
    results = {}
    
    for name, model in methods.items():
        print(f"\nTesting {name}:")
        
        # Fit model
        model.fit(panel_data, assignment, pre_period_end_str)
        
        # Calculate iROAS (original behavior)
        iroas_estimate = model.calculate_iroas(
            panel_data, eval_period_start_str, eval_period_end_str
        )
        
        # Calculate confidence interval (original behavior)
        iroas_lower, iroas_upper = model.confidence_interval(
            panel_data, eval_period_start_str, eval_period_end_str,
            confidence_level=0.95,
            n_bootstrap=100,  # Reduced for speed
            seed=42
        )
        
        ci_width = iroas_upper - iroas_lower
        significant = (iroas_lower > 0) or (iroas_upper < 0)
        coverage = (iroas_lower <= 0) and (iroas_upper >= 0)
        
        results[name] = {
            'iroas_estimate': iroas_estimate,
            'iroas_lower': iroas_lower,
            'iroas_upper': iroas_upper,
            'ci_width': ci_width,
            'significant': significant,
            'coverage': coverage
        }
        
        print(f"  iROAS estimate: {iroas_estimate:.4f}")
        print(f"  CI: [{iroas_lower:.4f}, {iroas_upper:.4f}]")
        print(f"  CI width: {ci_width:.4f}")
        print(f"  Significant: {significant}")
        print(f"  Coverage (contains 0): {coverage}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Original iROAS Calculation)")
    print("=" * 60)
    print(f"{'Method':<15} {'iROAS':<10} {'Significant':<12} {'Coverage':<10} {'CI Width':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<15} {result['iroas_estimate']:>8.4f} {str(result['significant']):>10} "
              f"{str(result['coverage']):>8} {result['ci_width']:>8.2f}")
    
    # Expected behavior check
    print("\n" + "=" * 60)
    print("EXPECTED vs ACTUAL (A/A Testing)")
    print("=" * 60)
    
    all_coverage = all(r['coverage'] for r in results.values())
    max_abs_iroas = max(abs(r['iroas_estimate']) for r in results.values())
    any_significant = any(r['significant'] for r in results.values())
    
    print(f"All methods have proper coverage (contain 0): {all_coverage} ✅" if all_coverage else f"Coverage issues detected: ❌")
    print(f"Maximum absolute iROAS: {max_abs_iroas:.4f} (should be small)")
    print(f"Any false positives: {any_significant} (should be False for A/A)")
    
    if all_coverage and max_abs_iroas < 2.0 and not any_significant:
        print("\n✅ ORIGINAL BEHAVIOR: Baseline methods working correctly!")
        return True
    else:
        print("\n❌ ISSUE DETECTED: Baseline methods not working as expected")
        return False

if __name__ == "__main__":
    success = test_original_iroas_calculation()
    if success:
        print("\n🎯 Baseline methods work correctly with original iROAS calculation.")
        print("   The issue is likely in the A/A component metrics calculation.")
    else:
        print("\n⚠️ Even original iROAS calculation shows issues.")
        print("   The problem might be deeper in the models themselves.")