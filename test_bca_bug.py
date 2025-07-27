#!/usr/bin/env python3
"""
Test BCa bootstrap bug - check if it returns constant intervals.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
import numpy as np
import pandas as pd

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel

def test_bca_bug():
    """Test if BCa bootstrap is returning constant intervals."""
    print("🔧 TESTING BCa BOOTSTRAP BUG")
    print("="*50)
    
    warnings.filterwarnings('ignore')
    
    # Test BCa on multiple different datasets
    ci_results = []
    
    for test_idx in range(3):
        print(f"\nTest {test_idx + 1}/3:")
        
        # Create different data each time
        config = DataConfig(n_geos=20, n_days=120, seed=42 + test_idx * 10)
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
        
        model.fit(panel_data, assignment_df, pre_period_end)
        iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
        
        # Test BCa bootstrap
        print(f"   Training model... iROAS: {iroas:.4f}")
        
        try:
            # Test with BCa enabled (the problematic setting)
            bca_lower, bca_upper = model.confidence_interval(
                panel_data, eval_start, eval_end,
                method='ensemble',
                ensemble_size=3,
                use_parallel=True,
                use_bca=True  # This is what playground uses!
            )
            
            ci_results.append({
                'test': test_idx + 1,
                'iroas': iroas,
                'bca_lower': bca_lower,
                'bca_upper': bca_upper,
                'bca_width': bca_upper - bca_lower
            })
            
            print(f"   BCa CI: [{bca_lower:.6f}, {bca_upper:.6f}] (width: {bca_upper - bca_lower:.6f})")
            
        except Exception as e:
            print(f"   ❌ BCa failed: {e}")
    
    # Analyze if CIs are constant
    print(f"\n📊 BCa BOOTSTRAP ANALYSIS:")
    print("="*40)
    
    if len(ci_results) >= 2:
        # Check if CI bounds are identical
        first_lower = ci_results[0]['bca_lower']
        first_upper = ci_results[0]['bca_upper']
        
        all_same_lower = all(abs(r['bca_lower'] - first_lower) < 1e-6 for r in ci_results)
        all_same_upper = all(abs(r['bca_upper'] - first_upper) < 1e-6 for r in ci_results)
        
        print("Individual results:")
        for result in ci_results:
            print(f"  Test {result['test']}: iROAS={result['iroas']:8.4f}, CI=[{result['bca_lower']:8.4f}, {result['bca_upper']:8.4f}]")
        
        if all_same_lower and all_same_upper:
            print(f"\n🚨 BCA BOOTSTRAP BUG CONFIRMED!")
            print(f"   ALL CI bounds are identical: [{first_lower:.6f}, {first_upper:.6f}]")
            print(f"   This explains constant CI width (0.0435) and broken FPR!")
            print(f"   BCa is ignoring the data and returning fixed values!")
        else:
            print(f"\n✅ BCa bootstrap CI bounds vary correctly with data")
            
        # Check if widths match playground problem
        avg_width = np.mean([r['bca_width'] for r in ci_results])
        if abs(avg_width - 0.0435) < 0.01:
            print(f"\n🎯 PLAYGROUND BUG MATCH:")
            print(f"   Average width: {avg_width:.4f} matches playground (0.0435)")
    
    return ci_results

if __name__ == "__main__":
    results = test_bca_bug()
    
    print(f"\n💡 DEBUGGING STEPS:")
    print("="*30)
    if len(results) > 0 and all(abs(r['bca_lower'] - results[0]['bca_lower']) < 1e-6 for r in results):
        print("1. ✅ Confirmed: BCa bootstrap returns constant CI bounds")
        print("2. 🔍 Next: Check _compute_bca_ci implementation")
        print("3. 🔧 Likely: BCa calculation has hardcoded values or wrong input")
        print("4. 🚨 Impact: This breaks all STGCN evaluation metrics")
    else:
        print("1. ❓ BCa bootstrap CI bounds vary - bug may be elsewhere")
        print("2. 🔍 Next: Check ensemble variance calculation")
        print("3. 🔧 Likely: Ensemble models are too similar")