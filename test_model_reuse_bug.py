#!/usr/bin/env python3
"""
Test if model reuse is causing the constant CI bug.

This simulates the playground's behavior of reusing the same model objects
across multiple simulations.
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

def test_model_reuse_bug():
    """Test if reusing model objects causes constant CIs."""
    print("🔧 TESTING MODEL REUSE BUG")
    print("="*50)
    
    warnings.filterwarnings('ignore')
    
    # Create models once (like playground does)
    stgcn_enhanced = STGCNReportingModel(
        hidden_dim=32,
        epochs=8,
        learning_rate=0.01,
        dropout=0.1,
        verbose=False,
        use_offset_calibration=True  # Enhanced
    )
    
    stgcn_regular = STGCNReportingModel(
        hidden_dim=32,
        epochs=8,
        learning_rate=0.01,
        dropout=0.1,
        verbose=False,
        use_offset_calibration=False  # Regular
    )
    
    models = {
        'Enhanced': stgcn_enhanced,
        'Regular': stgcn_regular
    }
    
    # Simulate playground loop with 3 simulations
    results = []
    
    for sim_idx in range(3):
        print(f"\nSimulation {sim_idx + 1}/3:")
        
        # Generate different data each time (like playground)
        config = DataConfig(n_geos=20, n_days=120, seed=42 + sim_idx)
        generator = SimpleNullGenerator(config)
        panel_data, geo_features = generator.generate()
        
        assignment_method = RandomAssignment()
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42 + sim_idx)
        
        dates = sorted(panel_data['date'].unique())
        pre_period_end = dates[99].strftime('%Y-%m-%d')
        eval_start = dates[100].strftime('%Y-%m-%d')
        eval_end = dates[119].strftime('%Y-%m-%d')
        
        # Test each model (like playground)
        for model_name, model in models.items():
            print(f"   Testing {model_name}...")
            
            try:
                # Fit model on new data
                model.fit(panel_data, assignment_df, pre_period_end)
                iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
                
                # Calculate CI (like playground)
                ci_lower, ci_upper = model.confidence_interval(
                    panel_data, eval_start, eval_end,
                    method='ensemble',
                    ensemble_size=3,
                    use_parallel=True,
                    use_bca=True
                )
                
                ci_width = ci_upper - ci_lower
                
                results.append({
                    'simulation': sim_idx + 1,
                    'model': model_name,
                    'iroas': iroas,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_width
                })
                
                print(f"      iROAS: {iroas:.4f}, CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
                
            except Exception as e:
                print(f"      ❌ Failed: {e}")
    
    # Analyze results
    print(f"\n📊 MODEL REUSE ANALYSIS:")
    print("="*50)
    
    # Group by model type
    enhanced_results = [r for r in results if r['model'] == 'Enhanced']
    regular_results = [r for r in results if r['model'] == 'Regular']
    
    def analyze_constancy(results, model_name):
        if len(results) < 2:
            return
            
        print(f"\n{model_name} Model:")
        for r in results:
            print(f"  Sim {r['simulation']}: iROAS={r['iroas']:8.4f}, CI=[{r['ci_lower']:8.6f}, {r['ci_upper']:8.6f}]")
        
        # Check if CI bounds are constant
        first_lower = results[0]['ci_lower']
        first_upper = results[0]['ci_upper']
        
        all_same_lower = all(abs(r['ci_lower'] - first_lower) < 1e-6 for r in results)
        all_same_upper = all(abs(r['ci_upper'] - first_upper) < 1e-6 for r in results)
        
        if all_same_lower and all_same_upper:
            print(f"  🚨 CONSTANT CI BOUNDS! [{first_lower:.6f}, {first_upper:.6f}]")
        else:
            print(f"  ✅ CI bounds vary correctly")
            
        # Check if widths match playground problem
        avg_width = np.mean([r['ci_width'] for r in results])
        if abs(avg_width - 0.0435) < 0.01:
            print(f"  🎯 MATCHES PLAYGROUND BUG: width ≈ 0.0435")
    
    analyze_constancy(enhanced_results, "Enhanced")
    analyze_constancy(regular_results, "Regular")
    
    # Check if Enhanced = Regular (both have calibration disabled)
    if len(enhanced_results) > 0 and len(regular_results) > 0:
        enhanced_avg = np.mean([r['iroas'] for r in enhanced_results])
        regular_avg = np.mean([r['iroas'] for r in regular_results])
        
        print(f"\n📊 Enhanced vs Regular:")
        print(f"   Enhanced avg iROAS: {enhanced_avg:.4f}")
        print(f"   Regular avg iROAS: {regular_avg:.4f}")
        
        if abs(enhanced_avg - regular_avg) < 0.01:
            print(f"   ✅ Both behave identically (calibration disabled)")
        else:
            print(f"   ⚠️ Different behavior (unexpected)")
    
    return results

if __name__ == "__main__":
    results = test_model_reuse_bug()
    
    print(f"\n💡 CONCLUSION:")
    print("="*30)
    print("If CI bounds are constant:")
    print("• Model reuse may be caching ensemble results")
    print("• Check if ensemble training is being skipped")
    print("• Look for cached model states or ensemble results")
    print("\nIf CI bounds vary:")
    print("• Model reuse is not the issue")  
    print("• Bug may be in playground's result aggregation")
    print("• Check how playground calculates summary statistics")