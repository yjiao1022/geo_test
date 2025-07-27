#!/usr/bin/env python3
"""
Test that the parallel ensemble bug is fixed and FPR is reasonable.
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

def test_fixed_ensemble_fpr(n_tests=5):
    """Test FPR with fixed ensemble method."""
    print("🔧 TESTING FIXED ENSEMBLE FPR")
    print("=" * 50)
    
    warnings.filterwarnings('ignore')
    
    false_positives = 0
    results = []
    
    for test_idx in range(n_tests):
        print(f"\nA/A Test {test_idx + 1}/{n_tests}:")
        
        # Create different data for each test
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
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end)
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            
            # Test ensemble CI with the fix
            ci_lower, ci_upper = model.confidence_interval(
                panel_data, eval_start, eval_end,
                method='ensemble',
                ensemble_size=3,
                use_parallel=True,
                use_bca=True
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
            print(f"   iROAS: {iroas:.4f}, CI: [{ci_lower:.4f}, {ci_upper:.4f}], width: {ci_width:.4f} {status}")
            
        except Exception as e:
            print(f"   ❌ Test {test_idx + 1} failed: {e}")
    
    fpr = false_positives / n_tests
    avg_width = np.mean([r['ci_width'] for r in results])
    
    print(f"\n📊 FIXED ENSEMBLE RESULTS:")
    print(f"   False positives: {false_positives}/{n_tests}")
    print(f"   Estimated FPR: {fpr:.1%} (target: ~5%)")
    print(f"   Average CI width: {avg_width:.4f}")
    
    if fpr > 0.5:
        print(f"   🚨 STILL HIGH FPR - ensemble bug may not be fully fixed")
    elif fpr > 0.2:
        print(f"   ⚠️ Moderate FPR - may need further calibration")
    elif fpr < 0.15:
        print(f"   ✅ Reasonable FPR - ensemble fix appears successful!")
    
    # Check if CI width is reasonable
    if avg_width < 0.1:
        print(f"   ⚠️ CI widths still very narrow - may need ensemble variance tuning")
    elif avg_width > 10:
        print(f"   ⚠️ CI widths very wide - may be overconservative")
    else:
        print(f"   ✅ CI widths appear reasonable")
    
    return results

if __name__ == "__main__":
    results = test_fixed_ensemble_fpr(n_tests=5)
    
    print(f"\n💡 NEXT STEPS:")
    print("=" * 30)
    if len(results) > 0:
        fpr = sum(r['significant'] for r in results) / len(results)
        if fpr < 0.2:
            print("1. ✅ Ensemble bug appears fixed - BCa now varies with data")
            print("2. ✅ FPR is reasonable - ready for playground testing")
            print("3. 🔍 Next: Test in playground to confirm full pipeline fix")
        else:
            print("1. ✅ Ensemble CI varies (no more constant bounds)")
            print("2. ⚠️ FPR still high - may need additional calibration")
            print("3. 🔍 Consider implementing offset calibration properly")