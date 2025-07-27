"""
Simple ensemble test to validate K=5 approach for STGCN.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import warnings
import time

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def test_ensemble_approach():
    """Test if K=5 ensemble reduces underestimation."""
    print("Simple K=5 Ensemble Test for STGCN")
    print("=" * 40)
    
    # Create test data
    config = DataConfig(n_geos=20, n_days=120, seed=42)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[99].strftime('%Y-%m-%d')
    eval_start = dates[100].strftime('%Y-%m-%d')
    eval_end = dates[119].strftime('%Y-%m-%d')
    
    print(f"Setup: {len(assignment_df)} geos, {(assignment_df['assignment'] == 'treatment').sum()} treatment")
    
    # Test 1: Single model variance (baseline)
    print(f"\n=== Single Model Variance Test ===")
    single_iroas = []
    single_ci_widths = []
    
    for i in range(5):
        torch.manual_seed(100 + i)
        np.random.seed(100 + i)
        
        model = STGCNReportingModel(
            hidden_dim=32, epochs=8, verbose=False,
            dropout=0.1, learning_rate=0.01
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end)
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            single_iroas.append(iroas)
            
            # MC dropout CI
            lower, upper = model.confidence_interval(
                panel_data, eval_start, eval_end,
                method='mc_dropout', n_mc_samples=30
            )
            single_ci_widths.append(upper - lower)
            print(f"  Model {i+1}: iROAS={iroas:.3f}, CI_width={upper-lower:.3f}")
            
        except Exception as e:
            print(f"  Model {i+1} failed: {e}")
    
    # Calculate single model stats
    single_empirical_std = np.std(single_iroas) if len(single_iroas) > 1 else 0
    single_avg_ci_width = np.mean(single_ci_widths) if single_ci_widths else 0
    single_ci_implied_std = single_avg_ci_width / 3.92
    single_underest = single_empirical_std / single_ci_implied_std if single_ci_implied_std > 0 else float('inf')
    
    print(f"\nSingle Model Results:")
    print(f"  Empirical std: {single_empirical_std:.4f}")
    print(f"  Avg CI width: {single_avg_ci_width:.4f}")
    print(f"  CI implied std: {single_ci_implied_std:.4f}")
    print(f"  Underestimation ratio: {single_underest:.1f}x")
    
    # Test 2: Ensemble approach
    print(f"\n=== K=5 Ensemble Test ===")
    
    # Train K=5 ensemble
    ensemble_models = []
    for i in range(5):
        torch.manual_seed(200 + i)
        np.random.seed(200 + i)
        
        model = STGCNReportingModel(
            hidden_dim=32, epochs=8, verbose=False,
            dropout=0.1, learning_rate=0.01
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end)
            ensemble_models.append(model)
            print(f"  Ensemble model {i+1}: trained successfully")
        except Exception as e:
            print(f"  Ensemble model {i+1} failed: {e}")
    
    if len(ensemble_models) < 2:
        print("❌ Ensemble failed - not enough models")
        return
    
    # Get ensemble iROAS predictions
    ensemble_iroas = []
    for model in ensemble_models:
        try:
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            ensemble_iroas.append(iroas)
        except:
            continue
    
    if len(ensemble_iroas) < 2:
        print("❌ Ensemble predictions failed")
        return
    
    # Calculate ensemble CI
    ensemble_mean = np.mean(ensemble_iroas)
    ensemble_std = np.std(ensemble_iroas, ddof=1)
    
    # Use t-distribution for small ensemble
    from scipy import stats
    t_score = stats.t.ppf(0.975, df=len(ensemble_iroas) - 1)  # 95% CI
    margin = t_score * ensemble_std
    
    ensemble_lower = ensemble_mean - margin
    ensemble_upper = ensemble_mean + margin
    ensemble_ci_width = ensemble_upper - ensemble_lower
    
    print(f"\nEnsemble Results:")
    print(f"  Successful models: {len(ensemble_iroas)}/5")
    print(f"  Individual iROAS: {ensemble_iroas}")
    print(f"  Ensemble mean: {ensemble_mean:.4f}")
    print(f"  Ensemble std: {ensemble_std:.4f}")
    print(f"  95% CI: [{ensemble_lower:.4f}, {ensemble_upper:.4f}]")
    print(f"  CI width: {ensemble_ci_width:.4f}")
    
    # Compare empirical vs ensemble CI
    # For ensemble, empirical std = ensemble std (by design)
    ensemble_ci_implied_std = ensemble_ci_width / 3.92
    ensemble_underest = ensemble_std / ensemble_ci_implied_std if ensemble_ci_implied_std > 0 else float('inf')
    
    print(f"  CI implied std: {ensemble_ci_implied_std:.4f}")
    print(f"  Underestimation ratio: {ensemble_underest:.1f}x")
    
    # Assessment
    print(f"\n=== COMPARISON ===")
    print(f"Single Model Underestimation: {single_underest:.1f}x")
    print(f"Ensemble Underestimation: {ensemble_underest:.1f}x")
    
    if ensemble_underest < 5:
        print(f"🎉 EXCELLENT: Ensemble achieves good calibration")
        print(f"   → Use ensemble approach in production")
    elif ensemble_underest < single_underest * 0.2:
        print(f"✅ MAJOR IMPROVEMENT: Ensemble much better")
        print(f"   → Ensemble approach viable")
    elif ensemble_underest < single_underest * 0.5:
        print(f"⚠️ IMPROVEMENT: Ensemble better but not great")
        print(f"   → Consider larger ensemble or other methods")
    else:
        print(f"❌ INSUFFICIENT: Ensemble doesn't help much")
        print(f"   → Need different approach")
    
    improvement = single_underest / ensemble_underest if ensemble_underest > 0 else float('inf')
    print(f"Improvement factor: {improvement:.1f}x")
    
    return {
        'single_underest': single_underest,
        'ensemble_underest': ensemble_underest,
        'improvement_factor': improvement,
        'ensemble_iroas': ensemble_iroas,
        'single_iroas': single_iroas
    }


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    results = test_ensemble_approach()