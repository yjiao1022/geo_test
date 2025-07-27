"""
Test the integrated ensemble method in STGCN.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def test_integrated_ensemble():
    """Test the integrated ensemble method."""
    print("Testing Integrated Ensemble STGCN")
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
    
    # Fit single STGCN model
    print(f"\nFitting STGCN model...")
    model = STGCNReportingModel(
        hidden_dim=32,
        epochs=8,
        learning_rate=0.01,
        dropout=0.1,
        verbose=True
    )
    
    model.fit(panel_data, assignment_df, pre_period_end)
    
    # Test different CI methods
    print(f"\n=== Testing CI Methods ===")
    
    # Method 1: Ensemble (new default)
    print(f"\n1. Ensemble Method (K=3 for speed):")
    ens_lower, ens_upper = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='ensemble',
        ensemble_size=3,
        confidence_level=0.95
    )
    ens_width = ens_upper - ens_lower
    print(f"   CI: [{ens_lower:.4f}, {ens_upper:.4f}]")
    print(f"   Width: {ens_width:.4f}")
    
    # Method 2: MC Dropout (old default)
    print(f"\n2. MC Dropout Method:")
    mc_lower, mc_upper = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='mc_dropout',
        n_mc_samples=50,
        confidence_level=0.95
    )
    mc_width = mc_upper - mc_lower
    print(f"   CI: [{mc_lower:.4f}, {mc_upper:.4f}]")
    print(f"   Width: {mc_width:.4f}")
    
    # Calculate base iROAS
    base_iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
    print(f"\nBase iROAS: {base_iroas:.4f}")
    
    # Analysis
    print(f"\n=== Analysis ===")
    width_ratio = ens_width / mc_width if mc_width > 0 else float('inf')
    print(f"Ensemble CI is {width_ratio:.1f}x wider than MC dropout")
    
    # Check if CI includes 0 (null hypothesis)
    ens_significant = (ens_lower > 0) or (ens_upper < 0)
    mc_significant = (mc_lower > 0) or (mc_upper < 0)
    
    print(f"Ensemble significant: {ens_significant}")
    print(f"MC Dropout significant: {mc_significant}")
    
    if not ens_significant and mc_significant:
        print(f"✅ GOOD: Ensemble correctly fails to reject null, MC dropout falsely significant")
    elif ens_significant and mc_significant:
        print(f"⚠️ Both methods significant (may be true effect or both overconfident)")
    elif not ens_significant and not mc_significant:
        print(f"✅ Both methods correctly fail to reject null")
    else:
        print(f"❓ Ensemble significant but MC dropout not (unusual)")
    
    print(f"\n💡 Summary:")
    print(f"  Ensemble method provides {width_ratio:.1f}x more realistic uncertainty")
    print(f"  This should dramatically reduce false positive rates")
    print(f"  Computational cost: ~{3}x training time for K=3 ensemble")
    
    return {
        'ensemble_ci': (ens_lower, ens_upper),
        'mc_dropout_ci': (mc_lower, mc_upper),
        'width_ratio': width_ratio,
        'ensemble_significant': ens_significant,
        'mc_significant': mc_significant,
        'base_iroas': base_iroas
    }


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    results = test_integrated_ensemble()