#!/usr/bin/env python3
"""
Debug BCa bootstrap by examining the exact inputs.
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

def debug_bca_inputs():
    """Debug what inputs BCa receives."""
    print("🔧 DEBUGGING BCa BOOTSTRAP INPUTS")
    print("="*50)
    
    warnings.filterwarnings('ignore')
    
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
    
    # Create custom STGCN model with debug BCa
    class DebugSTGCN(STGCNReportingModel):
        def _compute_bca_ci(self, estimates: np.ndarray, confidence_level: float):
            print(f"\n🔍 BCa DEBUG:")
            print(f"   Input estimates: {estimates}")
            print(f"   Estimates length: {len(estimates)}")
            print(f"   Estimates range: [{estimates.min():.6f}, {estimates.max():.6f}]")
            print(f"   Estimates std: {estimates.std():.6f}")
            print(f"   Confidence level: {confidence_level}")
            
            # Call original implementation with debug
            result = super()._compute_bca_ci(estimates, confidence_level)
            
            print(f"   Result CI: [{result[0]:.6f}, {result[1]:.6f}]")
            print(f"   Result width: {result[1] - result[0]:.6f}")
            
            return result
    
    # Train model
    model = DebugSTGCN(
        hidden_dim=32,
        epochs=8,
        learning_rate=0.01,
        dropout=0.1,
        verbose=True  # Enable verbose for BCa debug prints
    )
    
    model.fit(panel_data, assignment_df, pre_period_end)
    iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
    
    print(f"\nMain iROAS: {iroas:.6f}")
    
    # Test BCa CI
    print(f"\nTesting BCa CI...")
    bca_lower, bca_upper = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='ensemble',
        ensemble_size=3,
        use_parallel=True,
        use_bca=True
    )
    
    print(f"\nFinal result: [{bca_lower:.6f}, {bca_upper:.6f}]")
    
    # Test with different ensemble size to see if estimates change
    print(f"\n" + "="*50)
    print(f"Testing with different ensemble size (5 instead of 3)...")
    
    bca_lower2, bca_upper2 = model.confidence_interval(
        panel_data, eval_start, eval_end,
        method='ensemble',
        ensemble_size=5,
        use_parallel=True,
        use_bca=True
    )
    
    print(f"\nEnsemble size 5 result: [{bca_lower2:.6f}, {bca_upper2:.6f}]")
    
    # Check if results are identical
    if abs(bca_lower - bca_lower2) < 1e-10 and abs(bca_upper - bca_upper2) < 1e-10:
        print(f"🚨 IDENTICAL RESULTS WITH DIFFERENT ENSEMBLE SIZES!")
        print(f"   This confirms the BCa bug - should be different with different estimates")
    else:
        print(f"✅ Results differ with ensemble size - BCa may be working")

if __name__ == "__main__":
    debug_bca_inputs()