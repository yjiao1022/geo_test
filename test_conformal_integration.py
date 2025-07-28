#!/usr/bin/env python3
"""
Test conformal prediction integration with component metrics.
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
from evaluation.metrics import EvaluationRunner, EvaluationConfig

def test_conformal_integration():
    """Test conformal prediction with STGCN in A/A mode."""
    print("🧪 CONFORMAL INTEGRATION TEST")
    print("=" * 40)
    
    warnings.filterwarnings('ignore')
    
    # Create small test data
    config = DataConfig(n_geos=12, n_days=90, seed=123)
    generator = SimpleNullGenerator(config)
    
    # Create evaluation config for A/A conformal mode
    eval_config = EvaluationConfig(
        n_simulations=1,  # Just one simulation for testing
        pre_period_days=60,
        eval_period_days=20,
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,
        seed=123
    )
    
    # Create methods
    assignment_methods = {
        'Random': RandomAssignment()
    }
    
    reporting_methods = {
        'STGCN': STGCNReportingModel(
            hidden_dim=16, 
            epochs=5, 
            learning_rate=0.01, 
            dropout=0.1,
            verbose=False
        )
    }
    
    # Run evaluation
    print("Running A/A simulation with conformal prediction...")
    runner = EvaluationRunner(eval_config)
    
    try:
        results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
        
        if len(results_df) > 0:
            result = results_df.iloc[0]
            print(f"\n📊 CONFORMAL RESULTS:")
            print(f"  Incremental Sales: {result['incremental_sales']:.4f}")
            print(f"  Sales CI: [{result['sales_lower']:.4f}, {result['sales_upper']:.4f}]")
            print(f"  Sales Significant: {result['sales_significant']}")
            print(f"  CI Width: {result['sales_upper'] - result['sales_lower']:.4f}")
            
            # Check if results are reasonable
            sales_bias = abs(result['incremental_sales'])
            ci_width = result['sales_upper'] - result['sales_lower']
            
            print(f"\n✅ INTEGRATION TEST RESULTS:")
            print(f"  Sales bias magnitude: {sales_bias:.4f}")
            print(f"  CI width: {ci_width:.4f}")
            print(f"  Conformal interval valid: {result['sales_lower'] <= result['sales_upper']}")
            
            if sales_bias < 1000 and ci_width > 0:
                print("✅ SUCCESS: Conformal integration working correctly!")
                return True
            else:
                print("🚨 ISSUE: Results seem unreasonable")
                return False
        else:
            print("🚨 ERROR: No results generated")
            return False
            
    except Exception as e:
        print(f"🚨 ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_conformal_integration()
    if success:
        print("\n🎉 Ready to run full conformal diagnostic!")
    else:
        print("\n⚠️ Need to debug conformal integration first")