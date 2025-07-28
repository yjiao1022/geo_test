#!/usr/bin/env python3
"""
Run A/A simulations with conformal prediction for STGCN diagnostic.
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

def run_conformal_aa_diagnostic():
    """Run 3 A/A simulations with conformal method for STGCN metrics."""
    print("🎯 CONFORMAL A/A DIAGNOSTIC")
    print("=" * 50)
    
    warnings.filterwarnings('ignore')
    
    # Create evaluation config for A/A conformal mode
    eval_config = EvaluationConfig(
        n_simulations=3,  # Run 3 simulations as requested
        pre_period_days=60,
        eval_period_days=20,
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,
        seed=42
    )
    
    # Create data generator
    config = DataConfig(n_geos=20, n_days=90, seed=42)
    generator = SimpleNullGenerator(config)
    
    # Create methods
    assignment_methods = {
        'Random': RandomAssignment()
    }
    
    reporting_methods = {
        'STGCN': STGCNReportingModel(
            hidden_dim=32, 
            epochs=10, 
            learning_rate=0.01, 
            dropout=0.1,
            verbose=False
        )
    }
    
    # Run evaluation
    print("Running 3 A/A simulations with conformal prediction...")
    runner = EvaluationRunner(eval_config)
    
    try:
        results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
        
        # Filter STGCN results only
        stgcn_results = results_df[results_df['reporting_method'] == 'STGCN']
        
        if len(stgcn_results) == 3:
            print(f"\n📊 STGCN CONFORMAL RESULTS (3 simulations):")
            print("=" * 60)
            
            # Extract metrics for each simulation
            for i, (_, row) in enumerate(stgcn_results.iterrows()):
                print(f"\nSimulation {i+1}:")
                print(f"  Incremental Sales: {row['incremental_sales']:.4f}")
                print(f"  Sales CI: [{row['sales_lower']:.4f}, {row['sales_upper']:.4f}]")
                print(f"  Sales Significant: {row['sales_significant']}")
                print(f"  CI Width: {row['sales_upper'] - row['sales_lower']:.4f}")
            
            # Calculate aggregate metrics
            sales_bias = stgcn_results['incremental_sales'].values
            sales_lower = stgcn_results['sales_lower'].values
            sales_upper = stgcn_results['sales_upper'].values
            significant = stgcn_results['sales_significant'].values
            
            # 1. Mean incremental sales bias
            mean_sales_bias = np.mean(sales_bias)
            
            # 2. False positive rate on incremental sales (should be ~5%)
            false_positive_rate = np.mean(significant)
            
            # 3. Coverage rate on incremental sales (should be ~95%)
            # In A/A test, true incremental sales = 0
            true_sales = 0.0
            coverage = np.mean((sales_lower <= true_sales) & (sales_upper >= true_sales))
            
            # 4. Mean confidence interval width
            ci_widths = sales_upper - sales_lower
            mean_ci_width = np.mean(ci_widths)
            
            print(f"\n🎯 KEY METRICS FOR STGCN WITH CONFORMAL PREDICTION:")
            print("=" * 60)
            print(f"1. Mean incremental sales bias: {mean_sales_bias:.4f}")
            print(f"2. False positive rate: {false_positive_rate:.1%} (target: ~5%)")
            print(f"3. Coverage rate: {coverage:.1%} (target: ~95%)")
            print(f"4. Mean CI width: {mean_ci_width:.4f}")
            
            # Assessment
            print(f"\n📈 ASSESSMENT:")
            print("=" * 30)
            if abs(mean_sales_bias) < 1000:
                print(f"✅ Sales bias is reasonable: {mean_sales_bias:.1f}")
            else:
                print(f"🚨 Large sales bias detected: {mean_sales_bias:.1f}")
                
            if 0.01 <= false_positive_rate <= 0.15:
                print(f"✅ FPR is reasonable: {false_positive_rate:.1%}")
            else:
                print(f"⚠️ FPR outside expected range: {false_positive_rate:.1%}")
                
            if coverage >= 0.90:
                print(f"✅ Good coverage: {coverage:.1%}")
            else:
                print(f"🚨 Poor coverage: {coverage:.1%}")
                
            if mean_ci_width > 0:
                print(f"✅ CI width is positive: {mean_ci_width:.1f}")
            else:
                print(f"🚨 Invalid CI width: {mean_ci_width:.1f}")
            
            return {
                'mean_sales_bias': mean_sales_bias,
                'false_positive_rate': false_positive_rate,
                'coverage_rate': coverage,
                'mean_ci_width': mean_ci_width
            }
        else:
            print(f"🚨 ERROR: Expected 3 results, got {len(stgcn_results)}")
            return None
            
    except Exception as e:
        print(f"🚨 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    metrics = run_conformal_aa_diagnostic()
    if metrics:
        print(f"\n✅ Conformal diagnostic completed successfully!")
        print(f"📋 Copy these metrics for the report:")
        print(f"   Mean incremental sales bias: {metrics['mean_sales_bias']:.4f}")
        print(f"   False positive rate: {metrics['false_positive_rate']:.1%}")
        print(f"   Coverage rate: {metrics['coverage_rate']:.1%}")
        print(f"   CI width: {metrics['mean_ci_width']:.4f}")
    else:
        print(f"\n🚨 Conformal diagnostic failed!")