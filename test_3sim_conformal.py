#!/usr/bin/env python3
"""
3-simulation split-conformal test to record:
- mean incremental-sales bias
- false-positive rate  
- coverage
- CI width
"""

import numpy as np
import pandas as pd
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel, ReportingConfig
from evaluation.metrics import EvaluationRunner, EvaluationConfig

def run_3sim_conformal_test():
    """Run 3 simulations and record key metrics."""
    
    # Configuration for 3 simulations
    eval_config = EvaluationConfig(
        n_simulations=3,
        pre_period_days=60,
        eval_period_days=30,
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,  # Track component metrics
        seed=42
    )
    
    # Data configuration - A/A test with no true effect
    data_config = DataConfig(
        n_geos=20,
        n_days=90,
        base_sales_mean=1000,
        base_sales_std=100,
        daily_sales_noise=50,
        seed=42
    )
    
    # Set up generator and assignment method
    generator = SimpleNullGenerator(data_config)
    assignment_methods = {
        'Random': RandomAssignment()
    }
    
    # Set up STGCN reporting method
    reporting_config = ReportingConfig(use_observed_spend=False)
    stgcn_model = STGCNReportingModel(
        hidden_dim=16,
        epochs=20,
        verbose=False,  # Reduce output for cleaner results
        reporting_config=reporting_config
    )
    
    reporting_methods = {
        'STGCN': stgcn_model
    }
    
    # Run evaluation
    print("=== Running 3-Simulation Split-Conformal Test ===")
    runner = EvaluationRunner(eval_config)
    results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
    
    # Calculate key metrics
    print("\n=== Results Summary ===")
    
    # Mean incremental sales bias
    mean_sales_bias = results_df['incremental_sales'].mean()
    print(f"Mean incremental-sales bias: {mean_sales_bias:.1f}")
    
    # False positive rate (should be ~5% for 95% CI)
    false_positive_rate = results_df['sales_significant'].mean()
    print(f"False-positive rate: {false_positive_rate:.3f} ({false_positive_rate*100:.1f}%)")
    
    # Coverage rate (should be ~95%)
    # In A/A test, true incremental sales = 0, so coverage = CI contains 0
    coverage_rate = ((results_df['sales_lower'] <= 0) & (results_df['sales_upper'] >= 0)).mean()
    print(f"Coverage rate: {coverage_rate:.3f} ({coverage_rate*100:.1f}%)")
    
    # CI width
    mean_ci_width = (results_df['sales_upper'] - results_df['sales_lower']).mean()
    print(f"Mean CI width: {mean_ci_width:.1f}")
    
    print("\n=== Detailed Results ===")
    print(results_df[['simulation_id', 'incremental_sales', 'sales_lower', 'sales_upper', 'sales_significant']])
    
    return {
        'mean_sales_bias': mean_sales_bias,
        'false_positive_rate': false_positive_rate,
        'coverage_rate': coverage_rate,
        'mean_ci_width': mean_ci_width,
        'results_df': results_df
    }

if __name__ == "__main__":
    metrics = run_3sim_conformal_test()
    
    print(f"\n=== Final Metrics Summary ===")
    print(f"Mean incremental-sales bias: {metrics['mean_sales_bias']:.1f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.3f}")
    print(f"Coverage rate: {metrics['coverage_rate']:.3f}")
    print(f"CI width: {metrics['mean_ci_width']:.1f}")