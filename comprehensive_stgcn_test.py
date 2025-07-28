#!/usr/bin/env python3
"""
Comprehensive STGCN variant testing with full A/A evaluation metrics.
Tests: STGCN_Tiny, STGCN_Intermediate, STGCN_Full, and baseline methods.
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/yangjiao/Documents/Projects/geo_test')

from data_simulation.generators import IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel
from reporting.stgcn_model import STGCNReportingModel
from reporting.stgcn_shallow import STGCNShallowModel, STGCNIntermediateModel
from reporting.common_utils import ReportingConfig
from evaluation.metrics import EvaluationRunner, EvaluationConfig

def run_comprehensive_aa_test():
    """
    Run comprehensive A/A test with all STGCN variants.
    Returns the four key metrics: bias, FPR, coverage, CI width.
    """
    
    # Configuration - using IdenticalGeo for cleaner testing
    eval_config = EvaluationConfig(
        n_simulations=3,
        pre_period_days=120, 
        eval_period_days=60,
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,  # Track component metrics
        seed=42
    )
    
    data_config = DataConfig(
        n_geos=24,
        n_days=180,
        base_sales_mean=1000,
        base_sales_std=0,  # Identical baselines
        daily_sales_noise=100,
        seed=42
    )
    
    # Use IdenticalGeo generator for cleanest test
    generator = IdenticalGeoGenerator(data_config)
    
    # Assignment method
    assignment_methods = {'Random': RandomAssignment()}
    
    # Reporting methods - all STGCN variants plus baselines
    from reporting.models import TBRModel, SyntheticControlModel
    reporting_config = ReportingConfig(use_observed_spend=False)
    
    reporting_methods = {
        'MeanMatching': MeanMatchingModel(),
        'TBR': TBRModel(),
        'SCM': SyntheticControlModel(),
        'STGCN_Tiny': STGCNShallowModel(
            epochs=20,
            verbose=False,
            reporting_config=reporting_config
        ),
        'STGCN_Intermediate': STGCNIntermediateModel(
            epochs=20,
            verbose=False, 
            reporting_config=reporting_config
        ),
        'STGCN_Full': STGCNReportingModel(
            hidden_dim=32,
            num_st_blocks=2,
            epochs=20,
            verbose=False,
            reporting_config=reporting_config
        )
    }
    
    print("=" * 80)
    print("COMPREHENSIVE STGCN VARIANT A/A TEST")
    print("=" * 80)
    print(f"Generator: IdenticalGeo (pure daily noise)")
    print(f"Simulations: {eval_config.n_simulations}")
    print(f"Pre-period: {eval_config.pre_period_days} days")
    print(f"Eval-period: {eval_config.eval_period_days} days")
    print(f"Geos: {data_config.n_geos}")
    
    # Run evaluation
    runner = EvaluationRunner(eval_config)
    results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
    
    # Calculate key metrics for each method
    summary_metrics = {}
    
    for method in reporting_methods.keys():
        method_results = results_df[results_df['reporting_method'] == method]
        
        if len(method_results) > 0:
            # 1. Mean incremental sales bias
            mean_bias = method_results['incremental_sales'].mean()
            
            # 2. False positive rate (should be ~5% for 95% CI)
            false_positive_rate = method_results['sales_significant'].mean()
            
            # 3. Coverage rate (should be ~95%)
            # True value is 0, so coverage = CI contains 0
            coverage_rate = ((method_results['sales_lower'] <= 0) & 
                           (method_results['sales_upper'] >= 0)).mean()
            
            # 4. Mean CI width
            mean_ci_width = (method_results['sales_upper'] - method_results['sales_lower']).mean()
            
            # Additional metrics
            bias_std = method_results['incremental_sales'].std()
            
            summary_metrics[method] = {
                'mean_bias': mean_bias,
                'bias_std': bias_std,
                'false_positive_rate': false_positive_rate,
                'coverage_rate': coverage_rate,
                'mean_ci_width': mean_ci_width,
                'n_sims': len(method_results)
            }
    
    return summary_metrics, results_df

def print_summary_table(summary_metrics):
    """Print formatted summary table."""
    
    print("\n" + "=" * 100)
    print("SUMMARY METRICS TABLE")
    print("=" * 100)
    
    # Expected values
    print("Expected A/A performance:")
    print("  Mean incremental sales bias: ±2,000")
    print("  False positive rate: ~5%")
    print("  Coverage rate: ~95%") 
    print("  CI width: reasonable (similar to baselines)")
    
    print(f"\n{'Method':<20} {'Bias':<12} {'FPR':<8} {'Coverage':<10} {'CI Width':<12} {'Status':<15}")
    print("-" * 100)
    
    for method, metrics in summary_metrics.items():
        bias = metrics['mean_bias']
        fpr = metrics['false_positive_rate']
        coverage = metrics['coverage_rate']
        ci_width = metrics['mean_ci_width']
        
        # Determine status
        bias_ok = abs(bias) <= 2000
        fpr_ok = 0.01 <= fpr <= 0.15  # Allow some tolerance
        coverage_ok = coverage >= 0.85  # Allow some tolerance
        
        if bias_ok and fpr_ok and coverage_ok:
            status = "✅ GOOD"
        elif bias_ok:
            status = "⚠️ BIAS OK"
        else:
            status = "❌ BIASED"
        
        print(f"{method:<20} {bias:>8.0f} {fpr:>8.1%} {coverage:>8.1%} {ci_width:>8.0f} {status:<15}")

def analyze_stgcn_variants(summary_metrics):
    """Analyze patterns across STGCN variants."""
    
    print("\n" + "=" * 80)
    print("STGCN VARIANT ANALYSIS")
    print("=" * 80)
    
    stgcn_methods = [m for m in summary_metrics.keys() if 'STGCN' in m]
    
    print("\nBias progression across model complexity:")
    for method in ['STGCN_Tiny', 'STGCN_Intermediate', 'STGCN_Full']:
        if method in summary_metrics:
            bias = summary_metrics[method]['mean_bias']
            print(f"  {method:<20}: {bias:>8.0f}")
    
    # Check if bias grows with complexity
    tiny_bias = summary_metrics.get('STGCN_Tiny', {}).get('mean_bias', 0)
    full_bias = summary_metrics.get('STGCN_Full', {}).get('mean_bias', 0)
    
    print(f"\nDiagnostic conclusion:")
    if abs(tiny_bias) <= 2000:
        print("✅ STGCN_Tiny has acceptable bias - pipeline is sound")
        if abs(full_bias) > abs(tiny_bias) * 2:
            print("⚠️ Bias grows with model complexity - need regularization")
        else:
            print("✅ Bias stable across complexity - architecture is fine")
    else:
        print("❌ Even STGCN_Tiny shows large bias - systematic pipeline issue")
        print("   → Problem is in aggregation, residual measurement, or bias correction logic")

def main():
    """Run the comprehensive test and analysis."""
    
    summary_metrics, results_df = run_comprehensive_aa_test()
    
    # Print results
    print_summary_table(summary_metrics)
    analyze_stgcn_variants(summary_metrics)
    
    # Detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY SIMULATION")
    print("=" * 80)
    
    for method in summary_metrics.keys():
        method_results = results_df[results_df['reporting_method'] == method]
        print(f"\n{method}:")
        for _, row in method_results.iterrows():
            print(f"  Sim {row['simulation_id']}: "
                  f"Sales={row['incremental_sales']:>8.0f}, "
                  f"CI=[{row['sales_lower']:>6.0f}, {row['sales_upper']:>6.0f}], "
                  f"Sig={row['sales_significant']}")
    
    return summary_metrics, results_df

if __name__ == "__main__":
    summary_metrics, results_df = main()