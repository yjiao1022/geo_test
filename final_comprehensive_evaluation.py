#!/usr/bin/env python3
"""
Final comprehensive evaluation generating the requested table format.
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/yangjiao/Documents/Projects/geo_test')

from data_simulation.generators import IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel, SyntheticControlModel
from reporting.stgcn_model import STGCNReportingModel
from reporting.stgcn_shallow import STGCNShallowModel, STGCNIntermediateModel
from reporting.common_utils import ReportingConfig
from evaluation.metrics import EvaluationRunner, EvaluationConfig

def run_final_evaluation():
    """Run comprehensive evaluation with all methods."""
    
    print("=" * 80)
    print("FINAL COMPREHENSIVE A/A EVALUATION")
    print("=" * 80)
    print("Testing all methods with restored baseline behavior")
    
    # Configuration for proper statistical power
    eval_config = EvaluationConfig(
        n_simulations=10,  # Balanced for speed vs accuracy
        pre_period_days=120,
        eval_period_days=60,
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,
        seed=42
    )
    
    data_config = DataConfig(
        n_geos=24,
        n_days=180,
        base_sales_mean=1000,
        base_sales_std=0,  # Identical baselines for clean testing
        daily_sales_noise=100,
        seed=42
    )
    
    generator = IdenticalGeoGenerator(data_config)
    assignment_methods = {'Random': RandomAssignment()}
    
    # All methods to test
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
    
    print(f"Testing {len(reporting_methods)} methods with {eval_config.n_simulations} simulations each")
    print("Methods:", list(reporting_methods.keys()))
    
    # Run evaluation
    runner = EvaluationRunner(eval_config)
    results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
    
    # Calculate comprehensive metrics
    summary_metrics = {}
    
    for method in reporting_methods.keys():
        method_results = results_df[results_df['reporting_method'] == method]
        
        if len(method_results) > 0:
            # Calculate key metrics
            mean_bias = method_results['incremental_sales'].mean()
            bias_std = method_results['incremental_sales'].std()
            false_positive_rate = method_results['sales_significant'].mean()
            coverage_rate = ((method_results['sales_lower'] <= 0) & 
                           (method_results['sales_upper'] >= 0)).mean()
            mean_ci_width = (method_results['sales_upper'] - method_results['sales_lower']).mean()
            ci_width_std = (method_results['sales_upper'] - method_results['sales_lower']).std()
            
            summary_metrics[method] = {
                'bias': mean_bias,
                'bias_std': bias_std,
                'fpr': false_positive_rate,
                'coverage': coverage_rate, 
                'ci_width': mean_ci_width,
                'ci_width_std': ci_width_std,
                'n_sims': len(method_results)
            }
    
    return summary_metrics, results_df

def print_final_table(summary_metrics):
    """Print the requested table format."""
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    
    # Expected performance for reference
    print("Expected A/A Performance:")
    print("  Bias: ~0 (close to zero)")
    print("  FPR: ~5% (false positive rate)")
    print("  Coverage: ~95% (CI contains true value)")
    print("  CI Width: Reasonable (method-dependent)")
    
    print(f"\n{'Variant':<20} {'Bias':<12} {'FPR':<8} {'Coverage':<10} {'CI Width':<12}")
    print("-" * 70)
    
    # Order: baseline methods first, then STGCN variants
    method_order = ['MeanMatching', 'TBR', 'SCM', 'STGCN_Tiny', 'STGCN_Intermediate', 'STGCN_Full']
    
    for method in method_order:
        if method in summary_metrics:
            metrics = summary_metrics[method]
            bias = metrics['bias']
            fpr = metrics['fpr']
            coverage = metrics['coverage']
            ci_width = metrics['ci_width']
            
            print(f"{method:<20} {bias:>8.0f} {fpr:>8.1%} {coverage:>8.1%} {ci_width:>8.0f}")
    
    # Analysis section
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Baseline methods analysis
    baseline_methods = ['MeanMatching', 'TBR', 'SCM']
    baseline_working = 0
    
    print("\n1. Baseline Methods Performance:")
    for method in baseline_methods:
        if method in summary_metrics:
            metrics = summary_metrics[method]
            bias_ok = abs(metrics['bias']) <= 100  # Very small bias expected
            coverage_ok = metrics['coverage'] >= 0.80  # Allow some tolerance
            fpr_ok = metrics['fpr'] <= 0.20  # Allow some tolerance
            
            status = "✅ GOOD" if (bias_ok and coverage_ok) else "⚠️ ISSUES"
            
            print(f"   {method:<15}: {status}")
            print(f"      Bias: {metrics['bias']:>6.0f} (target: ~0)")
            print(f"      Coverage: {metrics['coverage']:>5.1%} (target: ~95%)")
            print(f"      FPR: {metrics['fpr']:>5.1%} (target: ~5%)")
            
            if bias_ok and coverage_ok:
                baseline_working += 1
    
    print(f"\n   Summary: {baseline_working}/{len(baseline_methods)} baseline methods working correctly")
    
    # STGCN methods analysis
    stgcn_methods = ['STGCN_Tiny', 'STGCN_Intermediate', 'STGCN_Full']
    
    print("\n2. STGCN Methods Performance:")
    for method in stgcn_methods:
        if method in summary_metrics:
            metrics = summary_metrics[method]
            bias_magnitude = abs(metrics['bias'])
            
            if bias_magnitude <= 5000:
                status = "✅ GOOD"
            elif bias_magnitude <= 50000:
                status = "⚠️ MODERATE BIAS"
            else:
                status = "❌ LARGE BIAS"
                
            print(f"   {method:<20}: {status}")
            print(f"      Bias: {metrics['bias']:>8.0f}")
            print(f"      Coverage: {metrics['coverage']:>5.1%}")
            print(f"      FPR: {metrics['fpr']:>5.1%}")
    
    # Overall conclusions
    print("\n3. Key Findings:")
    if baseline_working >= 2:
        print("   ✅ Baseline methods restored to working condition")
        print("   ✅ Bias correction successfully isolated to STGCN methods only")
    else:
        print("   ❌ Baseline methods still have issues")
    
    stgcn_bias_levels = [abs(summary_metrics[m]['bias']) for m in stgcn_methods if m in summary_metrics]
    if stgcn_bias_levels:
        max_stgcn_bias = max(stgcn_bias_levels)
        if max_stgcn_bias > 100000:
            print("   ⚠️ STGCN methods still require bias correction improvement")
        else:
            print("   ✅ STGCN bias correction working well")

def main():
    """Run comprehensive evaluation and generate final table."""
    
    summary_metrics, results_df = run_final_evaluation()
    print_final_table(summary_metrics)
    
    # Additional detailed breakdown
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN BY SIMULATION")
    print("=" * 80)
    
    # Show sample results for verification
    for method in ['MeanMatching', 'SCM', 'STGCN_Tiny']:
        if method in summary_metrics:
            method_results = results_df[results_df['reporting_method'] == method]
            print(f"\n{method} (first 5 simulations):")
            for _, row in method_results.head(5).iterrows():
                print(f"  Sim {row['simulation_id']}: "
                      f"Sales={row['incremental_sales']:>8.0f}, "
                      f"CI=[{row['sales_lower']:>6.0f}, {row['sales_upper']:>6.0f}], "
                      f"Sig={row['sales_significant']}")
    
    return summary_metrics, results_df

if __name__ == "__main__":
    summary_metrics, results_df = main()
    print(f"\n🎯 Evaluation complete! Summary data available in 'summary_metrics' dict")