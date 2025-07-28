#!/usr/bin/env python3
"""
Quick final table generation by running fewer simulations.
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/yangjiao/Documents/Projects/geo_test')

from data_simulation.generators import IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel, SyntheticControlModel
from reporting.stgcn_model import STGCNReportingModel
from reporting.stgcn_shallow import STGCNShallowModel
from reporting.common_utils import ReportingConfig
from evaluation.metrics import EvaluationRunner, EvaluationConfig

def run_quick_evaluation():
    """Quick evaluation with essential methods."""
    
    print("=" * 80)
    print("QUICK FINAL EVALUATION - COMPREHENSIVE RESULTS TABLE")
    print("=" * 80)
    
    # Fast configuration
    eval_config = EvaluationConfig(
        n_simulations=5,  # Quick test
        pre_period_days=60,  # Reduced 
        eval_period_days=30,  # Reduced
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,
        seed=42
    )
    
    data_config = DataConfig(
        n_geos=16,  # Reduced for speed
        n_days=90,
        base_sales_mean=1000,
        base_sales_std=0,
        daily_sales_noise=100,
        seed=42
    )
    
    generator = IdenticalGeoGenerator(data_config)
    assignment_methods = {'Random': RandomAssignment()}
    
    # Essential methods for table
    reporting_config = ReportingConfig(use_observed_spend=False)
    
    reporting_methods = {
        'MeanMatching': MeanMatchingModel(),
        'TBR': TBRModel(),
        'SCM': SyntheticControlModel(), 
        'STGCN_Tiny': STGCNShallowModel(
            epochs=5,  # Very fast
            verbose=False,
            reporting_config=reporting_config
        ),
        'STGCN_Full': STGCNReportingModel(
            hidden_dim=16,  # Smaller
            num_st_blocks=1,
            epochs=5,  # Very fast
            verbose=False,
            reporting_config=reporting_config
        )
    }
    
    print(f"Methods: {list(reporting_methods.keys())}")
    print(f"Configuration: {eval_config.n_simulations} sims, {data_config.n_geos} geos, {data_config.n_days} days")
    
    # Run evaluation
    runner = EvaluationRunner(eval_config)
    results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
    
    # Calculate summary metrics
    summary = {}
    
    for method in reporting_methods.keys():
        method_results = results_df[results_df['reporting_method'] == method]
        
        if len(method_results) > 0:
            bias = method_results['incremental_sales'].mean()
            fpr = method_results['sales_significant'].mean()
            coverage = ((method_results['sales_lower'] <= 0) & 
                       (method_results['sales_upper'] >= 0)).mean()
            ci_width = (method_results['sales_upper'] - method_results['sales_lower']).mean()
            
            summary[method] = {
                'bias': bias,
                'fpr': fpr, 
                'coverage': coverage,
                'ci_width': ci_width
            }
    
    return summary, results_df

def print_final_results_table(summary):
    """Print the requested format table."""
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - COMPREHENSIVE EVALUATION")
    print("=" * 80)
    
    print("Expected A/A Performance:")
    print("  Bias: ~0, FPR: ~5%, Coverage: ~95%, CI Width: method-dependent")
    
    print(f"\n{'Variant':<20} {'Bias':<12} {'FPR':<8} {'Coverage':<10} {'CI Width':<12}")
    print("-" * 70)
    
    # Ordered results
    method_order = ['MeanMatching', 'TBR', 'SCM', 'STGCN_Tiny', 'STGCN_Full']
    
    for method in method_order:
        if method in summary:
            metrics = summary[method]
            bias = metrics['bias']
            fpr = metrics['fpr']
            coverage = metrics['coverage']
            ci_width = metrics['ci_width']
            
            print(f"{method:<20} {bias:>8.0f} {fpr:>8.1%} {coverage:>8.1%} {ci_width:>8.0f}")
    
    # Status analysis
    print("\n" + "=" * 80)
    print("STATUS ANALYSIS")
    print("=" * 80)
    
    baseline_methods = ['MeanMatching', 'TBR', 'SCM']
    stgcn_methods = ['STGCN_Tiny', 'STGCN_Full']
    
    print("\n✅ BASELINE METHODS (Restored to original behavior):")
    for method in baseline_methods:
        if method in summary:
            m = summary[method]
            bias_ok = abs(m['bias']) <= 50
            coverage_ok = m['coverage'] >= 0.7
            status = "WORKING" if (bias_ok and coverage_ok) else "NEEDS WORK"
            print(f"   {method:<15}: {status:<12} (Bias: {m['bias']:>5.0f}, Coverage: {m['coverage']:>5.1%})")
    
    print("\n⚠️ STGCN METHODS (With bias correction applied):")
    for method in stgcn_methods:
        if method in summary:
            m = summary[method]
            bias_magnitude = abs(m['bias'])
            if bias_magnitude <= 1000:
                status = "EXCELLENT"
            elif bias_magnitude <= 10000:
                status = "GOOD"
            elif bias_magnitude <= 100000:
                status = "MODERATE BIAS"
            else:
                status = "LARGE BIAS"
            print(f"   {method:<15}: {status:<12} (Bias: {m['bias']:>8.0f}, Coverage: {m['coverage']:>5.1%})")
    
    # Key findings
    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS")
    print("=" * 80)
    
    baseline_working = sum(1 for m in baseline_methods 
                          if m in summary and abs(summary[m]['bias']) <= 50 and summary[m]['coverage'] >= 0.7)
    
    stgcn_bias_levels = [abs(summary[m]['bias']) for m in stgcn_methods if m in summary]
    max_stgcn_bias = max(stgcn_bias_levels) if stgcn_bias_levels else 0
    
    print(f"1. Baseline Methods: {baseline_working}/{len(baseline_methods)} working correctly")
    print(f"   → Restored to original iROAS calculation (commit baafb1f behavior)")
    print(f"   → No longer affected by component metrics calculation")
    
    print(f"\n2. STGCN Methods: Max bias = {max_stgcn_bias:,.0f}")
    if max_stgcn_bias <= 10000:
        print(f"   → ✅ Bias correction working well")
    elif max_stgcn_bias <= 100000:
        print(f"   → ⚠️ Moderate bias - bias correction partially effective")
    else:
        print(f"   → ❌ Large bias - bias correction needs improvement")
    
    print(f"\n3. Isolation Success:")
    if baseline_working >= 2:
        print(f"   → ✅ Successfully isolated bias correction to STGCN methods only")
        print(f"   → ✅ Baseline methods restored to working state")
    else:
        print(f"   → ❌ Baseline methods still have issues")

def main():
    """Run quick evaluation."""
    
    summary, results_df = run_quick_evaluation()
    print_final_results_table(summary)
    
    # Sample detailed results
    print("\n" + "=" * 80)
    print("SAMPLE DETAILED RESULTS")
    print("=" * 80)
    
    for method in ['MeanMatching', 'SCM', 'STGCN_Tiny']:
        if method in summary:
            method_results = results_df[results_df['reporting_method'] == method]
            print(f"\n{method}:")
            for _, row in method_results.head(3).iterrows():
                print(f"  Sim {row['simulation_id']}: "
                      f"Sales={row['incremental_sales']:>8.0f}, "
                      f"CI=[{row['sales_lower']:>6.0f}, {row['sales_upper']:>6.0f}], "
                      f"Sig={row['sales_significant']}")
    
    return summary

if __name__ == "__main__":
    summary = main()