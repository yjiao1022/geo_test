#!/usr/bin/env python3
"""
Multi-seed STGCN variant testing to check CI width consistency and other patterns.
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

def run_seed_test(seed_value):
    """Run test with specific seed."""
    
    eval_config = EvaluationConfig(
        n_simulations=3,
        pre_period_days=120,
        eval_period_days=60,
        confidence_level=0.95,
        uncertainty_method='conformal',
        aa_mode=True,
        seed=seed_value
    )
    
    data_config = DataConfig(
        n_geos=24,
        n_days=180,
        base_sales_mean=1000,
        base_sales_std=0,  # Identical baselines
        daily_sales_noise=100,
        seed=seed_value
    )
    
    generator = IdenticalGeoGenerator(data_config)
    assignment_methods = {'Random': RandomAssignment()}
    
    # All methods
    reporting_config = ReportingConfig(use_observed_spend=False)
    
    reporting_methods = {
        'MeanMatching': MeanMatchingModel(),
        'TBR': TBRModel(),
        'SCM': SyntheticControlModel(),
        'STGCN_Tiny': STGCNShallowModel(
            epochs=15,  # Reduced for speed
            verbose=False,
            reporting_config=reporting_config
        ),
        'STGCN_Intermediate': STGCNIntermediateModel(
            epochs=15,
            verbose=False,
            reporting_config=reporting_config
        ),
        'STGCN_Full': STGCNReportingModel(
            hidden_dim=16,  # Reduced for speed
            num_st_blocks=1,
            epochs=15,
            verbose=False,
            reporting_config=reporting_config
        )
    }
    
    print(f"\n{'='*60}")
    print(f"SEED {seed_value} TEST")
    print(f"{'='*60}")
    
    runner = EvaluationRunner(eval_config)
    results_df = runner.run_evaluation(generator, assignment_methods, reporting_methods)
    
    # Calculate metrics for each method
    summary = {}
    for method in reporting_methods.keys():
        method_results = results_df[results_df['reporting_method'] == method]
        
        if len(method_results) > 0:
            summary[method] = {
                'mean_bias': method_results['incremental_sales'].mean(),
                'bias_std': method_results['incremental_sales'].std(),
                'false_positive_rate': method_results['sales_significant'].mean(),
                'coverage_rate': ((method_results['sales_lower'] <= 0) & 
                                (method_results['sales_upper'] >= 0)).mean(),
                'mean_ci_width': (method_results['sales_upper'] - method_results['sales_lower']).mean(),
                'ci_width_std': (method_results['sales_upper'] - method_results['sales_lower']).std(),
                'seed': seed_value
            }
    
    return summary

def main():
    """Run tests with multiple seeds."""
    
    seeds = [42, 123, 789]  # Different seeds
    all_results = []
    
    for seed in seeds:
        seed_results = run_seed_test(seed)
        
        # Add to combined results
        for method, metrics in seed_results.items():
            metrics['method'] = method
            all_results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print comprehensive summary
    print("\n" + "="*100)
    print("MULTI-SEED COMPREHENSIVE RESULTS")
    print("="*100)
    
    print(f"\n{'Method':<20} {'Seed':<8} {'Bias':<12} {'FPR':<8} {'Coverage':<10} {'CI Width':<12} {'CI Std':<10}")
    print("-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"{row['method']:<20} {row['seed']:<8} {row['mean_bias']:>8.0f} "
              f"{row['false_positive_rate']:>8.1%} {row['coverage_rate']:>8.1%} "
              f"{row['mean_ci_width']:>8.0f} {row['ci_width_std']:>8.1f}")
    
    # Check CI width consistency within methods
    print("\n" + "="*80)
    print("CI WIDTH ANALYSIS")
    print("="*80)
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method]
        ci_widths = method_data['mean_ci_width'].values
        ci_mean = ci_widths.mean()
        ci_std = ci_widths.std()
        
        print(f"\n{method}:")
        print(f"  CI widths across seeds: {ci_widths}")
        print(f"  Mean: {ci_mean:.1f}, Std: {ci_std:.1f}")
        
        if ci_std < 1.0:  # Very low variation
            print(f"  ⚠️  Suspiciously consistent CI widths!")
        else:
            print(f"  ✅ Normal CI width variation")
    
    # Summary by method (averaged across seeds)
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE ACROSS SEEDS")
    print("="*80)
    
    method_summary = results_df.groupby('method').agg({
        'mean_bias': ['mean', 'std'],
        'false_positive_rate': 'mean',
        'coverage_rate': 'mean',
        'mean_ci_width': ['mean', 'std']
    }).round(2)
    
    print(method_summary)
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    stgcn_methods = [m for m in results_df['method'].unique() if 'STGCN' in m]
    
    print("\n1. Bias Consistency:")
    for method in ['MeanMatching', 'TBR', 'SCM'] + stgcn_methods:
        if method in results_df['method'].values:
            method_data = results_df[results_df['method'] == method]
            bias_std = method_data['mean_bias'].std()
            bias_mean = method_data['mean_bias'].mean()
            print(f"   {method}: {bias_mean:.0f} ± {bias_std:.0f}")
    
    print("\n2. CI Width Investigation:")
    for method in stgcn_methods:
        if method in results_df['method'].values:
            method_data = results_df[results_df['method'] == method]
            ci_widths = method_data['mean_ci_width'].values
            if len(set(ci_widths.round(0))) == 1:
                print(f"   {method}: IDENTICAL CI widths ({ci_widths[0]:.0f}) - SUSPICIOUS")
            else:
                print(f"   {method}: Variable CI widths - Normal")
    
    return results_df

if __name__ == "__main__":
    results_df = main()