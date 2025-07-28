#!/usr/bin/env python3
"""
3-simulation test with both data generators and all methods.
"""

import numpy as np
import pandas as pd
import sys

sys.path.append('/Users/yangjiao/Documents/Projects/geo_test')

from data_simulation.generators import SimpleNullGenerator, IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel
from reporting.stgcn_model import STGCNReportingModel
from reporting.common_utils import ReportingConfig

def run_single_simulation(generator, methods, config, sim_id):
    """Run a single simulation with all methods."""
    
    # Set seed for reproducibility
    np.random.seed(42 + sim_id)
    
    # Generate data
    panel_data, geo_features = generator.generate()
    
    # Random assignment
    assignment = RandomAssignment()
    assignment_df = assignment.assign(geo_features, treatment_ratio=0.5, seed=42 + sim_id)
    
    # Set periods
    dates = pd.to_datetime(panel_data['date'].unique())
    pre_period_end = dates[config['pre_days'] - 1]
    eval_period_start = dates[config['pre_days']]
    eval_period_end = dates[config['pre_days'] + config['eval_days'] - 1]
    
    # Convert to string format
    pre_period_end_str = pre_period_end.strftime('%Y-%m-%d')
    eval_period_start_str = eval_period_start.strftime('%Y-%m-%d')
    eval_period_end_str = eval_period_end.strftime('%Y-%m-%d')
    
    results = []
    
    for method_name, method in methods.items():
        try:
            print(f"  Sim {sim_id} - {method_name}...")
            
            # Fit model
            method.fit(panel_data, assignment_df, pre_period_end_str)
            
            # Get bias offset if STGCN
            bias_offset = getattr(method, 'daily_geo_bias', None)
            
            # Calculate incremental sales
            treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].values
            panel_data_copy = panel_data.copy()
            panel_data_copy['date'] = pd.to_datetime(panel_data_copy['date'])
            
            eval_data = panel_data_copy[
                (panel_data_copy['date'] >= pd.to_datetime(eval_period_start_str)) & 
                (panel_data_copy['date'] <= pd.to_datetime(eval_period_end_str))
            ]
            treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
            actual_sales = treatment_data['sales'].sum()
            actual_spend = treatment_data['spend'].sum()
            
            # Get counterfactual
            if hasattr(method, 'predict'):
                counterfactual = method.predict(panel_data, eval_period_start_str, eval_period_end_str)
                if isinstance(counterfactual['sales'], np.ndarray):
                    counterfactual_sales = counterfactual['sales'].sum()
                else:
                    counterfactual_sales = counterfactual['sales']
            else:
                # Use iROAS calculation
                iroas = method.calculate_iroas(panel_data, eval_period_start_str, eval_period_end_str)
                counterfactual_sales = actual_sales - (iroas * actual_spend)
            
            incremental_sales = actual_sales - counterfactual_sales
            
            results.append({
                'sim_id': sim_id,
                'method': method_name,
                'actual_sales': actual_sales,
                'counterfactual_sales': counterfactual_sales,
                'incremental_sales': incremental_sales,
                'bias_offset': bias_offset if bias_offset is not None else 0.0,
                'n_treat_geos': len(treatment_geos)
            })
            
        except Exception as e:
            print(f"    Error with {method_name}: {e}")
            continue
    
    return results

def run_multi_simulation_test():
    """Run 3 simulations with both generators and all methods."""
    
    # Configuration
    config = {
        'pre_days': 120,
        'eval_days': 60,
        'n_geos': 24,
        'total_days': 180
    }
    
    data_config = DataConfig(
        n_geos=config['n_geos'],
        n_days=config['total_days'],
        base_sales_mean=1000,
        base_sales_std=200,
        daily_sales_noise=100,
        seed=42
    )
    
    # Generators
    generators = {
        'SimpleNull': SimpleNullGenerator(data_config),
        'IdenticalGeo': IdenticalGeoGenerator(data_config)
    }
    
    # Methods
    reporting_config = ReportingConfig(use_observed_spend=False)
    
    all_results = []
    
    for gen_name, generator in generators.items():
        print(f"\n{'='*60}")
        print(f"Testing {gen_name} Generator")
        print(f"{'='*60}")
        
        for sim_id in range(3):
            print(f"\nSimulation {sim_id + 1}:")
            
            # Create fresh method instances for each simulation
            methods = {
                'MeanMatching': MeanMatchingModel(),
                'TBR': TBRModel(),
                'STGCN': STGCNReportingModel(
                    hidden_dim=16,
                    epochs=20,
                    verbose=False,
                    reporting_config=reporting_config
                )
            }
            
            sim_results = run_single_simulation(generator, methods, config, sim_id)
            
            for result in sim_results:
                result['generator'] = gen_name
                all_results.append(result)
    
    return pd.DataFrame(all_results)

def analyze_results(results_df):
    """Analyze and summarize the results."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-SIMULATION RESULTS")
    print("="*80)
    
    # Summary by generator and method
    summary = results_df.groupby(['generator', 'method']).agg({
        'incremental_sales': ['mean', 'std'],
        'bias_offset': 'mean'
    }).round(1)
    
    print("\nIncremental Sales Summary (Mean ± Std):")
    print(summary)
    
    # Detailed results
    print(f"\nDetailed Results:")
    for gen in results_df['generator'].unique():
        print(f"\n{gen} Generator:")
        gen_data = results_df[results_df['generator'] == gen]
        
        for method in gen_data['method'].unique():
            method_data = gen_data[gen_data['method'] == method]
            mean_sales = method_data['incremental_sales'].mean()
            std_sales = method_data['incremental_sales'].std()
            mean_bias = method_data['bias_offset'].mean()
            
            print(f"  {method}:")
            print(f"    Incremental sales: {mean_sales:.1f} ± {std_sales:.1f}")
            if mean_bias != 0:
                print(f"    Daily geo bias offset: {mean_bias:.4f}")
    
    # Expected A/A performance
    print(f"\nExpected A/A Performance:")
    print(f"- Target incremental sales: ~0")
    print(f"- Current performance:")
    
    for gen in results_df['generator'].unique():
        gen_data = results_df[results_df['generator'] == gen]
        print(f"  {gen}:")
        for method in gen_data['method'].unique():
            method_data = gen_data[gen_data['method'] == method]
            mean_sales = method_data['incremental_sales'].mean()
            print(f"    {method}: {mean_sales:.1f} (bias)")

if __name__ == "__main__":
    results_df = run_multi_simulation_test()
    analyze_results(results_df)