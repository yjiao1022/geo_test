#!/usr/bin/env python3
"""
Comprehensive bias correction evaluation with multiple data generators and methods.
Tests both SimpleNullGenerator (geo heterogeneity) and IdenticalGeoGenerator (pure noise).
"""

import numpy as np
import pandas as pd
from data_simulation.generators import SimpleNullGenerator, IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel, ReportingConfig
from reporting.models import MeanMatchingModel, TBRModel
from evaluation.metrics import EvaluationRunner, EvaluationConfig

def test_single_aa_simulation(generator_name, generator, method_name, method, config):
    """Test a single A/A simulation and return bias metrics."""
    
    panel_data, geo_features = generator.generate()
    
    # Random assignment
    assignment = RandomAssignment()
    assignment_df = assignment.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    # Set periods  
    dates = pd.to_datetime(panel_data['date'].unique())
    pre_period_end = dates[config['pre_days'] - 1]
    eval_period_start = dates[config['pre_days']]
    eval_period_end = dates[config['pre_days'] + config['eval_days'] - 1]
    
    # Convert to string format
    pre_period_end_str = pre_period_end.strftime('%Y-%m-%d')
    eval_period_start_str = eval_period_start.strftime('%Y-%m-%d')
    eval_period_end_str = eval_period_end.strftime('%Y-%m-%d')
    
    # Fit model
    print(f"\n=== {generator_name} + {method_name} ===")
    method.fit(panel_data, assignment_df, pre_period_end_str)
    
    # Store bias offset if STGCN
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
    
    # Get counterfactual with bias correction (if applicable)
    if hasattr(method, 'predict'):
        counterfactual = method.predict(panel_data, eval_period_start_str, eval_period_end_str)
        counterfactual_sales = counterfactual['sales'].sum()
    else:
        # For models without predict method
        iroas = method.calculate_iroas(panel_data, eval_period_start_str, eval_period_end_str)
        actual_spend = treatment_data['spend'].sum()
        counterfactual_sales = actual_sales - (iroas * actual_spend)
    
    incremental_sales = actual_sales - counterfactual_sales
    
    print(f"Actual treatment sales: {actual_sales:.1f}")
    print(f"Counterfactual sales: {counterfactual_sales:.1f}")
    print(f"Incremental sales: {incremental_sales:.1f}")
    if bias_offset is not None:
        print(f"Daily geo bias offset: {bias_offset:.4f}")
    
    return {
        'generator': generator_name,
        'method': method_name,
        'actual_sales': actual_sales,
        'counterfactual_sales': counterfactual_sales,
        'incremental_sales': incremental_sales,
        'bias_offset': bias_offset if bias_offset is not None else 0.0,
        'n_treat_geos': len(treatment_geos),
        'n_eval_days': config['eval_days']
    }

def run_comprehensive_evaluation():
    """Run comprehensive A/A evaluation with multiple generators and methods."""
    
    # Configuration with longer periods
    config = {
        'pre_days': 120,  # Increased from 60
        'eval_days': 60,  # Increased from 30
        'n_geos': 24,
        'total_days': 180
    }
    
    data_config = DataConfig(
        n_geos=config['n_geos'],
        n_days=config['total_days'],
        base_sales_mean=1000,
        base_sales_std=200,  # For SimpleNull
        daily_sales_noise=100,
        seed=42
    )
    
    # Data generators
    generators = {
        'SimpleNull': SimpleNullGenerator(data_config),
        'IdenticalGeo': IdenticalGeoGenerator(data_config)
    }
    
    # Reporting methods
    reporting_config = ReportingConfig(use_observed_spend=False)
    methods = {
        'STGCN': STGCNReportingModel(
            hidden_dim=16,
            epochs=30,
            verbose=False,
            reporting_config=reporting_config
        ),
        'MeanMatching': MeanMatchingModel(),
        'TBR': TBRModel()
    }
    
    # Run single simulations for each combination
    print("=" * 80)
    print("SINGLE A/A SIMULATION RESULTS")
    print("=" * 80)
    
    single_results = []
    for gen_name, generator in generators.items():
        for method_name, method in methods.items():
            try:
                result = test_single_aa_simulation(gen_name, generator, method_name, method, config)
                single_results.append(result)
            except Exception as e:
                print(f"Error with {gen_name} + {method_name}: {e}")
    
    # Convert to DataFrame for analysis
    single_df = pd.DataFrame(single_results)
    
    print("\n" + "=" * 80)
    print("SINGLE SIMULATION SUMMARY")
    print("=" * 80)
    print(single_df[['generator', 'method', 'incremental_sales', 'bias_offset']])
    
    # Run 3-simulation evaluation for each generator
    print("\n" + "=" * 80)
    print("3-SIMULATION CONFORMAL EVALUATION")
    print("=" * 80)
    
    multi_results = {}
    
    for gen_name, generator in generators.items():
        print(f"\n--- {gen_name} Generator ---")
        
        eval_config = EvaluationConfig(
            n_simulations=3,
            pre_period_days=config['pre_days'],
            eval_period_days=config['eval_days'],
            confidence_level=0.95,
            uncertainty_method='conformal',
            aa_mode=True,
            seed=42
        )
        
        assignment_methods = {'Random': RandomAssignment()}
        
        # Fresh instances for multi-simulation
        fresh_methods = {
            'STGCN': STGCNReportingModel(
                hidden_dim=16,
                epochs=30,
                verbose=False,
                reporting_config=reporting_config
            ),
            'MeanMatching': MeanMatchingModel(),
            'TBR': TBRModel()
        }
        
        runner = EvaluationRunner(eval_config)
        results_df = runner.run_evaluation(generator, assignment_methods, fresh_methods)
        
        # Calculate summary metrics
        summary = {}
        for method in fresh_methods.keys():
            method_results = results_df[results_df['reporting_method'] == method]
            if len(method_results) > 0:
                summary[method] = {
                    'mean_incremental_sales': method_results['incremental_sales'].mean(),
                    'std_incremental_sales': method_results['incremental_sales'].std(),
                    'false_positive_rate': method_results['sales_significant'].mean(),
                    'coverage_rate': ((method_results['sales_lower'] <= 0) & 
                                    (method_results['sales_upper'] >= 0)).mean(),
                    'mean_ci_width': (method_results['sales_upper'] - method_results['sales_lower']).mean()
                }
        
        multi_results[gen_name] = summary
        
        # Print results for this generator
        print(f"\n{gen_name} Results:")
        for method, metrics in summary.items():
            print(f"  {method}:")
            print(f"    Mean incremental sales: {metrics['mean_incremental_sales']:.1f}")
            print(f"    False positive rate: {metrics['false_positive_rate']:.3f}")
            print(f"    Coverage rate: {metrics['coverage_rate']:.3f}")
            print(f"    Mean CI width: {metrics['mean_ci_width']:.1f}")
    
    return single_df, multi_results

if __name__ == "__main__":
    single_results, multi_results = run_comprehensive_evaluation()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\nSingle Simulation Incremental Sales Bias:")
    for _, row in single_results.iterrows():
        print(f"  {row['generator']} + {row['method']}: {row['incremental_sales']:.1f}")
    
    print("\n3-Simulation Average Bias:")
    for gen_name, methods in multi_results.items():
        print(f"  {gen_name}:")
        for method, metrics in methods.items():
            print(f"    {method}: {metrics['mean_incremental_sales']:.1f}")