#!/usr/bin/env python3
"""
Simplified bias correction evaluation with multiple data generators and methods.
Tests both SimpleNullGenerator (geo heterogeneity) and IdenticalGeoGenerator (pure noise).
"""

import numpy as np
import pandas as pd
from data_simulation.generators import SimpleNullGenerator, IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel

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
    
    # Get counterfactual using predict method
    if hasattr(method, 'predict'):
        counterfactual = method.predict(panel_data, eval_period_start_str, eval_period_end_str)
        if isinstance(counterfactual['sales'], np.ndarray):
            counterfactual_sales = counterfactual['sales'].sum()
        else:
            # Handle case where it's a scalar 
            counterfactual_sales = counterfactual['sales'] * len(treatment_geos) * config['eval_days']
    else:
        # For models without predict method, use iROAS calculation
        iroas = method.calculate_iroas(panel_data, eval_period_start_str, eval_period_end_str)
        counterfactual_sales = actual_sales - (iroas * actual_spend)
    
    incremental_sales = actual_sales - counterfactual_sales
    
    print(f"Actual treatment sales: {actual_sales:.1f}")
    print(f"Counterfactual sales: {counterfactual_sales:.1f}")
    print(f"Incremental sales: {incremental_sales:.1f}")
    print(f"Number of treatment geos: {len(treatment_geos)}")
    print(f"Eval period days: {config['eval_days']}")
    
    return {
        'generator': generator_name,
        'method': method_name,
        'actual_sales': actual_sales,
        'counterfactual_sales': counterfactual_sales,
        'incremental_sales': incremental_sales,
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
    methods = {
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
                import traceback
                traceback.print_exc()
    
    # Convert to DataFrame for analysis
    single_df = pd.DataFrame(single_results)
    
    print("\n" + "=" * 80)
    print("SINGLE SIMULATION SUMMARY")
    print("=" * 80)
    print(single_df[['generator', 'method', 'incremental_sales']])
    
    return single_df

def analyze_data_generators():
    """Analyze the differences between data generators."""
    
    config = DataConfig(
        n_geos=24,
        n_days=180,
        base_sales_mean=1000,
        base_sales_std=200,
        daily_sales_noise=100,
        seed=42
    )
    
    print("=" * 80)
    print("DATA GENERATOR ANALYSIS")
    print("=" * 80)
    
    # Test SimpleNullGenerator
    simple_gen = SimpleNullGenerator(config)
    simple_panel, simple_geo = simple_gen.generate()
    
    print("\nSimpleNullGenerator (Geo Heterogeneity):")
    print(f"Base sales range: {simple_geo['base_sales'].min():.1f} - {simple_geo['base_sales'].max():.1f}")
    print(f"Base sales std: {simple_geo['base_sales'].std():.1f}")
    print(f"Sales std within geo: {simple_panel.groupby('geo')['sales'].std().mean():.1f}")
    print(f"Sales std across geos: {simple_panel.groupby('date')['sales'].std().mean():.1f}")
    
    # Test IdenticalGeoGenerator
    identical_gen = IdenticalGeoGenerator(config)
    identical_panel, identical_geo = identical_gen.generate()
    
    print("\nIdenticalGeoGenerator (Pure Daily Noise):")
    print(f"Base sales range: {identical_geo['base_sales'].min():.1f} - {identical_geo['base_sales'].max():.1f}")
    print(f"Base sales std: {identical_geo['base_sales'].std():.1f}")
    print(f"Sales std within geo: {identical_panel.groupby('geo')['sales'].std().mean():.1f}")
    print(f"Sales std across geos: {identical_panel.groupby('date')['sales'].std().mean():.1f}")

if __name__ == "__main__":
    # First analyze the data generators
    analyze_data_generators()
    
    # Then run the evaluation
    single_results = run_comprehensive_evaluation()
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print("\nIncremental Sales Bias by Generator and Method:")
    for _, row in single_results.iterrows():
        print(f"  {row['generator']} + {row['method']}: {row['incremental_sales']:.1f}")