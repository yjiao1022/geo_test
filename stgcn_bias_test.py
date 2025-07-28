#!/usr/bin/env python3
"""
Test STGCN bias correction with both data generators.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append('/Users/yangjiao/Documents/Projects/geo_test')

from data_simulation.generators import SimpleNullGenerator, IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment

def test_stgcn_bias_correction():
    """Test STGCN bias correction with different data generators."""
    
    # Try to import STGCN directly from file
    try:
        from reporting.stgcn_model import STGCNReportingModel
        print("Successfully imported STGCNReportingModel")
    except Exception as e:
        print(f"Could not import STGCN: {e}")
        return
    
    # Configuration with longer periods
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
    
    # Test both generators
    generators = {
        'SimpleNull': SimpleNullGenerator(data_config),
        'IdenticalGeo': IdenticalGeoGenerator(data_config)
    }
    
    results = []
    
    for gen_name, generator in generators.items():
        print(f"\n{'='*60}")
        print(f"Testing {gen_name} Generator with STGCN")
        print(f"{'='*60}")
        
        # Generate data
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
        
        # Create STGCN model with minimal config
        try:
            # Try simple initialization
            model = STGCNReportingModel(
                hidden_dim=16,
                epochs=20,
                verbose=True
            )
        except Exception as e:
            print(f"Could not initialize STGCN: {e}")
            continue
        
        # Fit model
        print("Fitting STGCN model...")
        try:
            model.fit(panel_data, assignment_df, pre_period_end_str)
            
            # Check if bias offset was calculated
            bias_offset = getattr(model, 'daily_geo_bias', None)
            print(f"Daily geo bias offset: {bias_offset}")
            
        except Exception as e:
            print(f"Error fitting model: {e}")
            continue
        
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
        
        # Get counterfactual predictions
        try:
            counterfactual = model.predict(panel_data, eval_period_start_str, eval_period_end_str)
            counterfactual_sales = counterfactual['sales'].sum()
            incremental_sales = actual_sales - counterfactual_sales
            
            print(f"Actual treatment sales: {actual_sales:.1f}")
            print(f"Counterfactual sales (with bias correction): {counterfactual_sales:.1f}")
            print(f"Incremental sales: {incremental_sales:.1f}")
            print(f"Number of treatment geos: {len(treatment_geos)}")
            print(f"Eval period days: {config['eval_days']}")
            
            results.append({
                'generator': gen_name,
                'method': 'STGCN',
                'actual_sales': actual_sales,
                'counterfactual_sales': counterfactual_sales,
                'incremental_sales': incremental_sales,
                'bias_offset': bias_offset if bias_offset is not None else 0.0,
                'n_treat_geos': len(treatment_geos),
                'n_eval_days': config['eval_days']
            })
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            continue
    
    return results

if __name__ == "__main__":
    results = test_stgcn_bias_correction()
    
    if results:
        print("\n" + "="*60)
        print("STGCN BIAS CORRECTION SUMMARY")
        print("="*60)
        
        for result in results:
            print(f"\n{result['generator']} Generator:")
            print(f"  Incremental sales: {result['incremental_sales']:.1f}")
            print(f"  Daily geo bias offset: {result['bias_offset']:.4f}")
            
            # Calculate expected total bias offset
            if result['bias_offset'] != 0:
                expected_total = result['bias_offset'] * result['n_treat_geos'] * result['n_eval_days']
                print(f"  Expected total offset: {expected_total:.1f}")
    else:
        print("No STGCN results obtained")