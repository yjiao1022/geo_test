#!/usr/bin/env python3
"""
Simple A/A test to verify per-geo-per-day bias correction is working.
Should print near-zero incremental sales.
"""

import numpy as np
import pandas as pd
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel, ReportingConfig

def test_aa_bias_correction():
    """Test that per-geo-per-day bias correction produces near-zero incremental sales."""
    
    # Set up A/A test 
    data_config = DataConfig(
        n_geos=20,
        n_days=90,
        base_sales_mean=1000,
        base_sales_std=100,
        daily_sales_noise=50,
        seed=42
    )
    
    generator = SimpleNullGenerator(data_config)
    panel_data, geo_features = generator.generate()
    
    # Random assignment
    assignment = RandomAssignment()
    assignment_df = assignment.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    # Set periods
    dates = pd.to_datetime(panel_data['date'].unique())
    pre_period_end = dates[59]  # 60 days pre-period
    eval_period_start = dates[60]
    eval_period_end = dates[89]   # 30 days eval period
    
    # Convert to string format
    pre_period_end_str = pre_period_end.strftime('%Y-%m-%d')
    eval_period_start_str = eval_period_start.strftime('%Y-%m-%d')
    eval_period_end_str = eval_period_end.strftime('%Y-%m-%d')
    
    # Initialize STGCN model
    reporting_config = ReportingConfig(use_observed_spend=False)
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=20,
        verbose=False,  # Reduce output
        reporting_config=reporting_config
    )
    
    # Fit model (this should calculate and store daily_geo_bias)
    print("=== Fitting STGCN Model ===")
    model.fit(panel_data, assignment_df, pre_period_end_str)
    
    # Get counterfactual predictions
    print("\n=== Prediction with Per-Geo-Per-Day Bias Correction ===")
    counterfactual = model.predict(panel_data, eval_period_start_str, eval_period_end_str)
    
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
    counterfactual_sales = counterfactual['sales'].sum()
    incremental_sales = actual_sales - counterfactual_sales
    
    print(f"Actual treatment sales: {actual_sales:.1f}")
    print(f"Counterfactual sales (with bias correction): {counterfactual_sales:.1f}")
    print(f"Final incremental sales: {incremental_sales:.1f}")
    
    print(f"\n=== Verification ===")
    print(f"A/A incremental sales should be close to 0: {incremental_sales:.1f}")
    print(f"Daily geo bias: {model.daily_geo_bias:.4f}")
    
    return {
        'daily_geo_bias': model.daily_geo_bias,
        'incremental_sales': incremental_sales,
        'actual_sales': actual_sales,
        'counterfactual_sales': counterfactual_sales
    }

if __name__ == "__main__":
    results = test_aa_bias_correction()
    print(f"\n=== Summary ===")
    for key, value in results.items():
        if key == 'daily_geo_bias':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.1f}")