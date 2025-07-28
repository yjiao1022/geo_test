#!/usr/bin/env python3
"""
Test script to verify STGCN bias correction is working properly.
This should print:
- mean control-pre residual
- stored self.offset_bias 
- new aggregate incremental sales
They should be within a few thousand dollars of each other.
"""

import numpy as np
import pandas as pd
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel, ReportingConfig

def test_bias_correction():
    """Test that bias correction is applied correctly in STGCN model."""
    
    # Set up A/A test with enhanced generator
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
        epochs=20,  # Reduced for fast testing
        verbose=True,
        reporting_config=reporting_config
    )
    
    # Fit model (this should calculate and store offset_bias)
    print("=== Fitting STGCN Model ===")
    model.fit(panel_data, assignment_df, pre_period_end_str)
    
    # Check that sales_bias_offset was calculated
    print(f"\n=== Bias Correction Results ===")
    print(f"Stored sales_bias_offset: {model.sales_bias_offset:.1f}")
    
    # Calculate mean control-pre residual manually for verification
    control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].values
    panel_data_copy = panel_data.copy()
    panel_data_copy['date'] = pd.to_datetime(panel_data_copy['date'])
    
    pre_data = panel_data_copy[
        (panel_data_copy['date'] <= pd.to_datetime(pre_period_end_str)) &
        (panel_data_copy['geo'].isin(control_geos))
    ]
    
    # Get model predictions for control geos in pre-period
    if len(pre_data) > 0:
        # This is a simplified verification - in practice the model calculates this more precisely
        actual_control_sales = pre_data.groupby('geo')['sales'].sum().mean()
        print(f"Mean control sales in pre-period: {actual_control_sales:.1f}")
    
    # Get counterfactual predictions and calculate incremental sales
    print("\n=== Prediction with Bias Correction ===")
    counterfactual = model.predict(panel_data, eval_period_start_str, eval_period_end_str)
    
    # Calculate incremental sales
    treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].values
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
    print(f"Number of treatment geos: {len(treatment_geos)}")
    print(f"Counterfactual sales array shape: {counterfactual['sales'].shape}")
    print(f"Counterfactual sales array: {counterfactual['sales']}")
    print(f"Final incremental sales: {incremental_sales:.1f}")
    
    # The bias correction should make incremental sales close to zero in A/A test
    print(f"\n=== Verification ===")
    print(f"A/A incremental sales should be close to 0: {incremental_sales:.1f}")
    print(f"Bias offset magnitude: {abs(model.sales_bias_offset):.1f}")
    
    return {
        'sales_bias_offset': model.sales_bias_offset,
        'incremental_sales': incremental_sales,
        'actual_sales': actual_sales,
        'counterfactual_sales': counterfactual_sales
    }

if __name__ == "__main__":
    results = test_bias_correction()
    print(f"\n=== Summary ===")
    for key, value in results.items():
        print(f"{key}: {value:.1f}")