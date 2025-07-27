#!/usr/bin/env python3
"""
Debug STGCN bias in A/A tests.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
import numpy as np
import pandas as pd

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel

def debug_stgcn_bias():
    """Debug why STGCN has bias in A/A tests."""
    print("🔧 DEBUGGING STGCN BIAS IN A/A TESTS")
    print("=" * 50)
    
    warnings.filterwarnings('ignore')
    
    # Create A/A test data
    config = DataConfig(n_geos=20, n_days=120, seed=42)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[99].strftime('%Y-%m-%d')
    eval_start = dates[100].strftime('%Y-%m-%d')
    eval_end = dates[119].strftime('%Y-%m-%d')
    
    print(f"Data setup:")
    print(f"  Geos: {len(geo_features)} ({len(assignment_df[assignment_df['assignment']=='treatment'])} treatment)")
    print(f"  Pre-period: 100 days, Eval period: 20 days")
    
    # Train STGCN model
    model = STGCNReportingModel(
        hidden_dim=32,
        epochs=10,
        learning_rate=0.01,
        dropout=0.1,
        verbose=True,
        use_offset_calibration=False  # No calibration
    )
    
    model.fit(panel_data, assignment_df, pre_period_end)
    iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
    
    print(f"\nSTGCN iROAS: {iroas:.4f} (should be ~0 for A/A test)")
    
    # Get counterfactual predictions to understand the bias
    counterfactual = model.predict(panel_data, eval_start, eval_end)
    
    # Get actual treatment group outcomes
    panel_data_copy = panel_data.copy()
    panel_data_copy['date'] = pd.to_datetime(panel_data_copy['date'])
    eval_start_dt = pd.to_datetime(eval_start)
    eval_end_dt = pd.to_datetime(eval_end)
    
    eval_data = panel_data_copy[
        (panel_data_copy['date'] >= eval_start_dt) & 
        (panel_data_copy['date'] <= eval_end_dt)
    ]
    
    treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].values
    treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
    
    # Calculate components
    actual_sales = treatment_data['sales'].sum()
    actual_spend = treatment_data['spend'].sum()
    counterfactual_sales = counterfactual['sales'].sum()
    counterfactual_spend = counterfactual['spend'].sum()
    
    incremental_sales = actual_sales - counterfactual_sales
    incremental_spend = actual_spend - counterfactual_spend
    
    print(f"\nDiagnostic breakdown:")
    print(f"  Actual sales: {actual_sales:.2f}")
    print(f"  Counterfactual sales: {counterfactual_sales:.2f}")
    print(f"  Incremental sales: {incremental_sales:.2f}")
    print(f"  Actual spend: {actual_spend:.2f}")
    print(f"  Counterfactual spend: {counterfactual_spend:.2f}")
    print(f"  Incremental spend: {incremental_spend:.2f}")
    print(f"  iROAS = {incremental_sales:.2f} / {incremental_spend:.2f} = {iroas:.4f}")
    
    # Check if counterfactual is systematically wrong
    print(f"\nBias analysis:")
    sales_bias = (counterfactual_sales - actual_sales) / actual_sales * 100
    spend_bias = (counterfactual_spend - actual_spend) / actual_spend * 100
    print(f"  Sales prediction bias: {sales_bias:.2f}%")
    print(f"  Spend prediction bias: {spend_bias:.2f}%")
    
    if abs(sales_bias) > 5:
        print(f"  🚨 Significant sales prediction bias!")
    if abs(spend_bias) > 5:
        print(f"  🚨 Significant spend prediction bias!")
    
    # Check counterfactual data structure
    print(f"\nCounterfactual data columns: {list(counterfactual.columns)}")
    print(f"Counterfactual data shape: {counterfactual.shape}")
    
    # The bias suggests STGCN is not perfectly predicting the null scenario
    # This could be due to:
    # 1. Model underfitting (needs more training)
    # 2. Temporal distribution shift between pre/eval periods
    # 3. Random assignment creating slightly unbalanced groups
    
    return {
        'iroas': iroas,
        'sales_bias': sales_bias,
        'spend_bias': spend_bias
    }

if __name__ == "__main__":
    result = debug_stgcn_bias()
    
    print(f"\n💡 BIAS ANALYSIS:")
    print("=" * 30)
    print("STGCN appears to have systematic prediction bias.")
    print("Possible causes:")
    print("1. Model underfitting - needs more epochs or complexity")
    print("2. Temporal shift - eval period differs from pre-period distribution")
    print("3. Assignment bias - treatment/control groups differ systematically")
    print("4. Model architecture issue - STGCN not suitable for this data")
    print("5. Data leakage - future information bleeding into predictions")