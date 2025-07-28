#!/usr/bin/env python3
"""
Debug script to understand how different models format their predictions.
"""

import numpy as np
import pandas as pd
from data_simulation.generators import IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel, SyntheticControlModel
from reporting.stgcn_shallow import STGCNShallowModel

print("🔍 DEBUGGING PREDICTION FORMATS")
print("="*50)

# Create simple test data
config = DataConfig(n_geos=4, n_days=30, base_sales_mean=1000, base_sales_std=0, seed=42)
generator = IdenticalGeoGenerator(config)
panel_data, geo_features = generator.generate()

# Create assignment
assignment = RandomAssignment().assign(geo_features, seed=42)

print(f"Test data: {len(geo_features)} geos, {len(panel_data)} observations")
print(f"Assignment: {(assignment['assignment'] == 'treatment').sum()} treatment, {(assignment['assignment'] == 'control').sum()} control")

# Define periods
dates = sorted(panel_data['date'].unique())
pre_period_end = dates[20].strftime('%Y-%m-%d')  # First 20 days for training
eval_start = dates[21].strftime('%Y-%m-%d')     # Next 9 days for evaluation  
eval_end = dates[-1].strftime('%Y-%m-%d')

print(f"Pre-period: {dates[0].strftime('%Y-%m-%d')} to {pre_period_end}")
print(f"Eval period: {eval_start} to {eval_end}")

# Test each model
models = {
    'MeanMatching': MeanMatchingModel(),
    'TBR': TBRModel(), 
    'SCM': SyntheticControlModel(),
    'STGCN_Tiny': STGCNShallowModel(epochs=3, verbose=False)
}

print(f"\n🔬 PREDICTION FORMAT ANALYSIS:")
print("="*50)

for name, model in models.items():
    print(f"\n{name}:")
    try:
        # Fit model
        model.fit(panel_data, assignment, pre_period_end)
        
        # Get predictions
        predictions = model.predict(panel_data, eval_start, eval_end)
        
        print(f"  Prediction keys: {list(predictions.keys())}")
        
        if 'sales' in predictions:
            sales_pred = predictions['sales']
            print(f"  Sales prediction type: {type(sales_pred)}")
            print(f"  Sales prediction shape: {np.array(sales_pred).shape}")
            print(f"  Sales prediction values: {np.array(sales_pred)[:5]}...")  # First 5 values
            print(f"  Sales prediction sum: {np.array(sales_pred).sum()}")
            
        # Get actual treatment data for comparison
        eval_mask = (panel_data['date'] >= eval_start) & (panel_data['date'] <= eval_end)
        eval_data = panel_data[eval_mask]
        treatment_geos = assignment[assignment['assignment'] == 'treatment']['geo'].values
        treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
        
        # Compare aggregation methods
        method1_actual = treatment_data['sales'].sum()  # Total sum
        method2_actual = treatment_data.groupby('date')['sales'].mean().sum()  # Daily mean then sum
        
        print(f"  Actual sales (total sum): {method1_actual}")
        print(f"  Actual sales (daily mean sum): {method2_actual}")
        print(f"  Difference: {method1_actual - method2_actual}")
        
        # Calculate incremental sales both ways
        pred_sum = np.array(sales_pred).sum()
        incremental1 = method1_actual - pred_sum  # My old approach
        incremental2 = method2_actual - pred_sum  # My new approach
        
        print(f"  Incremental (old method): {incremental1}")
        print(f"  Incremental (new method): {incremental2}")
        print(f"  Method difference: {incremental1 - incremental2}")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")

print(f"\n💡 ANALYSIS:")
print("The key insight is whether different models return predictions in")
print("different formats that affect the aggregation logic.")