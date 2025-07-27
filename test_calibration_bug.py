#!/usr/bin/env python3
"""
Quick test to identify the calibration bug.

This script creates a simple test case to verify that offset calibration
is working correctly.
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

def test_calibration_bug():
    """Test offset calibration implementation."""
    print("🔧 TESTING CALIBRATION BUG")
    print("="*50)
    
    warnings.filterwarnings('ignore')
    
    # 1. Create simple test data
    print("1. Creating simple test data...")
    config = DataConfig(n_geos=10, n_days=60, seed=123)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=123)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[40].strftime('%Y-%m-%d')  # 40 days pre-period
    eval_start = dates[41].strftime('%Y-%m-%d')
    eval_end = dates[59].strftime('%Y-%m-%d')        # 19 days eval
    
    print(f"   Data: {len(geo_features)} geos, {len(dates)} days")
    print(f"   Pre-period: {pre_period_end}, Eval: {eval_start} to {eval_end}")
    
    # 2. Test models with and without calibration
    models = {
        'No Calibration': STGCNReportingModel(
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01,
            verbose=True,
            use_offset_calibration=False
        ),
        'With Calibration': STGCNReportingModel(
            hidden_dim=16,
            epochs=5,
            learning_rate=0.01,
            verbose=True,
            use_offset_calibration=True
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n2. Testing {model_name}...")
        
        try:
            # Fit model
            model.fit(panel_data, assignment_df, pre_period_end)
            
            # Calculate iROAS
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            
            # Get some predictions for debugging
            predictions = model.predict(panel_data, eval_start, eval_end)
            
            results[model_name] = {
                'iroas': iroas,
                'predictions': predictions,
                'offset_bias': getattr(model, 'offset_bias', None),
                'success': True
            }
            
            print(f"   ✅ {model_name}: iROAS = {iroas:.4f}")
            
            if hasattr(model, 'offset_bias') and model.offset_bias is not None:
                print(f"      Offset bias: sales={model.offset_bias['sales']:.4f}, spend={model.offset_bias['spend']:.4f}")
            
            # Show prediction details
            if isinstance(predictions, dict):
                print(f"      Prediction format: {type(predictions)}")
                for key, val in predictions.items():
                    if hasattr(val, 'shape'):
                        print(f"         {key}: shape={val.shape}, type={type(val)}")
                    else:
                        print(f"         {key}: value={val}, type={type(val)}")
            
        except Exception as e:
            print(f"   ❌ {model_name} failed: {e}")
            results[model_name] = {'success': False, 'error': str(e)}
    
    # 3. Compare results
    print(f"\n3. COMPARISON ANALYSIS:")
    print("="*40)
    
    if all(r.get('success', False) for r in results.values()):
        no_cal = results['No Calibration']
        with_cal = results['With Calibration']
        
        print(f"No Calibration iROAS:   {no_cal['iroas']:.4f}")
        print(f"With Calibration iROAS: {with_cal['iroas']:.4f}")
        print(f"Difference:             {with_cal['iroas'] - no_cal['iroas']:.4f}")
        
        # Check if calibration made things worse
        no_cal_bias = abs(no_cal['iroas'])
        with_cal_bias = abs(with_cal['iroas'])
        
        if with_cal_bias > no_cal_bias:
            print(f"❌ CALIBRATION MADE BIAS WORSE!")
            print(f"   Bias increased by: {(with_cal_bias - no_cal_bias) / no_cal_bias:.1%}")
            
            # Debug the offset application
            if with_cal['offset_bias'] is not None:
                print(f"\n🔍 DEBUGGING OFFSET APPLICATION:")
                print(f"   Offset bias computed: {with_cal['offset_bias']}")
                
                # Check prediction shapes
                no_cal_pred = no_cal['predictions']
                with_cal_pred = with_cal['predictions']
                
                if isinstance(no_cal_pred, dict) and isinstance(with_cal_pred, dict):
                    for metric in ['sales', 'spend']:
                        if metric in no_cal_pred and metric in with_cal_pred:
                            no_cal_val = no_cal_pred[metric]
                            with_cal_val = with_cal_pred[metric]
                            
                            print(f"   {metric.capitalize()}:")
                            print(f"      No calibration:   {no_cal_val}")
                            print(f"      With calibration: {with_cal_val}")
                            print(f"      Offset applied:   {with_cal['offset_bias'][metric]}")
                            
                            # Check if difference matches offset
                            if hasattr(no_cal_val, '__len__') and hasattr(with_cal_val, '__len__'):
                                diff = with_cal_val - no_cal_val
                                print(f"      Actual difference: {diff}")
                                print(f"      Expected offset:   {with_cal['offset_bias'][metric]}")
        else:
            print(f"✅ Calibration reduced bias by: {(no_cal_bias - with_cal_bias) / no_cal_bias:.1%}")
    
    else:
        print("❌ Not all models ran successfully")
        for name, result in results.items():
            if not result.get('success', False):
                print(f"   {name}: {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    results = test_calibration_bug()
    
    print(f"\n💡 CALIBRATION BUG DIAGNOSIS:")
    print("="*50)
    print("The issue is likely in one of these areas:")
    print("1. Offset computation: per-unit vs aggregate scaling mismatch")
    print("2. Offset application: applied to wrong prediction format")
    print("3. iROAS calculation: calibration applied incorrectly")
    print("4. Ensemble inheritance: calibration not properly inherited")