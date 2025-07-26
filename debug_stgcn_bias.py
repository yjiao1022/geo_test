"""
Debug script to investigate STGCN false positive rate issue.

This script tests STGCN on null data scenarios to understand why 
it's producing 100% false positive rates instead of ~5%.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel
from reporting.models import MeanMatchingModel


def create_null_experiment(n_geos=20, n_days=90, seed=42):
    """Create a null experiment (A/A test) with no treatment effect."""
    config = DataConfig(
        n_geos=n_geos,
        n_days=n_days,
        seed=seed,
        base_sales_mean=10000,
        base_sales_std=2000,
        base_spend_mean=5000,
        base_spend_std=1000,
        daily_sales_noise=500,
        daily_spend_noise=200
    )
    
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    # Random assignment
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=seed)
    
    # Define time periods
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[59]  # Day 60 as pre-period end
    eval_period_start = dates[60]  # Day 61 start
    eval_period_end = dates[89]  # Day 90 end
    
    return panel_data, assignment_df, pre_period_end, eval_period_start, eval_period_end


def test_stgcn_bias_detection():
    """Test STGCN with enhanced bias detection."""
    print("=== STGCN Bias Detection Test ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_null_experiment(seed=123)
    
    print(f"Null experiment setup:")
    print(f"  Geos: {len(assignment_df)}")
    print(f"  Treatment: {(assignment_df['assignment'] == 'treatment').sum()}")
    print(f"  Control: {(assignment_df['assignment'] == 'control').sum()}")
    print(f"  Pre-period: Days 1-60")
    print(f"  Eval period: Days 61-90")
    
    # Test with verbose=True to see diagnostics
    stgcn_model = STGCNReportingModel(
        hidden_dim=32,
        num_st_blocks=1,
        epochs=10,  # Fewer epochs for testing
        learning_rate=0.01,
        normalize_data=True,
        verbose=True,  # Enable verbose output
        bias_threshold=0.05  # Lower threshold for bias detection
    )
    
    print(f"\n--- Fitting STGCN ---")
    stgcn_model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Calculate iROAS
    iroas = stgcn_model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
    print(f"\nSTGCN iROAS: {iroas:.4f}")
    
    # Get convergence summary
    convergence_summary = stgcn_model.get_convergence_summary()
    print(f"Convergence: {convergence_summary}")
    
    # Check bias
    bias_check = stgcn_model.check_null_scenario_bias()
    print(f"\nBias Analysis:")
    for key, value in bias_check.items():
        print(f"  {key}: {value}")
    
    # Compare with mean matching (baseline)
    print(f"\n--- Comparing with Mean Matching ---")
    mean_model = MeanMatchingModel()
    mean_model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    mean_iroas = mean_model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
    print(f"Mean Matching iROAS: {mean_iroas:.4f}")
    
    print(f"\n--- iROAS Comparison ---")
    print(f"STGCN:         {iroas:8.4f}")
    print(f"Mean Matching: {mean_iroas:8.4f}")
    print(f"Difference:    {abs(iroas - mean_iroas):8.4f}")
    
    return iroas, mean_iroas, bias_check


def test_multiple_seeds():
    """Test STGCN across multiple random seeds to see consistency of bias."""
    print("\n=== Multiple Seeds Test ===")
    
    seeds = [42, 123, 456, 789, 999]
    stgcn_iroas = []
    mean_iroas = []
    bias_levels = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_null_experiment(seed=seed)
        
        # STGCN with minimal verbosity
        stgcn_model = STGCNReportingModel(
            hidden_dim=24,
            num_st_blocks=1,
            epochs=5,
            learning_rate=0.01,
            normalize_data=True,
            verbose=False,  # Minimal output for multiple tests
            bias_threshold=0.05
        )
        
        stgcn_model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        stgcn_val = stgcn_model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        
        # Mean matching
        mean_model = MeanMatchingModel()
        mean_model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
        mean_val = mean_model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        
        # Bias check
        bias_check = stgcn_model.check_null_scenario_bias()
        bias_level = bias_check.get('bias_level', 'unknown')
        
        stgcn_iroas.append(stgcn_val)
        mean_iroas.append(mean_val)
        bias_levels.append(bias_level)
        
        print(f"  STGCN: {stgcn_val:7.4f}, Mean: {mean_val:7.4f}, Bias: {bias_level}")
    
    print(f"\n--- Summary across {len(seeds)} seeds ---")
    print(f"STGCN   - Mean: {np.mean(stgcn_iroas):7.4f}, Std: {np.std(stgcn_iroas):7.4f}")
    print(f"MeanM   - Mean: {np.mean(mean_iroas):7.4f}, Std: {np.std(mean_iroas):7.4f}")
    print(f"Bias levels: {bias_levels}")
    
    # Check if STGCN is systematically different from 0
    stgcn_mean = np.mean(stgcn_iroas)
    stgcn_std = np.std(stgcn_iroas)
    t_stat = stgcn_mean / (stgcn_std / np.sqrt(len(stgcn_iroas))) if stgcn_std > 0 else 0
    
    print(f"\nSTGCN Bias Test:")
    print(f"  Mean iROAS: {stgcn_mean:.6f}")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  Systematic bias: {'YES' if abs(t_stat) > 2 else 'NO'}")
    
    return stgcn_iroas, mean_iroas, bias_levels


def investigate_prediction_scale():
    """Investigate if STGCN predictions are at wrong scale."""
    print("\n=== Prediction Scale Investigation ===")
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_null_experiment(seed=777)
    
    # Get actual data statistics
    pre_data = panel_data[panel_data['date'] <= pre_period_end]
    eval_data = panel_data[
        (panel_data['date'] >= eval_start) & 
        (panel_data['date'] <= eval_end)
    ]
    
    treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].values
    control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].values
    
    # Actual statistics
    pre_treatment_sales = pre_data[pre_data['geo'].isin(treatment_geos)]['sales']
    eval_treatment_sales = eval_data[eval_data['geo'].isin(treatment_geos)]['sales']
    eval_control_sales = eval_data[eval_data['geo'].isin(control_geos)]['sales']
    
    print(f"Data Statistics:")
    print(f"  Pre-period treatment sales: {pre_treatment_sales.mean():.0f} ± {pre_treatment_sales.std():.0f}")
    print(f"  Eval-period treatment sales: {eval_treatment_sales.mean():.0f} ± {eval_treatment_sales.std():.0f}")
    print(f"  Eval-period control sales: {eval_control_sales.mean():.0f} ± {eval_control_sales.std():.0f}")
    
    # Fit STGCN and get predictions
    stgcn_model = STGCNReportingModel(
        hidden_dim=16,
        epochs=5,
        verbose=True,
        normalize_data=True
    )
    
    stgcn_model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Get STGCN predictions
    predictions = stgcn_model.predict(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
    pred_sales = predictions['sales']
    
    # Calculate evaluation period length for proper comparison
    eval_days = len(eval_data[eval_data['geo'] == treatment_geos[0]])
    
    print(f"\nSTGCN Predictions:")
    print(f"  Predicted total sales per geo: {pred_sales.mean():.0f} ± {pred_sales.std():.0f}")
    print(f"  Predicted daily sales per geo: {pred_sales.mean() / eval_days:.0f}")
    print(f"  Actual daily sales per geo: {eval_treatment_sales.mean():.0f}")
    print(f"  Prediction scale ratio: {(pred_sales.mean() / eval_days) / eval_treatment_sales.mean():.3f}")
    
    # Calculate what iROAS should be (approximately 0 for null)
    actual_treatment_total = eval_treatment_sales.sum()
    predicted_treatment_total = pred_sales.sum()  # Already total for all treatment geos
    
    print(f"\nPrediction Analysis:")
    print(f"  Actual treatment total: {actual_treatment_total:.0f}")
    print(f"  Predicted treatment total: {predicted_treatment_total:.0f}")
    print(f"  Sales difference: {actual_treatment_total - predicted_treatment_total:.0f}")
    
    return {
        'actual_daily_sales_mean': eval_treatment_sales.mean(),
        'predicted_daily_sales_mean': pred_sales.mean() / eval_days,
        'scale_ratio': (pred_sales.mean() / eval_days) / eval_treatment_sales.mean(),
        'sales_difference': actual_treatment_total - predicted_treatment_total
    }


def main():
    """Run all debugging tests."""
    print("STGCN False Positive Rate Debugging")
    print("=" * 50)
    
    # Test 1: Single experiment with detailed diagnostics
    iroas, mean_iroas, bias_check = test_stgcn_bias_detection()
    
    # Test 2: Multiple seeds to check consistency
    stgcn_vals, mean_vals, bias_levels = test_multiple_seeds()
    
    # Test 3: Investigate prediction scale
    scale_analysis = investigate_prediction_scale()
    
    print("\n" + "=" * 50)
    print("DEBUGGING SUMMARY")
    print("=" * 50)
    
    print(f"\n1. Single Test Results:")
    print(f"   STGCN iROAS: {iroas:.4f}")
    print(f"   Mean Matching iROAS: {mean_iroas:.4f}")
    print(f"   Bias detected: {bias_check.get('high_bias_detected', 'unknown')}")
    
    print(f"\n2. Multiple Seeds Results:")
    print(f"   STGCN mean: {np.mean(stgcn_vals):.4f} ± {np.std(stgcn_vals):.4f}")
    print(f"   Mean Matching mean: {np.mean(mean_vals):.4f} ± {np.std(mean_vals):.4f}")
    print(f"   Systematic STGCN bias: {'YES' if abs(np.mean(stgcn_vals)) > 0.1 else 'NO'}")
    
    print(f"\n3. Scale Analysis:")
    print(f"   Prediction scale ratio: {scale_analysis['scale_ratio']:.3f}")
    print(f"   Sales difference: {scale_analysis['sales_difference']:.0f}")
    
    # Diagnosis
    print(f"\n🔍 DIAGNOSIS:")
    if abs(np.mean(stgcn_vals)) > 0.5:
        print("   🚨 STGCN shows SYSTEMATIC BIAS in null scenarios")
        print("   🚨 This explains the 100% false positive rate")
        
        if scale_analysis['scale_ratio'] < 0.5 or scale_analysis['scale_ratio'] > 2.0:
            print("   🔧 Issue likely: PREDICTION SCALE PROBLEM")
        else:
            print("   🔧 Issue likely: MODEL ARCHITECTURE or TRAINING BIAS")
    else:
        print("   ✅ STGCN bias appears reasonable")
        print("   🤔 False positive issue may be elsewhere (CI calculation, etc.)")


if __name__ == "__main__":
    main()