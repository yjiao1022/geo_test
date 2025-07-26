"""
Detailed STGCN bias investigation to find root cause of persistent false positives.

This script investigates specific potential sources of bias:
1. Recursive prediction accumulation
2. Normalization/denormalization issues 
3. Training convergence quality
4. Model architecture issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def test_stgcn_per_timestep_bias():
    """Test if STGCN has systematic per-timestep bias that accumulates."""
    print("=== Per-Timestep Bias Investigation ===")
    
    config = DataConfig(
        n_geos=12, n_days=30, seed=42,
        base_sales_mean=10000, base_sales_std=1000,
        daily_sales_noise=200
    )
    
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    # Use shorter periods to minimize accumulation
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[19]  # Day 20 
    eval_start = dates[20]     # Day 21
    eval_end = dates[22]       # Day 23 (only 3 days)
    
    # Test with different configurations
    configs = [
        {"epochs": 20, "learning_rate": 0.005, "hidden_dim": 16, "normalize": True},
        {"epochs": 30, "learning_rate": 0.01, "hidden_dim": 24, "normalize": True},
        {"epochs": 15, "learning_rate": 0.02, "hidden_dim": 32, "normalize": False},
    ]
    
    results = []
    
    for i, config_params in enumerate(configs):
        print(f"\n--- Config {i+1}: {config_params} ---")
        
        model = STGCNReportingModel(
            hidden_dim=config_params["hidden_dim"],
            epochs=config_params["epochs"],
            learning_rate=config_params["learning_rate"],
            normalize_data=config_params["normalize"],
            verbose=False,
            bias_threshold=0.03
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
            
            # Get bias diagnostics
            bias_check = model.check_null_scenario_bias()
            convergence = model.get_convergence_summary()
            
            results.append({
                'config': i+1,
                'iroas': iroas,
                'bias_level': bias_check.get('bias_level', 'unknown'),
                'convergence': convergence,
                'relative_bias': bias_check.get('relative_bias', 0)
            })
            
            print(f"  iROAS: {iroas:.4f}")
            print(f"  Bias: {bias_check.get('relative_bias', 0):.3f} ({bias_check.get('bias_level', 'unknown')})")
            print(f"  Convergence: {convergence}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'config': i+1,
                'iroas': np.nan,
                'bias_level': 'error',
                'convergence': 'failed',
                'relative_bias': np.nan
            })
    
    # Summary
    print(f"\n--- Summary ---")
    valid_results = [r for r in results if not np.isnan(r['iroas'])]
    if valid_results:
        iroas_values = [r['iroas'] for r in valid_results]
        bias_values = [r['relative_bias'] for r in valid_results if not np.isnan(r['relative_bias'])]
        
        print(f"iROAS range: {min(iroas_values):.4f} to {max(iroas_values):.4f}")
        print(f"Mean iROAS: {np.mean(iroas_values):.4f} ± {np.std(iroas_values):.4f}")
        if bias_values:
            print(f"Bias range: {min(bias_values):.3f} to {max(bias_values):.3f}")
        
        # Check for systematic bias
        systematic_bias = abs(np.mean(iroas_values)) > 1.0  # Should be ~0 for null
        print(f"Systematic bias detected: {'YES' if systematic_bias else 'NO'}")
    
    return results


def test_stgcn_training_stability():
    """Test STGCN training stability across multiple random initializations."""
    print("\n=== Training Stability Investigation ===")
    
    config = DataConfig(
        n_geos=10, n_days=40, seed=123,
        base_sales_mean=8000, base_sales_std=1500
    )
    
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=123)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[29]
    eval_start = dates[30]
    eval_end = dates[39]
    
    # Test multiple random seeds (different model initializations)
    torch_seeds = [1, 42, 123, 456, 789]
    results = []
    
    for seed in torch_seeds:
        print(f"\n--- PyTorch seed {seed} ---")
        
        # Set random seeds for reproducible model initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = STGCNReportingModel(
            hidden_dim=20,
            epochs=15,
            learning_rate=0.01,
            normalize_data=True,
            verbose=False
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
            iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
            
            convergence = model.get_convergence_summary()
            print(f"  iROAS: {iroas:.4f}, Convergence: {convergence}")
            
            results.append(iroas)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(np.nan)
    
    # Analyze stability
    valid_results = [r for r in results if not np.isnan(r)]
    if len(valid_results) >= 3:
        mean_iroas = np.mean(valid_results)
        std_iroas = np.std(valid_results)
        
        print(f"\n--- Stability Analysis ---")
        print(f"Mean iROAS: {mean_iroas:.4f}")
        print(f"Std iROAS: {std_iroas:.4f}")
        print(f"Range: {min(valid_results):.4f} to {max(valid_results):.4f}")
        
        # Check consistency
        high_variance = std_iroas > 2.0
        systematic_non_zero = abs(mean_iroas) > 1.0
        
        print(f"High variance: {'YES' if high_variance else 'NO'}")
        print(f"Systematic non-zero bias: {'YES' if systematic_non_zero else 'NO'}")
        
        if systematic_non_zero:
            print("⚠️ STGCN shows systematic bias regardless of initialization")
        if high_variance:
            print("⚠️ STGCN shows unstable training across seeds")
    
    return results


def investigate_recursive_prediction():
    """Investigate if recursive prediction is accumulating bias."""
    print("\n=== Recursive Prediction Bias Investigation ===")
    
    config = DataConfig(
        n_geos=8, n_days=25, seed=999,
        base_sales_mean=12000, base_sales_std=800
    )
    
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=999)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[14]  # Day 15
    
    model = STGCNReportingModel(
        hidden_dim=16,
        epochs=10,
        learning_rate=0.015,
        normalize_data=True,
        verbose=True
    )
    
    model.fit(panel_data, assignment_df, pre_period_end.strftime('%Y-%m-%d'))
    
    # Test different prediction period lengths to see if bias accumulates
    period_lengths = [1, 3, 5, 7]
    bias_by_length = []
    
    for length in period_lengths:
        if 15 + length > len(dates):
            continue
            
        eval_start = dates[15]
        eval_end = dates[14 + length]
        
        iroas = model.calculate_iroas(panel_data, eval_start.strftime('%Y-%m-%d'), eval_end.strftime('%Y-%m-%d'))
        
        # Expected iROAS should be ~0, so bias is just the absolute value
        bias_by_length.append(abs(iroas))
        
        print(f"Period length {length} days: iROAS = {iroas:.4f}, |bias| = {abs(iroas):.4f}")
    
    # Check if bias increases with period length (indicating accumulation)
    if len(bias_by_length) >= 3:
        correlation = np.corrcoef(period_lengths[:len(bias_by_length)], bias_by_length)[0, 1]
        print(f"\nBias vs Period Length correlation: {correlation:.3f}")
        
        if correlation > 0.5:
            print("⚠️ Bias appears to INCREASE with prediction period length")
            print("   This suggests recursive prediction is accumulating bias")
        elif correlation < -0.1:
            print("✓ Bias DECREASES with longer periods (unexpectedly)")
        else:
            print("✓ Bias does not strongly correlate with period length")
    
    return bias_by_length


def main():
    """Run all detailed bias investigations."""
    print("STGCN Detailed Bias Investigation")
    print("=" * 60)
    
    # Test 1: Per-timestep bias with different configurations
    config_results = test_stgcn_per_timestep_bias()
    
    # Test 2: Training stability across random seeds
    stability_results = test_stgcn_training_stability()
    
    # Test 3: Recursive prediction bias accumulation
    recursive_bias = investigate_recursive_prediction()
    
    print("\n" + "=" * 60)
    print("DETAILED DIAGNOSIS")
    print("=" * 60)
    
    # Overall assessment
    all_configs_biased = all(r.get('relative_bias', 0) > 0.1 for r in config_results if not np.isnan(r.get('relative_bias', np.nan)))
    training_unstable = len(stability_results) > 0 and np.std([r for r in stability_results if not np.isnan(r)]) > 1.0
    
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    
    if all_configs_biased:
        print("   🚨 SYSTEMATIC BIAS across all configurations")
        print("   🚨 Issue is likely FUNDAMENTAL to STGCN architecture or implementation")
    else:
        print("   ⚠️ Bias varies by configuration - tuning may help")
    
    if training_unstable:
        print("   🚨 TRAINING INSTABILITY detected")
        print("   🔧 Try: Lower learning rate, more epochs, gradient clipping")
    else:
        print("   ✓ Training appears stable across seeds")
    
    if len(recursive_bias) >= 3:
        recursive_correlation = np.corrcoef(range(len(recursive_bias)), recursive_bias)[0, 1]
        if recursive_correlation > 0.3:
            print("   🚨 RECURSIVE PREDICTION BIAS accumulation detected")
            print("   🔧 Issue in iterative prediction loop or denormalization")
        else:
            print("   ✓ Recursive prediction bias does not accumulate significantly")
    
    print("\n📋 RECOMMENDED ACTIONS:")
    print("   1. Check STGCN model architecture for systematic bias sources")
    print("   2. Investigate normalization/denormalization implementation")
    print("   3. Review recursive prediction logic and anchoring method")
    print("   4. Consider alternative architectures or hybrid approaches")


if __name__ == "__main__":
    main()