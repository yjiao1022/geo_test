"""
Quick regularization test - focused on most promising configurations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import warnings

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def create_test_scenario(seed: int = 42):
    """Create test scenario."""
    config = DataConfig(n_geos=20, n_days=120, seed=seed)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=seed)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[99]
    eval_start = dates[100]
    eval_end = dates[119]
    
    return panel_data, assignment_df, pre_period_end, eval_start, eval_end


def test_config(config, panel_data, assignment_df, pre_period_end, eval_start, eval_end, n_seeds=5):
    """Test configuration with fewer seeds for speed."""
    iroas_estimates = []
    mc_ci_widths = []
    
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        np.random.seed(1000 + seed)
        
        model = STGCNReportingModel(
            hidden_dim=config['hidden_dim'],
            num_st_blocks=config['num_st_blocks'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate'],
            dropout=config['dropout'],
            normalize_data=True,
            verbose=False
        )
        
        try:
            model.fit(panel_data, assignment_df, pre_period_end)
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            iroas_estimates.append(iroas)
            
            mc_lower, mc_upper = model.confidence_interval(
                panel_data, eval_start, eval_end,
                method='mc_dropout',
                n_mc_samples=30,  # Reduced for speed
                confidence_level=0.95
            )
            mc_ci_widths.append(mc_upper - mc_lower)
            
        except Exception as e:
            print(f"    Seed {seed} failed: {e}")
            continue
    
    if len(iroas_estimates) < 2:
        return None
    
    empirical_std = np.std(iroas_estimates)
    avg_mc_width = np.mean(mc_ci_widths)
    mc_implied_std = avg_mc_width / 3.92
    underestimation_ratio = empirical_std / mc_implied_std if mc_implied_std > 0 else float('inf')
    
    return {
        'config': config,
        'n_successful': len(iroas_estimates),
        'empirical_std': empirical_std,
        'mc_implied_std': mc_implied_std,
        'underestimation_ratio': underestimation_ratio,
        'avg_mc_width': avg_mc_width
    }


def main():
    """Quick regularization test."""
    print("Quick STGCN Regularization Test")
    print("=" * 40)
    
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_scenario(seed=42)
    
    # Test key configurations
    configs = [
        # Baseline
        {'name': 'baseline', 'hidden_dim': 32, 'num_st_blocks': 2, 'epochs': 5, 'learning_rate': 0.01, 'dropout': 0.1},
        
        # High dropout
        {'name': 'high_dropout', 'hidden_dim': 32, 'num_st_blocks': 2, 'epochs': 5, 'learning_rate': 0.01, 'dropout': 0.5},
        
        # Small model
        {'name': 'small_model', 'hidden_dim': 16, 'num_st_blocks': 1, 'epochs': 5, 'learning_rate': 0.01, 'dropout': 0.3},
        
        # Conservative
        {'name': 'conservative', 'hidden_dim': 8, 'num_st_blocks': 1, 'epochs': 8, 'learning_rate': 0.005, 'dropout': 0.7},
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        result = test_config(config, panel_data, assignment_df, pre_period_end, eval_start, eval_end, n_seeds=5)
        
        if result:
            results.append(result)
            print(f"  Empirical std: {result['empirical_std']:.2f}")
            print(f"  Underestimation ratio: {result['underestimation_ratio']:.1f}x")
            
            if result['underestimation_ratio'] < 20:
                print(f"  🎉 PROMISING!")
            elif result['underestimation_ratio'] < 100:
                print(f"  ✅ IMPROVED")
            else:
                print(f"  ❌ STILL HIGH")
        else:
            print(f"  ❌ FAILED")
    
    print(f"\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    
    if results:
        # Sort by underestimation ratio
        results.sort(key=lambda x: x['underestimation_ratio'])
        
        for result in results:
            config = result['config']
            ratio = result['underestimation_ratio']
            print(f"{config['name']:12} | {ratio:7.1f}x | dropout={config['dropout']} hidden={config['hidden_dim']}")
        
        best = results[0]
        print(f"\n🏆 Best configuration: {best['config']['name']}")
        print(f"   Underestimation ratio: {best['underestimation_ratio']:.1f}x")
        
        if best['underestimation_ratio'] < 10:
            print(f"   ✅ Good enough for MC dropout!")
        elif best['underestimation_ratio'] < 50:
            print(f"   ⚠️ Better, but may need small ensemble")
        else:
            print(f"   ❌ Still need full ensemble approach")
    else:
        print("No successful configurations")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()