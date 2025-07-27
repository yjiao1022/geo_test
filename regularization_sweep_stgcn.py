"""
Step 1: Quick regularization sweep to reduce STGCN training variance.

Test various regularization configurations to see if we can collapse the 
catastrophic 3,345x variance underestimation to a manageable level.

Based on the action ladder:
1. Quick regularisation sweep (dropout, weight-decay, hidden dims ↓)
2. Re-measure across 10 seeds  
3. If std-dev ≲ data-driven CI width → stop; MC-dropout may be "good enough"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import warnings
from itertools import product

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


def create_test_scenario(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """Create consistent test scenario."""
    config = DataConfig(n_geos=20, n_days=120, seed=seed)
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=seed)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[99]  # 100 days training
    eval_start = dates[100]
    eval_end = dates[119]  # 20 days eval
    
    return panel_data, assignment_df, pre_period_end, eval_start, eval_end


def test_regularization_config(
    config: Dict[str, Any],
    panel_data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    pre_period_end: str,
    eval_start: str,
    eval_end: str,
    n_seeds: int = 10
) -> Dict[str, Any]:
    """Test a specific regularization configuration across multiple seeds."""
    
    iroas_estimates = []
    mc_ci_widths = []
    convergence_info = []
    
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        np.random.seed(1000 + seed)
        
        # Create model with regularization config
        model = STGCNReportingModel(
            hidden_dim=config['hidden_dim'],
            num_st_blocks=config['num_st_blocks'],
            epochs=config['epochs'],
            window_size=5,
            learning_rate=config['learning_rate'],
            dropout=config['dropout'],
            normalize_data=True,
            verbose=False
        )
        
        # Add weight decay if specified
        if 'weight_decay' in config and config['weight_decay'] > 0:
            # We'll need to modify the model to use weight decay
            # For now, we'll track this in the config but STGCN doesn't directly support it
            pass
        
        try:
            # Fit model
            model.fit(panel_data, assignment_df, pre_period_end)
            
            # Check convergence
            diagnostics = model.get_training_diagnostics()
            convergence = diagnostics.get('convergence_assessment', 'unknown')
            convergence_info.append(convergence)
            
            # Get iROAS estimate
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            iroas_estimates.append(iroas)
            
            # Get MC dropout CI width
            try:
                mc_lower, mc_upper = model.confidence_interval(
                    panel_data, eval_start, eval_end,
                    method='mc_dropout',
                    n_mc_samples=50,
                    confidence_level=0.95
                )
                mc_ci_widths.append(mc_upper - mc_lower)
            except:
                mc_ci_widths.append(np.nan)
                
        except Exception as e:
            # Model training failed
            iroas_estimates.append(np.nan)
            mc_ci_widths.append(np.nan)
            convergence_info.append('failed')
    
    # Calculate statistics
    valid_iroas = [x for x in iroas_estimates if np.isfinite(x)]
    valid_mc_widths = [x for x in mc_ci_widths if np.isfinite(x)]
    
    empirical_std = np.std(valid_iroas) if len(valid_iroas) > 1 else np.nan
    avg_mc_width = np.mean(valid_mc_widths) if len(valid_mc_widths) > 0 else np.nan
    mc_implied_std = avg_mc_width / 3.92 if np.isfinite(avg_mc_width) else np.nan
    
    # Calculate underestimation ratio
    underestimation_ratio = empirical_std / mc_implied_std if (
        np.isfinite(empirical_std) and np.isfinite(mc_implied_std) and mc_implied_std > 0
    ) else np.inf
    
    success_rate = len(valid_iroas) / n_seeds
    convergence_rate = sum(1 for c in convergence_info if c == 'good') / n_seeds
    
    return {
        'config': config,
        'n_successful': len(valid_iroas),
        'success_rate': success_rate,
        'convergence_rate': convergence_rate,
        'empirical_std': empirical_std,
        'empirical_mean': np.mean(valid_iroas) if valid_iroas else np.nan,
        'mc_implied_std': mc_implied_std,
        'avg_mc_width': avg_mc_width,
        'underestimation_ratio': underestimation_ratio,
        'iroas_estimates': valid_iroas,
        'mc_ci_widths': valid_mc_widths
    }


def run_regularization_sweep() -> pd.DataFrame:
    """Run comprehensive regularization sweep."""
    
    print("Setting up test scenario...")
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_scenario(seed=42)
    
    # Define regularization configurations to test
    configs = []
    
    # Baseline (current problematic config)
    configs.append({
        'name': 'baseline',
        'hidden_dim': 32,
        'num_st_blocks': 2,
        'epochs': 10,
        'learning_rate': 0.01,
        'dropout': 0.1
    })
    
    # High dropout configurations
    for dropout in [0.3, 0.5, 0.7]:
        configs.append({
            'name': f'high_dropout_{dropout}',
            'hidden_dim': 32,
            'num_st_blocks': 2,
            'epochs': 10,
            'learning_rate': 0.01,
            'dropout': dropout
        })
    
    # Reduced complexity configurations
    for hidden_dim in [16, 8]:
        configs.append({
            'name': f'small_hidden_{hidden_dim}',
            'hidden_dim': hidden_dim,
            'num_st_blocks': 1,  # Also reduce blocks
            'epochs': 10,
            'learning_rate': 0.01,
            'dropout': 0.3  # Moderate dropout
        })
    
    # Conservative configurations (high dropout + small model)
    configs.append({
        'name': 'conservative_1',
        'hidden_dim': 16,
        'num_st_blocks': 1,
        'epochs': 15,  # More epochs to compensate for regularization
        'learning_rate': 0.005,  # Lower LR
        'dropout': 0.5
    })
    
    configs.append({
        'name': 'conservative_2',
        'hidden_dim': 8,
        'num_st_blocks': 1,
        'epochs': 20,
        'learning_rate': 0.005,
        'dropout': 0.7
    })
    
    # Extremely conservative (minimal complexity)
    configs.append({
        'name': 'minimal',
        'hidden_dim': 4,
        'num_st_blocks': 1,
        'epochs': 25,
        'learning_rate': 0.001,
        'dropout': 0.8
    })
    
    results = []
    
    print(f"Testing {len(configs)} regularization configurations...")
    
    for i, config in enumerate(configs):
        print(f"\nTesting config {i+1}/{len(configs)}: {config['name']}")
        print(f"  hidden_dim={config['hidden_dim']}, dropout={config['dropout']}, blocks={config['num_st_blocks']}")
        
        result = test_regularization_config(
            config, panel_data, assignment_df, pre_period_end, eval_start, eval_end, n_seeds=10
        )
        
        results.append(result)
        
        # Quick feedback
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Empirical std: {result['empirical_std']:.2f}")
        print(f"  Underestimation ratio: {result['underestimation_ratio']:.1f}x")
        
        # Early assessment
        if result['underestimation_ratio'] < 10:
            print(f"  🎉 PROMISING: Underestimation ratio < 10x!")
        elif result['underestimation_ratio'] < 100:
            print(f"  ✅ BETTER: Underestimation ratio < 100x")
        elif result['underestimation_ratio'] < 1000:
            print(f"  ⚠️ IMPROVED: Still high but better than baseline")
        else:
            print(f"  ❌ POOR: Still severe underestimation")
    
    return pd.DataFrame([{
        'config_name': r['config']['name'],
        'hidden_dim': r['config']['hidden_dim'],
        'num_st_blocks': r['config']['num_st_blocks'],
        'dropout': r['config']['dropout'],
        'learning_rate': r['config']['learning_rate'],
        'epochs': r['config']['epochs'],
        'success_rate': r['success_rate'],
        'convergence_rate': r['convergence_rate'],
        'empirical_std': r['empirical_std'],
        'mc_implied_std': r['mc_implied_std'],
        'underestimation_ratio': r['underestimation_ratio'],
        'avg_mc_width': r['avg_mc_width']
    } for r in results])


def analyze_regularization_results(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze regularization sweep results."""
    
    # Remove failed configurations
    valid_results = results_df[results_df['success_rate'] >= 0.5].copy()
    
    if len(valid_results) == 0:
        return {'status': 'all_failed', 'message': 'All configurations failed'}
    
    # Sort by underestimation ratio (lower is better)
    valid_results = valid_results.sort_values('underestimation_ratio')
    
    best_config = valid_results.iloc[0]
    baseline_config = results_df[results_df['config_name'] == 'baseline']
    
    analysis = {
        'status': 'success',
        'n_valid_configs': len(valid_results),
        'best_config': best_config.to_dict(),
        'improvement_summary': {}
    }
    
    if len(baseline_config) > 0:
        baseline_ratio = baseline_config['underestimation_ratio'].iloc[0]
        best_ratio = best_config['underestimation_ratio']
        
        improvement_factor = baseline_ratio / best_ratio if best_ratio > 0 else float('inf')
        
        analysis['improvement_summary'] = {
            'baseline_underestimation': baseline_ratio,
            'best_underestimation': best_ratio,
            'improvement_factor': improvement_factor
        }
    
    # Categorize results
    excellent = valid_results[valid_results['underestimation_ratio'] < 5]
    good = valid_results[valid_results['underestimation_ratio'] < 20]
    better = valid_results[valid_results['underestimation_ratio'] < 100]
    
    analysis['categories'] = {
        'excellent_configs': len(excellent),
        'good_configs': len(good), 
        'better_configs': len(better)
    }
    
    return analysis


def main():
    """Run regularization sweep analysis."""
    print("STGCN Regularization Sweep")
    print("=" * 50)
    print("Goal: Reduce training variance through regularization")
    print("Target: Underestimation ratio < 10x (ideally < 5x)")
    
    # Run sweep
    results_df = run_regularization_sweep()
    
    # Analyze results
    analysis = analyze_regularization_results(results_df)
    
    print(f"\n" + "=" * 60)
    print("REGULARIZATION SWEEP RESULTS")
    print("=" * 60)
    
    if analysis['status'] == 'all_failed':
        print("❌ All configurations failed")
        return
    
    print(f"\n📊 SUMMARY:")
    print(f"  Valid configurations: {analysis['n_valid_configs']}")
    print(f"  Excellent (<5x): {analysis['categories']['excellent_configs']}")
    print(f"  Good (<20x): {analysis['categories']['good_configs']}")
    print(f"  Better (<100x): {analysis['categories']['better_configs']}")
    
    # Show top configurations
    valid_results = results_df[results_df['success_rate'] >= 0.5].sort_values('underestimation_ratio')
    
    print(f"\n🏆 TOP CONFIGURATIONS:")
    display_cols = ['config_name', 'hidden_dim', 'dropout', 'underestimation_ratio', 'success_rate']
    top_5 = valid_results[display_cols].head(5)
    print(top_5.round(2).to_string(index=False))
    
    # Best configuration analysis
    best_config = analysis['best_config']
    print(f"\n🎯 BEST CONFIGURATION: {best_config['config_name']}")
    print(f"  Hidden dim: {best_config['hidden_dim']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  ST blocks: {best_config['num_st_blocks']}")
    print(f"  Underestimation ratio: {best_config['underestimation_ratio']:.1f}x")
    print(f"  Success rate: {best_config['success_rate']:.1%}")
    
    # Improvement assessment
    if 'improvement_summary' in analysis:
        imp = analysis['improvement_summary']
        print(f"\n📈 IMPROVEMENT vs BASELINE:")
        print(f"  Baseline underestimation: {imp['baseline_underestimation']:.1f}x")
        print(f"  Best underestimation: {imp['best_underestimation']:.1f}x")
        print(f"  Improvement factor: {imp['improvement_factor']:.1f}x better")
    
    # Decision guidance
    print(f"\n💡 NEXT STEPS:")
    best_ratio = best_config['underestimation_ratio']
    
    if best_ratio < 5:
        print(f"  ✅ EXCELLENT: Underestimation ratio < 5x")
        print(f"     → MC dropout may be good enough with this regularization")
        print(f"     → Test on full evaluation pipeline")
        print(f"     → Consider this the production configuration")
        
    elif best_ratio < 20:
        print(f"  ✅ GOOD: Underestimation ratio < 20x")
        print(f"     → Significant improvement but may still need ensembles")
        print(f"     → Try minimal ensemble (K=3) with this config")
        print(f"     → Worth testing on full pipeline")
        
    elif best_ratio < 100:
        print(f"  ⚠️ BETTER: Improved but still high underestimation")
        print(f"     → Regularization helps but insufficient alone")
        print(f"     → Proceed to ensemble methods (K=5)")
        print(f"     → Use this config as base for ensemble")
        
    else:
        print(f"  ❌ INSUFFICIENT: Regularization doesn't solve the problem")
        print(f"     → Training variance still too high")
        print(f"     → Must use ensemble methods")
        print(f"     → Consider simpler architectures")
    
    # Export results
    results_df.to_csv('stgcn_regularization_sweep_results.csv', index=False)
    print(f"\n📁 Results exported to: stgcn_regularization_sweep_results.csv")
    
    return results_df, analysis


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    results_df, analysis = main()