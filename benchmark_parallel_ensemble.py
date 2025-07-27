"""
Performance benchmark for parallel vs sequential ensemble training.

This script provides detailed performance analysis of parallel ensemble training
across different system configurations and parameter settings.

Usage:
    python benchmark_parallel_ensemble.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import warnings
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import required modules
from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel
from reporting.parallel_ensemble import ParallelEnsembleSTGCN, benchmark_parallel_vs_sequential


def create_benchmark_data(scenario: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Create test data for different benchmark scenarios.
    
    Args:
        scenario: "small", "standard", "large", or "xlarge"
        
    Returns:
        panel_data, assignment_df, pre_period_end
    """
    scenario_configs = {
        "small": {"n_geos": 12, "n_days": 50},
        "standard": {"n_geos": 20, "n_days": 80},
        "large": {"n_geos": 30, "n_days": 100},
        "xlarge": {"n_geos": 50, "n_days": 120}
    }
    
    config_params = scenario_configs.get(scenario, scenario_configs["standard"])
    
    config = DataConfig(
        n_geos=config_params["n_geos"],
        n_days=config_params["n_days"],
        seed=42
    )
    
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=42)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[int(config_params["n_days"] * 0.7) - 1].strftime('%Y-%m-%d')
    
    return panel_data, assignment_df, pre_period_end


def benchmark_ensemble_sizes():
    """Benchmark different ensemble sizes."""
    print("📊 BENCHMARK 1: Ensemble Size Scaling")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end = create_benchmark_data("standard")
    
    # Model configuration optimized for benchmarking
    model_config = {
        'hidden_dim': 32,
        'num_st_blocks': 2,
        'epochs': 5,  # Moderate epochs for realistic timing
        'learning_rate': 0.01,
        'dropout': 0.1,
        'verbose': False
    }
    
    ensemble_sizes = [3, 5, 8, 10]
    results = []
    
    for ensemble_size in ensemble_sizes:
        print(f"\nTesting ensemble size: {ensemble_size}")
        
        # Sequential timing
        print(f"  Sequential training...")
        seq_start = time.time()
        try:
            seq_ensemble = ParallelEnsembleSTGCN(
                ensemble_size=ensemble_size,
                **model_config
            )
            seq_ensemble.fit_sequential(panel_data, assignment_df, pre_period_end, seed=100)
            seq_time = time.time() - seq_start
            seq_success = True
        except Exception as e:
            print(f"    Sequential failed: {e}")
            seq_time = np.nan
            seq_success = False
        
        # Parallel timing
        print(f"  Parallel training...")
        par_start = time.time()
        try:
            par_ensemble = ParallelEnsembleSTGCN(
                ensemble_size=ensemble_size,
                n_jobs=-1,  # Auto-detect
                **model_config
            )
            par_ensemble.fit_parallel(panel_data, assignment_df, pre_period_end, seed=100)
            par_time = time.time() - par_start
            par_success = True
            n_jobs_used = par_ensemble.n_jobs
        except Exception as e:
            print(f"    Parallel failed: {e}")
            par_time = np.nan
            par_success = False
            n_jobs_used = 0
        
        # Calculate metrics
        speedup = seq_time / par_time if (seq_success and par_success and par_time > 0) else np.nan
        efficiency = speedup / n_jobs_used if (not np.isnan(speedup) and n_jobs_used > 0) else np.nan
        
        results.append({
            'ensemble_size': ensemble_size,
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'n_jobs_used': n_jobs_used,
            'seq_success': seq_success,
            'par_success': par_success
        })
        
        if seq_success and par_success:
            print(f"    Results: {seq_time:.1f}s → {par_time:.1f}s ({speedup:.1f}x speedup, {efficiency:.1%} efficiency)")
        else:
            print(f"    Results: seq_success={seq_success}, par_success={par_success}")
    
    # Create visualization
    results_df = pd.DataFrame(results)
    successful_results = results_df[(results_df['seq_success']) & (results_df['par_success'])]
    
    if len(successful_results) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Training time comparison
        axes[0].plot(successful_results['ensemble_size'], successful_results['sequential_time'], 
                    'o-', label='Sequential', linewidth=2, markersize=8)
        axes[0].plot(successful_results['ensemble_size'], successful_results['parallel_time'], 
                    'o-', label='Parallel', linewidth=2, markersize=8)
        axes[0].set_xlabel('Ensemble Size')
        axes[0].set_ylabel('Training Time (seconds)')
        axes[0].set_title('Training Time vs Ensemble Size')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Speedup
        axes[1].plot(successful_results['ensemble_size'], successful_results['speedup'], 
                    'o-', color='green', linewidth=2, markersize=8)
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
        axes[1].set_xlabel('Ensemble Size')
        axes[1].set_ylabel('Speedup (x)')
        axes[1].set_title('Parallel Speedup vs Ensemble Size')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Efficiency
        axes[2].plot(successful_results['ensemble_size'], successful_results['efficiency'], 
                    'o-', color='orange', linewidth=2, markersize=8)
        axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect efficiency')
        axes[2].set_xlabel('Ensemble Size')
        axes[2].set_ylabel('Efficiency (speedup/cores)')
        axes[2].set_title('Parallel Efficiency vs Ensemble Size')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ensemble_size_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n📈 ENSEMBLE SIZE BENCHMARK SUMMARY:")
        print(f"  Average speedup: {successful_results['speedup'].mean():.1f}x")
        print(f"  Best speedup: {successful_results['speedup'].max():.1f}x (K={successful_results.loc[successful_results['speedup'].idxmax(), 'ensemble_size']})")
        print(f"  Average efficiency: {successful_results['efficiency'].mean():.1%}")
        
    return results_df


def benchmark_system_resources():
    """Benchmark different numbers of parallel jobs."""
    print("\n📊 BENCHMARK 2: System Resource Utilization")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end = create_benchmark_data("standard")
    
    model_config = {
        'hidden_dim': 32,
        'num_st_blocks': 2,
        'epochs': 4,
        'verbose': False
    }
    
    ensemble_size = 8
    max_cores = mp.cpu_count()
    job_settings = [1, 2, 4, max_cores//2, max_cores, -1]  # -1 = auto
    
    print(f"System: {max_cores} CPU cores available")
    print(f"Testing ensemble size: {ensemble_size}")
    
    results = []
    
    for n_jobs in job_settings:
        print(f"\nTesting n_jobs={n_jobs}...")
        
        try:
            start_time = time.time()
            
            ensemble = ParallelEnsembleSTGCN(
                ensemble_size=ensemble_size,
                n_jobs=n_jobs,
                **model_config
            )
            
            ensemble.fit_parallel(panel_data, assignment_df, pre_period_end, seed=200)
            
            elapsed = time.time() - start_time
            diagnostics = ensemble.get_training_diagnostics()
            
            actual_jobs = diagnostics['n_jobs_used']
            success_rate = diagnostics['success_rate']
            estimated_speedup = diagnostics.get('speedup_estimate', np.nan)
            
            results.append({
                'n_jobs_requested': n_jobs,
                'n_jobs_actual': actual_jobs,
                'training_time': elapsed,
                'success_rate': success_rate,
                'estimated_speedup': estimated_speedup
            })
            
            print(f"  ✅ n_jobs={n_jobs} → actual={actual_jobs}, time={elapsed:.1f}s, "
                  f"success_rate={success_rate:.1%}, speedup={estimated_speedup:.1f}x")
            
        except Exception as e:
            print(f"  ❌ n_jobs={n_jobs} failed: {e}")
            results.append({
                'n_jobs_requested': n_jobs,
                'n_jobs_actual': 0,
                'training_time': np.nan,
                'success_rate': 0,
                'estimated_speedup': np.nan
            })
    
    # Analyze results
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success_rate'] > 0]
    
    if len(successful_results) > 0:
        # Find optimal number of jobs
        valid_times = successful_results[~successful_results['training_time'].isna()]
        if len(valid_times) > 0:
            optimal_idx = valid_times['training_time'].idxmin()
            optimal_jobs = valid_times.loc[optimal_idx, 'n_jobs_actual']
            optimal_time = valid_times.loc[optimal_idx, 'training_time']
            
            print(f"\n🎯 RESOURCE UTILIZATION ANALYSIS:")
            print(f"  Optimal n_jobs: {optimal_jobs} (time: {optimal_time:.1f}s)")
            print(f"  Available cores: {max_cores}")
            print(f"  Utilization efficiency: {optimal_jobs/max_cores:.1%}")
            
            # Create visualization
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(successful_results['n_jobs_actual'], successful_results['training_time'], 'o-')
            plt.axvline(x=optimal_jobs, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {optimal_jobs} jobs')
            plt.xlabel('Number of Parallel Jobs')
            plt.ylabel('Training Time (seconds)')
            plt.title('Training Time vs Parallel Jobs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(successful_results['n_jobs_actual'], successful_results['estimated_speedup'], 'o-', color='green')
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
            plt.xlabel('Number of Parallel Jobs')
            plt.ylabel('Estimated Speedup (x)')
            plt.title('Speedup vs Parallel Jobs')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('resource_utilization_benchmark.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    return results_df


def benchmark_model_complexity():
    """Benchmark different model complexities."""
    print("\n📊 BENCHMARK 3: Model Complexity Impact")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end = create_benchmark_data("standard")
    
    model_configs = [
        {'name': 'Small', 'hidden_dim': 16, 'num_st_blocks': 1, 'epochs': 4},
        {'name': 'Medium', 'hidden_dim': 32, 'num_st_blocks': 2, 'epochs': 4},
        {'name': 'Large', 'hidden_dim': 64, 'num_st_blocks': 3, 'epochs': 4}
    ]
    
    ensemble_size = 5
    results = []
    
    for config in model_configs:
        print(f"\nTesting {config['name']} model:")
        print(f"  Config: hidden_dim={config['hidden_dim']}, blocks={config['num_st_blocks']}")
        
        model_params = {k: v for k, v in config.items() if k != 'name'}
        model_params.update({'verbose': False, 'learning_rate': 0.01, 'dropout': 0.1})
        
        # Sequential timing
        seq_start = time.time()
        try:
            seq_ensemble = ParallelEnsembleSTGCN(ensemble_size=ensemble_size, **model_params)
            seq_ensemble.fit_sequential(panel_data, assignment_df, pre_period_end, seed=300)
            seq_time = time.time() - seq_start
            seq_success = True
        except Exception as e:
            print(f"    Sequential failed: {e}")
            seq_time = np.nan
            seq_success = False
        
        # Parallel timing
        par_start = time.time()
        try:
            par_ensemble = ParallelEnsembleSTGCN(ensemble_size=ensemble_size, n_jobs=-1, **model_params)
            par_ensemble.fit_parallel(panel_data, assignment_df, pre_period_end, seed=300)
            par_time = time.time() - par_start
            par_success = True
        except Exception as e:
            print(f"    Parallel failed: {e}")
            par_time = np.nan
            par_success = False
        
        # Calculate metrics
        speedup = seq_time / par_time if (seq_success and par_success and par_time > 0) else np.nan
        
        results.append({
            'model_name': config['name'],
            'hidden_dim': config['hidden_dim'],
            'num_st_blocks': config['num_st_blocks'],
            'sequential_time': seq_time,
            'parallel_time': par_time,
            'speedup': speedup,
            'seq_success': seq_success,
            'par_success': par_success
        })
        
        if seq_success and par_success:
            print(f"    Results: {seq_time:.1f}s → {par_time:.1f}s ({speedup:.1f}x speedup)")
        else:
            print(f"    Results: seq_success={seq_success}, par_success={par_success}")
    
    # Create visualization
    results_df = pd.DataFrame(results)
    successful_results = results_df[(results_df['seq_success']) & (results_df['par_success'])]
    
    if len(successful_results) > 0:
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(successful_results))
        width = 0.35
        
        plt.bar(x - width/2, successful_results['sequential_time'], width, 
               label='Sequential', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, successful_results['parallel_time'], width, 
               label='Parallel', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Model Complexity')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time by Model Complexity')
        plt.xticks(x, successful_results['model_name'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add speedup annotations
        for i, (_, row) in enumerate(successful_results.iterrows()):
            plt.annotate(f'{row["speedup"]:.1f}x', 
                        xy=(i, max(row['sequential_time'], row['parallel_time']) + 1),
                        ha='center', fontweight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig('model_complexity_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n🎯 MODEL COMPLEXITY ANALYSIS:")
        for _, row in successful_results.iterrows():
            print(f"  {row['model_name']}: {row['speedup']:.1f}x speedup")
    
    return results_df


def benchmark_integrated_stgcn():
    """Benchmark integrated STGCN confidence interval methods."""
    print("\n📊 BENCHMARK 4: Integrated STGCN CI Methods")
    print("=" * 60)
    
    panel_data, assignment_df, pre_period_end = create_benchmark_data("standard")
    
    # Prepare evaluation period
    dates = sorted(panel_data['date'].unique())
    eval_start = dates[int(len(dates) * 0.7)].strftime('%Y-%m-%d')
    eval_end = dates[-1].strftime('%Y-%m-%d')
    
    # Train main STGCN model
    print("Training main STGCN model...")
    main_model = STGCNReportingModel(
        hidden_dim=32,
        num_st_blocks=2,
        epochs=5,
        verbose=True
    )
    main_model.fit(panel_data, assignment_df, pre_period_end)
    
    # Test different CI methods
    ci_methods = {
        'MC Dropout': {
            'method': 'mc_dropout',
            'params': {'n_mc_samples': 50}
        },
        'Sequential Ensemble': {
            'method': 'ensemble',
            'params': {'ensemble_size': 5, 'use_parallel': False}
        },
        'Parallel Ensemble': {
            'method': 'ensemble',
            'params': {'ensemble_size': 5, 'use_parallel': True, 'n_jobs': -1}
        }
    }
    
    results = []
    
    for method_name, config in ci_methods.items():
        print(f"\nTesting {method_name}...")
        
        try:
            start_time = time.time()
            
            lower, upper = main_model.confidence_interval(
                panel_data, eval_start, eval_end,
                method=config['method'],
                **config['params']
            )
            
            elapsed = time.time() - start_time
            ci_width = upper - lower
            includes_zero = (lower <= 0 <= upper)
            
            results.append({
                'method': method_name,
                'time': elapsed,
                'ci_lower': lower,
                'ci_upper': upper,
                'ci_width': ci_width,
                'includes_zero': includes_zero,
                'success': True
            })
            
            print(f"  ✅ {method_name}: {elapsed:.1f}s, CI=[{lower:.4f}, {upper:.4f}], width={ci_width:.4f}")
            
        except Exception as e:
            results.append({
                'method': method_name,
                'time': np.nan,
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ {method_name} failed: {e}")
    
    # Analyze results
    results_df = pd.DataFrame(results)
    successful_results = results_df[results_df['success']]
    
    if len(successful_results) >= 2:
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training time comparison
        axes[0].bar(successful_results['method'], successful_results['time'], 
                   color=['red', 'blue', 'green'], alpha=0.7)
        axes[0].set_ylabel('Time (seconds)')
        axes[0].set_title('CI Calculation Time by Method')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: CI width comparison
        axes[1].bar(successful_results['method'], successful_results['ci_width'], 
                   color=['red', 'blue', 'green'], alpha=0.7)
        axes[1].set_ylabel('CI Width')
        axes[1].set_title('Confidence Interval Width by Method')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('integrated_stgcn_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Calculate speedups
        if 'Sequential Ensemble' in successful_results['method'].values and 'Parallel Ensemble' in successful_results['method'].values:
            seq_time = successful_results[successful_results['method'] == 'Sequential Ensemble']['time'].iloc[0]
            par_time = successful_results[successful_results['method'] == 'Parallel Ensemble']['time'].iloc[0]
            speedup = seq_time / par_time if par_time > 0 else float('inf')
            
            print(f"\n🎯 INTEGRATED CI METHOD ANALYSIS:")
            print(f"  Parallel ensemble speedup: {speedup:.1f}x vs sequential ensemble")
            
            if speedup > 2:
                print(f"  ✅ Excellent speedup achieved!")
            elif speedup > 1.5:
                print(f"  ✅ Good speedup achieved!")
            elif speedup > 1.1:
                print(f"  ⚠️ Modest speedup")
            else:
                print(f"  ❌ No significant speedup")
    
    return results_df


def main():
    """Run comprehensive parallel ensemble benchmarks."""
    print("🚀 COMPREHENSIVE PARALLEL ENSEMBLE BENCHMARK SUITE")
    print("=" * 80)
    print(f"System: {mp.cpu_count()} CPU cores, {os.cpu_count()} logical processors")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # Create results directory
        os.makedirs('benchmark_results', exist_ok=True)
        
        # Run all benchmarks
        print(f"\n🔬 Running comprehensive benchmark suite...")
        
        ensemble_results = benchmark_ensemble_sizes()
        resource_results = benchmark_system_resources()
        complexity_results = benchmark_model_complexity()
        integrated_results = benchmark_integrated_stgcn()
        
        # Save all results
        ensemble_results.to_csv('benchmark_results/ensemble_size_results.csv', index=False)
        resource_results.to_csv('benchmark_results/resource_utilization_results.csv', index=False)
        complexity_results.to_csv('benchmark_results/model_complexity_results.csv', index=False)
        integrated_results.to_csv('benchmark_results/integrated_stgcn_results.csv', index=False)
        
        # Generate final summary
        print(f"\n🎉 BENCHMARK SUITE COMPLETED")
        print("=" * 60)
        
        print(f"✅ All benchmark results saved to 'benchmark_results/' directory")
        print(f"✅ Visualizations saved as PNG files")
        
        # Calculate overall recommendations
        successful_ensemble = ensemble_results[(ensemble_results['seq_success']) & (ensemble_results['par_success'])]
        if len(successful_ensemble) > 0:
            avg_speedup = successful_ensemble['speedup'].mean()
            best_ensemble_size = successful_ensemble.loc[successful_ensemble['speedup'].idxmax(), 'ensemble_size']
            
            print(f"\n💡 PERFORMANCE RECOMMENDATIONS:")
            print(f"• Average parallel speedup: {avg_speedup:.1f}x")
            print(f"• Best ensemble size: K={best_ensemble_size}")
            print(f"• Use parallel training for K≥5 ensembles")
            print(f"• Set n_jobs=-1 for automatic resource optimization")
            print(f"• Expect {avg_speedup:.1f}x speedup on similar hardware")
        
    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()