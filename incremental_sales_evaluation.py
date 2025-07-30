#!/usr/bin/env python3
"""
Incremental sales evaluation utilities for A/A testing.

This module provides comprehensive evaluation functions for testing geo-experiment
models in A/A (null effect) scenarios. It focuses on incremental sales evaluation
rather than iROAS to avoid noise from spend estimates.

Key Features:
- A/A testing framework for bias detection
- Comprehensive metrics: bias, standard error, MSE, FPR, coverage
- Support for all model types with consistent aggregation
- Statistical validation of confidence interval performance

Usage:
    # Quick A/A test
    results_df, summary = run_aa_incremental_sales_test()
    
    # Custom evaluation
    results = evaluate_incremental_sales_aa(methods, data_config, n_simulations=10)
    summary = summarize_incremental_sales_results(results)

Critical for Development:
    Run A/A tests before each commit to ensure model reliability and catch
    systematic bias issues early in development.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from dataclasses import dataclass

from data_simulation.generators import IdenticalGeoGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.models import MeanMatchingModel, TBRModel, SyntheticControlModel, GBRModel
from reporting.stgcn_shallow import STGCNShallowModel, STGCNIntermediateModel
from reporting.stgcn_model import STGCNReportingModel


@dataclass
class IncrementalSalesResult:
    """
    Results from a single incremental sales evaluation.
    
    This dataclass stores the outcomes from evaluating one method on one simulation.
    All metrics are calculated relative to the true incremental sales value (0 in A/A testing).
    
    Attributes:
        method_name: Name of the evaluation method (e.g., 'STGCN_Tiny', 'MeanMatching')
        incremental_sales: Estimated incremental sales (actual - predicted)
        actual_sales: Observed sales in treatment group (aggregated appropriately)
        predicted_sales: Model's counterfactual prediction (aggregated appropriately) 
        bias: Systematic error = incremental_sales - true_value (0 for A/A)
        abs_bias: Absolute bias for easier aggregation
        squared_error: Squared bias for MSE calculation
        ci_lower: Lower bound of confidence interval (optional, for future use)
        ci_upper: Upper bound of confidence interval (optional, for future use)
    """
    method_name: str
    incremental_sales: float
    actual_sales: float
    predicted_sales: float
    bias: float  # incremental_sales - true_value (0 for A/A)
    abs_bias: float
    squared_error: float  # (incremental_sales - true_value)^2
    ci_lower: float = None  # Confidence interval lower bound
    ci_upper: float = None  # Confidence interval upper bound


def calculate_incremental_sales(
    model: Any,
    panel_data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    eval_period_start: str,
    eval_period_end: str
) -> Tuple[float, float, float]:
    """
    Calculate incremental sales for a fitted model using consistent aggregation.
    
    This function computes the core metric for A/A testing: incremental sales,
    which is the numerator of iROAS. It uses daily mean aggregation to ensure
    consistency across all model types.
    
    Aggregation Logic:
        - All models return daily predictions (shape: n_days)
        - Actual sales: mean across treatment geos per day, then sum across days
        - Predicted sales: sum of daily predictions
        - Incremental sales: actual_sales - predicted_sales
    
    Args:
        model: Fitted reporting model (any type: traditional or STGCN)
        panel_data: Long-format panel data with columns ['geo', 'date', 'sales', 'spend']
        assignment_df: Assignment data with columns ['geo', 'assignment']
        eval_period_start: Start date of evaluation period (string format)
        eval_period_end: End date of evaluation period (string format)
        
    Returns:
        Tuple of (incremental_sales, actual_sales, predicted_sales)
        - incremental_sales: The bias estimate (should be ~0 in A/A testing)
        - actual_sales: Total observed sales in treatment group (for validation)
        - predicted_sales: Total predicted sales (for validation)
    """
    # Step 1: Get counterfactual prediction from the fitted model
    # All models now return daily predictions (shape: n_evaluation_days)
    counterfactual = model.predict(panel_data, eval_period_start, eval_period_end)
    
    # Step 2: Prepare evaluation period data
    # Convert dates to datetime for proper filtering
    panel_data_copy = panel_data.copy()
    panel_data_copy['date'] = pd.to_datetime(panel_data_copy['date'])
    eval_start_dt = pd.to_datetime(eval_period_start)
    eval_end_dt = pd.to_datetime(eval_period_end)
    
    # Filter to evaluation period only
    eval_data = panel_data_copy[
        (panel_data_copy['date'] >= eval_start_dt) & 
        (panel_data_copy['date'] <= eval_end_dt)
    ]
    
    # Step 3: Extract treatment group data
    treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].values
    treatment_data = eval_data[eval_data['geo'].isin(treatment_geos)]
    
    # Step 4: Calculate actual sales using consistent aggregation
    # Daily mean aggregation: mean across treatment geos per day, then sum across days
    # This matches the approach used by traditional models in their calculate_iroas methods
    actual_sales = treatment_data.groupby('date')['sales'].mean().sum()
    
    # Step 5: Calculate predicted sales from counterfactual
    # Handle different counterfactual return formats (all should be arrays now)
    if isinstance(counterfactual.get('sales'), np.ndarray):
        # Sum daily predictions to get total predicted sales over evaluation period
        predicted_sales = counterfactual['sales'].sum()
    else:
        # Fallback for non-array formats (shouldn't happen with current implementation)
        predicted_sales = counterfactual.get('sales', 0)
    
    # Step 6: Calculate incremental sales (the key A/A testing metric)
    # In A/A testing, this should be approximately 0 (no treatment effect)
    incremental_sales = actual_sales - predicted_sales
    
    return incremental_sales, actual_sales, predicted_sales


def evaluate_incremental_sales_aa(
    methods: Dict[str, Any],
    data_config: DataConfig = None,
    n_simulations: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate multiple methods on incremental sales in A/A setting.
    
    Args:
        methods: Dictionary of method_name -> model instances
        data_config: Data configuration
        n_simulations: Number of simulations
        seed: Random seed
        
    Returns:
        DataFrame with evaluation results
    """
    if data_config is None:
        data_config = DataConfig(
            n_geos=20,
            n_days=60,
            base_sales_mean=1000,
            base_sales_std=0,  # Identical baselines
            daily_sales_noise=100,
            seed=seed
        )
    
    generator = IdenticalGeoGenerator(data_config)
    results = []
    
    for sim_id in range(n_simulations):
        # Generate data
        sim_seed = seed + sim_id if seed is not None else None
        data_config.seed = sim_seed
        panel_data, geo_features = generator.generate()
        
        # Create assignment
        assignment = RandomAssignment().assign(geo_features, 0.5, seed=sim_seed)
        
        # Define periods
        dates = pd.to_datetime(panel_data['date'].unique())
        pre_period_end = dates[data_config.n_days // 2 - 1]  # Half for pre-period
        eval_period_start = dates[data_config.n_days // 2]
        eval_period_end = dates[-1]
        
        pre_period_end_str = pre_period_end.strftime('%Y-%m-%d')
        eval_period_start_str = eval_period_start.strftime('%Y-%m-%d')
        eval_period_end_str = eval_period_end.strftime('%Y-%m-%d')
        
        # Test each method
        for method_name, model in methods.items():
            try:
                # Fit model
                model.fit(panel_data, assignment, pre_period_end_str)
                
                # Calculate incremental sales
                incremental_sales, actual_sales, predicted_sales = calculate_incremental_sales(
                    model, panel_data, assignment, eval_period_start_str, eval_period_end_str
                )
                
                # For A/A testing, true incremental sales = 0
                bias = incremental_sales - 0.0
                abs_bias = abs(bias)
                squared_error = bias ** 2
                
                # Calculate confidence interval to assess model uncertainty
                try:
                    # NEW: Try incremental sales CI first
                    if hasattr(model, 'incremental_sales_confidence_interval'):
                        ci_lower, ci_upper = model.incremental_sales_confidence_interval(
                            panel_data,
                            eval_period_start_str,
                            eval_period_end_str,
                            confidence_level=0.95,
                            n_bootstrap=None  # Let method choose smart default
                        )
                    else:
                        # FALLBACK: Use old iROAS CI method (will show unit mismatch warning)
                        print(f"⚠️ {method_name}: Using iROAS CI (incremental_sales_confidence_interval not implemented)")
                        ci_lower, ci_upper = model.confidence_interval(
                            panel_data,
                            eval_period_start_str,
                            eval_period_end_str,
                            confidence_level=0.95
                        )
                        
                except AttributeError as attr_e:
                    print(f"⚠️ {method_name}: No CI method available - {attr_e}")
                    ci_lower, ci_upper = np.nan, np.nan
                except Exception as ci_e:
                    print(f"Could not compute CI for {method_name}: {ci_e}")
                    ci_lower, ci_upper = np.nan, np.nan

                results.append(IncrementalSalesResult(
                    method_name=method_name,
                    incremental_sales=incremental_sales,
                    actual_sales=actual_sales,
                    predicted_sales=predicted_sales,
                    bias=bias,
                    abs_bias=abs_bias,
                    squared_error=squared_error,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper
                ))
                
            except Exception as e:
                print(f"Error with {method_name} in simulation {sim_id}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'simulation_id': i // len(methods),
            'method': r.method_name,
            'incremental_sales': r.incremental_sales,
            'actual_sales': r.actual_sales,
            'predicted_sales': r.predicted_sales,
            'bias': r.bias,
            'abs_bias': r.abs_bias,
            'squared_error': r.squared_error,
            'ci_lower': r.ci_lower,
            'ci_upper': r.ci_upper
        }
        for i, r in enumerate(results)
    ])
    
    return results_df


def summarize_incremental_sales_results(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Summarize incremental sales evaluation results with comprehensive statistical metrics.
    
    This function computes key performance metrics for A/A testing, focusing on:
    1. Bias detection (mean bias, standard error, MSE)
    2. Confidence interval performance (coverage, false positive rate)
    
    The metrics follow standard statistical definitions:
    - Coverage: Proportion of CIs that correctly contain the true value (0)
    - FPR: Proportion of CIs that incorrectly miss the true value = 1 - Coverage
    - Standard Error: An estimate of the model's prediction uncertainty, derived from the CI width.
    
    Args:
        results_df: Results DataFrame from evaluate_incremental_sales_aa()
                   Must contain columns: ['method', 'bias', 'squared_error', 'ci_lower', 'ci_upper']
        alpha: Significance level for confidence intervals (default: 0.05 for 95% CIs)
        
    Returns:
        Summary DataFrame with columns:
        - method: Method name
        - mean_bias: Average bias across simulations (should be ~0 for A/A)
        - mean(std err): Mean of the model's standard error of incremental sales.
        - mean_cihw: Mean half-width of the confidence intervals.
        - mean(MSE): Mean squared error (overall accuracy)
        - FPR: False positive rate (proportion of CIs missing true value)
        - Coverage: Coverage rate (proportion of CIs containing true value)
        
    Note:
        FPR + Coverage = 1.0 exactly (complementary metrics)
    """
    import scipy.stats as stats
    
    summary_stats = []
    
    for method in results_df['method'].unique():
        method_data = results_df[results_df['method'] == method].copy()
        n_sims = len(method_data)
        
        # Basic statistics
        mean_bias = method_data['bias'].mean()
        mean_mse = method_data['squared_error'].mean()
        
        # Model uncertainty metrics from confidence intervals
        method_data['ci_half_width'] = (method_data['ci_upper'] - method_data['ci_lower']) / 2.0
        mean_cihw = method_data['ci_half_width'].mean()
        
        # Approximate standard error from CI width, assuming normality
        z_critical = stats.norm.ppf(1 - alpha / 2)
        method_data['std_error_est'] = method_data['ci_half_width'] / z_critical
        mean_std_error = method_data['std_error_est'].mean()

        # Coverage and FPR: Based on confidence intervals containing true value (0)
        truth = 0.0
        contained = (method_data['ci_lower'] <= truth) & (truth <= method_data['ci_upper'])
        coverage = contained.mean()
        fpr = 1.0 - coverage

        summary_stats.append({
            'method': method,
            'mean_bias': round(mean_bias, 1),
            'mean(std err)': round(mean_std_error, 1),
            'mean_cihw': round(mean_cihw, 1),
            'mean(MSE)': round(mean_mse, 1),
            'FPR': round(fpr, 3),
            'Coverage': round(coverage, 3)
        })
    
    return pd.DataFrame(summary_stats)


def run_aa_incremental_sales_test():
    """
    Quick A/A test focusing on incremental sales.
    """
    print("=" * 60)
    print("A/A INCREMENTAL SALES TEST")
    print("=" * 60)
    print("Testing models on incremental sales (numerator of iROAS)")
    print("Expected result: All methods should have bias ≈ 0")
    
    # Test configuration
    data_config = DataConfig(
        n_geos=16,
        n_days=40,  # Small for speed
        base_sales_mean=1000,
        base_sales_std=0,  # Identical baselines
        daily_sales_noise=100,
        seed=42
    )
    
    # Methods to test
    methods = {
        'MeanMatching': MeanMatchingModel(),
        'GBR': GBRModel(),
        'TBR': TBRModel(),
        'SCM': SyntheticControlModel(),
        'STGCN_Tiny': STGCNShallowModel(
            epochs=5,  # Very fast
            verbose=False
        ),
        'STGCN_Intermediate': STGCNIntermediateModel(
            epochs=8,  # Moderate training
            verbose=False
        ),
        'STGCN_Full': STGCNReportingModel(
            hidden_dim=32,
            num_st_blocks=2,
            epochs=10,  # More training for full model
            verbose=False
        )
    }
    
    print(f"Configuration: {data_config.n_geos} geos, {data_config.n_days} days")
    print(f"Methods: {list(methods.keys())}")
    
    # Run evaluation
    results_df = evaluate_incremental_sales_aa(
        methods, data_config, n_simulations=3, seed=42
    )
    
    # Summarize results
    summary = summarize_incremental_sales_results(results_df)
    
    print(f"\n{'Method':<15} {'Mean Bias':<12} {'mean(std err)':<15} {'mean(MSE)':<12} {'FPR':<8} {'Coverage':<10}")
    print("-" * 80)
    
    for _, row in summary.iterrows():
        method = row['method']
        mean_bias = row['mean_bias']
        mean_std_err = row['mean(std err)']
        mean_mse = row['mean(MSE)']
        fpr = row['FPR']
        coverage = row['Coverage']
        
        coverage_str = f"{coverage:.3f}" if not np.isnan(coverage) else "N/A"
        print(f"{method:<15} {mean_bias:>8.1f} {mean_std_err:>12.1f} {mean_mse:>12.0f} {fpr:>6.3f} {coverage_str:>8}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    for _, row in summary.iterrows():
        method = row['method']
        mean_bias = row['mean_bias']
        fpr = row['FPR']
        coverage = row['Coverage']
        
        if abs(mean_bias) <= 100:
            bias_status = "✅ GOOD"
        elif abs(mean_bias) <= 1000:
            bias_status = "⚠️ MODERATE BIAS"
        else:
            bias_status = "❌ LARGE BIAS"
            
        fpr_status = "✅ GOOD" if fpr <= 0.1 else "❌ HIGH FPR"
        coverage_status = "✅ GOOD" if not np.isnan(coverage) and coverage >= 0.9 else "❌ POOR COVERAGE"
        
        print(f"{method:<15}: {bias_status}, {fpr_status}, {coverage_status}")
        coverage_formatted = f"{coverage:.3f}" if not np.isnan(coverage) else "N/A"
        print(f"                Bias: {mean_bias:>8.1f}, FPR: {fpr:.3f}, Coverage: {coverage_formatted}")
    
    return results_df, summary



if __name__ == "__main__":
    results_df, summary = run_aa_incremental_sales_test()