"""
Diagnose STGCN overconfidence issue by analyzing prediction variance vs CI width.

Based on prediction quality analysis, STGCN has reasonable bias but high variance.
This script investigates whether the bootstrap/MC dropout methods are capturing
this variance correctly, or if they're producing overconfident CIs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import warnings

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel
from reporting.models import MeanMatchingModel


def create_test_scenario(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """Create test scenario matching user's setup."""
    config = DataConfig(
        n_geos=20,
        n_days=120,
        seed=seed
    )
    
    generator = SimpleNullGenerator(config)
    panel_data, geo_features = generator.generate()
    
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=seed)
    
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[99]  # 100 days training
    eval_start = dates[100]
    eval_end = dates[119]  # 20 days eval
    
    return panel_data, assignment_df, pre_period_end, eval_start, eval_end


def analyze_prediction_vs_ci_relationship(
    panel_data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    pre_period_end: str,
    eval_start: str,
    eval_end: str,
    n_trials: int = 20
) -> Dict[str, Any]:
    """
    Analyze relationship between prediction variance and CI width.
    
    This tests if STGCN's CI methods are capturing the true prediction uncertainty.
    """
    
    # Get multiple iROAS estimates and CIs from different model fits
    iroas_estimates = []
    ci_widths_mc = []
    ci_widths_bootstrap = []
    prediction_variants = []
    
    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}")
        
        # Fit new model with different random seed
        torch.manual_seed(100 + trial)
        np.random.seed(100 + trial)
        
        model = STGCNReportingModel(
            hidden_dim=32,
            epochs=10,
            window_size=5,
            learning_rate=0.01,
            normalize_data=True,
            verbose=False,
            dropout=0.2
        )
        
        try:
            # Fit model
            model.fit(panel_data, assignment_df, pre_period_end)
            
            # Get iROAS estimate
            iroas = model.calculate_iroas(panel_data, eval_start, eval_end)
            iroas_estimates.append(iroas)
            
            # Get CI using MC dropout
            try:
                mc_lower, mc_upper = model.confidence_interval(
                    panel_data, eval_start, eval_end,
                    method='mc_dropout',
                    n_mc_samples=50,
                    confidence_level=0.95
                )
                ci_widths_mc.append(mc_upper - mc_lower)
            except:
                ci_widths_mc.append(np.nan)
            
            # Get CI using smaller bootstrap for speed
            try:
                bs_lower, bs_upper = model.confidence_interval(
                    panel_data, eval_start, eval_end,
                    method='model_aware_bootstrap',
                    n_bootstrap=5,  # Small for speed
                    confidence_level=0.95
                )
                ci_widths_bootstrap.append(bs_upper - bs_lower)
            except:
                ci_widths_bootstrap.append(np.nan)
            
            # Get prediction variant (not used for CI, just raw prediction variation)
            try:
                counterfactual = model.predict(panel_data, eval_start, eval_end)
                prediction_variants.append({
                    'sales_total': counterfactual['sales'].sum(),
                    'spend_total': counterfactual['spend'].sum()
                })
            except:
                prediction_variants.append({'sales_total': np.nan, 'spend_total': np.nan})
                
        except Exception as e:
            print(f"  Trial {trial} failed: {e}")
            iroas_estimates.append(np.nan)
            ci_widths_mc.append(np.nan)
            ci_widths_bootstrap.append(np.nan)
            prediction_variants.append({'sales_total': np.nan, 'spend_total': np.nan})
    
    # Remove NaN values for analysis
    valid_iroas = [x for x in iroas_estimates if np.isfinite(x)]
    valid_mc_widths = [x for x in ci_widths_mc if np.isfinite(x)]
    valid_bs_widths = [x for x in ci_widths_bootstrap if np.isfinite(x)]
    
    # Compute empirical variance from multiple fits
    empirical_iroas_std = np.std(valid_iroas) if len(valid_iroas) > 1 else np.nan
    empirical_iroas_var = np.var(valid_iroas) if len(valid_iroas) > 1 else np.nan
    
    # Average CI widths from methods
    avg_mc_width = np.mean(valid_mc_widths) if len(valid_mc_widths) > 0 else np.nan
    avg_bs_width = np.mean(valid_bs_widths) if len(valid_bs_widths) > 0 else np.nan
    
    # Compare empirical variance to CI-implied variance
    # For 95% CI, width ≈ 3.92 * std (for normal distribution)
    ci_implied_std_mc = avg_mc_width / 3.92 if np.isfinite(avg_mc_width) else np.nan
    ci_implied_std_bs = avg_bs_width / 3.92 if np.isfinite(avg_bs_width) else np.nan
    
    results = {
        'n_successful_trials': len(valid_iroas),
        'empirical_iroas_mean': np.mean(valid_iroas) if len(valid_iroas) > 0 else np.nan,
        'empirical_iroas_std': empirical_iroas_std,
        'empirical_iroas_var': empirical_iroas_var,
        'avg_mc_ci_width': avg_mc_width,
        'avg_bootstrap_ci_width': avg_bs_width,
        'ci_implied_std_mc': ci_implied_std_mc,
        'ci_implied_std_bs': ci_implied_std_bs,
        'iroas_estimates': valid_iroas,
        'mc_ci_widths': valid_mc_widths,
        'bootstrap_ci_widths': valid_bs_widths
    }
    
    return results


def compare_with_baseline_method(
    panel_data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    pre_period_end: str,
    eval_start: str,
    eval_end: str,
    n_trials: int = 20
) -> Dict[str, Any]:
    """Compare STGCN confidence vs simple baseline method."""
    
    # Compare with Mean Matching (known to have reasonable CI performance)
    mm_iroas_estimates = []
    mm_ci_widths = []
    
    for trial in range(n_trials):
        np.random.seed(200 + trial)
        
        mm_model = MeanMatchingModel()
        
        try:
            mm_model.fit(panel_data, assignment_df, pre_period_end)
            mm_iroas = mm_model.calculate_iroas(panel_data, eval_start, eval_end)
            mm_iroas_estimates.append(mm_iroas)
            
            # Get CI
            mm_lower, mm_upper = mm_model.confidence_interval(
                panel_data, eval_start, eval_end,
                confidence_level=0.95,
                n_bootstrap=10  # Small for speed
            )
            mm_ci_widths.append(mm_upper - mm_lower)
            
        except Exception as e:
            print(f"  Mean Matching trial {trial} failed: {e}")
            mm_iroas_estimates.append(np.nan)
            mm_ci_widths.append(np.nan)
    
    valid_mm_iroas = [x for x in mm_iroas_estimates if np.isfinite(x)]
    valid_mm_widths = [x for x in mm_ci_widths if np.isfinite(x)]
    
    mm_empirical_std = np.std(valid_mm_iroas) if len(valid_mm_iroas) > 1 else np.nan
    mm_avg_width = np.mean(valid_mm_widths) if len(valid_mm_widths) > 0 else np.nan
    mm_ci_implied_std = mm_avg_width / 3.92 if np.isfinite(mm_avg_width) else np.nan
    
    return {
        'mm_n_trials': len(valid_mm_iroas),
        'mm_empirical_std': mm_empirical_std,
        'mm_avg_ci_width': mm_avg_width,
        'mm_ci_implied_std': mm_ci_implied_std,
        'mm_iroas_estimates': valid_mm_iroas,
        'mm_ci_widths': valid_mm_widths
    }


def diagnose_overconfidence_sources(
    stgcn_results: Dict[str, Any],
    baseline_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Diagnose sources of STGCN overconfidence.
    
    Compares empirical variance (from multiple fits) vs CI-implied variance.
    """
    
    diagnosis = {}
    
    # Extract key metrics
    stgcn_emp_std = stgcn_results['empirical_iroas_std']
    stgcn_ci_std_mc = stgcn_results['ci_implied_std_mc']
    stgcn_ci_std_bs = stgcn_results['ci_implied_std_bs']
    
    mm_emp_std = baseline_results['mm_empirical_std']
    mm_ci_std = baseline_results['mm_ci_implied_std']
    
    # Check if STGCN CIs underestimate empirical variance
    if np.isfinite(stgcn_emp_std) and np.isfinite(stgcn_ci_std_mc):
        mc_underestimation_ratio = stgcn_emp_std / stgcn_ci_std_mc
        diagnosis['mc_underestimation_ratio'] = mc_underestimation_ratio
        diagnosis['mc_underestimated'] = mc_underestimation_ratio > 1.5
    else:
        diagnosis['mc_underestimation_ratio'] = np.nan
        diagnosis['mc_underestimated'] = False
    
    if np.isfinite(stgcn_emp_std) and np.isfinite(stgcn_ci_std_bs):
        bs_underestimation_ratio = stgcn_emp_std / stgcn_ci_std_bs
        diagnosis['bs_underestimation_ratio'] = bs_underestimation_ratio
        diagnosis['bs_underestimated'] = bs_underestimation_ratio > 1.5
    else:
        diagnosis['bs_underestimation_ratio'] = np.nan
        diagnosis['bs_underestimated'] = False
    
    # Compare STGCN vs baseline variance
    if np.isfinite(stgcn_emp_std) and np.isfinite(mm_emp_std):
        variance_ratio = stgcn_emp_std / mm_emp_std
        diagnosis['stgcn_vs_baseline_variance_ratio'] = variance_ratio
        diagnosis['stgcn_more_variable'] = variance_ratio > 2.0
    else:
        diagnosis['stgcn_vs_baseline_variance_ratio'] = np.nan
        diagnosis['stgcn_more_variable'] = False
    
    # Check baseline calibration
    if np.isfinite(mm_emp_std) and np.isfinite(mm_ci_std):
        mm_calibration_ratio = mm_emp_std / mm_ci_std
        diagnosis['baseline_calibration_ratio'] = mm_calibration_ratio
        diagnosis['baseline_well_calibrated'] = 0.8 <= mm_calibration_ratio <= 1.2
    else:
        diagnosis['baseline_calibration_ratio'] = np.nan
        diagnosis['baseline_well_calibrated'] = False
    
    return diagnosis


def main():
    """Run comprehensive overconfidence diagnosis."""
    print("STGCN Overconfidence Diagnosis")
    print("=" * 50)
    
    print("Creating test scenario...")
    panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_test_scenario(seed=42)
    
    print(f"Setup:")
    print(f"  Geos: {len(assignment_df)}")
    print(f"  Treatment: {(assignment_df['assignment'] == 'treatment').sum()}")
    print(f"  Control: {(assignment_df['assignment'] == 'control').sum()}")
    print(f"  Analysis: Empirical variance vs CI-implied variance")
    
    # Run STGCN analysis
    print(f"\n🔬 Analyzing STGCN prediction variance vs CI width...")
    stgcn_results = analyze_prediction_vs_ci_relationship(
        panel_data, assignment_df, pre_period_end, eval_start, eval_end, n_trials=10
    )
    
    # Run baseline comparison
    print(f"\n📊 Analyzing baseline method for comparison...")
    baseline_results = compare_with_baseline_method(
        panel_data, assignment_df, pre_period_end, eval_start, eval_end, n_trials=10
    )
    
    # Diagnose overconfidence
    print(f"\n🎯 Diagnosing overconfidence sources...")
    diagnosis = diagnose_overconfidence_sources(stgcn_results, baseline_results)
    
    # Report results
    print(f"\n" + "=" * 60)
    print("OVERCONFIDENCE DIAGNOSIS RESULTS")
    print("=" * 60)
    
    print(f"\n📈 STGCN Variance Analysis:")
    print(f"  Successful trials: {stgcn_results['n_successful_trials']}")
    print(f"  Empirical iROAS std: {stgcn_results['empirical_iroas_std']:.4f}")
    print(f"  MC Dropout CI-implied std: {stgcn_results['ci_implied_std_mc']:.4f}")
    print(f"  Bootstrap CI-implied std: {stgcn_results['ci_implied_std_bs']:.4f}")
    print(f"  Average MC CI width: {stgcn_results['avg_mc_ci_width']:.4f}")
    print(f"  Average Bootstrap CI width: {stgcn_results['avg_bootstrap_ci_width']:.4f}")
    
    print(f"\n📊 Baseline (Mean Matching) Analysis:")
    print(f"  Successful trials: {baseline_results['mm_n_trials']}")
    print(f"  Empirical iROAS std: {baseline_results['mm_empirical_std']:.4f}")
    print(f"  CI-implied std: {baseline_results['mm_ci_implied_std']:.4f}")
    print(f"  Average CI width: {baseline_results['mm_avg_ci_width']:.4f}")
    
    print(f"\n🔍 OVERCONFIDENCE DIAGNOSIS:")
    
    # MC Dropout assessment
    if diagnosis['mc_underestimated']:
        print(f"  🚨 MC DROPOUT UNDERESTIMATES VARIANCE by {diagnosis['mc_underestimation_ratio']:.1f}x")
        print(f"     Empirical std ({stgcn_results['empirical_iroas_std']:.4f}) >> CI-implied std ({stgcn_results['ci_implied_std_mc']:.4f})")
    else:
        print(f"  ✅ MC Dropout variance estimation reasonable ({diagnosis['mc_underestimation_ratio']:.1f}x)")
    
    # Bootstrap assessment
    if diagnosis['bs_underestimated']:
        print(f"  🚨 BOOTSTRAP UNDERESTIMATES VARIANCE by {diagnosis['bs_underestimation_ratio']:.1f}x")
        print(f"     Empirical std ({stgcn_results['empirical_iroas_std']:.4f}) >> CI-implied std ({stgcn_results['ci_implied_std_bs']:.4f})")
    else:
        print(f"  ✅ Bootstrap variance estimation reasonable ({diagnosis['bs_underestimation_ratio']:.1f}x)")
    
    # Variance comparison
    if diagnosis['stgcn_more_variable']:
        print(f"  ⚠️ STGCN IS {diagnosis['stgcn_vs_baseline_variance_ratio']:.1f}x MORE VARIABLE than baseline")
        print(f"     STGCN std: {stgcn_results['empirical_iroas_std']:.4f} vs Baseline std: {baseline_results['mm_empirical_std']:.4f}")
    else:
        print(f"  ✅ STGCN variance similar to baseline ({diagnosis['stgcn_vs_baseline_variance_ratio']:.1f}x)")
    
    # Baseline calibration
    if diagnosis['baseline_well_calibrated']:
        print(f"  ✅ Baseline method well-calibrated ({diagnosis['baseline_calibration_ratio']:.1f}x)")
    else:
        print(f"  ⚠️ Baseline method calibration issues ({diagnosis['baseline_calibration_ratio']:.1f}x)")
    
    print(f"\n💡 ROOT CAUSE ASSESSMENT:")
    
    # Determine primary issue
    mc_problem = diagnosis.get('mc_underestimated', False)
    bs_problem = diagnosis.get('bs_underestimated', False)
    high_variance = diagnosis.get('stgcn_more_variable', False)
    
    if mc_problem or bs_problem:
        print(f"  🎯 PRIMARY ISSUE: CI METHODS UNDERESTIMATE UNCERTAINTY")
        print(f"     - STGCN predictions vary more across model fits than CIs capture")
        print(f"     - This leads to overconfident CIs → high false positive rates")
        
        if mc_problem:
            print(f"     - MC Dropout underestimates by {diagnosis['mc_underestimation_ratio']:.1f}x")
        if bs_problem:
            print(f"     - Bootstrap underestimates by {diagnosis['bs_underestimation_ratio']:.1f}x")
            
    elif high_variance:
        print(f"  🎯 PRIMARY ISSUE: STGCN HAS INHERENTLY HIGH VARIANCE")
        print(f"     - STGCN is {diagnosis['stgcn_vs_baseline_variance_ratio']:.1f}x more variable than simple baselines")
        print(f"     - Model complexity may be causing overfitting")
        
    else:
        print(f"  🤔 INCONCLUSIVE: Need more investigation")
        print(f"     - Sample sizes may be too small for reliable variance estimation")
    
    print(f"\n🔧 RECOMMENDED FIXES:")
    
    if mc_problem or bs_problem:
        print(f"  1. VARIANCE INFLATION: Multiply CI widths by {max(diagnosis.get('mc_underestimation_ratio', 1), diagnosis.get('bs_underestimation_ratio', 1)):.1f}x")
        print(f"  2. ENSEMBLE METHODS: Use multiple model fits in CI calculation")
        print(f"  3. EMPIRICAL CALIBRATION: Adjust CI based on historical performance")
        
    if high_variance:
        print(f"  4. MODEL REGULARIZATION: Increase dropout, reduce complexity")
        print(f"  5. ENSEMBLE AVERAGING: Average predictions from multiple fits")
        print(f"  6. SIMPLER ARCHITECTURE: Consider spatial regression instead of STGCN")
    
    print(f"  7. VALIDATION: Test fixes on independent validation set")
    
    # Export detailed results
    results_export = {
        'stgcn': stgcn_results,
        'baseline': baseline_results,
        'diagnosis': diagnosis
    }
    
    np.save('stgcn_overconfidence_diagnosis.npy', results_export)
    print(f"\n📁 Detailed results saved to: stgcn_overconfidence_diagnosis.npy")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()