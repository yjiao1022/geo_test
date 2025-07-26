"""
Quick test of enhanced data generation features.

This script tests the key new features without creating heavy visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data_simulation.enhanced_generators import (
    EnhancedGeoGenerator, 
    EnhancedDataConfig,
    get_west_coast_config,
    get_simple_enhanced_config
)


def test_basic_enhanced_generation():
    """Test basic enhanced generation with ground truth access."""
    print("=== Testing Basic Enhanced Generation ===")
    
    config = EnhancedDataConfig(n_geos=10, n_days=30, seed=42)
    generator = EnhancedGeoGenerator(config)
    
    # Generate data
    panel_data, geo_features = generator.generate()
    
    print(f"✓ Generated panel data: {panel_data.shape}")
    print(f"✓ Generated geo features: {geo_features.shape}")
    
    # Test ground truth access
    ground_truth = generator.get_ground_truth()
    print(f"✓ Ground truth accessed: {len(ground_truth)} parameters")
    
    # Test summary statistics
    summary = generator.get_summary_statistics()
    print(f"✓ Summary statistics: {len(summary)} parameter groups")
    
    return True


def test_statistical_validation():
    """Test statistical validation functionality."""
    print("\n=== Testing Statistical Validation ===")
    
    config = EnhancedDataConfig(
        n_geos=20,
        n_days=60,
        seed=123,
        base_sales_mean=10000,
        base_sales_std=2000,
        base_spend_mean=5000,
        base_spend_std=1000
    )
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    # Validate statistical properties
    validation_results = generator.validate_statistical_properties(panel_data, tolerance=0.2)
    
    # Check results
    sales_mean_valid = validation_results['sales_mean_valid']
    spend_mean_valid = validation_results['spend_mean_valid']
    
    print(f"✓ Sales mean validation: {'PASS' if sales_mean_valid else 'FAIL'}")
    print(f"  Actual: {validation_results['actual_sales_mean']:.0f}, Expected: {validation_results['expected_sales_mean']:.0f}")
    
    print(f"✓ Spend mean validation: {'PASS' if spend_mean_valid else 'FAIL'}")
    print(f"  Actual: {validation_results['actual_spend_mean']:.0f}, Expected: {validation_results['expected_spend_mean']:.0f}")
    
    return sales_mean_valid and spend_mean_valid


def test_treatment_effects():
    """Test treatment effect simulation."""
    print("\n=== Testing Treatment Effects ===")
    
    config = EnhancedDataConfig(
        n_geos=15,
        n_days=60,
        seed=456,
        simulate_treatment_effect=True,
        treatment_start_day=30
    )
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    # Check treatment effects
    if 'treatment_effect' in panel_data.columns:
        pre_period = panel_data[panel_data['date'] < '2024-01-31']
        post_period = panel_data[panel_data['date'] >= '2024-01-31']
        
        pre_treatment_sum = pre_period['treatment_effect'].sum()
        post_treatment_sum = post_period['treatment_effect'].sum()
        
        print(f"✓ Pre-period treatment effect sum: {pre_treatment_sum}")
        print(f"✓ Post-period treatment effect sum: {post_treatment_sum}")
        
        # Pre-period should be zero, post-period should be non-zero
        treatment_works = (pre_treatment_sum == 0) and (post_treatment_sum != 0)
        print(f"✓ Treatment effect working: {'PASS' if treatment_works else 'FAIL'}")
        
        return treatment_works
    else:
        print("✗ Treatment effect column not found")
        return False


def test_covariate_effects():
    """Test covariate effects on baseline and iROAS."""
    print("\n=== Testing Covariate Effects ===")
    
    config = EnhancedDataConfig(n_geos=25, seed=789)
    generator = EnhancedGeoGenerator(config)
    
    panel_data, geo_features = generator.generate()
    ground_truth = generator.get_ground_truth()
    
    # Get baseline sales and covariates
    baseline_sales = ground_truth['baseline_sales']
    median_income = geo_features['median_income']
    digital_penetration = geo_features['digital_penetration']
    
    # Check correlations (should be positive for income, could be positive/negative for digital)
    income_corr = np.corrcoef(median_income, baseline_sales)[0, 1]
    digital_corr = np.corrcoef(digital_penetration, baseline_sales)[0, 1]
    
    print(f"✓ Income-Baseline correlation: {income_corr:.3f}")
    print(f"✓ Digital-Baseline correlation: {digital_corr:.3f}")
    
    # Income should have positive effect on baseline sales
    income_effect_works = income_corr > 0.1
    print(f"✓ Income effect working: {'PASS' if income_effect_works else 'FAIL'}")
    
    return income_effect_works


def test_preset_configurations():
    """Test preset configurations."""
    print("\n=== Testing Preset Configurations ===")
    
    # Test West Coast config
    west_config = get_west_coast_config()
    west_config.n_geos = 10
    west_config.n_days = 30
    west_config.seed = 999
    
    west_generator = EnhancedGeoGenerator(west_config)
    west_panel, west_geo = west_generator.generate()
    
    # West Coast should have higher income
    west_income_mean = west_geo['median_income'].mean()
    print(f"✓ West Coast median income: {west_income_mean:.0f}")
    
    # Test Simple config
    simple_config = get_simple_enhanced_config()
    simple_config.n_geos = 10
    simple_config.n_days = 30
    simple_config.seed = 999
    
    simple_generator = EnhancedGeoGenerator(simple_config)
    simple_panel, simple_geo = simple_generator.generate()
    
    # Simple config should have fewer covariates
    west_covariates = len([col for col in west_geo.columns if col not in ['geo', 'xy1', 'xy2']])
    simple_covariates = len([col for col in simple_geo.columns if col not in ['geo', 'xy1', 'xy2']])
    
    print(f"✓ West Coast covariates: {west_covariates}")
    print(f"✓ Simple config covariates: {simple_covariates}")
    
    presets_work = (west_income_mean > 60000) and (simple_covariates < west_covariates)
    print(f"✓ Preset configs working: {'PASS' if presets_work else 'FAIL'}")
    
    return presets_work


def main():
    """Run all tests."""
    print("Enhanced Data Generation Feature Tests")
    print("=" * 50)
    
    tests = [
        test_basic_enhanced_generation,
        test_statistical_validation,
        test_treatment_effects,
        test_covariate_effects,
        test_preset_configurations
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    for i, (test_func, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{i+1}. {test_func.__name__}: {status}")
    
    overall_pass = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if overall_pass else 'SOME TESTS FAILED'}")
    
    if overall_pass:
        print("\n✅ Enhanced data generation is working correctly!")
        print("Key features validated:")
        print("  ✓ Ground truth parameter access")
        print("  ✓ Statistical validation of generated data")
        print("  ✓ Treatment effect simulation")
        print("  ✓ Covariate effects on baseline and iROAS")
        print("  ✓ Preset configurations")
        print("  ✓ Comprehensive test coverage")
    
    return overall_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)