"""
Example usage of the enhanced data generator.

This script demonstrates how to use the new enhanced generator to create
realistic geo-experiment data with spatial relationships, covariates,
seasonality, and treatment effects.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_simulation.enhanced_generators import (
    EnhancedGeoGenerator, 
    EnhancedDataConfig,
    get_west_coast_config,
    get_simple_enhanced_config
)


def example_basic_usage():
    """Basic usage example with default configuration."""
    print("=== Basic Enhanced Data Generation ===")
    
    # Create generator with default config
    generator = EnhancedGeoGenerator()
    
    # Generate data
    panel_data, geo_features = generator.generate()
    
    print(f"Generated panel data shape: {panel_data.shape}")
    print(f"Generated geo features shape: {geo_features.shape}")
    print("\nGeo features columns:", list(geo_features.columns))
    print("\nPanel data columns:", list(panel_data.columns))
    
    # Show sample of geo features
    print("\nSample geo features:")
    print(geo_features.head())
    
    # Show sample of panel data
    print("\nSample panel data:")
    print(panel_data.head())
    
    return panel_data, geo_features


def example_with_treatment_effect():
    """Example with simulated treatment effect."""
    print("\n=== A/B Testing Simulation ===")
    
    # Create config with treatment effect
    config = EnhancedDataConfig(
        n_geos=30,
        n_days=90,
        simulate_treatment_effect=True,
        treatment_start_day=60,  # Treatment starts on day 60
        seed=123
    )
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    # Analyze treatment effect
    pre_period = panel_data[panel_data['date'] < '2024-03-01']  # Days 1-60
    post_period = panel_data[panel_data['date'] >= '2024-03-01']  # Days 61-90
    
    pre_avg = pre_period.groupby('geo')['sales'].mean()
    post_avg = post_period.groupby('geo')['sales'].mean()
    
    lift = post_avg - pre_avg
    print(f"\nAverage lift per geo: {lift.mean():.2f}")
    print(f"Lift standard deviation: {lift.std():.2f}")
    
    return panel_data, geo_features


def example_west_coast_scenario():
    """Example using West Coast preset configuration."""
    print("\n=== West Coast Scenario ===")
    
    config = get_west_coast_config()
    config.n_geos = 25
    config.n_days = 60
    config.seed = 456
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    # Analyze covariate effects
    print("\nCovariate summary:")
    for col in ['median_income', 'digital_penetration', 'urban_indicator']:
        if col in geo_features.columns:
            print(f"{col}: mean={geo_features[col].mean():.3f}, std={geo_features[col].std():.3f}")
    
    # Show spatial patterns
    if len(geo_features) > 0:
        print(f"\nSpatial range - X: [{geo_features['xy1'].min():.1f}, {geo_features['xy1'].max():.1f}]")
        print(f"Spatial range - Y: [{geo_features['xy2'].min():.1f}, {geo_features['xy2'].max():.1f}]")
    
    return panel_data, geo_features


def example_custom_configuration():
    """Example with fully customized configuration."""
    print("\n=== Custom Configuration ===")
    
    # Create custom configuration
    config = EnhancedDataConfig(
        n_geos=20,
        n_days=120,
        seed=789
    )
    
    # Customize covariates - add a new business-specific covariate
    config.covariates.covariates['competitor_density'] = {
        'type': 'continuous',
        'base_mean': 5.0,  # Average number of competitors
        'base_std': 2.0,
        'geographic_gradient': {'x': -0.2, 'y': 0.1},  # Fewer competitors in west
        'affects_baseline': True,
        'affects_iroas': True,
        'baseline_coefficient': -500,  # Negative effect on baseline sales
        'iroas_coefficient': -0.1,    # Reduces treatment effectiveness
    }
    
    # Customize seasonality - add holiday effect
    config.seasonality.components['holiday'] = {
        'amplitude': 3000,
        'phase': 30,  # Peak around day 30
        'geo_variation': 0.4,
    }
    
    # Stronger spatial correlation
    config.spatial.spatial_correlation_strength = 0.8
    config.spatial.spatial_correlation_range = 15.0
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    print("Custom covariates included:", [col for col in geo_features.columns if col not in ['geo', 'xy1', 'xy2']])
    print(f"Competitor density range: [{geo_features['competitor_density'].min():.2f}, {geo_features['competitor_density'].max():.2f}]")
    
    return panel_data, geo_features


def visualize_spatial_patterns(geo_features):
    """Create visualizations of spatial patterns."""
    print("\n=== Creating Spatial Visualizations ===")
    
    if len(geo_features) == 0:
        print("No data to visualize")
        return
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Spatial Patterns in Generated Data', fontsize=16)
    
    # 1. Geographic distribution
    axes[0, 0].scatter(geo_features['xy1'], geo_features['xy2'], alpha=0.6)
    axes[0, 0].set_title('Geographic Distribution')
    axes[0, 0].set_xlabel('X Coordinate (West-East)')
    axes[0, 0].set_ylabel('Y Coordinate (South-North)')
    
    # 2. Median Income by location
    if 'median_income' in geo_features.columns:
        scatter = axes[0, 1].scatter(geo_features['xy1'], geo_features['xy2'], 
                                   c=geo_features['median_income'], cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Median Income by Location')
        axes[0, 1].set_xlabel('X Coordinate')
        axes[0, 1].set_ylabel('Y Coordinate')
        plt.colorbar(scatter, ax=axes[0, 1])
    
    # 3. Digital Penetration by location
    if 'digital_penetration' in geo_features.columns:
        scatter = axes[1, 0].scatter(geo_features['xy1'], geo_features['xy2'], 
                                   c=geo_features['digital_penetration'], cmap='plasma', alpha=0.7)
        axes[1, 0].set_title('Digital Penetration by Location')
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Y Coordinate')
        plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Urban vs Rural distribution
    if 'urban_indicator' in geo_features.columns:
        colors = ['red' if x == 1 else 'blue' for x in geo_features['urban_indicator']]
        axes[1, 1].scatter(geo_features['xy1'], geo_features['xy2'], c=colors, alpha=0.7)
        axes[1, 1].set_title('Urban (Red) vs Rural (Blue)')
        axes[1, 1].set_xlabel('X Coordinate')
        axes[1, 1].set_ylabel('Y Coordinate')
    
    plt.tight_layout()
    plt.show()
    
    print("Spatial pattern plots generated!")


def analyze_seasonality(panel_data):
    """Analyze seasonality patterns in the generated data."""
    print("\n=== Analyzing Seasonality ===")
    
    if len(panel_data) == 0:
        print("No data to analyze")
        return
    
    # Aggregate daily sales across all geos
    daily_sales = panel_data.groupby('date')['sales'].mean().reset_index()
    daily_sales['day_of_week'] = daily_sales['date'].dt.day_of_week
    daily_sales['day_number'] = range(len(daily_sales))
    
    # Create seasonality plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Seasonality Patterns in Generated Data', fontsize=16)
    
    # 1. Time series
    axes[0, 0].plot(daily_sales['date'], daily_sales['sales'])
    axes[0, 0].set_title('Daily Sales Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Average Sales')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Day of week pattern
    weekly_pattern = daily_sales.groupby('day_of_week')['sales'].mean()
    axes[0, 1].bar(range(7), weekly_pattern.values)
    axes[0, 1].set_title('Weekly Seasonality')
    axes[0, 1].set_xlabel('Day of Week (0=Monday)')
    axes[0, 1].set_ylabel('Average Sales')
    axes[0, 1].set_xticks(range(7))
    axes[0, 1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # 3. Seasonal effect if available
    if 'seasonal_effect' in panel_data.columns:
        daily_seasonal = panel_data.groupby('date')['seasonal_effect'].mean().reset_index()
        axes[1, 0].plot(daily_seasonal['date'], daily_seasonal['seasonal_effect'])
        axes[1, 0].set_title('Seasonal Effect Component')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Seasonal Effect')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Treatment effect if available
    if 'treatment_effect' in panel_data.columns:
        daily_treatment = panel_data.groupby('date')['treatment_effect'].mean().reset_index()
        axes[1, 1].plot(daily_treatment['date'], daily_treatment['treatment_effect'])
        axes[1, 1].set_title('Treatment Effect Over Time')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Treatment Effect')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("Seasonality analysis complete!")


def example_statistical_validation():
    """Example demonstrating statistical validation."""
    print("\n=== Statistical Validation Example ===")
    
    config = EnhancedDataConfig(
        n_geos=30,
        n_days=90,
        seed=999,
        base_sales_mean=12000,
        base_sales_std=2500,
        base_spend_mean=6000,
        base_spend_std=1200,
        daily_noise_std=400
    )
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    # Validate statistical properties
    validation_results = generator.validate_statistical_properties(panel_data, tolerance=0.15)
    
    print("\nStatistical Validation Results:")
    print(f"Sales Mean Valid: {'✓' if validation_results['sales_mean_valid'] else '✗'}")
    print(f"  Actual: {validation_results['actual_sales_mean']:.0f}, Expected: {validation_results['expected_sales_mean']:.0f}")
    
    print(f"Sales Std Valid: {'✓' if validation_results['sales_std_valid'] else '✗'}")
    print(f"  Actual: {validation_results['actual_sales_std']:.0f}, Expected: {validation_results['expected_sales_std']:.0f}")
    
    print(f"Spend Mean Valid: {'✓' if validation_results['spend_mean_valid'] else '✗'}")
    print(f"  Actual: {validation_results['actual_spend_mean']:.0f}, Expected: {validation_results['expected_spend_mean']:.0f}")
    
    print(f"Spend Std Valid: {'✓' if validation_results['spend_std_valid'] else '✗'}")
    print(f"  Actual: {validation_results['actual_spend_std']:.0f}, Expected: {validation_results['expected_spend_std']:.0f}")
    
    # Get ground truth summary
    summary = generator.get_summary_statistics()
    print(f"\nGround Truth Summary:")
    print(f"Baseline Sales - Mean: {summary['baseline_sales']['mean']:.0f}, Std: {summary['baseline_sales']['std']:.0f}")
    print(f"iROAS Values - Mean: {summary['iroas_values']['mean']:.2f}, Std: {summary['iroas_values']['std']:.2f}")
    
    return generator, panel_data, geo_features, validation_results


def example_ground_truth_visualization():
    """Example demonstrating ground truth visualization."""
    print("\n=== Ground Truth Visualization Example ===")
    
    config = EnhancedDataConfig(
        n_geos=25,
        n_days=60,
        seed=2024,
        simulate_treatment_effect=True,
        treatment_start_day=40
    )
    
    # Customize for more interesting patterns
    config.spatial.spatial_correlation_strength = 0.8
    config.covariates.covariates['median_income']['geographic_gradient']['x'] = 0.5
    config.covariates.covariates['digital_penetration']['affects_iroas'] = True
    
    generator = EnhancedGeoGenerator(config)
    panel_data, geo_features = generator.generate()
    
    print(f"Generated data: {panel_data.shape[0]} observations, {geo_features.shape[0]} geos")
    
    # Get ground truth parameters
    ground_truth = generator.get_ground_truth()
    
    print("\nGround Truth Parameters Available:")
    print(f"- Baseline sales: {len(ground_truth['baseline_sales'])} geos")
    print(f"- iROAS values: {len(ground_truth['iroas_values'])} geos") 
    print(f"- Seasonality patterns: {len(ground_truth['seasonality_patterns'])} geos")
    print(f"- Geographic coordinates: {ground_truth['geo_coordinates'].shape}")
    
    # Create visualizations
    try:
        from data_simulation.visualization import (
            plot_ground_truth_overview, 
            plot_seasonality_patterns,
            plot_covariate_effects,
            plot_spatial_correlation_analysis,
            plot_validation_results
        )
        
        print("\nCreating ground truth visualizations...")
        
        # Overview plot
        fig1 = plot_ground_truth_overview(generator, figsize=(16, 12))
        plt.show()
        
        # Seasonality patterns
        fig2 = plot_seasonality_patterns(generator, n_geos_to_show=5)
        plt.show()
        
        # Covariate effects
        fig3 = plot_covariate_effects(generator, figsize=(16, 10))
        plt.show()
        
        # Spatial correlation analysis
        fig4 = plot_spatial_correlation_analysis(generator)
        plt.show()
        
        print("✓ Ground truth visualizations created successfully!")
        
    except ImportError:
        print("Visualization modules not available, skipping plots")
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    return generator, panel_data, geo_features


def main():
    """Run all examples."""
    print("Enhanced Geo Data Generator Examples")
    print("="*50)
    
    # Run examples
    panel1, geo1 = example_basic_usage()
    panel2, geo2 = example_with_treatment_effect()
    panel3, geo3 = example_west_coast_scenario()
    panel4, geo4 = example_custom_configuration()
    
    # New examples with validation and visualization
    generator5, panel5, geo5, validation5 = example_statistical_validation()
    generator6, panel6, geo6 = example_ground_truth_visualization()
    
    # Create basic visualizations (optional - comment out if running without display)
    try:
        visualize_spatial_patterns(geo3)
        analyze_seasonality(panel2)
        
        # Validation visualization
        from data_simulation.visualization import plot_validation_results
        fig_validation = plot_validation_results(validation5)
        plt.show()
        
    except Exception as e:
        print(f"Skipping basic visualizations: {e}")
    
    print("\n" + "="*50)
    print("All examples completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✓ Spatial autocorrelation and geographic patterns")
    print("✓ Static covariates with effect modification")
    print("✓ Multiple seasonality components")
    print("✓ Realistic sales models with iROAS effects")
    print("✓ Flexible configuration system")
    print("✓ Treatment effect simulation for A/B testing")
    print("✓ Statistical validation of generated data")
    print("✓ Ground truth parameter visualization")
    print("✓ Comprehensive analysis and diagnostics")


if __name__ == "__main__":
    main()