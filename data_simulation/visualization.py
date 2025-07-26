"""
Visualization utilities for enhanced data generators.

This module provides functions to visualize ground truth parameters
and generated data patterns for analysis and validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from data_simulation.enhanced_generators import EnhancedGeoGenerator


def plot_ground_truth_overview(generator: EnhancedGeoGenerator, 
                               figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create comprehensive overview of ground truth parameters.
    
    Args:
        generator: EnhancedGeoGenerator instance (after generate() called)
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    ground_truth = generator.get_ground_truth()
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle('Ground Truth Parameter Overview', fontsize=16, y=0.98)
    
    # 1. Geographic distribution
    coords = ground_truth['geo_coordinates']
    axes[0, 0].scatter(coords['xy1'], coords['xy2'], alpha=0.7, s=50)
    axes[0, 0].set_title('Geographic Distribution')
    axes[0, 0].set_xlabel('X Coordinate (West-East)')
    axes[0, 0].set_ylabel('Y Coordinate (South-North)')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Baseline sales by location
    baseline_sales = ground_truth['baseline_sales']
    scatter = axes[0, 1].scatter(coords['xy1'], coords['xy2'], 
                                c=baseline_sales, cmap='viridis', 
                                alpha=0.8, s=60)
    axes[0, 1].set_title('Baseline Sales by Location')
    axes[0, 1].set_xlabel('X Coordinate')
    axes[0, 1].set_ylabel('Y Coordinate')
    plt.colorbar(scatter, ax=axes[0, 1], label='Baseline Sales')
    
    # 3. iROAS values by location
    iroas_values = ground_truth['iroas_values']
    scatter = axes[0, 2].scatter(coords['xy1'], coords['xy2'], 
                                c=iroas_values, cmap='plasma', 
                                alpha=0.8, s=60)
    axes[0, 2].set_title('iROAS Values by Location')
    axes[0, 2].set_xlabel('X Coordinate')
    axes[0, 2].set_ylabel('Y Coordinate')
    plt.colorbar(scatter, ax=axes[0, 2], label='iROAS')
    
    # 4. Baseline sales distribution
    axes[1, 0].hist(baseline_sales, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(baseline_sales.mean(), color='red', linestyle='--', 
                      label=f'Mean: {baseline_sales.mean():.0f}')
    axes[1, 0].set_title('Baseline Sales Distribution')
    axes[1, 0].set_xlabel('Baseline Sales')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # 5. iROAS distribution
    axes[1, 1].hist(iroas_values, bins=20, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].axvline(iroas_values.mean(), color='red', linestyle='--',
                      label=f'Mean: {iroas_values.mean():.2f}')
    axes[1, 1].set_title('iROAS Distribution')
    axes[1, 1].set_xlabel('iROAS')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Baseline sales vs iROAS relationship
    axes[1, 2].scatter(baseline_sales, iroas_values, alpha=0.7)
    axes[1, 2].set_title('Baseline Sales vs iROAS')
    axes[1, 2].set_xlabel('Baseline Sales')
    axes[1, 2].set_ylabel('iROAS')
    axes[1, 2].grid(alpha=0.3)
    
    # Calculate correlation
    correlation = np.corrcoef(baseline_sales, iroas_values)[0, 1]
    axes[1, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[1, 2].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 7-9. Top covariates by location
    geo_features = ground_truth['covariate_values']
    covariate_names = [col for col in geo_features.columns 
                      if col not in ['geo', 'xy1', 'xy2']][:3]
    
    for i, cov_name in enumerate(covariate_names):
        if i < 3:  # Only plot first 3 covariates
            row, col = 2, i
            values = geo_features[cov_name]
            scatter = axes[row, col].scatter(coords['xy1'], coords['xy2'], 
                                           c=values, cmap='coolwarm', 
                                           alpha=0.8, s=60)
            axes[row, col].set_title(f'{cov_name.replace("_", " ").title()} by Location')
            axes[row, col].set_xlabel('X Coordinate')
            axes[row, col].set_ylabel('Y Coordinate')
            plt.colorbar(scatter, ax=axes[row, col], label=cov_name)
    
    # Remove empty subplots if fewer than 3 covariates
    for i in range(len(covariate_names), 3):
        axes[2, i].remove()
    
    plt.tight_layout()
    return fig


def plot_seasonality_patterns(generator: EnhancedGeoGenerator,
                             n_geos_to_show: int = 5,
                             figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Plot seasonality patterns for selected geos.
    
    Args:
        generator: EnhancedGeoGenerator instance (after generate() called)
        n_geos_to_show: Number of geos to show patterns for
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    ground_truth = generator.get_ground_truth()
    seasonality_patterns = ground_truth['seasonality_patterns']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Seasonality Patterns', fontsize=16)
    
    # Get sample of geos to display
    geo_ids = list(seasonality_patterns.keys())[:n_geos_to_show]
    dates = pd.date_range("2024-01-01", periods=generator.config.n_days)
    
    # 1. Individual seasonality patterns
    for geo_id in geo_ids:
        pattern = seasonality_patterns[geo_id]
        axes[0, 0].plot(dates, pattern, alpha=0.7, label=geo_id)
    
    axes[0, 0].set_title('Individual Seasonality Patterns')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Seasonal Effect')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Average seasonality pattern
    all_patterns = np.array([seasonality_patterns[geo_id] for geo_id in geo_ids])
    mean_pattern = all_patterns.mean(axis=0)
    std_pattern = all_patterns.std(axis=0)
    
    axes[0, 1].plot(dates, mean_pattern, 'b-', linewidth=2, label='Mean')
    axes[0, 1].fill_between(dates, mean_pattern - std_pattern, mean_pattern + std_pattern, 
                           alpha=0.3, label='±1 Std')
    axes[0, 1].set_title('Average Seasonality ± Standard Deviation')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Seasonal Effect')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Weekly pattern (first 28 days)
    if generator.config.n_days >= 28:
        weekly_dates = dates[:28]
        axes[1, 0].plot(weekly_dates, mean_pattern[:28], 'g-', linewidth=2)
        axes[1, 0].set_title('Weekly Pattern (First 4 Weeks)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Seasonal Effect')
        axes[1, 0].grid(alpha=0.3)
        
        # Add day of week labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i in range(0, 28, 7):
            if i < len(weekly_dates):
                axes[1, 0].axvline(weekly_dates[i], color='red', alpha=0.3, linestyle='--')
                axes[1, 0].text(weekly_dates[i], axes[1, 0].get_ylim()[1] * 0.9, 
                               day_names[i % 7], rotation=45, fontsize=8)
    
    # 4. Seasonality magnitude by geo
    magnitudes = [np.std(seasonality_patterns[geo_id]) for geo_id in geo_ids]
    axes[1, 1].bar(range(len(geo_ids)), magnitudes)
    axes[1, 1].set_title('Seasonality Magnitude by Geo')
    axes[1, 1].set_xlabel('Geo Index')
    axes[1, 1].set_ylabel('Standard Deviation of Seasonal Effect')
    axes[1, 1].set_xticks(range(len(geo_ids)))
    axes[1, 1].set_xticklabels([geo_id.split('_')[1] for geo_id in geo_ids])
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_covariate_effects(generator: EnhancedGeoGenerator,
                          figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Visualize covariate effects on baseline sales and iROAS.
    
    Args:
        generator: EnhancedGeoGenerator instance (after generate() called)
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    ground_truth = generator.get_ground_truth()
    geo_features = ground_truth['covariate_values']
    baseline_sales = ground_truth['baseline_sales']
    iroas_values = ground_truth['iroas_values']
    
    # Get covariates that affect baseline or iROAS
    affecting_covariates = []
    for cov_name, cov_config in generator.config.covariates.covariates.items():
        if cov_config.get('affects_baseline', False) or cov_config.get('affects_iroas', False):
            affecting_covariates.append((cov_name, cov_config))
    
    n_covariates = len(affecting_covariates)
    fig, axes = plt.subplots(2, max(n_covariates, 1), figsize=figsize)
    if n_covariates == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Covariate Effects on Baseline Sales and iROAS', fontsize=16)
    
    for i, (cov_name, cov_config) in enumerate(affecting_covariates):
        if cov_name in geo_features.columns:
            cov_values = geo_features[cov_name]
            
            # Baseline sales effect (top row)
            if cov_config.get('affects_baseline', False):
                axes[0, i].scatter(cov_values, baseline_sales, alpha=0.7)
                
                # Add trend line
                z = np.polyfit(cov_values, baseline_sales, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(cov_values.min(), cov_values.max(), 100)
                axes[0, i].plot(x_trend, p(x_trend), "r--", alpha=0.8)
                
                # Calculate and display correlation
                correlation = np.corrcoef(cov_values, baseline_sales)[0, 1]
                axes[0, i].text(0.05, 0.95, f'r = {correlation:.3f}', 
                               transform=axes[0, i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[0, i].text(0.5, 0.5, 'No Baseline Effect', 
                               transform=axes[0, i].transAxes, ha='center', va='center',
                               fontsize=12, alpha=0.5)
            
            axes[0, i].set_title(f'{cov_name.replace("_", " ").title()} vs Baseline Sales')
            axes[0, i].set_xlabel(cov_name.replace("_", " ").title())
            axes[0, i].set_ylabel('Baseline Sales')
            axes[0, i].grid(alpha=0.3)
            
            # iROAS effect (bottom row)
            if cov_config.get('affects_iroas', False):
                axes[1, i].scatter(cov_values, iroas_values, alpha=0.7, color='orange')
                
                # Add trend line
                z = np.polyfit(cov_values, iroas_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(cov_values.min(), cov_values.max(), 100)
                axes[1, i].plot(x_trend, p(x_trend), "r--", alpha=0.8)
                
                # Calculate and display correlation
                correlation = np.corrcoef(cov_values, iroas_values)[0, 1]
                axes[1, i].text(0.05, 0.95, f'r = {correlation:.3f}', 
                               transform=axes[1, i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                axes[1, i].text(0.5, 0.5, 'No iROAS Effect', 
                               transform=axes[1, i].transAxes, ha='center', va='center',
                               fontsize=12, alpha=0.5)
            
            axes[1, i].set_title(f'{cov_name.replace("_", " ").title()} vs iROAS')
            axes[1, i].set_xlabel(cov_name.replace("_", " ").title())
            axes[1, i].set_ylabel('iROAS')
            axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_spatial_correlation_analysis(generator: EnhancedGeoGenerator,
                                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Analyze and visualize spatial correlation patterns.
    
    Args:
        generator: EnhancedGeoGenerator instance (after generate() called)
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    ground_truth = generator.get_ground_truth()
    coords = ground_truth['geo_coordinates']
    baseline_sales = ground_truth['baseline_sales']
    iroas_values = ground_truth['iroas_values']
    
    # Calculate distance matrix
    from scipy.spatial.distance import cdist
    coordinates = coords[['xy1', 'xy2']].values
    distance_matrix = cdist(coordinates, coordinates)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Spatial Correlation Analysis', fontsize=16)
    
    # 1. Distance vs Baseline Sales Similarity
    # Calculate pairwise similarities
    n_geos = len(baseline_sales)
    distances = []
    sales_similarities = []
    iroas_similarities = []
    
    for i in range(n_geos):
        for j in range(i+1, n_geos):
            distances.append(distance_matrix[i, j])
            sales_similarities.append(abs(baseline_sales[i] - baseline_sales[j]))
            iroas_similarities.append(abs(iroas_values[i] - iroas_values[j]))
    
    distances = np.array(distances)
    sales_similarities = np.array(sales_similarities)
    iroas_similarities = np.array(iroas_similarities)
    
    # Plot distance vs baseline sales similarity
    axes[0, 0].scatter(distances, sales_similarities, alpha=0.5)
    axes[0, 0].set_title('Geographic Distance vs Baseline Sales Difference')
    axes[0, 0].set_xlabel('Geographic Distance')
    axes[0, 0].set_ylabel('Absolute Baseline Sales Difference')
    axes[0, 0].grid(alpha=0.3)
    
    # Add correlation
    correlation = np.corrcoef(distances, sales_similarities)[0, 1]
    axes[0, 0].text(0.05, 0.95, f'r = {correlation:.3f}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Distance vs iROAS Similarity
    axes[0, 1].scatter(distances, iroas_similarities, alpha=0.5, color='orange')
    axes[0, 1].set_title('Geographic Distance vs iROAS Difference')
    axes[0, 1].set_xlabel('Geographic Distance')
    axes[0, 1].set_ylabel('Absolute iROAS Difference')
    axes[0, 1].grid(alpha=0.3)
    
    # Add correlation
    correlation = np.corrcoef(distances, iroas_similarities)[0, 1]
    axes[0, 1].text(0.05, 0.95, f'r = {correlation:.3f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Spatial autocorrelation function for baseline sales
    # Bin distances and calculate average similarity within each bin
    max_distance = distances.max()
    distance_bins = np.linspace(0, max_distance, 20)
    bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
    binned_sales_similarities = []
    
    for i in range(len(distance_bins)-1):
        mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
        if mask.sum() > 0:
            binned_sales_similarities.append(sales_similarities[mask].mean())
        else:
            binned_sales_similarities.append(np.nan)
    
    axes[1, 0].plot(bin_centers, binned_sales_similarities, 'bo-', label='Observed')
    axes[1, 0].set_title('Spatial Autocorrelation Function (Baseline Sales)')
    axes[1, 0].set_xlabel('Distance')
    axes[1, 0].set_ylabel('Average Absolute Difference')
    axes[1, 0].grid(alpha=0.3)
    
    # Add theoretical exponential decay if spatial correlation is enabled
    if generator.config.spatial.spatial_correlation_strength > 0:
        range_param = generator.config.spatial.spatial_correlation_range
        strength = generator.config.spatial.spatial_correlation_strength
        
        # Theoretical correlation function (inverted for difference)
        theoretical_correlation = strength * np.exp(-bin_centers / range_param)
        # Convert correlation to expected difference (approximate)
        baseline_std = baseline_sales.std()
        theoretical_diff = baseline_std * np.sqrt(2 * (1 - theoretical_correlation))
        
        axes[1, 0].plot(bin_centers, theoretical_diff, 'r--', 
                       label=f'Theoretical (range={range_param})')
        axes[1, 0].legend()
    
    # 4. Spatial autocorrelation function for iROAS
    binned_iroas_similarities = []
    for i in range(len(distance_bins)-1):
        mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
        if mask.sum() > 0:
            binned_iroas_similarities.append(iroas_similarities[mask].mean())
        else:
            binned_iroas_similarities.append(np.nan)
    
    axes[1, 1].plot(bin_centers, binned_iroas_similarities, 'bo-', label='Observed')
    axes[1, 1].set_title('Spatial Autocorrelation Function (iROAS)')
    axes[1, 1].set_xlabel('Distance')  
    axes[1, 1].set_ylabel('Average Absolute Difference')
    axes[1, 1].grid(alpha=0.3)
    
    # Add theoretical if iROAS spatial correlation is enabled
    if generator.config.treatment.iroas_spatial_correlation > 0:
        iroas_corr_strength = generator.config.treatment.iroas_spatial_correlation
        range_param = generator.config.spatial.spatial_correlation_range
        
        theoretical_correlation = iroas_corr_strength * np.exp(-bin_centers / range_param)
        iroas_std = iroas_values.std()
        theoretical_diff = iroas_std * np.sqrt(2 * (1 - theoretical_correlation))
        
        axes[1, 1].plot(bin_centers, theoretical_diff, 'r--', 
                       label=f'Theoretical (strength={iroas_corr_strength})')
        axes[1, 1].legend()
    
    plt.tight_layout()
    return fig


def plot_validation_results(validation_results: Dict[str, any],
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Visualize statistical validation results.
    
    Args:
        validation_results: Results from generator.validate_statistical_properties()
        figsize: Figure size tuple
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Statistical Validation Results', fontsize=16)
    
    # 1. Mean validation
    metrics = ['sales_mean', 'spend_mean']
    actual_values = [validation_results[f'actual_{metric}'] for metric in metrics]
    expected_values = [validation_results[f'expected_{metric}'] for metric in metrics]
    valid_flags = [validation_results[f'{metric}_valid'] for metric in metrics]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar(x_pos - width/2, actual_values, width, 
                       label='Actual', alpha=0.8)
    bars2 = axes[0].bar(x_pos + width/2, expected_values, width, 
                       label='Expected', alpha=0.8)
    
    # Color bars based on validation
    for i, (bar1, bar2, valid) in enumerate(zip(bars1, bars2, valid_flags)):
        color = 'green' if valid else 'red'
        bar1.set_edgecolor(color)
        bar1.set_linewidth(2)
        bar2.set_edgecolor(color)
        bar2.set_linewidth(2)
    
    axes[0].set_title('Mean Values Validation')
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Values')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Add validation status text
    for i, valid in enumerate(valid_flags):
        status = '✓' if valid else '✗'
        axes[0].text(i, max(actual_values[i], expected_values[i]) * 1.05, 
                    status, ha='center', fontsize=16, 
                    color='green' if valid else 'red', weight='bold')
    
    # 2. Standard deviation validation
    std_metrics = ['sales_std', 'spend_std']
    actual_stds = [validation_results[f'actual_{metric}'] for metric in std_metrics]
    expected_stds = [validation_results[f'expected_{metric}'] for metric in std_metrics]
    std_valid_flags = [validation_results[f'{metric}_valid'] for metric in std_metrics]
    
    bars1 = axes[1].bar(x_pos - width/2, actual_stds, width, 
                       label='Actual', alpha=0.8)
    bars2 = axes[1].bar(x_pos + width/2, expected_stds, width, 
                       label='Expected', alpha=0.8)
    
    # Color bars based on validation
    for i, (bar1, bar2, valid) in enumerate(zip(bars1, bars2, std_valid_flags)):
        color = 'green' if valid else 'red'
        bar1.set_edgecolor(color)
        bar1.set_linewidth(2)
        bar2.set_edgecolor(color)
        bar2.set_linewidth(2)
    
    axes[1].set_title('Standard Deviation Validation')
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([m.replace('_', ' ').title().replace('Std', 'Std Dev') for m in std_metrics])
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Add validation status text
    for i, valid in enumerate(std_valid_flags):
        status = '✓' if valid else '✗'
        axes[1].text(i, max(actual_stds[i], expected_stds[i]) * 1.05, 
                    status, ha='center', fontsize=16, 
                    color='green' if valid else 'red', weight='bold')
    
    plt.tight_layout()
    return fig


def create_full_ground_truth_report(generator: EnhancedGeoGenerator,
                                   save_path: Optional[str] = None) -> List[plt.Figure]:
    """
    Create comprehensive ground truth visualization report.
    
    Args:
        generator: EnhancedGeoGenerator instance (after generate() called)
        save_path: Optional path to save figures (without extension)
        
    Returns:
        List of matplotlib figure objects
    """
    figures = []
    
    # 1. Overview plot
    fig1 = plot_ground_truth_overview(generator)
    figures.append(fig1)
    if save_path:
        fig1.savefig(f"{save_path}_overview.png", dpi=300, bbox_inches='tight')
    
    # 2. Seasonality patterns
    fig2 = plot_seasonality_patterns(generator)
    figures.append(fig2)
    if save_path:
        fig2.savefig(f"{save_path}_seasonality.png", dpi=300, bbox_inches='tight')
    
    # 3. Covariate effects
    fig3 = plot_covariate_effects(generator)
    figures.append(fig3)
    if save_path:
        fig3.savefig(f"{save_path}_covariates.png", dpi=300, bbox_inches='tight')
    
    # 4. Spatial correlation analysis
    fig4 = plot_spatial_correlation_analysis(generator)
    figures.append(fig4)
    if save_path:
        fig4.savefig(f"{save_path}_spatial.png", dpi=300, bbox_inches='tight')
    
    return figures