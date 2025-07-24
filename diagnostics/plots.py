"""
Plotting and visualization functions for geo-experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


class DiagnosticPlotter:
    """
    Utility class for creating diagnostic plots for geo-experiments.
    """
    
    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the plotter.
        
        Args:
            style: Seaborn style to use
            figsize: Default figure size
        """
        sns.set_style(style)
        self.default_figsize = figsize
    
    def plot_assignment_balance(self, geo_features: pd.DataFrame, 
                              assignment_df: pd.DataFrame,
                              feature_cols: Optional[List[str]] = None,
                              figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot balance of covariates across treatment and control groups.
        
        Args:
            geo_features: DataFrame with geo-level features
            assignment_df: Assignment DataFrame
            feature_cols: List of feature columns to plot. If None, uses numeric columns.
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        # Merge assignment with features
        merged = geo_features.merge(assignment_df, on='geo')
        
        if feature_cols is None:
            feature_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != 'geo']
        
        n_features = len(feature_cols)
        if n_features == 0:
            raise ValueError("No numeric features found to plot")
        
        figsize = figsize or (12, 3 * n_features)
        fig, axes = plt.subplots(n_features, 2, figsize=figsize)
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(feature_cols):
            # Box plot
            sns.boxplot(data=merged, x='assignment', y=feature, ax=axes[i, 0])
            axes[i, 0].set_title(f'{feature} - Distribution by Group')
            
            # Histogram
            for assignment in merged['assignment'].unique():
                data = merged[merged['assignment'] == assignment][feature]
                axes[i, 1].hist(data, alpha=0.6, label=assignment, bins=15)
            axes[i, 1].set_xlabel(feature)
            axes[i, 1].set_ylabel('Frequency')
            axes[i, 1].set_title(f'{feature} - Histogram by Group')
            axes[i, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, panel_data: pd.DataFrame,
                        assignment_df: pd.DataFrame,
                        metric: str = 'sales',
                        figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot time series of metrics by treatment/control groups.
        
        Args:
            panel_data: Panel data with time series
            assignment_df: Assignment DataFrame
            metric: Column name to plot ('sales' or 'spend')
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        # Merge with assignment
        merged = panel_data.merge(assignment_df, on='geo')
        
        # Calculate daily means by group
        daily_means = merged.groupby(['date', 'assignment'])[metric].mean().reset_index()
        
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        for assignment in daily_means['assignment'].unique():
            data = daily_means[daily_means['assignment'] == assignment]
            ax.plot(data['date'], data[metric], label=assignment, linewidth=2)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Mean {metric.capitalize()}')
        ax.set_title(f'{metric.capitalize()} Time Series by Group')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def plot_evaluation_results(self, results_df: pd.DataFrame,
                               figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot evaluation results across method combinations.
        
        Args:
            results_df: Results DataFrame from evaluation
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or (15, 10)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. iROAS estimates distribution
        sns.boxplot(data=results_df, x='assignment_method', y='iroas_estimate', 
                   hue='reporting_method', ax=axes[0, 0])
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('iROAS Estimates Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Confidence interval widths
        sns.boxplot(data=results_df, x='assignment_method', y='ci_width',
                   hue='reporting_method', ax=axes[0, 1])
        axes[0, 1].set_title('Confidence Interval Widths')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. False positive rates
        fp_rates = results_df.groupby(['assignment_method', 'reporting_method'])['significant'].mean().reset_index()
        fp_pivot = fp_rates.pivot(index='assignment_method', columns='reporting_method', values='significant')
        sns.heatmap(fp_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1, 0])
        axes[1, 0].set_title('False Positive Rates')
        
        # 4. Mean CI widths heatmap
        ci_means = results_df.groupby(['assignment_method', 'reporting_method'])['ci_width'].mean().reset_index()
        ci_pivot = ci_means.pivot(index='assignment_method', columns='reporting_method', values='ci_width')
        sns.heatmap(ci_pivot, annot=True, fmt='.3f', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Mean CI Widths')
        
        plt.tight_layout()
        return fig
    
    def plot_bootstrap_distribution(self, bootstrap_estimates: np.ndarray,
                                   point_estimate: float,
                                   confidence_interval: Tuple[float, float],
                                   true_value: float = 0.0,
                                   figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """
        Plot bootstrap distribution with confidence intervals.
        
        Args:
            bootstrap_estimates: Array of bootstrap estimates
            point_estimate: Point estimate
            confidence_interval: Tuple of (lower, upper) bounds
            true_value: True value to compare against
            figsize: Figure size override
            
        Returns:
            Matplotlib figure
        """
        figsize = figsize or self.default_figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogram of bootstrap estimates
        ax.hist(bootstrap_estimates, bins=50, alpha=0.7, density=True, 
                color='skyblue', edgecolor='black')
        
        # Add vertical lines
        ax.axvline(point_estimate, color='blue', linestyle='-', linewidth=2, 
                  label=f'Point Estimate: {point_estimate:.3f}')
        ax.axvline(true_value, color='red', linestyle='--', linewidth=2,
                  label=f'True Value: {true_value:.3f}')
        ax.axvline(confidence_interval[0], color='green', linestyle=':', linewidth=2,
                  label=f'CI Lower: {confidence_interval[0]:.3f}')
        ax.axvline(confidence_interval[1], color='green', linestyle=':', linewidth=2,
                  label=f'CI Upper: {confidence_interval[1]:.3f}')
        
        # Fill CI region
        ax.fill_betweenx([0, ax.get_ylim()[1]], confidence_interval[0], confidence_interval[1],
                        alpha=0.2, color='green')
        
        ax.set_xlabel('iROAS Estimate')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap Distribution of iROAS Estimates')
        ax.legend()
        
        plt.tight_layout()
        return fig