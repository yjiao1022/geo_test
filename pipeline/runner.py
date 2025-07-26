"""
Main experiment runner for geo-experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any
import os
from datetime import datetime

from data_simulation.generators import SimpleNullGenerator
from assignment.methods import BaseAssignment, RandomAssignment
from reporting.models import (
    BaseModel, MeanMatchingModel, GBRModel, TBRModel, SyntheticControlModel
)
from reporting.stgcn_model import STGCNReportingModel
from evaluation.metrics import EvaluationRunner
from diagnostics.plots import DiagnosticPlotter
from .config import ExperimentConfig


class ExperimentRunner:
    """
    High-level runner for complete geo-experiment evaluations.
    
    This class orchestrates the entire pipeline from data generation
    to results visualization and summary.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration. If None, uses defaults.
        """
        self.config = config or ExperimentConfig()
        self.plotter = DiagnosticPlotter()
        
        # Initialize default methods
        self.assignment_methods = {
            'Random': RandomAssignment()
        }
        
        self.reporting_methods = {
            'MeanMatching': MeanMatchingModel(),
            'GBR': GBRModel(),
            'TBR': TBRModel(),
            'SCM': SyntheticControlModel(),
            'STGCN': STGCNReportingModel()
        }
    
    def add_assignment_method(self, name: str, method: BaseAssignment) -> 'ExperimentRunner':
        """
        Add an assignment method to evaluate.
        
        Args:
            name: Name for the method
            method: Assignment method instance
            
        Returns:
            Self for method chaining
        """
        self.assignment_methods[name] = method
        return self
    
    def add_reporting_method(self, name: str, method: BaseModel) -> 'ExperimentRunner':
        """
        Add a reporting method to evaluate.
        
        Args:
            name: Name for the method
            method: Reporting method instance
            
        Returns:
            Self for method chaining
        """
        self.reporting_methods[name] = method
        return self
    
    def run_single_experiment(self, 
                             show_plots: bool = True, 
                             save_plots: bool = False,
                             output_dir: str = "single_experiment_results") -> Dict[str, Any]:
        """
        Run a single experiment for demonstration/debugging.
        
        Args:
            show_plots: Whether to display diagnostic plots
            save_plots: Whether to save diagnostic plots to files
            output_dir: Directory to save plots (if save_plots=True)
            
        Returns:
            Dictionary with experiment results and data
        """
        # Generate data
        data_generator = SimpleNullGenerator(self.config.to_data_config())
        panel_data, geo_features = data_generator.generate()
        
        # Use first methods for single experiment
        assignment_method = list(self.assignment_methods.values())[0]
        reporting_method = list(self.reporting_methods.values())[0]
        
        # Assignment
        assignment_df = assignment_method.assign(
            geo_features, 
            self.config.treatment_ratio,
            seed=self.config.seed
        )
        
        # Define periods
        dates = pd.to_datetime(panel_data['date'].unique())
        pre_period_end = dates[self.config.pre_period_days - 1].strftime('%Y-%m-%d')
        eval_period_start = dates[self.config.pre_period_days].strftime('%Y-%m-%d')
        eval_period_end = dates[self.config.pre_period_days + self.config.eval_period_days - 1].strftime('%Y-%m-%d')
        
        # Fit model and calculate results
        reporting_method.fit(panel_data, assignment_df, pre_period_end)
        iroas_estimate = reporting_method.calculate_iroas(panel_data, eval_period_start, eval_period_end)
        iroas_lower, iroas_upper = reporting_method.confidence_interval(
            panel_data, eval_period_start, eval_period_end,
            confidence_level=self.config.confidence_level,
            n_bootstrap=self.config.n_bootstrap,
            seed=self.config.seed
        )
        
        results = {
            'panel_data': panel_data,
            'geo_features': geo_features,
            'assignment_df': assignment_df,
            'iroas_estimate': iroas_estimate,
            'iroas_ci': (iroas_lower, iroas_upper),
            'ci_width': iroas_upper - iroas_lower,
            'significant': bool((iroas_lower > 0) or (iroas_upper < 0))
        }
        
        if show_plots or save_plots:
            if save_plots:
                # Create output directory
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Balance plot
            balance_fig = self.plotter.plot_assignment_balance(
                geo_features, assignment_df, ['base_sales', 'base_spend', 'covariate']
            )
            if save_plots:
                balance_path = os.path.join(output_dir, f"balance_plot_{timestamp}.png")
                balance_fig.savefig(balance_path, dpi=300, bbox_inches='tight')
                results['balance_plot_path'] = balance_path
            if show_plots:
                plt.show()
            else:
                plt.close(balance_fig)
            
            # Time series plot
            ts_fig = self.plotter.plot_time_series(panel_data, assignment_df, 'sales')
            if save_plots:
                ts_path = os.path.join(output_dir, f"timeseries_plot_{timestamp}.png")
                ts_fig.savefig(ts_path, dpi=300, bbox_inches='tight')
                results['timeseries_plot_path'] = ts_path
            if show_plots:
                plt.show()
            else:
                plt.close(ts_fig)
        
        return results
    
    def run_full_evaluation(self, 
                          verbose: bool = True,
                          save_csv: bool = False,
                          save_plots: bool = False,
                          output_dir: str = "evaluation_results") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full evaluation across all methods and simulations.
        
        Args:
            verbose: Whether to print progress and summary
            save_csv: Whether to save results as CSV files
            save_plots: Whether to save plots as image files
            output_dir: Directory to save outputs (created if doesn't exist)
            
        Returns:
            Tuple of (detailed_results, summary_results)
        """
        if verbose:
            print(f"Running evaluation with {self.config.n_simulations} simulations...")
            print(f"Assignment methods: {list(self.assignment_methods.keys())}")
            print(f"Reporting methods: {list(self.reporting_methods.keys())}")
        
        # Set up evaluation
        data_generator = SimpleNullGenerator(self.config.to_data_config())
        evaluation_runner = EvaluationRunner(self.config.to_evaluation_config())
        
        # Run evaluation
        detailed_results = evaluation_runner.run_evaluation(
            data_generator, self.assignment_methods, self.reporting_methods
        )
        
        # Generate summary
        summary_results = evaluation_runner.summarize_results(detailed_results)
        coverage_results = evaluation_runner.get_coverage_rate(detailed_results)
        
        # Merge coverage into summary
        summary_results = summary_results.merge(
            coverage_results, 
            on=['assignment_method', 'reporting_method'],
            how='left'
        )
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(summary_results.to_string(index=False))
            print("\nNotes:")
            print("- false_positive_rate should be ~0.05 for well-calibrated methods")
            print("- coverage_rate should be ~0.95 for well-calibrated CIs")
            print("- Lower mean_ci_width indicates more precise estimates")
        
        # Save outputs if requested
        if save_csv or save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                if verbose:
                    print(f"\n📁 Created output directory: {output_dir}")
            
            if save_csv:
                # Save detailed results
                detailed_csv_path = os.path.join(output_dir, f"detailed_results_{timestamp}.csv")
                detailed_results.to_csv(detailed_csv_path, index=False)
                
                # Save summary results
                summary_csv_path = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
                summary_results.to_csv(summary_csv_path, index=False)
                
                if verbose:
                    print(f"💾 Saved CSV files:")
                    print(f"  - Detailed results: {detailed_csv_path}")
                    print(f"  - Summary results: {summary_csv_path}")
            
            if save_plots:
                # Generate and save plots
                plots_saved = self._save_evaluation_plots(
                    detailed_results, summary_results, output_dir, timestamp, verbose
                )
                
                if verbose and plots_saved:
                    print(f"📊 Saved {plots_saved} plot files to {output_dir}")
        
        return detailed_results, summary_results
    
    def _save_evaluation_plots(self, 
                              detailed_results: pd.DataFrame,
                              summary_results: pd.DataFrame,
                              output_dir: str,
                              timestamp: str,
                              verbose: bool = True) -> int:
        """
        Save evaluation plots to files.
        
        Args:
            detailed_results: Detailed results DataFrame
            summary_results: Summary results DataFrame
            output_dir: Directory to save plots
            timestamp: Timestamp string for filenames
            verbose: Whether to print progress
            
        Returns:
            Number of plots saved
        """
        plots_saved = 0
        
        try:
            # 1. Main evaluation results plot
            fig = self.plotter.plot_evaluation_results(detailed_results)
            main_plot_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.png")
            fig.savefig(main_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots_saved += 1
            
            # 2. Summary metrics bar chart
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Evaluation Summary Metrics', fontsize=16, fontweight='bold')
            
            # False Positive Rate
            summary_results.plot(x='assignment_method', y='false_positive_rate', 
                               kind='bar', ax=axes[0, 0], color='skyblue')
            axes[0, 0].set_title('False Positive Rate')
            axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target (0.05)')
            axes[0, 0].set_ylabel('Rate')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Coverage Rate
            summary_results.plot(x='assignment_method', y='coverage_rate', 
                               kind='bar', ax=axes[0, 1], color='lightgreen')
            axes[0, 1].set_title('Coverage Rate')
            axes[0, 1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target (0.95)')
            axes[0, 1].set_ylabel('Rate')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Mean CI Width
            summary_results.plot(x='assignment_method', y='mean_ci_width', 
                               kind='bar', ax=axes[1, 0], color='orange')
            axes[1, 0].set_title('Mean CI Width')
            axes[1, 0].set_ylabel('Width')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Mean iROAS (should be ~0 for null data)
            summary_results.plot(x='assignment_method', y='mean_iroas', 
                               kind='bar', ax=axes[1, 1], color='pink')
            axes[1, 1].set_title('Mean iROAS Estimate')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='True Value (0)')
            axes[1, 1].set_ylabel('iROAS')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            summary_plot_path = os.path.join(output_dir, f"summary_metrics_{timestamp}.png")
            fig.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots_saved += 1
            
            # 3. Method comparison heatmap (if multiple methods)
            if len(summary_results) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create heatmap data
                metrics = ['false_positive_rate', 'coverage_rate', 'mean_ci_width', 'mean_iroas']
                heatmap_data = summary_results.set_index('assignment_method')[metrics]
                
                # Normalize each metric for better visualization
                normalized_data = heatmap_data.copy()
                for col in metrics:
                    col_min, col_max = heatmap_data[col].min(), heatmap_data[col].max()
                    if col_max != col_min:
                        normalized_data[col] = (heatmap_data[col] - col_min) / (col_max - col_min)
                
                import seaborn as sns
                sns.heatmap(normalized_data.T, annot=heatmap_data.T, fmt='.3f', 
                           cmap='RdYlBu_r', ax=ax, cbar_kws={'label': 'Normalized Score'})
                ax.set_title('Method Performance Comparison\n(Values show original metrics)')
                ax.set_xlabel('Assignment Method')
                ax.set_ylabel('Metric')
                
                plt.tight_layout()
                heatmap_path = os.path.join(output_dir, f"method_comparison_{timestamp}.png")
                fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                plots_saved += 1
                
        except Exception as e:
            if verbose:
                print(f"⚠️ Warning: Error saving some plots: {e}")
        
        return plots_saved
    
    def plot_results(self, detailed_results: pd.DataFrame, 
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive results visualization.
        
        Args:
            detailed_results: Results from run_full_evaluation
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = self.plotter.plot_evaluation_results(detailed_results)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def quick_run(self, show_plots: bool = True, verbose: bool = True) -> Dict[str, Any]:
        """
        Convenience method for quick testing with default settings.
        
        Args:
            show_plots: Whether to show diagnostic plots
            verbose: Whether to print results
            
        Returns:
            Dictionary with all results
        """
        # Run single experiment
        single_results = self.run_single_experiment(show_plots=show_plots)
        
        if verbose:
            print(f"\nSingle Experiment Results:")
            print(f"iROAS Estimate: {single_results['iroas_estimate']:.4f}")
            print(f"95% CI: [{single_results['iroas_ci'][0]:.4f}, {single_results['iroas_ci'][1]:.4f}]")
            print(f"CI Width: {single_results['ci_width']:.4f}")
            print(f"Significant: {single_results['significant']}")
        
        # Run evaluation with fewer simulations for speed
        quick_config = self.config.update(n_simulations=20, n_bootstrap=100)
        quick_runner = ExperimentRunner(quick_config)
        detailed_results, summary_results = quick_runner.run_full_evaluation(verbose=verbose)
        
        return {
            'single_experiment': single_results,
            'detailed_results': detailed_results,
            'summary_results': summary_results
        }