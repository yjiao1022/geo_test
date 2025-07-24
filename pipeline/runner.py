"""
Main experiment runner for geo-experiments.
"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, Any

from data_simulation.generators import SimpleNullGenerator
from assignment.methods import BaseAssignment, RandomAssignment
from reporting.models import BaseModel, MeanMatchingModel
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
            'MeanMatching': MeanMatchingModel()
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
    
    def run_single_experiment(self, show_plots: bool = True) -> Dict[str, Any]:
        """
        Run a single experiment for demonstration/debugging.
        
        Args:
            show_plots: Whether to display diagnostic plots
            
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
            'significant': (iroas_lower > 0) or (iroas_upper < 0)
        }
        
        if show_plots:
            # Balance plot
            balance_fig = self.plotter.plot_assignment_balance(
                geo_features, assignment_df, ['base_sales', 'base_spend', 'covariate']
            )
            plt.show()
            
            # Time series plot
            ts_fig = self.plotter.plot_time_series(panel_data, assignment_df, 'sales')
            plt.show()
        
        return results
    
    def run_full_evaluation(self, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run full evaluation across all methods and simulations.
        
        Args:
            verbose: Whether to print progress and summary
            
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
        
        return detailed_results, summary_results
    
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