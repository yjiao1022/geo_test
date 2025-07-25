"""
Evaluation metrics and runners for geo-experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import BaseAssignment
from reporting.models import BaseModel


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runner."""
    n_simulations: int = 100
    pre_period_days: int = 60
    eval_period_days: int = 30
    confidence_level: float = 0.95
    n_bootstrap: int = 500
    treatment_ratio: float = 0.5
    seed: Optional[int] = None


@dataclass
class EvaluationResult:
    """Results from a single simulation."""
    simulation_id: int
    assignment_method: str
    reporting_method: str
    iroas_estimate: float
    iroas_lower: float
    iroas_upper: float
    ci_width: float
    significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'simulation_id': self.simulation_id,
            'assignment_method': self.assignment_method,
            'reporting_method': self.reporting_method,
            'iroas_estimate': self.iroas_estimate,
            'iroas_lower': self.iroas_lower,
            'iroas_upper': self.iroas_upper,
            'ci_width': self.ci_width,
            'significant': self.significant
        }


class EvaluationRunner:
    """
    Runner for evaluating geo-experiment methods across multiple simulations.
    
    This class runs multiple simulations to evaluate the performance of different
    combinations of assignment and reporting methods.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration. If None, uses defaults.
        """
        self.config = config or EvaluationConfig()
        self.results = []
    
    def run_evaluation(self, 
                      data_generator: SimpleNullGenerator,
                      assignment_methods: Dict[str, BaseAssignment],
                      reporting_methods: Dict[str, BaseModel]) -> pd.DataFrame:
        """
        Run full evaluation across multiple simulations.
        
        Args:
            data_generator: Data generator for simulations
            assignment_methods: Dictionary of assignment methods to evaluate
            reporting_methods: Dictionary of reporting methods to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        self.results = []
        
        # Total number of combinations
        total_combinations = (len(assignment_methods) * len(reporting_methods) * 
                            self.config.n_simulations)
        
        with tqdm(total=total_combinations, desc="Running evaluations") as pbar:
            for sim_id in range(self.config.n_simulations):
                # Generate new data for each simulation
                sim_seed = None if self.config.seed is None else self.config.seed + sim_id
                data_generator.config.seed = sim_seed
                panel_data, geo_features = data_generator.generate()
                
                # Define time periods
                dates = pd.to_datetime(panel_data['date'].unique())
                pre_period_end = dates[self.config.pre_period_days - 1]
                eval_period_start = dates[self.config.pre_period_days]
                eval_period_end = dates[self.config.pre_period_days + self.config.eval_period_days - 1]
                
                # Convert to string format for API consistency
                pre_period_end_str = pre_period_end.strftime('%Y-%m-%d')
                eval_period_start_str = eval_period_start.strftime('%Y-%m-%d')
                eval_period_end_str = eval_period_end.strftime('%Y-%m-%d')
                
                # Iterate over all method combinations
                for assign_name, assign_method in assignment_methods.items():
                    for report_name, report_method in reporting_methods.items():
                        try:
                            result = self._run_single_simulation(
                                sim_id, assign_name, report_name,
                                panel_data, geo_features,
                                assign_method, report_method,
                                pre_period_end_str, eval_period_start_str, eval_period_end_str
                            )
                            self.results.append(result)
                        except Exception as e:
                            print(f"Error in sim {sim_id}, {assign_name}+{report_name}: {e}")
                        
                        pbar.update(1)
        
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def _run_single_simulation(self, 
                              sim_id: int,
                              assign_name: str,
                              report_name: str,
                              panel_data: pd.DataFrame,
                              geo_features: pd.DataFrame,
                              assign_method: BaseAssignment,
                              report_method: BaseModel,
                              pre_period_end: str,
                              eval_period_start: str,
                              eval_period_end: str) -> EvaluationResult:
        """Run a single simulation with specific method combination."""
        
        # Assignment
        assignment_df = assign_method.assign(
            geo_features, 
            self.config.treatment_ratio,
            seed=sim_id  # Use sim_id as seed for reproducibility
        )
        
        # Fit reporting model
        report_method.fit(panel_data, assignment_df, pre_period_end)
        
        # Calculate iROAS and confidence interval
        iroas_estimate = report_method.calculate_iroas(
            panel_data, eval_period_start, eval_period_end
        )
        
        iroas_lower, iroas_upper = report_method.confidence_interval(
            panel_data, eval_period_start, eval_period_end,
            confidence_level=self.config.confidence_level,
            n_bootstrap=self.config.n_bootstrap,
            seed=sim_id
        )
        
        ci_width = iroas_upper - iroas_lower
        significant = (iroas_lower > 0) or (iroas_upper < 0)
        
        return EvaluationResult(
            simulation_id=sim_id,
            assignment_method=assign_name,
            reporting_method=report_name,
            iroas_estimate=iroas_estimate,
            iroas_lower=iroas_lower,
            iroas_upper=iroas_upper,
            ci_width=ci_width,
            significant=significant
        )
    
    def summarize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize evaluation results across simulations.
        
        Args:
            results_df: Results DataFrame from run_evaluation
            
        Returns:
            Summary DataFrame with key metrics by method combination
        """
        summary = results_df.groupby(['assignment_method', 'reporting_method']).agg({
            'iroas_estimate': ['mean', 'std'],
            'ci_width': 'mean',
            'significant': 'mean'  # False positive rate for null data
        }).round(4)
        
        # Flatten column names
        summary.columns = ['mean_iroas', 'std_iroas', 'mean_ci_width', 'false_positive_rate']
        summary = summary.reset_index()
        
        return summary
    
    def get_coverage_rate(self, results_df: pd.DataFrame, true_iroas: float = 0.0) -> pd.DataFrame:
        """
        Calculate confidence interval coverage rates.
        
        Args:
            results_df: Results DataFrame from run_evaluation
            true_iroas: True iROAS value (default: 0.0 for null simulations)
            
        Returns:
            DataFrame with coverage rates by method combination
        """
        coverage = results_df.groupby(['assignment_method', 'reporting_method']).apply(
            lambda x: ((x['iroas_lower'] <= true_iroas) & (x['iroas_upper'] >= true_iroas)).mean(),
            include_groups=False
        ).reset_index()
        coverage.columns = ['assignment_method', 'reporting_method', 'coverage_rate']
        
        return coverage