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
    uncertainty_method: str = 'percentile'  # 'percentile', 'bca', 'jackknife', 'conformal'
    aa_mode: bool = True  # If True, track component metrics; if False, traditional iROAS


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
    half_ci_width: float
    significant: bool
    # Component metrics for A/A diagnostics
    incremental_sales: float = 0.0
    incremental_spend: float = 0.0
    sales_lower: float = 0.0
    sales_upper: float = 0.0
    spend_lower: float = 0.0
    spend_upper: float = 0.0
    sales_significant: bool = False
    spend_significant: bool = False
    
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
            'half_ci_width': self.half_ci_width,
            'significant': self.significant,
            'incremental_sales': self.incremental_sales,
            'incremental_spend': self.incremental_spend,
            'sales_lower': self.sales_lower,
            'sales_upper': self.sales_upper,
            'spend_lower': self.spend_lower,
            'spend_upper': self.spend_upper,
            'sales_significant': self.sales_significant,
            'spend_significant': self.spend_significant
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
        
        # Calculate metrics based on mode
        if self.config.aa_mode:
            # A/A mode: track component metrics separately
            incremental_sales, incremental_spend = self._calculate_component_metrics(
                report_method, panel_data, assignment_df, eval_period_start, eval_period_end
            )
            
            
            # Calculate component confidence intervals
            from reporting.conformal_utils import calculate_component_confidence_intervals
            
            component_cis = calculate_component_confidence_intervals(
                report_method, panel_data, assignment_df, pre_period_end,
                eval_period_start, eval_period_end, 
                confidence_level=self.config.confidence_level,
                uncertainty_method=self.config.uncertainty_method
            )
            
            sales_lower, sales_upper = component_cis['sales']
            spend_lower, spend_upper = component_cis['spend']
            
            # Significance based on components
            sales_significant = (sales_lower > 0) or (sales_upper < 0)
            spend_significant = (spend_lower > 0) or (spend_upper < 0)
            significant = sales_significant  # Use sales significance for overall
