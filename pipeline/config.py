"""
Configuration classes for experiment pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ExperimentConfig:
    """
    Configuration for complete experiment pipeline.
    
    This class combines all configuration needed for running
    a complete geo-experiment evaluation.
    """
    
    # Data generation parameters
    n_geos: int = 50
    n_days: int = 90
    base_sales_mean: float = 10000
    base_sales_std: float = 2000
    base_spend_mean: float = 5000
    base_spend_std: float = 1000
    daily_sales_noise: float = 500
    daily_spend_noise: float = 200
    
    # Experiment design parameters
    pre_period_days: int = 60
    eval_period_days: int = 30
    treatment_ratio: float = 0.5
    
    # Evaluation parameters
    n_simulations: int = 100
    confidence_level: float = 0.95
    n_bootstrap: int = 500
    
    # Random seed
    seed: Optional[int] = None
    
    def to_data_config(self):
        """Convert to DataConfig for data generation."""
        from data_simulation.generators import DataConfig
        return DataConfig(
            n_geos=self.n_geos,
            n_days=self.n_days,
            seed=self.seed,
            base_sales_mean=self.base_sales_mean,
            base_sales_std=self.base_sales_std,
            base_spend_mean=self.base_spend_mean,
            base_spend_std=self.base_spend_std,
            daily_sales_noise=self.daily_sales_noise,
            daily_spend_noise=self.daily_spend_noise
        )
    
    def to_evaluation_config(self):
        """Convert to EvaluationConfig for evaluation."""
        from evaluation.metrics import EvaluationConfig
        return EvaluationConfig(
            n_simulations=self.n_simulations,
            pre_period_days=self.pre_period_days,
            eval_period_days=self.eval_period_days,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            treatment_ratio=self.treatment_ratio,
            seed=self.seed
        )
    
    def update(self, **kwargs) -> 'ExperimentConfig':
        """
        Create a new config with updated parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            New ExperimentConfig instance with updated parameters
        """
        params = {
            'n_geos': self.n_geos,
            'n_days': self.n_days,
            'base_sales_mean': self.base_sales_mean,
            'base_sales_std': self.base_sales_std,
            'base_spend_mean': self.base_spend_mean,
            'base_spend_std': self.base_spend_std,
            'daily_sales_noise': self.daily_sales_noise,
            'daily_spend_noise': self.daily_spend_noise,
            'pre_period_days': self.pre_period_days,
            'eval_period_days': self.eval_period_days,
            'treatment_ratio': self.treatment_ratio,
            'n_simulations': self.n_simulations,
            'confidence_level': self.confidence_level,
            'n_bootstrap': self.n_bootstrap,
            'seed': self.seed
        }
        params.update(kwargs)
        return ExperimentConfig(**params)