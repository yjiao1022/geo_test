"""
Reporting models for geo-experiments.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Base class for all reporting models."""
    
    def __init__(self):
        self.is_fitted = False
        self.model_params = {}
    
    @abstractmethod
    def fit(self, panel_data: pd.DataFrame, assignment_df: pd.DataFrame,
            pre_period_end: str) -> 'BaseModel':
        """
        Fit the model using pre-period data.
        
        Args:
            panel_data: Long-format panel data
            assignment_df: Assignment of geos to treatment/control
            pre_period_end: End date of pre-period (exclusive)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, panel_data: pd.DataFrame, period_start: str,
                period_end: str) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual predictions for treatment group.
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of prediction period
            period_end: End date of prediction period
            
        Returns:
            Dictionary with 'sales' and 'spend' counterfactual predictions
        """
        pass
    
    @abstractmethod
    def calculate_iroas(self, panel_data: pd.DataFrame, period_start: str,
                       period_end: str) -> float:
        """
        Calculate incremental ROAS point estimate.
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of evaluation period
            period_end: End date of evaluation period
            
        Returns:
            iROAS point estimate
        """
        pass
    
    @abstractmethod
    def confidence_interval(self, panel_data: pd.DataFrame, period_start: str,
                           period_end: str, confidence_level: float = 0.95,
                           n_bootstrap: int = 500, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate confidence interval for iROAS using model-specific method.
        
        Args:
            panel_data: Long-format panel data
            period_start: Start date of evaluation period
            period_end: End date of evaluation period
            confidence_level: Confidence level (default: 0.95)
            n_bootstrap: Number of bootstrap samples
            seed: Random seed
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        pass


class MeanMatchingModel(BaseModel):
    """
    Simple mean matching model using control group average as counterfactual.
    
    This model uses the mean of the control group as the counterfactual
    for the treatment group. It's the simplest possible synthetic control method.
    """
    
    def __init__(self):
        super().__init__()
        self.control_geos = None
        self.treatment_geos = None
    
    def fit(self, panel_data: pd.DataFrame, assignment_df: pd.DataFrame,
            pre_period_end: str) -> 'MeanMatchingModel':
        """
        Fit the mean matching model.
        
        Args:
            panel_data: Panel data with columns ['geo', 'date', 'sales', 'spend']
            assignment_df: Assignment with columns ['geo', 'assignment']
            pre_period_end: End of pre-period (format: 'YYYY-MM-DD')
        """
        # Validate inputs
        required_panel_cols = {'geo', 'date', 'sales', 'spend'}
        if not required_panel_cols.issubset(panel_data.columns):
            raise ValueError(f"panel_data must contain columns: {required_panel_cols}")
        
        required_assign_cols = {'geo', 'assignment'}
        if not required_assign_cols.issubset(assignment_df.columns):
            raise ValueError(f"assignment_df must contain columns: {required_assign_cols}")
        
        # Store geo assignments
        self.control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].tolist()
        self.treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].tolist()
        
        if len(self.control_geos) == 0:
            raise ValueError("No control geos found in assignment")
        if len(self.treatment_geos) == 0:
            raise ValueError("No treatment geos found in assignment")
        
        # Store model parameters (for mean matching, there aren't really any to fit)
        pre_period_data = panel_data[panel_data['date'] < pre_period_end]
        control_pre_data = pre_period_data[pre_period_data['geo'].isin(self.control_geos)]
        
        self.model_params = {
            'pre_period_end': pre_period_end,
            'n_control_geos': len(self.control_geos),
            'n_treatment_geos': len(self.treatment_geos),
            'pre_period_control_sales_mean': control_pre_data['sales'].mean(),
            'pre_period_control_spend_mean': control_pre_data['spend'].mean()
        }
        
        self.is_fitted = True
        return self
    
    def predict(self, panel_data: pd.DataFrame, period_start: str,
                period_end: str) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual using control group mean.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get evaluation period data for control group
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask]
        control_data = eval_data[eval_data['geo'].isin(self.control_geos)]
        
        # Calculate daily means for control group
        control_daily_means = control_data.groupby('date').agg({
            'sales': 'mean',
            'spend': 'mean'
        }).reset_index()
        
        return {
            'sales': control_daily_means['sales'].values,
            'spend': control_daily_means['spend'].values,
            'dates': control_daily_means['date'].values
        }
    
    def calculate_iroas(self, panel_data: pd.DataFrame, period_start: str,
                       period_end: str) -> float:
        """
        Calculate iROAS using mean matching.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating iROAS")
        
        # Get counterfactual predictions
        counterfactual = self.predict(panel_data, period_start, period_end)
        
        # Get actual treatment group data
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask]
        treatment_data = eval_data[eval_data['geo'].isin(self.treatment_geos)]
        
        treatment_daily_means = treatment_data.groupby('date').agg({
            'sales': 'mean',
            'spend': 'mean'
        }).reset_index()
        
        # Calculate incremental effects
        delta_sales = treatment_daily_means['sales'].sum() - counterfactual['sales'].sum()
        delta_spend = treatment_daily_means['spend'].sum() - counterfactual['spend'].sum()
        
        # Calculate iROAS
        if abs(delta_spend) < 1e-10:  # Avoid division by zero
            return 0.0
        
        return delta_sales / delta_spend
    
    def confidence_interval(self, panel_data: pd.DataFrame, period_start: str,
                           period_end: str, confidence_level: float = 0.95,
                           n_bootstrap: int = 500, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Bootstrap confidence interval by resampling control geos.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating confidence interval")
        
        if seed is not None:
            np.random.seed(seed)
        
        bootstrap_iroas = []
        
        # Get evaluation period data
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask]
        
        treatment_data = eval_data[eval_data['geo'].isin(self.treatment_geos)]
        control_data = eval_data[eval_data['geo'].isin(self.control_geos)]
        
        treatment_daily_means = treatment_data.groupby('date').agg({
            'sales': 'mean',
            'spend': 'mean'
        }).reset_index()
        
        for _ in range(n_bootstrap):
            # Bootstrap sample control geos
            bootstrap_control_geos = np.random.choice(
                self.control_geos, 
                size=len(self.control_geos), 
                replace=True
            )
            
            # Calculate counterfactual with bootstrap sample
            bootstrap_control_data = control_data[control_data['geo'].isin(bootstrap_control_geos)]
            bootstrap_counterfactual = bootstrap_control_data.groupby('date').agg({
                'sales': 'mean',
                'spend': 'mean'
            }).reset_index()
            
            # Calculate bootstrap iROAS
            delta_sales = treatment_daily_means['sales'].sum() - bootstrap_counterfactual['sales'].sum()
            delta_spend = treatment_daily_means['spend'].sum() - bootstrap_counterfactual['spend'].sum()
            
            if abs(delta_spend) > 1e-10:
                bootstrap_iroas.append(delta_sales / delta_spend)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        bootstrap_iroas = np.array(bootstrap_iroas)
        lower_bound = np.percentile(bootstrap_iroas, lower_percentile)
        upper_bound = np.percentile(bootstrap_iroas, upper_percentile)
        
        return lower_bound, upper_bound