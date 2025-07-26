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


class GBRModel(BaseModel):
    """
    Geographical-Based Regression model (GBR).

    This model fits a cross-sectional regression on data aggregated at the geo
    level. The iROAS is estimated directly as a coefficient in the model,
    following the methodology from Google's GeoexperimentsResearch package.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize the model.

        Args:
            alpha: Regularization strength for Ridge regression.
        """
        super().__init__()
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha)

    def fit(self, panel_data: pd.DataFrame, assignment_df: pd.DataFrame,
            pre_period_end: str) -> 'GBRModel':
        """
        Fit the geo-based regression model.
        """
        # 1. Aggregate data to geo-level
        pre_data = panel_data[panel_data['date'] < pre_period_end]
        post_data = panel_data[panel_data['date'] >= pre_period_end]

        geo_agg = pre_data.groupby('geo').agg(pre_sales=('sales', 'sum'))
        post_agg = post_data.groupby('geo').agg(
            post_sales=('sales', 'sum'),
            post_spend=('spend', 'sum')
        )
        
        df = geo_agg.join(post_agg, how='outer').fillna(0).reset_index()
        df = df.merge(assignment_df, on='geo')
        df['is_treatment'] = (df['assignment'] == 'treatment').astype(int)

        # 2. Create interaction term for iROAS
        df['treatment_spend'] = df['is_treatment'] * df['post_spend']

        # 3. Fit regression
        # Potential improvement: The original R code uses lm.cluster for robust
        # standard errors. For a more faithful implementation, using statsmodels
        # to allow for clustered standard errors by geo would be an improvement.
        X = df[['pre_sales', 'post_spend', 'treatment_spend']]
        y = df['post_sales']
        
        self.model.fit(X, y)
        self.is_fitted = True

        # 4. Store coefficients and data for CI calculation
        self.model_params['iroas_estimate'] = self.model.coef_[2] # Coeff of treatment_spend
        self.model_params['X'] = X
        self.model_params['y'] = y
        
        return self

    def calculate_iroas(self, panel_data: pd.DataFrame, period_start: str,
                       period_end: str) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating iROAS.")
        return self.model_params['iroas_estimate']

    def confidence_interval(self, panel_data: pd.DataFrame, period_start: str,
                           period_end: str, confidence_level: float = 0.95,
                           n_bootstrap: int = 500, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate CI for the iROAS coefficient using bootstrap on the geos.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating CI.")
        
        if seed is not None:
            np.random.seed(seed)

        X = self.model_params['X']
        y = self.model_params['y']
        
        bootstrap_iroas = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]
            
            # Refit the model on the bootstrap sample
            model_sample = self.model.fit(X_sample, y_sample)
            bootstrap_iroas.append(model_sample.coef_[2])

        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_iroas, (alpha / 2) * 100)
        upper_bound = np.percentile(bootstrap_iroas, (1 - alpha / 2) * 100)
        return lower_bound, upper_bound

    def predict(self, panel_data: pd.DataFrame, period_start: str,
                period_end: str) -> Dict[str, np.ndarray]:
        """
        Predict is not used for iROAS calculation in this model.
        This method is for API consistency.
        """
        return {'sales': np.array([]), 'spend': np.array([]), 'dates': np.array([])}


class TBRModel(BaseModel):
    """
    Time-Based Regression model (TBR).

    This model is a form of Difference-in-Differences (DiD). It models the
    difference between the treatment and control groups over time.
    """

    def fit(self, panel_data: pd.DataFrame, assignment_df: pd.DataFrame,
            pre_period_end: str) -> 'TBRModel':
        """
        Calculate the mean and variance of the difference in the pre-period.
        """
        control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].tolist()
        treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].tolist()

        if not control_geos or not treatment_geos:
            raise ValueError("Both control and treatment groups must have at least one geo.")

        pre_data = panel_data[panel_data['date'] < pre_period_end].copy()
        
        control_sales = pre_data[pre_data['geo'].isin(control_geos)].groupby('date')['sales'].mean()
        treatment_sales = pre_data[pre_data['geo'].isin(treatment_geos)].groupby('date')['sales'].mean()
        
        # Align series and fill missing dates with 0, then calculate difference
        pre_period_diff = treatment_sales.subtract(control_sales, fill_value=0)
        
        self.model_params = {
            'pre_period_diff_mean': pre_period_diff.mean(),
            'pre_period_diff_var': pre_period_diff.var(),
            'control_geos': control_geos,
            'treatment_geos': treatment_geos
        }
        self.is_fitted = True
        return self

    def _get_incremental_sales_and_spend(self, panel_data: pd.DataFrame, period_start: str, period_end: str) -> Tuple[float, float]:
        """
        Helper to calculate incremental sales and spend.
        """
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask]

        control_daily = eval_data[eval_data['geo'].isin(self.model_params['control_geos'])].groupby('date').mean(numeric_only=True)
        treatment_daily = eval_data[eval_data['geo'].isin(self.model_params['treatment_geos'])].groupby('date').mean(numeric_only=True)

        # Counterfactual sales = control sales + avg pre-period difference
        counterfactual_sales_sum = (control_daily['sales'] + self.model_params['pre_period_diff_mean']).sum()
        actual_sales_sum = treatment_daily['sales'].sum()

        delta_sales = actual_sales_sum - counterfactual_sales_sum
        delta_spend = treatment_daily['spend'].sum() - control_daily['spend'].sum()
        
        return delta_sales, delta_spend

    def calculate_iroas(self, panel_data: pd.DataFrame, period_start: str,
                       period_end: str) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating iROAS.")

        delta_sales, delta_spend = self._get_incremental_sales_and_spend(panel_data, period_start, period_end)

        if abs(delta_spend) < 1e-10:
            return 0.0
        return delta_sales / delta_spend

    def confidence_interval(self, panel_data: pd.DataFrame, period_start: str,
                           period_end: str, confidence_level: float = 0.95,
                           n_bootstrap: int = 500, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate CI based on the variance of the pre-period difference.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating CI.")

        from scipy.stats import norm
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        n_days = panel_data[period_mask]['date'].nunique()

        delta_sales, delta_spend = self._get_incremental_sales_and_spend(panel_data, period_start, period_end)

        if abs(delta_spend) < 1e-10:
            return 0.0, 0.0

        # Variance of the lift estimate
        # Potential improvement: The R code includes a correction for the variance
        # if the pre- and post-periods have different lengths. This implementation
        # assumes they are comparable.
        lift_variance = (self.model_params['pre_period_diff_var']) * n_days
        lift_std_err = np.sqrt(lift_variance)

        # Z-score for confidence interval
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        # CI for the lift
        lower_lift = delta_sales - z_score * lift_std_err
        upper_lift = delta_sales + z_score * lift_std_err

        # Convert to iROAS CI
        lower_bound = lower_lift / delta_spend
        upper_bound = upper_lift / delta_spend

        # Handle case where spend is negative
        if delta_spend < 0:
            return upper_bound, lower_bound
        return lower_bound, upper_bound

    def predict(self, panel_data: pd.DataFrame, period_start: str,
                period_end: str) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual by adding the pre-period difference to the control group.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask].copy()
        
        control_daily = eval_data[eval_data['geo'].isin(self.model_params['control_geos'])].groupby('date').mean(numeric_only=True)
        
        predicted_sales = control_daily['sales'] + self.model_params['pre_period_diff_mean']

        return {
            'sales': predicted_sales.values,
            'spend': control_daily['spend'].values,
            'dates': control_daily.index.values
        }


class SyntheticControlModel(BaseModel):
    """
    Synthetic Control Model (SCM).

    This model creates a synthetic control group by finding an optimal weighted
    average of the control geos that best reproduces the pre-treatment trend
    of the treatment group.
    """

    def fit(self, panel_data: pd.DataFrame, assignment_df: pd.DataFrame,
            pre_period_end: str) -> 'SyntheticControlModel':
        """
        Find the optimal weights for the synthetic control.
        """
        from scipy.optimize import minimize

        control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].tolist()
        treatment_geos = assignment_df[assignment_df['assignment'] == 'treatment']['geo'].tolist()

        if not control_geos or not treatment_geos:
            raise ValueError("Both control and treatment groups must have at least one geo.")

        pre_data = panel_data[panel_data['date'] < pre_period_end].copy()
        sales_pivot = pre_data.pivot(index='date', columns='geo', values='sales').fillna(0)

        X_pre = sales_pivot[control_geos].values
        y_pre = sales_pivot[treatment_geos].mean(axis=1).values

        def loss_function(weights):
            return np.sqrt(np.mean((y_pre - X_pre.dot(weights))**2))

        # Initial guess: equal weights
        w0 = np.ones(len(control_geos)) / len(control_geos)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0, 1) for _ in range(len(control_geos)))

        # Find optimal weights
        result = minimize(loss_function, w0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.model_params = {
            'weights': result.x,
            'control_geos': control_geos,
            'treatment_geos': treatment_geos
        }
        self.is_fitted = True
        return self

    def predict(self, panel_data: pd.DataFrame, period_start: str,
                period_end: str) -> Dict[str, np.ndarray]:
        """
        Generate counterfactual predictions using the synthetic control.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask].copy()
        sales_pivot = eval_data.pivot(index='date', columns='geo', values='sales').fillna(0)

        X_eval = sales_pivot[self.model_params['control_geos']].values
        predicted_sales = X_eval.dot(self.model_params['weights'])

        # For spend, we use a simple mean of the control group as a counterfactual
        spend_pivot = eval_data.pivot(index='date', columns='geo', values='spend').fillna(0)
        control_spend = spend_pivot[self.model_params['control_geos']].mean(axis=1)

        return {
            'sales': predicted_sales,
            'spend': control_spend.values,
            'dates': sales_pivot.index.values
        }

    def calculate_iroas(self, panel_data: pd.DataFrame, period_start: str,
                       period_end: str) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating iROAS.")

        counterfactual = self.predict(panel_data, period_start, period_end)
        
        period_mask = (panel_data['date'] >= period_start) & (panel_data['date'] <= period_end)
        eval_data = panel_data[period_mask]
        treatment_data = eval_data[eval_data['geo'].isin(self.model_params['treatment_geos'])]
        
        actual_sales = treatment_data.groupby('date')['sales'].mean().sum()
        actual_spend = treatment_data.groupby('date')['spend'].mean().sum()
        
        counterfactual_sales = counterfactual['sales'].sum()
        counterfactual_spend = counterfactual['spend'].sum()

        delta_sales = actual_sales - counterfactual_sales
        delta_spend = actual_spend - counterfactual_spend

        if abs(delta_spend) < 1e-10:
            return 0.0
        return delta_sales / delta_spend

    def confidence_interval(self, panel_data: pd.DataFrame, period_start: str,
                           period_end: str, confidence_level: float = 0.95,
                           n_bootstrap: int = 500, seed: Optional[int] = None) -> Tuple[float, float]:
        """
        Calculate CI using placebo tests on control geos.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating CI.")

        if seed is not None:
            np.random.seed(seed)

        # Placebo test: iterate through control geos, treat one as a pseudo-treatment group
        placebo_lifts = []
        control_geos = self.model_params['control_geos']
        
        for geo in control_geos:
            # Define pseudo-treatment and pseudo-control groups
            pseudo_treat_geos = [geo]
            pseudo_control_geos = [g for g in control_geos if g != geo]

            if not pseudo_control_geos:
                continue

            # Refit the model on the placebo groups
            placebo_assignment = pd.DataFrame({
                'geo': pseudo_treat_geos + pseudo_control_geos,
                'assignment': ['treatment'] * len(pseudo_treat_geos) + ['control'] * len(pseudo_control_geos)
            })
            
            try:
                placebo_model = SyntheticControlModel()
                placebo_model.fit(panel_data, placebo_assignment, period_start)
                lift = placebo_model.calculate_iroas(panel_data, period_start, period_end)
                placebo_lifts.append(lift)
            except Exception:
                continue # Skip if model fails for a placebo

        if not placebo_lifts:
            return np.nan, np.nan # Cannot compute CI

        # Calculate the CI from the distribution of placebo lifts
        point_estimate = self.calculate_iroas(panel_data, period_start, period_end)
        lift_variance = np.var(placebo_lifts)
        
        from scipy.stats import norm
        alpha = 1 - confidence_level
        z_score = norm.ppf(1 - alpha / 2)

        lower_bound = point_estimate - z_score * np.sqrt(lift_variance)
        upper_bound = point_estimate + z_score * np.sqrt(lift_variance)

        return lower_bound, upper_bound
