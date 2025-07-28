"""
Data generators for geo-experiment simulations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data generation."""
    n_geos: int = 50
    n_days: int = 90
    seed: Optional[int] = None
    base_sales_mean: float = 10000
    base_sales_std: float = 2000
    base_spend_mean: float = 5000
    base_spend_std: float = 1000
    daily_sales_noise: float = 500
    daily_spend_noise: float = 200


class SimpleNullGenerator:
    """
    Simple data generator for A/A testing (null effect).
    
    Generates synthetic geo-experiment data with:
    - No treatment effect (iROAS = 0)
    - Independent normal distributions for sales and spend
    - Optional covariates that don't affect the outcome
    - No spatial or temporal correlations
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the generator.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or DataConfig()
    
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic geo-experiment data.
        
        Returns:
            Tuple of (panel_data, geo_features):
            - panel_data: Long-format DataFrame with columns 
              ['geo', 'date', 'sales', 'spend', 'covariate']
            - geo_features: DataFrame with geo-level features
              ['geo', 'base_sales', 'base_spend', 'covariate']
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Generate geo identifiers and dates
        geo_ids = [f"geo_{i:03d}" for i in range(self.config.n_geos)]
        dates = pd.date_range("2024-01-01", periods=self.config.n_days)
        
        # Generate geo-level base metrics
        base_sales = np.random.normal(
            self.config.base_sales_mean, 
            self.config.base_sales_std, 
            self.config.n_geos
        )
        base_spend = np.random.normal(
            self.config.base_spend_mean,
            self.config.base_spend_std,
            self.config.n_geos
        )
        
        # Ensure positive values
        base_sales = np.maximum(base_sales, 1000)
        base_spend = np.maximum(base_spend, 100)
        
        # Generate independent covariate (not affecting outcomes)
        covariate = np.random.normal(0, 1, self.config.n_geos)
        
        # Create panel data
        records = []
        for geo_idx, geo in enumerate(geo_ids):
            for date in dates:
                # Add daily noise to base levels
                sales = base_sales[geo_idx] + np.random.normal(0, self.config.daily_sales_noise)
                spend = base_spend[geo_idx] + np.random.normal(0, self.config.daily_spend_noise)
                
                # Ensure non-negative values
                sales = max(sales, 0)
                spend = max(spend, 0)
                
                records.append({
                    "geo": geo,
                    "date": date,
                    "sales": sales,
                    "spend": spend,
                    "covariate": covariate[geo_idx]
                })
        
        panel_data = pd.DataFrame(records)
        
        # Create geo features table
        geo_features = pd.DataFrame({
            "geo": geo_ids,
            "base_sales": base_sales,
            "base_spend": base_spend,
            "covariate": covariate
        })
        
        return panel_data, geo_features
    
    def validate_data(self, panel_data: pd.DataFrame, geo_features: pd.DataFrame) -> bool:
        """
        Validate generated data meets expected properties.
        
        Args:
            panel_data: Generated panel data
            geo_features: Generated geo features
            
        Returns:
            True if data passes validation
        """
        # Check basic structure
        expected_panel_cols = {"geo", "date", "sales", "spend", "covariate"}
        expected_geo_cols = {"geo", "base_sales", "base_spend", "covariate"}
        
        if not expected_panel_cols.issubset(panel_data.columns):
            return False
        
        if not expected_geo_cols.issubset(geo_features.columns):
            return False
        
        # Check dimensions
        expected_panel_rows = self.config.n_geos * self.config.n_days
        if len(panel_data) != expected_panel_rows:
            return False
        
        if len(geo_features) != self.config.n_geos:
            return False
        
        # Check for non-negative values
        if (panel_data["sales"] < 0).any() or (panel_data["spend"] < 0).any():
            return False
        
        return True


class IdenticalGeoGenerator:
    """
    Even simpler data generator for A/A testing with identical geo baselines.
    
    Generates synthetic geo-experiment data with:
    - No treatment effect (iROAS = 0)
    - Identical baseline sales/spend across all geos
    - Only daily random noise varies across days
    - No spatial, temporal, or geo-level correlations
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize the generator.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or DataConfig()
    
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic geo-experiment data with identical geo baselines.
        
        Returns:
            Tuple of (panel_data, geo_features):
            - panel_data: Long-format DataFrame with columns 
              ['geo', 'date', 'sales', 'spend', 'covariate']
            - geo_features: DataFrame with geo-level features
              ['geo', 'base_sales', 'base_spend', 'covariate']
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
        
        # Create geo identifiers
        geo_ids = [f"geo_{i:03d}" for i in range(self.config.n_geos)]
        
        # Generate date range
        dates = pd.date_range(start="2023-01-01", periods=self.config.n_days, freq="D")
        
        # All geos have identical base metrics
        base_sales = np.full(self.config.n_geos, self.config.base_sales_mean)
        base_spend = np.full(self.config.n_geos, self.config.base_spend_mean)
        covariate = np.full(self.config.n_geos, 1.0)  # Identical covariates
        
        # Generate panel data
        records = []
        for date in dates:
            for geo_idx, geo in enumerate(geo_ids):
                # Sales and spend with daily noise only
                sales = base_sales[geo_idx] + np.random.normal(0, self.config.daily_sales_noise)
                spend = base_spend[geo_idx] + np.random.normal(0, self.config.daily_spend_noise)
                
                # Ensure non-negative values
                sales = max(sales, 0)
                spend = max(spend, 0)
                
                records.append({
                    "geo": geo,
                    "date": date,
                    "sales": sales,
                    "spend": spend,
                    "covariate": covariate[geo_idx]
                })
        
        panel_data = pd.DataFrame(records)
        
        # Create geo features table
        geo_features = pd.DataFrame({
            "geo": geo_ids,
            "base_sales": base_sales,
            "base_spend": base_spend,
            "covariate": covariate
        })
        
        return panel_data, geo_features
    
    def validate_data(self, panel_data: pd.DataFrame, geo_features: pd.DataFrame) -> bool:
        """
        Validate generated data meets expected properties.
        
        Args:
            panel_data: Generated panel data
            geo_features: Generated geo features
            
        Returns:
            True if data passes validation
        """
        # Check basic structure
        expected_panel_cols = {"geo", "date", "sales", "spend", "covariate"}
        expected_geo_cols = {"geo", "base_sales", "base_spend", "covariate"}
        
        if not expected_panel_cols.issubset(panel_data.columns):
            return False
        
        if not expected_geo_cols.issubset(geo_features.columns):
            return False
        
        # Check dimensions
        expected_panel_rows = self.config.n_geos * self.config.n_days
        if len(panel_data) != expected_panel_rows:
            return False
        
        if len(geo_features) != self.config.n_geos:
            return False
        
        # Check for non-negative values
        if (panel_data["sales"] < 0).any() or (panel_data["spend"] < 0).any():
            return False
        
        # Check that all geos have identical baselines
        unique_base_sales = geo_features["base_sales"].nunique()
        unique_base_spend = geo_features["base_spend"].nunique()
        if unique_base_sales != 1 or unique_base_spend != 1:
            return False
        
        return True