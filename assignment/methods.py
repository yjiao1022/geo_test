"""
Assignment methods for geo-experiments.
"""

import numpy as np
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod


class BaseAssignment(ABC):
    """Base class for all assignment methods."""
    
    @abstractmethod
    def assign(self, geo_features: pd.DataFrame, treatment_ratio: float = 0.5, 
               seed: Optional[int] = None) -> pd.DataFrame:
        """
        Assign geos to treatment and control groups.
        
        Args:
            geo_features: DataFrame with geo-level features
            treatment_ratio: Proportion of geos to assign to treatment
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns ['geo', 'assignment'] where assignment
            is either 'treatment' or 'control'
        """
        pass


class RandomAssignment(BaseAssignment):
    """
    Simple random assignment of geos to treatment and control.
    
    This is the simplest assignment method that randomly assigns each geo
    to treatment or control with specified probability.
    """
    
    def assign(self, geo_features: pd.DataFrame, treatment_ratio: float = 0.5,
               seed: Optional[int] = None) -> pd.DataFrame:
        """
        Randomly assign geos to treatment and control groups.
        
        Args:
            geo_features: DataFrame with geo-level features (must have 'geo' column)
            treatment_ratio: Proportion of geos to assign to treatment (default: 0.5)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns ['geo', 'assignment']
            
        Raises:
            ValueError: If geo_features doesn't have 'geo' column
        """
        if 'geo' not in geo_features.columns:
            raise ValueError("geo_features must contain 'geo' column")
        
        if not 0 < treatment_ratio < 1:
            raise ValueError("treatment_ratio must be between 0 and 1")
        
        if seed is not None:
            np.random.seed(seed)
        
        geos = geo_features['geo'].tolist()
        n_geos = len(geos)
        n_treatment = int(n_geos * treatment_ratio)
        
        # Randomly select treatment geos
        treatment_geos = np.random.choice(geos, size=n_treatment, replace=False)
        
        # Create assignment DataFrame
        assignments = []
        for geo in geos:
            assignment = 'treatment' if geo in treatment_geos else 'control'
            assignments.append({'geo': geo, 'assignment': assignment})
        
        return pd.DataFrame(assignments)
    
    def validate_assignment(self, assignment_df: pd.DataFrame, 
                          expected_treatment_ratio: float = 0.5,
                          tolerance: float = 0.1) -> bool:
        """
        Validate that assignment meets expected criteria.
        
        Args:
            assignment_df: DataFrame with assignment results
            expected_treatment_ratio: Expected proportion in treatment
            tolerance: Allowed deviation from expected ratio
            
        Returns:
            True if assignment is valid
        """
        if not {'geo', 'assignment'}.issubset(assignment_df.columns):
            return False
        
        valid_assignments = {'treatment', 'control'}
        if not set(assignment_df['assignment']).issubset(valid_assignments):
            return False
        
        # Check treatment ratio
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        total_count = len(assignment_df)
        actual_ratio = treatment_count / total_count
        
        if abs(actual_ratio - expected_treatment_ratio) > tolerance:
            return False
        
        return True