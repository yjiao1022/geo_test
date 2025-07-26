"""
Enhanced data generators for realistic geo-experiment simulations.

This module provides flexible data generators that support:
- Spatial relationships and autocorrelation
- Static covariates with geographic patterns
- Effect modifiers for baseline and treatment effects
- Multiple seasonality components
- Realistic sales models with iROAS effects
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal


@dataclass
class SpatialConfig:
    """Configuration for spatial structure."""
    # Geographic bounds (can represent lat/lon or abstract coordinates)
    x_bounds: Tuple[float, float] = (0.0, 100.0)
    y_bounds: Tuple[float, float] = (0.0, 100.0)
    
    # Spatial autocorrelation parameters
    spatial_correlation_range: float = 20.0  # Distance at which correlation decays
    spatial_correlation_strength: float = 0.7  # Maximum correlation
    spatial_nugget: float = 0.1  # Independent noise component
    
    # Geographic patterns (e.g., east/west coast effects)
    enable_geographic_gradients: bool = True
    gradient_directions: List[str] = field(default_factory=lambda: ['x', 'y'])  # x=east-west, y=north-south


@dataclass 
class CovariateConfig:
    """Configuration for static covariates."""
    # Covariate definitions
    covariates: Dict[str, Dict] = field(default_factory=lambda: {
        'median_income': {
            'type': 'continuous',
            'base_mean': 50000,
            'base_std': 15000,
            'geographic_gradient': {'x': 0.3, 'y': 0.1},  # Higher on east coast, slightly north
            'affects_baseline': True,
            'affects_iroas': False,
            'baseline_coefficient': 0.1,  # Sales increase per $1000 income
        },
        'digital_penetration': {
            'type': 'continuous', 
            'base_mean': 0.7,
            'base_std': 0.15,
            'geographic_gradient': {'x': 0.2, 'y': 0.0},
            'affects_baseline': True,
            'affects_iroas': True,
            'baseline_coefficient': 5000,  # Sales increase per unit penetration
            'iroas_coefficient': 0.5,     # iROAS increase per unit penetration
        },
        'population_density': {
            'type': 'continuous',
            'base_mean': 1000,
            'base_std': 800,
            'geographic_gradient': {'x': 0.0, 'y': 0.3},  # Higher in north
            'affects_baseline': True,
            'affects_iroas': False,
            'baseline_coefficient': 2.0,
        },
        'urban_indicator': {
            'type': 'binary',
            'base_prob': 0.6,
            'geographic_gradient': {'x': 0.1, 'y': 0.2},
            'affects_baseline': True,
            'affects_iroas': True,
            'baseline_coefficient': 3000,
            'iroas_coefficient': 0.3,
        }
    })


@dataclass
class SeasonalityConfig:
    """Configuration for temporal patterns."""
    components: Dict[str, Dict] = field(default_factory=lambda: {
        'weekly': {
            'amplitude': 1000,
            'phase': 0,  # Phase shift in days
            'geo_variation': 0.2,  # How much amplitude varies by geo
        },
        'monthly': {
            'amplitude': 2000,
            'phase': 0,
            'geo_variation': 0.3,
        },
        'yearly': {
            'amplitude': 5000,
            'phase': 0,
            'geo_variation': 0.1,
        }
    })


@dataclass
class TreatmentConfig:
    """Configuration for treatment effects."""
    # Base iROAS parameters
    base_iroas_mean: float = 2.0
    base_iroas_std: float = 0.5
    
    # Spatial correlation in treatment effects
    iroas_spatial_correlation: float = 0.4
    
    # Effect modification by covariates (defined in CovariateConfig)
    enable_effect_modification: bool = True


@dataclass
class EnhancedDataConfig:
    """Comprehensive configuration for enhanced data generation."""
    # Basic parameters
    n_geos: int = 50
    n_days: int = 90
    seed: Optional[int] = None
    
    # Component configurations
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    covariates: CovariateConfig = field(default_factory=CovariateConfig)
    seasonality: SeasonalityConfig = field(default_factory=SeasonalityConfig)
    treatment: TreatmentConfig = field(default_factory=TreatmentConfig)
    
    # Base noise parameters
    base_sales_mean: float = 10000
    base_sales_std: float = 2000
    base_spend_mean: float = 5000  
    base_spend_std: float = 1000
    daily_noise_std: float = 500
    
    # Treatment simulation
    simulate_treatment_effect: bool = False  # Set to True for A/B testing
    treatment_start_day: int = 60  # Day when treatment starts


class EnhancedGeoGenerator:
    """
    Enhanced generator for realistic geo-experiment data.
    
    This generator creates synthetic data with:
    - Spatial autocorrelation and geographic patterns
    - Static covariates affecting baseline and treatment effects
    - Multiple seasonality components
    - Realistic sales models incorporating iROAS
    """
    
    def __init__(self, config: Optional[EnhancedDataConfig] = None):
        """Initialize the enhanced generator."""
        self.config = config or EnhancedDataConfig()
        
        # Storage for ground truth parameters (populated after generate())
        self.ground_truth = {
            'baseline_sales': None,
            'iroas_values': None,
            'seasonality_patterns': None,
            'geo_coordinates': None,
            'covariate_values': None,
            'expected_sales_mean': None,
            'expected_sales_std': None,
            'expected_spend_mean': None,
            'expected_spend_std': None,
        }
        
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate comprehensive geo-experiment data.
        
        Returns:
            Tuple of (panel_data, geo_features):
            - panel_data: Long-format DataFrame with all temporal data
            - geo_features: DataFrame with geo-level static features
        """
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            
        # Step 1: Generate spatial structure
        geo_coords = self._generate_spatial_coordinates()
        
        # Step 2: Generate static covariates with geographic patterns
        geo_features = self._generate_static_covariates(geo_coords)
        
        # Step 3: Generate baseline sales levels
        baseline_sales = self._generate_baseline_sales(geo_features)
        
        # Step 4: Generate iROAS values (with effect modification)
        iroas_values = self._generate_iroas_values(geo_features, geo_coords)
        
        # Step 5: Generate time series data
        panel_data = self._generate_panel_data(geo_features, baseline_sales, iroas_values)
        
        # Step 6: Store ground truth parameters for validation and visualization
        self._store_ground_truth(geo_coords, geo_features, baseline_sales, iroas_values, panel_data)
        
        return panel_data, geo_features
    
    def _generate_spatial_coordinates(self) -> pd.DataFrame:
        """Generate spatial coordinates for geos."""
        geo_ids = [f"geo_{i:03d}" for i in range(self.config.n_geos)]
        
        # Generate coordinates (uniform random or clustered)
        x_coords = np.random.uniform(
            self.config.spatial.x_bounds[0], 
            self.config.spatial.x_bounds[1], 
            self.config.n_geos
        )
        y_coords = np.random.uniform(
            self.config.spatial.y_bounds[0],
            self.config.spatial.y_bounds[1],
            self.config.n_geos
        )
        
        return pd.DataFrame({
            'geo': geo_ids,
            'xy1': x_coords,
            'xy2': y_coords
        })
    
    def _generate_static_covariates(self, geo_coords: pd.DataFrame) -> pd.DataFrame:
        """Generate static covariates with geographic patterns."""
        result = geo_coords.copy()
        
        for cov_name, cov_config in self.config.covariates.covariates.items():
            if cov_config['type'] == 'continuous':
                values = self._generate_continuous_covariate(geo_coords, cov_config)
            elif cov_config['type'] == 'binary':
                values = self._generate_binary_covariate(geo_coords, cov_config)
            else:
                raise ValueError(f"Unknown covariate type: {cov_config['type']}")
                
            result[cov_name] = values
            
        return result
    
    def _generate_continuous_covariate(self, geo_coords: pd.DataFrame, config: Dict) -> np.ndarray:
        """Generate continuous covariate with geographic gradient."""
        base_values = np.random.normal(config['base_mean'], config['base_std'], self.config.n_geos)
        
        if self.config.spatial.enable_geographic_gradients:
            # Add geographic gradients
            if 'x' in config.get('geographic_gradient', {}):
                x_normalized = (geo_coords['xy1'] - self.config.spatial.x_bounds[0]) / \
                             (self.config.spatial.x_bounds[1] - self.config.spatial.x_bounds[0])
                x_effect = config['geographic_gradient']['x'] * config['base_std'] * x_normalized
                base_values += x_effect
                
            if 'y' in config.get('geographic_gradient', {}):
                y_normalized = (geo_coords['xy2'] - self.config.spatial.y_bounds[0]) / \
                             (self.config.spatial.y_bounds[1] - self.config.spatial.y_bounds[0])  
                y_effect = config['geographic_gradient']['y'] * config['base_std'] * y_normalized
                base_values += y_effect
        
        # Add spatial autocorrelation
        if self.config.spatial.spatial_correlation_strength > 0:
            base_values = self._add_spatial_correlation(base_values, geo_coords)
            
        return base_values
    
    def _generate_binary_covariate(self, geo_coords: pd.DataFrame, config: Dict) -> np.ndarray:
        """Generate binary covariate with geographic patterns."""
        base_prob = config['base_prob']
        probs = np.full(self.config.n_geos, base_prob)
        
        if self.config.spatial.enable_geographic_gradients:
            # Modify probabilities based on geography
            if 'x' in config.get('geographic_gradient', {}):
                x_normalized = (geo_coords['xy1'] - self.config.spatial.x_bounds[0]) / \
                             (self.config.spatial.x_bounds[1] - self.config.spatial.x_bounds[0])
                probs += config['geographic_gradient']['x'] * x_normalized
                
            if 'y' in config.get('geographic_gradient', {}):
                y_normalized = (geo_coords['xy2'] - self.config.spatial.y_bounds[0]) / \
                             (self.config.spatial.y_bounds[1] - self.config.spatial.y_bounds[0])
                probs += config['geographic_gradient']['y'] * y_normalized
                
        # Ensure probabilities are in [0, 1]
        probs = np.clip(probs, 0.01, 0.99)
        
        return np.random.binomial(1, probs, self.config.n_geos)
    
    def _add_spatial_correlation(self, values: np.ndarray, geo_coords: pd.DataFrame) -> np.ndarray:
        """Add spatial autocorrelation to values."""
        coords = geo_coords[['xy1', 'xy2']].values
        distances = cdist(coords, coords)
        
        # Exponential correlation function
        correlation_matrix = self.config.spatial.spatial_correlation_strength * \
                           np.exp(-distances / self.config.spatial.spatial_correlation_range)
        
        # Add nugget effect (independent noise)
        np.fill_diagonal(correlation_matrix, 1.0)
        correlation_matrix = (1 - self.config.spatial.spatial_nugget) * correlation_matrix + \
                           self.config.spatial.spatial_nugget * np.eye(len(values))
        
        # Generate spatially correlated residuals
        try:
            residuals = multivariate_normal.rvs(mean=np.zeros(len(values)), cov=correlation_matrix)
            # Standardize and add to original values
            residuals = residuals / np.std(residuals) * np.std(values) * 0.5
            return values + residuals
        except:
            # Fallback if correlation matrix is not positive definite
            return values
    
    def _generate_baseline_sales(self, geo_features: pd.DataFrame) -> np.ndarray:
        """Generate baseline sales incorporating covariate effects."""
        base_sales = np.random.normal(
            self.config.base_sales_mean,
            self.config.base_sales_std,
            self.config.n_geos
        )
        
        # Add covariate effects
        for cov_name, cov_config in self.config.covariates.covariates.items():
            if cov_config.get('affects_baseline', False):
                coeff = cov_config.get('baseline_coefficient', 0)
                base_sales += coeff * geo_features[cov_name].values
        
        return np.maximum(base_sales, 1000)  # Ensure positive
    
    def _generate_iroas_values(self, geo_features: pd.DataFrame, geo_coords: pd.DataFrame) -> np.ndarray:
        """Generate iROAS values with effect modification."""
        base_iroas = np.random.normal(
            self.config.treatment.base_iroas_mean,
            self.config.treatment.base_iroas_std,
            self.config.n_geos
        )
        
        # Add covariate effect modification
        if self.config.treatment.enable_effect_modification:
            for cov_name, cov_config in self.config.covariates.covariates.items():
                if cov_config.get('affects_iroas', False):
                    coeff = cov_config.get('iroas_coefficient', 0)
                    base_iroas += coeff * geo_features[cov_name].values
        
        # Add spatial correlation to iROAS
        if self.config.treatment.iroas_spatial_correlation > 0:
            # Temporarily adjust config for iROAS spatial correlation
            orig_strength = self.config.spatial.spatial_correlation_strength
            self.config.spatial.spatial_correlation_strength = self.config.treatment.iroas_spatial_correlation
            base_iroas = self._add_spatial_correlation(base_iroas, geo_coords)
            self.config.spatial.spatial_correlation_strength = orig_strength
        
        return np.maximum(base_iroas, 0.1)  # Ensure positive iROAS
    
    def _generate_panel_data(self, geo_features: pd.DataFrame, baseline_sales: np.ndarray, 
                           iroas_values: np.ndarray) -> pd.DataFrame:
        """Generate complete panel dataset."""
        dates = pd.date_range("2024-01-01", periods=self.config.n_days)
        records = []
        
        for geo_idx, geo in enumerate(geo_features['geo']):
            # Generate seasonality pattern for this geo
            seasonality_pattern = self._generate_seasonality_pattern(geo_idx, dates)
            
            for day_idx, date in enumerate(dates):
                # Base spend (with some daily variation)
                base_spend = self.config.base_spend_mean + \
                           np.random.normal(0, self.config.base_spend_std)
                base_spend = max(base_spend, 100)
                
                # Seasonality effect
                seasonal_effect = seasonality_pattern[day_idx]
                
                # Treatment effect (if enabled and in treatment period)
                treatment_effect = 0
                if (self.config.simulate_treatment_effect and 
                    day_idx >= self.config.treatment_start_day):
                    # Incremental spend during treatment
                    incremental_spend = base_spend * 0.2  # 20% spend increase
                    treatment_effect = incremental_spend * iroas_values[geo_idx]
                    base_spend += incremental_spend
                
                # Final sales calculation
                sales = (baseline_sales[geo_idx] + 
                        seasonal_effect + 
                        treatment_effect +
                        np.random.normal(0, self.config.daily_noise_std))
                
                sales = max(sales, 0)  # Ensure non-negative
                
                # Create record with all geo features
                record = {
                    'geo': geo,
                    'date': date,
                    'sales': sales,
                    'spend': base_spend,
                    'seasonal_effect': seasonal_effect,
                    'treatment_effect': treatment_effect
                }
                
                # Add all covariates to the record for easy access
                for col in geo_features.columns:
                    if col not in ['geo']:
                        record[col] = geo_features.loc[geo_features['geo'] == geo, col].iloc[0]
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def _generate_seasonality_pattern(self, geo_idx: int, dates: pd.DatetimeIndex) -> np.ndarray:
        """Generate seasonality pattern for a specific geo."""
        pattern = np.zeros(len(dates))
        
        for season_name, season_config in self.config.seasonality.components.items():
            base_amplitude = season_config['amplitude']
            
            # Vary amplitude by geo
            geo_variation = season_config.get('geo_variation', 0)
            geo_multiplier = 1 + np.random.normal(0, geo_variation)
            amplitude = base_amplitude * geo_multiplier
            
            # Calculate seasonal component
            if season_name == 'weekly':
                period_days = 7
            elif season_name == 'monthly':
                period_days = 30
            elif season_name == 'yearly':
                period_days = 365
            else:
                continue
                
            phase = season_config.get('phase', 0)
            day_numbers = np.arange(len(dates))
            seasonal_component = amplitude * np.sin(2 * np.pi * (day_numbers + phase) / period_days)
            
            pattern += seasonal_component
            
        return pattern
    
    def _store_ground_truth(self, geo_coords: pd.DataFrame, geo_features: pd.DataFrame,
                           baseline_sales: np.ndarray, iroas_values: np.ndarray, 
                           panel_data: pd.DataFrame) -> None:
        """Store ground truth parameters for validation and visualization."""
        # Basic ground truth parameters
        self.ground_truth['baseline_sales'] = baseline_sales.copy()
        self.ground_truth['iroas_values'] = iroas_values.copy()
        self.ground_truth['geo_coordinates'] = geo_coords.copy()
        self.ground_truth['covariate_values'] = geo_features.copy()
        
        # Calculate expected statistical properties
        self._calculate_expected_statistics(panel_data)
        
        # Store seasonality patterns for each geo
        dates = pd.date_range("2024-01-01", periods=self.config.n_days)
        seasonality_patterns = {}
        for geo_idx in range(self.config.n_geos):
            pattern = self._generate_seasonality_pattern(geo_idx, dates)
            geo_id = f"geo_{geo_idx:03d}"
            seasonality_patterns[geo_id] = pattern
        self.ground_truth['seasonality_patterns'] = seasonality_patterns
    
    def _calculate_expected_statistics(self, panel_data: pd.DataFrame) -> None:
        """Calculate expected mean and variance for validation."""
        # Expected spend statistics (base spend + noise)
        expected_spend_mean = self.config.base_spend_mean
        expected_spend_std = np.sqrt(self.config.base_spend_std**2 + self.config.daily_noise_std**2/4)
        
        # Expected sales statistics (more complex due to multiple components)
        # Base sales contribution
        sales_base_mean = self.ground_truth['baseline_sales'].mean()
        sales_base_var = self.ground_truth['baseline_sales'].var()
        
        # Seasonality contribution (approximately zero mean, but adds variance)
        seasonality_var = 0
        for season_config in self.config.seasonality.components.values():
            amp = season_config['amplitude']
            geo_var = season_config.get('geo_variation', 0)
            # Variance of seasonal component: ~(amplitude^2)/2 + geo_variation_effect
            seasonality_var += (amp**2)/2 * (1 + geo_var**2)
        
        # Treatment effect contribution (if enabled)
        treatment_var = 0
        if self.config.simulate_treatment_effect:
            treatment_days = max(0, self.config.n_days - self.config.treatment_start_day)
            treatment_fraction = treatment_days / self.config.n_days
            # Approximate treatment effect variance
            treatment_var = (self.ground_truth['iroas_values'].var() * 
                           (self.config.base_spend_mean * 0.2)**2 * treatment_fraction)
        
        # Daily noise contribution
        noise_var = self.config.daily_noise_std**2
        
        # Total expected statistics
        expected_sales_mean = sales_base_mean  # Seasonality and noise have zero mean
        expected_sales_var = sales_base_var + seasonality_var + treatment_var + noise_var
        expected_sales_std = np.sqrt(expected_sales_var)
        
        self.ground_truth['expected_sales_mean'] = expected_sales_mean
        self.ground_truth['expected_sales_std'] = expected_sales_std
        self.ground_truth['expected_spend_mean'] = expected_spend_mean
        self.ground_truth['expected_spend_std'] = expected_spend_std
    
    def get_ground_truth(self) -> Dict:
        """Get ground truth parameters for validation and visualization."""
        if self.ground_truth['baseline_sales'] is None:
            raise ValueError("Must call generate() first to populate ground truth data")
        return self.ground_truth.copy()
    
    def validate_statistical_properties(self, panel_data: pd.DataFrame, 
                                       tolerance: float = 0.15) -> Dict[str, bool]:
        """
        Validate that generated data matches expected statistical properties.
        
        Args:
            panel_data: Generated panel data
            tolerance: Relative tolerance for mean/std validation (default: 15%)
            
        Returns:
            Dictionary with validation results for each metric
        """
        if self.ground_truth['expected_sales_mean'] is None:
            raise ValueError("Must call generate() first to populate ground truth data")
        
        validation_results = {}
        
        # Calculate actual statistics
        actual_sales_mean = panel_data['sales'].mean()
        actual_sales_std = panel_data['sales'].std()
        actual_spend_mean = panel_data['spend'].mean()
        actual_spend_std = panel_data['spend'].std()
        
        # Expected statistics
        expected_sales_mean = self.ground_truth['expected_sales_mean']
        expected_sales_std = self.ground_truth['expected_sales_std']
        expected_spend_mean = self.ground_truth['expected_spend_mean']
        expected_spend_std = self.ground_truth['expected_spend_std']
        
        # Validation checks
        validation_results['sales_mean_valid'] = (
            abs(actual_sales_mean - expected_sales_mean) / expected_sales_mean <= tolerance
        )
        validation_results['sales_std_valid'] = (
            abs(actual_sales_std - expected_sales_std) / expected_sales_std <= tolerance
        )
        validation_results['spend_mean_valid'] = (
            abs(actual_spend_mean - expected_spend_mean) / expected_spend_mean <= tolerance
        )
        validation_results['spend_std_valid'] = (
            abs(actual_spend_std - expected_spend_std) / expected_spend_std <= tolerance
        )
        
        # Store actual values for debugging
        validation_results['actual_sales_mean'] = actual_sales_mean
        validation_results['expected_sales_mean'] = expected_sales_mean
        validation_results['actual_sales_std'] = actual_sales_std
        validation_results['expected_sales_std'] = expected_sales_std
        validation_results['actual_spend_mean'] = actual_spend_mean
        validation_results['expected_spend_mean'] = expected_spend_mean
        validation_results['actual_spend_std'] = actual_spend_std
        validation_results['expected_spend_std'] = expected_spend_std
        
        return validation_results
    
    def get_summary_statistics(self) -> Dict[str, Dict]:
        """Get summary statistics of ground truth parameters."""
        if self.ground_truth['baseline_sales'] is None:
            raise ValueError("Must call generate() first to populate ground truth data")
        
        baseline_sales = self.ground_truth['baseline_sales']
        iroas_values = self.ground_truth['iroas_values']
        geo_features = self.ground_truth['covariate_values']
        
        summary = {
            'baseline_sales': {
                'mean': float(baseline_sales.mean()),
                'std': float(baseline_sales.std()),
                'min': float(baseline_sales.min()),
                'max': float(baseline_sales.max())
            },
            'iroas_values': {
                'mean': float(iroas_values.mean()),
                'std': float(iroas_values.std()),
                'min': float(iroas_values.min()),
                'max': float(iroas_values.max())
            }
        }
        
        # Add covariate summaries
        for cov_name in self.config.covariates.covariates.keys():
            if cov_name in geo_features.columns:
                values = geo_features[cov_name]
                summary[cov_name] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max())
                }
        
        return summary


# Preset configurations for common scenarios
def get_west_coast_config() -> EnhancedDataConfig:
    """Configuration simulating West Coast US geos."""
    config = EnhancedDataConfig()
    
    # Modify for West Coast characteristics
    config.covariates.covariates['median_income']['base_mean'] = 70000
    config.covariates.covariates['digital_penetration']['base_mean'] = 0.85
    config.covariates.covariates['urban_indicator']['base_prob'] = 0.8
    
    # Stronger geographic gradients
    config.spatial.enable_geographic_gradients = True
    
    return config


def get_east_coast_config() -> EnhancedDataConfig:
    """Configuration simulating East Coast US geos."""
    config = EnhancedDataConfig()
    
    # Modify for East Coast characteristics  
    config.covariates.covariates['median_income']['base_mean'] = 65000
    config.covariates.covariates['population_density']['base_mean'] = 1500
    config.covariates.covariates['urban_indicator']['base_prob'] = 0.75
    
    return config


def get_simple_enhanced_config() -> EnhancedDataConfig:
    """Simplified configuration with basic enhancements."""
    config = EnhancedDataConfig()
    
    # Keep only essential covariates
    config.covariates.covariates = {
        'median_income': config.covariates.covariates['median_income'],
        'digital_penetration': config.covariates.covariates['digital_penetration']
    }
    
    # Simpler seasonality
    config.seasonality.components = {
        'weekly': config.seasonality.components['weekly']
    }
    
    return config