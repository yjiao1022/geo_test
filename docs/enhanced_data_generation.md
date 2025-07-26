# Enhanced Data Generation for Geo-Experiments

This document describes the enhanced data generation system for creating realistic geo-experiment simulations with spatial relationships, covariates, seasonality, and treatment effects.

## Overview

The enhanced data generator (`EnhancedGeoGenerator`) addresses limitations of the simple null generator by incorporating:

- **Spatial autocorrelation**: Similar locations have similar characteristics
- **Static covariates**: Geographic features that affect baseline sales and treatment effects
- **Geographic patterns**: East/West coast effects, urban/rural differences
- **Flexible seasonality**: Weekly, monthly, yearly patterns
- **Realistic sales model**: `sales = baseline + (spend × iROAS) + seasonality + spatial_effects + noise`

## Key Components

### 1. Spatial Structure (`SpatialConfig`)

```python
spatial_config = SpatialConfig(
    x_bounds=(0.0, 100.0),  # Geographic coordinate bounds
    y_bounds=(0.0, 100.0),
    spatial_correlation_range=20.0,      # Distance at which correlation decays
    spatial_correlation_strength=0.7,    # Maximum correlation
    enable_geographic_gradients=True     # East/west coast effects
)
```

**Features:**
- Geographic coordinates (xy1, xy2) representing location
- Exponential spatial correlation: `corr = strength × exp(-distance/range)`
- Geographic gradients for covariates (e.g., higher income on coasts)

### 2. Static Covariates (`CovariateConfig`)

Default covariates included:

| Covariate | Type | Description | Effects |
|-----------|------|-------------|---------|
| `median_income` | Continuous | Household income by geo | ✓ Baseline, ✗ iROAS |
| `digital_penetration` | Continuous | Digital adoption rate | ✓ Baseline, ✓ iROAS |
| `population_density` | Continuous | Population per sq mile | ✓ Baseline, ✗ iROAS |
| `urban_indicator` | Binary | Urban vs rural classification | ✓ Baseline, ✓ iROAS |

**Customization Example:**
```python
config.covariates.covariates['custom_metric'] = {
    'type': 'continuous',
    'base_mean': 100,
    'base_std': 25,
    'geographic_gradient': {'x': 0.5, 'y': 0.1},  # Higher in east and north
    'affects_baseline': True,
    'affects_iroas': True,
    'baseline_coefficient': 10,    # Sales increase per unit
    'iroas_coefficient': 0.2,      # iROAS increase per unit
}
```

### 3. Seasonality (`SeasonalityConfig`)

Multiple seasonal components with geo-specific variation:

```python
seasonality_config = SeasonalityConfig(
    components={
        'weekly': {
            'amplitude': 1000,     # Base effect size
            'phase': 0,            # Phase shift in days
            'geo_variation': 0.2,  # How much amplitude varies by geo
        },
        'monthly': {'amplitude': 2000, 'phase': 0, 'geo_variation': 0.3},
        'yearly': {'amplitude': 5000, 'phase': 0, 'geo_variation': 0.1}
    }
)
```

### 4. Treatment Effects (`TreatmentConfig`)

```python
treatment_config = TreatmentConfig(
    base_iroas_mean=2.0,                    # Average iROAS across geos
    base_iroas_std=0.5,                     # iROAS variation
    iroas_spatial_correlation=0.4,          # Spatial correlation in treatment effects
    enable_effect_modification=True         # Allow covariates to modify iROAS
)
```

## Usage Examples

### Basic Usage

```python
from data_simulation.enhanced_generators import EnhancedGeoGenerator, EnhancedDataConfig

# Create generator with default config
generator = EnhancedGeoGenerator()
panel_data, geo_features = generator.generate()

print(f"Panel data shape: {panel_data.shape}")
print(f"Geo features: {list(geo_features.columns)}")
```

### A/B Testing Simulation

```python
config = EnhancedDataConfig(
    n_geos=30,
    n_days=90,
    simulate_treatment_effect=True,
    treatment_start_day=60,  # Treatment starts on day 60
    seed=123
)

generator = EnhancedGeoGenerator(config)
panel_data, geo_features = generator.generate()

# Analyze treatment effect
pre_period = panel_data[panel_data['date'] < '2024-03-01']
post_period = panel_data[panel_data['date'] >= '2024-03-01']
lift = post_period.groupby('geo')['sales'].mean() - pre_period.groupby('geo')['sales'].mean()
print(f"Average lift: {lift.mean():.2f}")
```

### Custom Configuration

```python
config = EnhancedDataConfig(n_geos=25, n_days=120, seed=456)

# Add custom covariate
config.covariates.covariates['competitor_density'] = {
    'type': 'continuous',
    'base_mean': 5.0,
    'base_std': 2.0,
    'geographic_gradient': {'x': -0.2},  # Fewer competitors in west
    'affects_baseline': True,
    'affects_iroas': True,
    'baseline_coefficient': -500,   # Negative effect on sales
    'iroas_coefficient': -0.1,      # Reduces treatment effectiveness
}

# Stronger spatial correlation
config.spatial.spatial_correlation_strength = 0.8

generator = EnhancedGeoGenerator(config)
panel_data, geo_features = generator.generate()
```

### Preset Configurations

```python
from data_simulation.enhanced_generators import get_west_coast_config, get_simple_enhanced_config

# West Coast scenario (higher income, digital penetration)
west_config = get_west_coast_config()
west_generator = EnhancedGeoGenerator(west_config)

# Simplified configuration (fewer covariates)
simple_config = get_simple_enhanced_config() 
simple_generator = EnhancedGeoGenerator(simple_config)
```

## Integration with Existing Framework

The enhanced generator produces data compatible with existing assignment and reporting methods:

```python
from assignment.methods import RandomAssignment, KMeansEmbeddingAssignment
from reporting.models import MeanMatchingModel

# Generate data
generator = EnhancedGeoGenerator(EnhancedDataConfig(n_geos=20, seed=123))
panel_data, geo_features = generator.generate()

# Use with assignment methods
random_assign = RandomAssignment()
assignment_df = random_assign.assign(geo_features, treatment_ratio=0.5, seed=123)

# Use with reporting models
model = MeanMatchingModel()
model.fit(panel_data, assignment_df, '2024-02-01')
iroas = model.calculate_iroas(panel_data, '2024-02-02', '2024-03-01')
```

## Data Structure

### Panel Data Output
Long-format DataFrame with columns:
- `geo`: Geographic unit identifier
- `date`: Time dimension
- `sales`: Outcome variable (incorporating all effects)
- `spend`: Media spend variable
- `seasonal_effect`: Seasonal component (for analysis)
- `treatment_effect`: Treatment effect component
- All covariates (replicated from geo_features for convenience)

### Geo Features Output
Cross-sectional DataFrame with columns:
- `geo`: Geographic unit identifier
- `xy1`, `xy2`: Spatial coordinates
- All static covariates (income, digital_penetration, etc.)

## Sales Model Details

The underlying sales model is:

```
sales_it = baseline_sales_i + (spend_it × iROAS_i) + seasonality_it + spatial_effect_it + noise_it
```

Where:
- `baseline_sales_i` = f(base_sales, covariates) varies by geo
- `iROAS_i` = f(base_iROAS, effect_modifying_covariates) varies by geo  
- `seasonality_it` = Σ amplitude_k × sin(2π × (t + phase_k) / period_k)
- `spatial_effect_it` incorporated through correlated baseline and iROAS values
- `noise_it` ~ N(0, σ²)

## Validation and Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/test_enhanced_generators.py -v
```

See examples in action:

```bash
cd examples
python enhanced_data_generation_example.py
```

## Performance Considerations

- **Small datasets** (≤50 geos, ≤90 days): Fast generation (~1 second)
- **Medium datasets** (≤200 geos, ≤365 days): Moderate generation (~10 seconds)
- **Large datasets** (>500 geos, >365 days): May take longer due to spatial correlation calculations

For large datasets, consider:
- Reducing `spatial_correlation_strength` to speed up calculations
- Using fewer covariates
- Disabling `enable_geographic_gradients` if not needed

## Extending the System

### Adding New Covariate Types

```python
def _generate_categorical_covariate(self, geo_coords: pd.DataFrame, config: Dict) -> np.ndarray:
    \"\"\"Add support for categorical covariates.\"\"\"
    categories = config['categories']  # e.g., ['A', 'B', 'C']
    probs = config['probabilities']    # e.g., [0.3, 0.5, 0.2]
    return np.random.choice(categories, size=self.config.n_geos, p=probs)
```

### Adding New Seasonality Types

```python
# In seasonality config
'holiday': {
    'amplitude': 3000,
    'peak_days': [25, 359],  # Christmas, New Year's Eve
    'duration': 3,           # Days around peak
    'geo_variation': 0.5
}
```

### Custom Spatial Correlation Functions

```python
def custom_correlation_function(distances):
    \"\"\"Custom spatial correlation function.\"\"\"
    return np.exp(-0.5 * (distances / range_param)**2)  # Gaussian instead of exponential
```

This enhanced data generation system provides the flexibility to create realistic geo-experiment data for thorough method evaluation and comparison.