# Geo-Experiment Evaluation Framework

A modular Python framework for evaluating geo-experiment methodologies using synthetic control methods.

## Overview

This framework provides a comprehensive toolkit for:
- **Data Simulation**: Generate synthetic geo-experiment data with configurable parameters
- **Assignment Methods**: Various strategies for assigning geos to treatment/control groups  
- **Reporting Models**: Different approaches for counterfactual estimation and effect measurement
- **Evaluation**: Statistical evaluation across multiple simulations with false positive rate analysis
- **Visualization**: Diagnostic plots and results visualization

## Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from pipeline.runner import ExperimentRunner
from pipeline.config import ExperimentConfig

# Create configuration
config = ExperimentConfig(
    n_geos=50,
    n_days=90,
    pre_period_days=60,
    eval_period_days=30,
    seed=42
)

# Run single experiment
runner = ExperimentRunner(config)
results = runner.run_single_experiment(show_plots=True)

print(f"iROAS Estimate: {results['iroas_estimate']:.4f}")
print(f"95% CI: [{results['iroas_ci'][0]:.4f}, {results['iroas_ci'][1]:.4f}]")
```

### Full Evaluation

```python
# Run evaluation across multiple simulations
detailed_results, summary_results = runner.run_full_evaluation()

# View summary
print(summary_results)

# Plot results
fig = runner.plot_results(detailed_results)
```

## Architecture

### Core Modules

- **`data_simulation/`**: Synthetic data generation
  - `SimpleNullGenerator`: A/A testing data with null effects
  
- **`assignment/`**: Geo assignment methods
  - `RandomAssignment`: Simple random assignment
  
- **`reporting/`**: Counterfactual models
  - `MeanMatchingModel`: Control group mean as counterfactual
  
- **`evaluation/`**: Cross-simulation evaluation
  - `EvaluationRunner`: Orchestrates multiple simulations
  
- **`diagnostics/`**: Visualization and diagnostics
  - `DiagnosticPlotter`: Standard plots for geo-experiments
  
- **`pipeline/`**: High-level orchestration
  - `ExperimentRunner`: End-to-end experiment execution

### Key Features

#### Two-Level Inference
- **Model-Level**: Each reporting model provides its own confidence intervals
- **Evaluation-Level**: Cross-simulation statistical analysis for method comparison

#### Modular Design
- Easy to add new assignment methods, reporting models, or data generators
- Consistent interfaces across all components
- Comprehensive unit tests

#### Configuration-Driven
- All parameters controlled through config classes
- Easy to run parameter sweeps and sensitivity analyses

## Interactive Playground

Use `playground.ipynb` for interactive experimentation:

```bash
jupyter notebook playground.ipynb
```

The notebook includes:
- Quick start examples
- Parameter exploration
- Method comparison
- Template for custom experiments

## Testing

Run basic functionality tests:

```bash
python test_basic.py
```

Run full unit test suite (requires pytest):

```bash
pytest tests/ -v
```

## Extending the Framework

### Adding New Assignment Methods

```python
from assignment.methods import BaseAssignment

class MyAssignmentMethod(BaseAssignment):
    def assign(self, geo_features, treatment_ratio=0.5, seed=None):
        # Your implementation here
        return assignment_df
```

### Adding New Reporting Models

```python
from reporting.models import BaseModel

class MyReportingModel(BaseModel):
    def fit(self, panel_data, assignment_df, pre_period_end):
        # Fit your model
        pass
    
    def predict(self, panel_data, period_start, period_end):
        # Generate counterfactual
        pass
    
    def calculate_iroas(self, panel_data, period_start, period_end):
        # Calculate point estimate
        pass
    
    def confidence_interval(self, panel_data, period_start, period_end, **kwargs):
        # Calculate confidence interval
        pass
```

## Current Limitations

This is a simplified framework focused on A/A testing (null effects). Future extensions could include:

- Non-null data generators with actual treatment effects
- More sophisticated assignment methods (stratified, matched pairs, etc.)
- Additional reporting models (synthetic control, neural methods, etc.)
- Power analysis capabilities
- More evaluation metrics beyond false positive rates

## Contributing

The modular design makes it easy to contribute new methods:

1. Implement new classes following the base class interfaces
2. Add corresponding unit tests
3. Update the playground notebook with examples

## Legacy Code

The `legacy/` folder contains the original notebook-based implementation for reference. The new modular code supersedes this but maintains the same core methodologies.