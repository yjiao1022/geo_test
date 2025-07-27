"""
Validate STGCN prediction quality by testing control geo predictions.

This script isolates the prediction quality issue by:
1. Training models only on control geos
2. Predicting control geo outcomes in eval period
3. Comparing predictions vs ground truth
4. Computing bias, variance, MSE
5. Including baseline models for comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_simulation.generators import SimpleNullGenerator, DataConfig
from assignment.methods import RandomAssignment
from reporting.stgcn_model import STGCNReportingModel


class BaselinePredictor:
    """Base class for baseline prediction methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit(self, train_data: pd.DataFrame) -> 'BaselinePredictor':
        """Fit the model on training data."""
        raise NotImplementedError
    
    def predict(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict sales and spend for test period."""
        raise NotImplementedError


class HistoricalMeanPredictor(BaselinePredictor):
    """Predict using historical mean of each geo."""
    
    def __init__(self):
        super().__init__("HistoricalMean")
        self.geo_means = {}
    
    def fit(self, train_data: pd.DataFrame) -> 'HistoricalMeanPredictor':
        """Compute historical means for each geo."""
        self.geo_means = train_data.groupby('geo')[['sales', 'spend']].mean().to_dict()
        self.is_fitted = True
        return self
    
    def predict(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict using geo-level historical means."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = {'sales': [], 'spend': []}
        
        for _, row in test_data.iterrows():
            geo = row['geo']
            # Use historical mean for this geo
            pred_sales = self.geo_means['sales'].get(geo, 0)
            pred_spend = self.geo_means['spend'].get(geo, 0)
            
            predictions['sales'].append(pred_sales)
            predictions['spend'].append(pred_spend)
        
        return {k: np.array(v) for k, v in predictions.items()}


class LinearTrendPredictor(BaselinePredictor):
    """Predict using linear trend for each geo."""
    
    def __init__(self):
        super().__init__("LinearTrend")
        self.geo_models = {}
    
    def fit(self, train_data: pd.DataFrame) -> 'LinearTrendPredictor':
        """Fit linear models for each geo."""
        self.geo_models = {}
        
        for geo in train_data['geo'].unique():
            geo_data = train_data[train_data['geo'] == geo].copy()
            geo_data = geo_data.sort_values('date')
            geo_data['day_index'] = range(len(geo_data))
            
            # Fit separate models for sales and spend
            models = {}
            for target in ['sales', 'spend']:
                if len(geo_data) >= 2:  # Need at least 2 points for trend
                    model = LinearRegression()
                    X = geo_data[['day_index']]
                    y = geo_data[target]
                    model.fit(X, y)
                    models[target] = model
                else:
                    # Fallback to mean
                    models[target] = geo_data[target].mean()
            
            self.geo_models[geo] = models
        
        self.is_fitted = True
        return self
    
    def predict(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict using linear trend extrapolation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        predictions = {'sales': [], 'spend': []}
        
        # Group test data by geo to get proper day indices
        for geo in test_data['geo'].unique():
            geo_test_data = test_data[test_data['geo'] == geo].copy()
            geo_test_data = geo_test_data.sort_values('date')
            
            if geo in self.geo_models:
                models = self.geo_models[geo]
                
                # Predict based on day continuation
                start_day = len(test_data)  # Assume test starts after training
                day_indices = np.arange(start_day, start_day + len(geo_test_data))
                
                for target in ['sales', 'spend']:
                    if hasattr(models[target], 'predict'):
                        # Linear model
                        geo_preds = models[target].predict(day_indices.reshape(-1, 1))
                    else:
                        # Fallback mean
                        geo_preds = np.full(len(geo_test_data), models[target])
                    
                    predictions[target].extend(geo_preds)
            else:
                # No training data for this geo, predict 0
                predictions['sales'].extend([0] * len(geo_test_data))
                predictions['spend'].extend([0] * len(geo_test_data))
        
        return {k: np.array(v) for k, v in predictions.items()}


class RandomForestPredictor(BaselinePredictor):
    """Predict using Random Forest with engineered features."""
    
    def __init__(self):
        super().__init__("RandomForest")
        self.models = {}
        self.geo_encoder = {}
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features for prediction."""
        features = data.copy()
        
        # Convert date to day index
        features['date'] = pd.to_datetime(features['date'])
        min_date = features['date'].min()
        features['day_index'] = (features['date'] - min_date).dt.days
        
        # Encode geo as integer
        unique_geos = sorted(features['geo'].unique())
        if not self.geo_encoder:  # Only create mapping during training
            self.geo_encoder = {geo: i for i, geo in enumerate(unique_geos)}
        
        features['geo_encoded'] = features['geo'].map(self.geo_encoder).fillna(-1)
        
        # Add day of week, month features
        features['day_of_week'] = features['date'].dt.dayofweek
        features['month'] = features['date'].dt.month
        
        # Add lagged features within each geo
        feature_cols = ['day_index', 'geo_encoded', 'day_of_week', 'month']
        
        for geo in features['geo'].unique():
            mask = features['geo'] == geo
            geo_data = features[mask].sort_values('date')
            
            # Add simple lag features
            if len(geo_data) > 1:
                for target in ['sales', 'spend']:
                    if target in geo_data.columns:
                        lag1 = geo_data[target].shift(1)
                        features.loc[mask, f'{target}_lag1'] = lag1
                        feature_cols.append(f'{target}_lag1')
        
        return features[feature_cols].fillna(0)
    
    def fit(self, train_data: pd.DataFrame) -> 'RandomForestPredictor':
        """Fit Random Forest models."""
        train_features = self._create_features(train_data)
        
        for target in ['sales', 'spend']:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Use lagged features (exclude current target)
            feature_cols = [col for col in train_features.columns if not col.startswith(target)]
            X = train_features[feature_cols]
            y = train_data[target]
            
            model.fit(X, y)
            self.models[target] = (model, feature_cols)
        
        self.is_fitted = True
        return self
    
    def predict(self, test_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict using Random Forest."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        test_features = self._create_features(test_data)
        predictions = {}
        
        for target in ['sales', 'spend']:
            model, feature_cols = self.models[target]
            X = test_features[feature_cols]
            pred = model.predict(X)
            predictions[target] = pred
        
        return predictions


def create_experiment_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, str, str, str]:
    """Create experiment data for prediction validation."""
    data_config = DataConfig(
        n_geos=config['n_geos'],
        n_days=config['n_days'],
        seed=config['seed']
    )
    
    generator = SimpleNullGenerator(data_config)
    panel_data, geo_features = generator.generate()
    
    # Create assignment for control/treatment split
    assignment_method = RandomAssignment()
    assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.5, seed=config['seed'])
    
    # Define periods
    dates = sorted(panel_data['date'].unique())
    pre_period_end = dates[config['pre_period_days'] - 1]
    eval_period_start = dates[config['pre_period_days']]
    eval_period_end = dates[config['pre_period_days'] + config['eval_period_days'] - 1]
    
    return panel_data, assignment_df, pre_period_end, eval_period_start, eval_period_end


def evaluate_stgcn_prediction(
    panel_data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    pre_period_end: str,
    eval_period_start: str,
    eval_period_end: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """Evaluate STGCN prediction quality on control geos."""
    
    # Get control geos only
    control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].values
    control_data = panel_data[panel_data['geo'].isin(control_geos)].copy()
    
    # Split into train/test
    control_data['date'] = pd.to_datetime(control_data['date'])
    pre_period_end = pd.to_datetime(pre_period_end)
    eval_period_start = pd.to_datetime(eval_period_start)
    eval_period_end = pd.to_datetime(eval_period_end)
    
    train_data = control_data[control_data['date'] <= pre_period_end]
    test_data = control_data[
        (control_data['date'] >= eval_period_start) & 
        (control_data['date'] <= eval_period_end)
    ]
    
    if len(test_data) == 0:
        return {'error': 'No test data available'}
    
    # Create a temporary assignment with all control geos as "treatment" for STGCN prediction
    temp_assignment = pd.DataFrame({
        'geo': control_geos,
        'assignment': 'treatment'  # Treat all as treatment for prediction
    })
    
    # Fit STGCN model
    stgcn_model = STGCNReportingModel(
        hidden_dim=32,
        epochs=10,
        window_size=5,
        learning_rate=0.01,
        normalize_data=True,
        verbose=verbose
    )
    
    try:
        stgcn_model.fit(control_data, temp_assignment, pre_period_end.strftime('%Y-%m-%d'))
        
        # Get STGCN predictions
        stgcn_predictions = stgcn_model.predict(
            control_data,
            eval_period_start.strftime('%Y-%m-%d'),
            eval_period_end.strftime('%Y-%m-%d')
        )
        
        # Convert to per-row predictions to match test_data
        n_test_rows = len(test_data)
        n_geos = len(control_geos)
        n_days = len(test_data['date'].unique())
        
        # STGCN returns total for each geo, we need daily averages
        stgcn_daily_sales = np.repeat(stgcn_predictions['sales'] / n_days, n_days)
        stgcn_daily_spend = np.repeat(stgcn_predictions['spend'] / n_days, n_days)
        
        # Ensure predictions match test data length
        if len(stgcn_daily_sales) != n_test_rows:
            # If mismatch, tile the predictions to match
            stgcn_daily_sales = np.tile(stgcn_daily_sales, n_test_rows // len(stgcn_daily_sales) + 1)[:n_test_rows]
            stgcn_daily_spend = np.tile(stgcn_daily_spend, n_test_rows // len(stgcn_daily_spend) + 1)[:n_test_rows]
        
        stgcn_pred_dict = {
            'sales': stgcn_daily_sales,
            'spend': stgcn_daily_spend
        }
        
    except Exception as e:
        if verbose:
            print(f"STGCN prediction failed: {e}")
        return {'error': f'STGCN failed: {e}'}
    
    # Ground truth
    actual = {
        'sales': test_data['sales'].values,
        'spend': test_data['spend'].values
    }
    
    # Compute metrics
    results = {'model': 'STGCN'}
    
    for target in ['sales', 'spend']:
        pred = stgcn_pred_dict[target]
        true = actual[target]
        
        # Basic metrics
        bias = np.mean(pred - true)
        variance = np.var(pred - true)
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        
        # Relative metrics
        true_mean = np.mean(true)
        relative_bias = bias / true_mean if true_mean != 0 else np.inf
        relative_mse = mse / (true_mean ** 2) if true_mean != 0 else np.inf
        
        results[f'{target}_bias'] = bias
        results[f'{target}_variance'] = variance
        results[f'{target}_mse'] = mse
        results[f'{target}_mae'] = mae
        results[f'{target}_relative_bias'] = relative_bias
        results[f'{target}_relative_mse'] = relative_mse
        results[f'{target}_true_mean'] = true_mean
        results[f'{target}_pred_mean'] = np.mean(pred)
    
    return results


def evaluate_baseline_prediction(
    panel_data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    pre_period_end: str,
    eval_period_start: str,
    eval_period_end: str,
    model: BaselinePredictor
) -> Dict[str, Any]:
    """Evaluate baseline prediction model on control geos."""
    
    # Get control geos only
    control_geos = assignment_df[assignment_df['assignment'] == 'control']['geo'].values
    control_data = panel_data[panel_data['geo'].isin(control_geos)].copy()
    
    # Split into train/test
    control_data['date'] = pd.to_datetime(control_data['date'])
    pre_period_end = pd.to_datetime(pre_period_end)
    eval_period_start = pd.to_datetime(eval_period_start)
    eval_period_end = pd.to_datetime(eval_period_end)
    
    train_data = control_data[control_data['date'] <= pre_period_end]
    test_data = control_data[
        (control_data['date'] >= eval_period_start) & 
        (control_data['date'] <= eval_period_end)
    ]
    
    if len(test_data) == 0:
        return {'error': 'No test data available'}
    
    try:
        # Fit and predict
        model.fit(train_data)
        predictions = model.predict(test_data)
        
        # Ground truth
        actual = {
            'sales': test_data['sales'].values,
            'spend': test_data['spend'].values
        }
        
        # Compute metrics
        results = {'model': model.name}
        
        for target in ['sales', 'spend']:
            pred = predictions[target]
            true = actual[target]
            
            # Ensure same length
            min_len = min(len(pred), len(true))
            pred = pred[:min_len]
            true = true[:min_len]
            
            # Basic metrics
            bias = np.mean(pred - true)
            variance = np.var(pred - true)
            mse = mean_squared_error(true, pred)
            mae = mean_absolute_error(true, pred)
            
            # Relative metrics
            true_mean = np.mean(true)
            relative_bias = bias / true_mean if true_mean != 0 else np.inf
            relative_mse = mse / (true_mean ** 2) if true_mean != 0 else np.inf
            
            results[f'{target}_bias'] = bias
            results[f'{target}_variance'] = variance
            results[f'{target}_mse'] = mse
            results[f'{target}_mae'] = mae
            results[f'{target}_relative_bias'] = relative_bias
            results[f'{target}_relative_mse'] = relative_mse
            results[f'{target}_true_mean'] = true_mean
            results[f'{target}_pred_mean'] = np.mean(pred)
        
        return results
        
    except Exception as e:
        return {'error': f'{model.name} failed: {e}'}


def run_prediction_validation(config: Dict[str, Any], n_simulations: int = 10) -> pd.DataFrame:
    """Run prediction validation across multiple simulations."""
    
    results = []
    
    # Define baseline models
    baseline_models = [
        HistoricalMeanPredictor(),
        LinearTrendPredictor(),
        RandomForestPredictor()
    ]
    
    for sim in range(n_simulations):
        print(f"\nSimulation {sim + 1}/{n_simulations}")
        
        # Create data with different seed
        sim_config = config.copy()
        sim_config['seed'] = config['seed'] + sim
        
        panel_data, assignment_df, pre_period_end, eval_start, eval_end = create_experiment_data(sim_config)
        
        # Evaluate STGCN
        print("  Evaluating STGCN...")
        stgcn_result = evaluate_stgcn_prediction(
            panel_data, assignment_df, pre_period_end, eval_start, eval_end, verbose=False
        )
        
        if 'error' not in stgcn_result:
            stgcn_result['simulation'] = sim
            results.append(stgcn_result)
        else:
            print(f"    STGCN failed: {stgcn_result['error']}")
        
        # Evaluate baseline models
        for model in baseline_models:
            print(f"  Evaluating {model.name}...")
            try:
                baseline_result = evaluate_baseline_prediction(
                    panel_data, assignment_df, pre_period_end, eval_start, eval_end, model
                )
                
                if 'error' not in baseline_result:
                    baseline_result['simulation'] = sim
                    results.append(baseline_result)
                else:
                    print(f"    {model.name} failed: {baseline_result['error']}")
                    
            except Exception as e:
                print(f"    {model.name} failed with exception: {e}")
    
    return pd.DataFrame(results)


def summarize_prediction_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize prediction results across simulations."""
    
    if len(results_df) == 0:
        print("No results to summarize")
        return pd.DataFrame()
    
    summary_metrics = []
    
    for model in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model]
        
        if len(model_results) == 0:
            continue
            
        summary = {'model': model}
        
        for target in ['sales', 'spend']:
            metrics = ['bias', 'variance', 'mse', 'mae', 'relative_bias', 'relative_mse']
            
            for metric in metrics:
                col = f'{target}_{metric}'
                if col in model_results.columns:
                    values = model_results[col].dropna()
                    if len(values) > 0:
                        summary[f'{target}_{metric}_mean'] = values.mean()
                        summary[f'{target}_{metric}_std'] = values.std()
                        summary[f'{target}_{metric}_median'] = values.median()
        
        summary_metrics.append(summary)
    
    return pd.DataFrame(summary_metrics)


def main():
    """Run comprehensive prediction validation."""
    print("STGCN Prediction Quality Validation")
    print("=" * 50)
    
    # Configuration matching the user's setup
    config = {
        'n_geos': 20,
        'n_days': 120,
        'pre_period_days': 100,
        'eval_period_days': 20,
        'seed': 42
    }
    
    print(f"Configuration:")
    print(f"  Geos: {config['n_geos']}")
    print(f"  Days: {config['n_days']} (training: {config['pre_period_days']}, eval: {config['eval_period_days']})")
    print(f"  Focus: Control geo prediction quality")
    
    # Run validation
    print(f"\nRunning prediction validation...")
    results_df = run_prediction_validation(config, n_simulations=5)
    
    if len(results_df) == 0:
        print("❌ No successful predictions to analyze")
        return
    
    print(f"\n📊 Raw Results Summary:")
    print(f"  Total successful predictions: {len(results_df)}")
    print(f"  Models evaluated: {results_df['model'].unique()}")
    
    # Summary statistics
    summary_df = summarize_prediction_results(results_df)
    
    print(f"\n📈 Prediction Quality Summary:")
    print("=" * 80)
    
    # Focus on key metrics for readability
    key_metrics = ['sales_bias_mean', 'sales_relative_bias_mean', 'sales_mse_mean',
                   'spend_bias_mean', 'spend_relative_bias_mean', 'spend_mse_mean']
    
    display_cols = ['model'] + [col for col in key_metrics if col in summary_df.columns]
    display_df = summary_df[display_cols].round(4)
    print(display_df.to_string(index=False))
    
    # Detailed analysis for STGCN
    if 'STGCN' in results_df['model'].values:
        stgcn_results = results_df[results_df['model'] == 'STGCN']
        
        print(f"\n🔍 STGCN Detailed Analysis:")
        print(f"  Successful predictions: {len(stgcn_results)}")
        
        for target in ['sales', 'spend']:
            bias_col = f'{target}_bias'
            rel_bias_col = f'{target}_relative_bias'
            mse_col = f'{target}_mse'
            
            if bias_col in stgcn_results.columns:
                bias_vals = stgcn_results[bias_col].dropna()
                rel_bias_vals = stgcn_results[rel_bias_col].dropna()
                mse_vals = stgcn_results[mse_col].dropna()
                
                print(f"\n  {target.upper()} Predictions:")
                print(f"    Bias: {bias_vals.mean():.2f} ± {bias_vals.std():.2f}")
                print(f"    Relative Bias: {rel_bias_vals.mean():.2%} ± {rel_bias_vals.std():.2%}")
                print(f"    MSE: {mse_vals.mean():.2f} ± {mse_vals.std():.2f}")
                
                # Check for systematic bias
                if abs(rel_bias_vals.mean()) > 0.1:
                    print(f"    🚨 SYSTEMATIC BIAS DETECTED: {rel_bias_vals.mean():.1%}")
                elif abs(rel_bias_vals.mean()) > 0.05:
                    print(f"    ⚠️ Moderate bias: {rel_bias_vals.mean():.1%}")
                else:
                    print(f"    ✅ Low bias: {rel_bias_vals.mean():.1%}")
    
    # Model comparison
    print(f"\n🏆 Model Ranking (by Sales MSE):")
    if 'sales_mse_mean' in summary_df.columns:
        ranking = summary_df.sort_values('sales_mse_mean')[['model', 'sales_mse_mean', 'sales_relative_bias_mean']]
        print(ranking.round(4).to_string(index=False))
    
    print(f"\n💡 Recommendations:")
    
    if 'STGCN' in summary_df['model'].values:
        stgcn_summary = summary_df[summary_df['model'] == 'STGCN'].iloc[0]
        
        sales_rel_bias = stgcn_summary.get('sales_relative_bias_mean', 0)
        if abs(sales_rel_bias) > 0.2:
            print(f"  🚨 STGCN has severe systematic bias ({sales_rel_bias:.1%})")
            print(f"     - Check data normalization and model architecture")
            print(f"     - Consider reducing model complexity")
            print(f"     - Investigate training convergence")
        elif abs(sales_rel_bias) > 0.05:
            print(f"  ⚠️ STGCN has moderate bias ({sales_rel_bias:.1%})")
            print(f"     - Fine-tune hyperparameters")
            print(f"     - Consider ensemble methods")
        else:
            print(f"  ✅ STGCN bias is acceptable ({sales_rel_bias:.1%})")
    
    # Export results
    results_df.to_csv('stgcn_prediction_validation_results.csv', index=False)
    summary_df.to_csv('stgcn_prediction_validation_summary.csv', index=False)
    print(f"\n📁 Results exported to:")
    print(f"  - stgcn_prediction_validation_results.csv")
    print(f"  - stgcn_prediction_validation_summary.csv")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()