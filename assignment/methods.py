"""
Assignment methods for geo-experiments.
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
from .spatial_utils import standardize_features, add_spectral_spatial_embedding
from .stratified_utils import stratified_assignment_within_clusters


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


class KMeansEmbeddingAssignment(BaseAssignment):
    """
    K-means clustering assignment using engineered features.
    
    This method standardizes key features and uses K-means clustering to create
    balanced strata, then performs stratified random assignment within each cluster.
    """
    
    def __init__(self, feature_cols: Optional[List[str]] = None, n_clusters: int = 4):
        """
        Initialize K-means embedding assignment.
        
        Args:
            feature_cols: List of feature columns to use. If None, uses default set.
            n_clusters: Number of clusters to create for stratification (default: 4)
        """
        self.feature_cols = feature_cols
        self.n_clusters = n_clusters
    
    def assign(self, geo_features: pd.DataFrame, treatment_ratio: float = 0.5,
               seed: Optional[int] = None) -> pd.DataFrame:
        """
        Assign geos using stratified K-means clustering.
        
        Args:
            geo_features: DataFrame with geo-level features
            treatment_ratio: Proportion of geos to assign to treatment within each cluster
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns ['geo', 'assignment', 'cluster']
        """
        if 'geo' not in geo_features.columns:
            raise ValueError("geo_features must contain 'geo' column")
        
        # Determine feature columns to use
        if self.feature_cols is None:
            # Use default features that should be available in simple data
            available_cols = geo_features.columns.tolist()
            default_cols = ['base_sales', 'base_spend', 'covariate']
            self.feature_cols = [col for col in default_cols if col in available_cols]
            
            if len(self.feature_cols) == 0:
                raise ValueError("No suitable feature columns found. Available columns: " + 
                               str(available_cols))
        
        # Check that all required columns exist
        missing_cols = set(self.feature_cols) - set(geo_features.columns)
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Extract and standardize features
        features = geo_features[self.feature_cols].values
        features_std = standardize_features(features)
        
        # Apply K-means clustering to create strata
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(features_std)
        
        # Use stratified assignment within clusters
        assignment_df = stratified_assignment_within_clusters(
            geo_features=geo_features,
            cluster_labels=cluster_labels,
            treatment_ratio=treatment_ratio,
            seed=seed
        )
        
        return assignment_df


class PrognosticScoreAssignment(BaseAssignment):
    """
    Prognostic score assignment using OLS prediction of outcomes.
    
    This method fits an OLS model to predict outcomes, creates strata based on
    prognostic scores, then performs stratified random assignment within strata.
    """
    
    def __init__(self, 
                 outcome_col: str = 'sales',
                 covariate_cols: Optional[List[str]] = None,
                 pre_period_fraction: float = 0.5,
                 n_strata: int = 4):
        """
        Initialize prognostic score assignment.
        
        Args:
            outcome_col: Column name for outcome variable (default: 'sales')
            covariate_cols: List of covariate columns. If None, uses default set.
            pre_period_fraction: Fraction of time periods to use for pre-period
            n_strata: Number of prognostic score strata for assignment (default: 4)
        """
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols
        self.pre_period_fraction = pre_period_fraction
        self.n_strata = n_strata
    
    def assign(self, geo_features: pd.DataFrame, treatment_ratio: float = 0.5,
               seed: Optional[int] = None, panel_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Assign geos using prognostic scores from OLS model.
        
        Args:
            geo_features: DataFrame with geo-level features
            treatment_ratio: Proportion of geos to assign to treatment
            seed: Random seed for reproducibility
            panel_data: Panel data for computing pre-period outcomes (optional)
            
        Returns:
            DataFrame with columns ['geo', 'assignment']
            
        Note:
            If panel_data is not provided, this method falls back to using base_sales
            from geo_features as the prognostic target.
        """
        if 'geo' not in geo_features.columns:
            raise ValueError("geo_features must contain 'geo' column")
        
        # Determine covariate columns
        if self.covariate_cols is None:
            available_cols = geo_features.columns.tolist()
            default_cols = ['base_sales', 'base_spend', 'covariate']
            self.covariate_cols = [col for col in default_cols if col in available_cols]
            
            if len(self.covariate_cols) == 0:
                raise ValueError("No suitable covariate columns found")
        
        # Check columns exist
        missing_cols = set(self.covariate_cols) - set(geo_features.columns)
        if missing_cols:
            raise ValueError(f"Missing covariate columns: {missing_cols}")
        
        # Get target variable (prognostic score target)
        if panel_data is not None and self.outcome_col in panel_data.columns:
            # Use pre-period data to compute target
            dates = sorted(panel_data['date'].unique())
            pre_dates = dates[:int(len(dates) * self.pre_period_fraction)]
            pre_data = panel_data[panel_data['date'].isin(pre_dates)]
            target_data = pre_data.groupby('geo')[self.outcome_col].mean()
            
            # Align with geo_features
            geo_target = geo_features.set_index('geo')[self.covariate_cols]
            geo_target = geo_target.loc[target_data.index]
            y = target_data.values
            X = geo_target.values
        else:
            # Fall back to using base_sales as target
            if 'base_sales' not in geo_features.columns:
                raise ValueError("Need either panel_data or 'base_sales' column in geo_features")
            
            X = geo_features[self.covariate_cols].values
            y = geo_features['base_sales'].values
        
        # Standardize features and fit OLS
        X_std = standardize_features(X)
        model = LinearRegression()
        model.fit(X_std, y)
        
        # Predict prognostic scores for all geos
        X_all = geo_features[self.covariate_cols].values
        X_all_std = standardize_features(X_all)
        prognostic_scores = model.predict(X_all_std)
        
        # Create prognostic score strata using quantiles
        quantiles = np.linspace(0, 1, self.n_strata + 1)
        score_thresholds = np.quantile(prognostic_scores, quantiles)
        
        # Assign strata labels (0 to n_strata-1)
        strata_labels = np.digitize(prognostic_scores, score_thresholds[1:-1])
        
        # Use stratified assignment within strata
        assignment_df = stratified_assignment_within_clusters(
            geo_features=geo_features,
            cluster_labels=strata_labels,
            treatment_ratio=treatment_ratio,
            seed=seed
        )
        
        return assignment_df


class SimpleEmbeddingNet(nn.Module):
    """Simple neural network for learning feature embeddings."""
    
    def __init__(self, input_dim: int, embedding_dim: int = 8, hidden_dim: int = 16):
        super(SimpleEmbeddingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Bounded output
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction


class EmbeddingBasedAssignment(BaseAssignment):
    """
    General embedding-based assignment using neural embeddings + spectral spatial embeddings.
    
    This method combines different types of features through embeddings:
    1. Neural embeddings learned from static geo features via autoencoder
    2. Spectral spatial embeddings from geographic coordinates
    3. K-means clustering on combined embeddings for stratified assignment
    
    This is a general embedding approach suitable for static feature data.
    """
    
    def __init__(self, 
                 feature_cols: Optional[List[str]] = None,
                 spatial_cols: List[str] = ['xy1', 'xy2'],
                 neural_embedding_dim: int = 8,
                 spatial_embedding_dim: int = 2,
                 n_clusters: int = 4,
                 spatial_neighbors: int = 8,
                 neural_epochs: int = 50,
                 neural_lr: float = 0.01):
        """
        Initialize hybrid embedding assignment.
        
        Args:
            feature_cols: List of feature columns for neural embedding. If None, uses default set.
            spatial_cols: List of spatial coordinate columns (default: ['xy1', 'xy2'])
            neural_embedding_dim: Dimension of neural embeddings (default: 8)
            spatial_embedding_dim: Dimension of spatial embeddings (default: 2)
            n_clusters: Number of clusters for stratified assignment (default: 4)
            spatial_neighbors: Number of neighbors for spatial embedding (default: 8)
            neural_epochs: Training epochs for neural network (default: 50)
            neural_lr: Learning rate for neural network (default: 0.01)
        """
        self.feature_cols = feature_cols
        self.spatial_cols = spatial_cols
        self.neural_embedding_dim = neural_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.n_clusters = n_clusters
        self.spatial_neighbors = spatial_neighbors
        self.neural_epochs = neural_epochs
        self.neural_lr = neural_lr
    
    def _train_neural_embeddings(self, features: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Train neural network to learn feature embeddings."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(features)
        
        # Initialize network
        net = SimpleEmbeddingNet(
            input_dim=features.shape[1],
            embedding_dim=self.neural_embedding_dim
        )
        
        # Training setup
        optimizer = optim.Adam(net.parameters(), lr=self.neural_lr)
        criterion = nn.MSELoss()
        
        # Training loop
        net.train()
        for epoch in range(self.neural_epochs):
            optimizer.zero_grad()
            embeddings, reconstructions = net(X_tensor)
            loss = criterion(reconstructions, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Extract final embeddings
        net.eval()
        with torch.no_grad():
            embeddings, _ = net(X_tensor)
            return embeddings.numpy()
    
    def assign(self, geo_features: pd.DataFrame, treatment_ratio: float = 0.5,
               seed: Optional[int] = None) -> pd.DataFrame:
        """
        Assign geos using hybrid neural + spatial embeddings.
        
        Args:
            geo_features: DataFrame with geo-level features and spatial coordinates
            treatment_ratio: Proportion of geos to assign to treatment within each cluster
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with columns ['geo', 'assignment', 'cluster']
        """
        if 'geo' not in geo_features.columns:
            raise ValueError("geo_features must contain 'geo' column")
        
        # Check for spatial columns
        missing_spatial_cols = set(self.spatial_cols) - set(geo_features.columns)
        if missing_spatial_cols:
            raise ValueError(f"Missing spatial columns: {missing_spatial_cols}")
        
        # Determine feature columns for neural embedding
        if self.feature_cols is None:
            available_cols = geo_features.columns.tolist()
            default_cols = ['base_sales', 'base_spend', 'covariate']
            self.feature_cols = [col for col in default_cols if col in available_cols]
            
            if len(self.feature_cols) == 0:
                raise ValueError("No suitable feature columns found for neural embedding")
        
        # Check that all required columns exist
        missing_feature_cols = set(self.feature_cols) - set(geo_features.columns)
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")
        
        # 1. Create neural embeddings from features
        features = geo_features[self.feature_cols].values
        features_std = standardize_features(features)
        neural_embeddings = self._train_neural_embeddings(features_std, seed=seed)
        
        # 2. Create spectral spatial embeddings
        geo_with_spatial = add_spectral_spatial_embedding(
            geo_features=geo_features,
            spatial_cols=self.spatial_cols,
            spatial_emb_dim=self.spatial_embedding_dim,
            spatial_neighbors=self.spatial_neighbors,
            seed=seed
        )
        
        spatial_emb_cols = [f'spatial_emb_{i}' for i in range(self.spatial_embedding_dim)]
        spatial_embeddings = geo_with_spatial[spatial_emb_cols].values
        
        # 3. Combine neural and spatial embeddings
        combined_embeddings = np.hstack([neural_embeddings, spatial_embeddings])
        combined_embeddings_std = standardize_features(combined_embeddings)
        
        # 4. Apply K-means clustering on combined embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_embeddings_std)
        
        # 5. Use stratified assignment within clusters
        assignment_df = stratified_assignment_within_clusters(
            geo_features=geo_features,
            cluster_labels=cluster_labels,
            treatment_ratio=treatment_ratio,
            seed=seed
        )
        
        return assignment_df


class PredictiveGeoEmbedder(nn.Module):
    """
    Semi-supervised neural network for learning predictive geo embeddings.
    
    Combines reconstruction loss (unsupervised) with prediction loss (supervised)
    to create embeddings that are both representative and predictive of future outcomes.
    """
    
    def __init__(self, n_time_steps: int, emb_dim: int = 8, hidden_dim: int = 32):
        super(PredictiveGeoEmbedder, self).__init__()
        
        # Encoder: Time series -> Embedding
        self.encoder = nn.Sequential(
            nn.Linear(n_time_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
            nn.Tanh()  # Bounded embedding space
        )
        
        # Decoder: Embedding -> Reconstructed time series
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_time_steps)
        )
        
        # Predictor: Embedding -> Future outcome prediction
        self.predictor = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Predict single future sales value
        )
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        prediction = self.predictor(embedding)
        return embedding, reconstruction, prediction


class HybridEmbeddingAssignment(BaseAssignment):
    """
    Hybrid (semi-supervised) embedding assignment with predictive capability.
    
    This method uses a hybrid loss function combining:
    1. Reconstruction loss (unsupervised): Learn to represent pre-period time series
    2. Prediction loss (supervised): Learn to predict future sales outcomes
    3. Spatial spectral embeddings from geographic coordinates
    4. K-means clustering on combined embeddings for stratified assignment
    
    The "hybrid" refers to the multi-objective loss function that makes assignments
    both balanced (unsupervised) and prediction-aware (supervised).
    
    Requires panel data with time series. Best for cases where you want
    treatment assignments that consider likely future outcomes.
    """
    
    def __init__(self,
                 # Data and prediction settings
                 prediction_fraction: float = 0.3,        # Use 30% of timeline for prediction targets
                 prediction_outcome: str = 'sales',       # Primary outcome to predict
                 pre_period_fraction: float = 0.7,        # Use 70% of data for pre-period
                 
                 # Loss function weights (should sum to 1.0)
                 reconstruction_weight: float = 0.5,      # Unsupervised reconstruction 
                 prediction_weight: float = 0.25,         # Supervised prediction
                 regularization_weight: float = 0.25,     # Embedding regularization
                 
                 # Neural architecture
                 neural_embedding_dim: int = 8,
                 hidden_dim: int = 32,
                 
                 # Spatial embedding settings
                 spatial_cols: List[str] = ['xy1', 'xy2'],
                 spatial_embedding_dim: int = 2,
                 spatial_neighbors: int = 8,
                 
                 # Clustering
                 n_clusters: int = 4,
                 
                 # Training parameters
                 neural_epochs: int = 100,
                 neural_lr: float = 0.01):
        """
        Initialize hybrid embedding assignment method.
        
        Args:
            prediction_fraction: Fraction of timeline to use for prediction targets
            prediction_outcome: Outcome variable to predict ('sales' or 'spend_dollars')
            pre_period_fraction: Fraction of timeline to use as pre-period for training
            reconstruction_weight: Weight for reconstruction loss (unsupervised)
            prediction_weight: Weight for prediction loss (supervised)  
            regularization_weight: Weight for embedding regularization
            neural_embedding_dim: Dimension of neural embeddings
            hidden_dim: Hidden layer dimension for neural network
            spatial_cols: Column names for spatial coordinates
            spatial_embedding_dim: Dimension of spatial embeddings
            spatial_neighbors: Number of neighbors for spatial embedding
            n_clusters: Number of clusters for stratified assignment
            neural_epochs: Training epochs for neural network
            neural_lr: Learning rate for neural network training
        """
        # Validate loss weights
        total_weight = reconstruction_weight + prediction_weight + regularization_weight
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Loss weights must sum to 1.0, got {total_weight}")
        
        self.prediction_fraction = prediction_fraction
        self.prediction_outcome = prediction_outcome
        self.pre_period_fraction = pre_period_fraction
        
        # Loss weights
        self.reconstruction_weight = reconstruction_weight
        self.prediction_weight = prediction_weight
        self.regularization_weight = regularization_weight
        
        # Architecture parameters
        self.neural_embedding_dim = neural_embedding_dim
        self.hidden_dim = hidden_dim
        
        # Spatial parameters
        self.spatial_cols = spatial_cols
        self.spatial_embedding_dim = spatial_embedding_dim
        self.spatial_neighbors = spatial_neighbors
        
        # Clustering
        self.n_clusters = n_clusters
        
        # Training parameters
        self.neural_epochs = neural_epochs
        self.neural_lr = neural_lr
    
    def _prepare_time_series_data(self, panel_data: pd.DataFrame) -> tuple:
        """
        Prepare time series data for semi-supervised learning.
        
        Returns:
            pre_period_data: Sales time series for pre-period [n_geos, n_time_steps]
            prediction_targets: Future sales values for prediction [n_geos]
        """
        # Get unique dates and split timeline
        dates = np.sort(panel_data['date'].unique())
        n_dates = len(dates)
        
        pre_end_idx = int(n_dates * self.pre_period_fraction)
        pred_start_idx = pre_end_idx
        pred_end_idx = min(pre_end_idx + int(n_dates * self.prediction_fraction), n_dates)
        
        pre_dates = dates[:pre_end_idx]
        pred_dates = dates[pred_start_idx:pred_end_idx]
        
        # Create pre-period time series matrix
        pre_data = panel_data[panel_data['date'].isin(pre_dates)]
        pre_pivot = pre_data.pivot(index='geo', columns='date', values=self.prediction_outcome)
        pre_pivot = pre_pivot.fillna(pre_pivot.mean(axis=1, skipna=True).mean())  # Fill NaN with overall mean
        
        # Create prediction targets (mean of prediction period)
        pred_data = panel_data[panel_data['date'].isin(pred_dates)]
        pred_targets = pred_data.groupby('geo')[self.prediction_outcome].mean()
        
        # Align geos
        common_geos = pre_pivot.index.intersection(pred_targets.index)
        pre_period_data = pre_pivot.loc[common_geos].values
        prediction_targets = pred_targets.loc[common_geos].values
        
        return pre_period_data, prediction_targets, common_geos
    
    def _train_hybrid_embeddings(self, pre_period_data: np.ndarray, 
                                prediction_targets: np.ndarray, 
                                seed: Optional[int] = None) -> np.ndarray:
        """Train semi-supervised neural network to learn predictive embeddings."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(pre_period_data)
        y_tensor = torch.FloatTensor(prediction_targets)
        
        # Initialize network
        n_time_steps = pre_period_data.shape[1]
        net = PredictiveGeoEmbedder(
            n_time_steps=n_time_steps,
            emb_dim=self.neural_embedding_dim,
            hidden_dim=self.hidden_dim
        )
        
        # Training setup
        optimizer = optim.Adam(net.parameters(), lr=self.neural_lr)
        
        # Training loop
        net.train()
        for epoch in range(self.neural_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings, reconstructions, predictions = net(X_tensor)
            
            # Multi-objective loss
            reconstruction_loss = nn.MSELoss()(reconstructions, X_tensor)
            prediction_loss = nn.MSELoss()(predictions.squeeze(), y_tensor)
            regularization_loss = torch.mean(embeddings ** 2)
            
            # Combine losses with weights
            total_loss = (self.reconstruction_weight * reconstruction_loss +
                         self.prediction_weight * prediction_loss +
                         self.regularization_weight * regularization_loss)
            
            total_loss.backward()
            optimizer.step()
        
        # Extract final embeddings
        net.eval()
        with torch.no_grad():
            embeddings, _, _ = net(X_tensor)
            return embeddings.numpy()
    
    def assign(self, geo_features: pd.DataFrame, treatment_ratio: float = 0.5,
               seed: Optional[int] = None, panel_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Assign geos using hybrid (semi-supervised) embeddings.
        
        Args:
            geo_features: DataFrame with geo-level features and spatial coordinates
            treatment_ratio: Proportion of geos to assign to treatment within each cluster
            seed: Random seed for reproducibility
            panel_data: Required panel data with time series for prediction-aware assignment
            
        Returns:
            DataFrame with columns ['geo', 'assignment', 'cluster']
            
        Raises:
            ValueError: If panel_data is not provided or required columns are missing
        """
        if panel_data is None:
            raise ValueError("panel_data is required for HybridEmbeddingAssignment")
        
        if 'geo' not in geo_features.columns:
            raise ValueError("geo_features must contain 'geo' column")
        
        # Check for spatial columns
        missing_spatial_cols = set(self.spatial_cols) - set(geo_features.columns)
        if missing_spatial_cols:
            raise ValueError(f"Missing spatial columns: {missing_spatial_cols}")
        
        # Check panel data columns
        required_panel_cols = {'geo', 'date', self.prediction_outcome}
        missing_panel_cols = required_panel_cols - set(panel_data.columns)
        if missing_panel_cols:
            raise ValueError(f"Missing panel data columns: {missing_panel_cols}")
        
        # 1. Prepare time series data for semi-supervised learning
        pre_period_data, prediction_targets, common_geos = self._prepare_time_series_data(panel_data)
        
        if len(common_geos) == 0:
            raise ValueError("No common geos found between geo_features and panel_data")
        
        # 2. Train hybrid neural embeddings (reconstruction + prediction)
        neural_embeddings = self._train_hybrid_embeddings(
            pre_period_data, prediction_targets, seed=seed
        )
        
        # 3. Create spectral spatial embeddings for geos with time series data
        geo_features_subset = geo_features[geo_features['geo'].isin(common_geos)].copy()
        geo_with_spatial = add_spectral_spatial_embedding(
            geo_features=geo_features_subset,
            spatial_cols=self.spatial_cols,
            spatial_emb_dim=self.spatial_embedding_dim,
            spatial_neighbors=self.spatial_neighbors,
            seed=seed
        )
        
        # Align spatial embeddings with neural embeddings
        geo_with_spatial = geo_with_spatial.set_index('geo').loc[common_geos].reset_index()
        spatial_emb_cols = [f'spatial_emb_{i}' for i in range(self.spatial_embedding_dim)]
        spatial_embeddings = geo_with_spatial[spatial_emb_cols].values
        
        # 4. Combine neural and spatial embeddings
        combined_embeddings = np.hstack([neural_embeddings, spatial_embeddings])
        combined_embeddings_std = standardize_features(combined_embeddings)
        
        # 5. Apply K-means clustering on combined embeddings
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=seed, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_embeddings_std)
        
        # 6. Create assignment DataFrame with common geos
        assignment_geo_features = pd.DataFrame({'geo': common_geos})
        assignment_df = stratified_assignment_within_clusters(
            geo_features=assignment_geo_features,
            cluster_labels=cluster_labels,
            treatment_ratio=treatment_ratio,
            seed=seed
        )
        
        # TODO: Add validation component to measure prediction accuracy on holdout data
        # when more training data is available (future enhancement)
        
        return assignment_df