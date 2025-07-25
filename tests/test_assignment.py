"""
Tests for assignment module.
"""

import pytest
import pandas as pd
import numpy as np

from assignment.methods import RandomAssignment, KMeansEmbeddingAssignment, PrognosticScoreAssignment
from assignment.spatial_utils import add_spectral_spatial_embedding, standardize_features


class TestRandomAssignment:
    """Test cases for RandomAssignment."""
    
    @pytest.fixture
    def sample_geo_features(self):
        """Sample geo features for testing."""
        return pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(20)],
            'base_sales': np.random.normal(10000, 2000, 20),
            'base_spend': np.random.normal(5000, 1000, 20),
            'covariate': np.random.normal(0, 1, 20)
        })
    
    def test_basic_assignment(self, sample_geo_features):
        """Test basic random assignment."""
        assignment_method = RandomAssignment()
        
        assignment_df = assignment_method.assign(
            sample_geo_features, 
            treatment_ratio=0.5,
            seed=42
        )
        
        # Check structure - RandomAssignment only returns geo and assignment columns
        expected_cols = {'geo', 'assignment'}
        assert set(assignment_df.columns) == expected_cols
        assert len(assignment_df) == 20
        
        # Check assignments are valid
        assignments = set(assignment_df['assignment'])
        assert assignments == {'treatment', 'control'}
        
        # Check treatment ratio (approximately)
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        assert treatment_count == 10  # Exactly half with this seed/size
    
    def test_reproducibility(self, sample_geo_features):
        """Test that same seed produces same assignment."""
        assignment_method = RandomAssignment()
        
        assignment1 = assignment_method.assign(
            sample_geo_features, treatment_ratio=0.4, seed=123
        )
        assignment2 = assignment_method.assign(
            sample_geo_features, treatment_ratio=0.4, seed=123
        )
        
        pd.testing.assert_frame_equal(assignment1, assignment2)
    
    def test_different_treatment_ratios(self, sample_geo_features):
        """Test assignment with different treatment ratios."""
        assignment_method = RandomAssignment()
        
        # Test 30% treatment
        assignment_df = assignment_method.assign(
            sample_geo_features, treatment_ratio=0.3, seed=42
        )
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        assert treatment_count == 6  # 30% of 20
        
        # Test 70% treatment
        assignment_df = assignment_method.assign(
            sample_geo_features, treatment_ratio=0.7, seed=42
        )
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        assert treatment_count == 14  # 70% of 20
    
    def test_validation(self, sample_geo_features):
        """Test assignment validation."""
        assignment_method = RandomAssignment()
        
        assignment_df = assignment_method.assign(
            sample_geo_features, treatment_ratio=0.5, seed=42
        )
        
        # Should pass validation
        assert assignment_method.validate_assignment(
            assignment_df, expected_treatment_ratio=0.5
        ) is True
        
        # Should fail with wrong expected ratio
        assert assignment_method.validate_assignment(
            assignment_df, expected_treatment_ratio=0.8, tolerance=0.1
        ) is False
    
    def test_error_handling(self, sample_geo_features):
        """Test error handling for invalid inputs."""
        assignment_method = RandomAssignment()
        
        # Missing 'geo' column
        bad_features = sample_geo_features.drop('geo', axis=1)
        with pytest.raises(ValueError, match="must contain 'geo' column"):
            assignment_method.assign(bad_features)
        
        # Invalid treatment ratio
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            assignment_method.assign(sample_geo_features, treatment_ratio=1.5)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            assignment_method.assign(sample_geo_features, treatment_ratio=0)


class TestKMeansEmbeddingAssignment:
    """Test cases for KMeansEmbeddingAssignment."""
    
    @pytest.fixture
    def sample_geo_features(self):
        """Sample geo features with various columns."""
        np.random.seed(42)
        return pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(20)],
            'base_sales': np.random.normal(10000, 2000, 20),
            'base_spend': np.random.normal(5000, 1000, 20),
            'covariate': np.random.normal(0, 1, 20)
        })
    
    def test_basic_assignment(self, sample_geo_features):
        """Test basic K-means assignment."""
        assignment_method = KMeansEmbeddingAssignment()
        
        assignment_df = assignment_method.assign(sample_geo_features, seed=42)
        
        # Check structure - now includes cluster column
        expected_cols = {'geo', 'assignment', 'cluster'}
        assert set(assignment_df.columns) == expected_cols
        assert len(assignment_df) == 20
        
        # Check valid assignments
        assignments = set(assignment_df['assignment'])
        assert assignments == {'treatment', 'control'}
        
        # Should have some geos in each group
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        assert 0 < treatment_count < 20
        
        # Check cluster information
        clusters = assignment_df['cluster'].unique()
        assert len(clusters) <= 4  # Default n_clusters
    
    def test_custom_feature_columns(self, sample_geo_features):
        """Test assignment with custom feature columns."""
        assignment_method = KMeansEmbeddingAssignment(feature_cols=['base_sales', 'base_spend'])
        
        assignment_df = assignment_method.assign(sample_geo_features, seed=42)
        
        assert len(assignment_df) == 20
        assert set(assignment_df['assignment']) == {'treatment', 'control'}
        assert 'cluster' in assignment_df.columns
    
    def test_custom_n_clusters(self, sample_geo_features):
        """Test assignment with custom number of clusters."""
        assignment_method = KMeansEmbeddingAssignment(n_clusters=3)
        
        assignment_df = assignment_method.assign(sample_geo_features, seed=42)
        
        clusters = assignment_df['cluster'].unique()
        assert len(clusters) <= 3
    
    def test_reproducibility(self, sample_geo_features):
        """Test that same seed produces same assignment."""
        assignment_method = KMeansEmbeddingAssignment()
        
        assignment1 = assignment_method.assign(sample_geo_features, seed=123)
        assignment2 = assignment_method.assign(sample_geo_features, seed=123)
        
        pd.testing.assert_frame_equal(assignment1, assignment2)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        assignment_method = KMeansEmbeddingAssignment()
        
        # Missing geo column
        bad_features = pd.DataFrame({'bad_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="must contain 'geo' column"):
            assignment_method.assign(bad_features)
        
        # No suitable feature columns
        minimal_features = pd.DataFrame({'geo': ['geo_1', 'geo_2']})
        with pytest.raises(ValueError, match="No suitable feature columns found"):
            assignment_method.assign(minimal_features)


class TestPrognosticScoreAssignment:
    """Test cases for PrognosticScoreAssignment."""
    
    @pytest.fixture
    def sample_geo_features(self):
        """Sample geo features for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(15)],
            'base_sales': np.random.normal(10000, 2000, 15),
            'base_spend': np.random.normal(5000, 1000, 15),
            'covariate': np.random.normal(0, 1, 15)
        })
    
    @pytest.fixture  
    def sample_panel_data(self):
        """Sample panel data for prognostic scoring."""
        np.random.seed(42)
        geos = [f'geo_{i:03d}' for i in range(15)]
        dates = pd.date_range('2024-01-01', periods=20)
        
        records = []
        for geo in geos:
            base_sales = np.random.normal(10000, 1000)
            for date in dates:
                records.append({
                    'geo': geo,
                    'date': date,
                    'sales': base_sales + np.random.normal(0, 500)
                })
        
        return pd.DataFrame(records)
    
    def test_basic_assignment_with_base_sales(self, sample_geo_features):
        """Test basic prognostic assignment using base_sales."""
        assignment_method = PrognosticScoreAssignment()
        
        assignment_df = assignment_method.assign(sample_geo_features, seed=42)
        
        # Check structure - now includes cluster column
        expected_cols = {'geo', 'assignment', 'cluster'}
        assert set(assignment_df.columns) == expected_cols
        assert len(assignment_df) == 15
        assert set(assignment_df['assignment']) == {'treatment', 'control'}
        
        # Check strata information
        strata = assignment_df['cluster'].unique()
        assert len(strata) <= 4  # Default n_strata
    
    def test_assignment_with_panel_data(self, sample_geo_features, sample_panel_data):
        """Test prognostic assignment using panel data."""
        assignment_method = PrognosticScoreAssignment()
        
        assignment_df = assignment_method.assign(
            sample_geo_features, 
            seed=42,
            panel_data=sample_panel_data
        )
        
        assert len(assignment_df) == 15
        assert set(assignment_df['assignment']) == {'treatment', 'control'}
        assert 'cluster' in assignment_df.columns
    
    def test_custom_parameters(self, sample_geo_features):
        """Test assignment with custom parameters."""
        assignment_method = PrognosticScoreAssignment(
            covariate_cols=['base_sales', 'covariate'],
            pre_period_fraction=0.3,
            n_strata=3
        )
        
        assignment_df = assignment_method.assign(sample_geo_features, seed=42)
        
        assert len(assignment_df) == 15
        assert set(assignment_df['assignment']) == {'treatment', 'control'}
        
        # Check custom n_strata
        strata = assignment_df['cluster'].unique()
        assert len(strata) <= 3
    
    def test_reproducibility(self, sample_geo_features):
        """Test reproducibility with same seed."""
        assignment_method = PrognosticScoreAssignment()
        
        assignment1 = assignment_method.assign(sample_geo_features, seed=123)
        assignment2 = assignment_method.assign(sample_geo_features, seed=123)
        
        pd.testing.assert_frame_equal(assignment1, assignment2)
    
    def test_error_handling(self):
        """Test error handling."""
        assignment_method = PrognosticScoreAssignment()
        
        # Missing geo column
        bad_features = pd.DataFrame({'bad_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="must contain 'geo' column"):
            assignment_method.assign(bad_features)
        
        # No suitable covariate columns
        minimal_features = pd.DataFrame({'geo': ['geo_1', 'geo_2']})
        with pytest.raises(ValueError, match="No suitable covariate columns found"):
            assignment_method.assign(minimal_features)


class TestSpatialUtils:
    """Test cases for spatial utilities."""
    
    @pytest.fixture
    def sample_spatial_features(self):
        """Sample features with spatial coordinates."""
        np.random.seed(42)
        return pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(10)],
            'xy1': np.random.uniform(0, 100, 10),
            'xy2': np.random.uniform(0, 100, 10),
            'other_feature': np.random.normal(0, 1, 10)
        })
    
    def test_add_spectral_spatial_embedding(self, sample_spatial_features):
        """Test adding spectral spatial embedding."""
        result_df = add_spectral_spatial_embedding(
            sample_spatial_features,
            spatial_emb_dim=2,
            spatial_neighbors=3,
            seed=42
        )
        
        # Check that embedding columns were added
        expected_cols = set(sample_spatial_features.columns) | {'spatial_emb_0', 'spatial_emb_1'}
        assert set(result_df.columns) == expected_cols
        assert len(result_df) == 10
        
        # Check that embedding values are numeric
        assert result_df['spatial_emb_0'].dtype in [np.float32, np.float64]
        assert result_df['spatial_emb_1'].dtype in [np.float32, np.float64]
    
    def test_spatial_embedding_error_handling(self):
        """Test error handling in spatial embedding."""
        # Missing spatial columns
        bad_df = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'other': [1, 2]
        })
        
        with pytest.raises(ValueError, match="Missing spatial columns"):
            add_spectral_spatial_embedding(bad_df)
        
        # Too few geos for neighbors
        small_df = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'xy1': [1, 2],
            'xy2': [3, 4]
        })
        
        with pytest.raises(ValueError, match="Need at least"):
            add_spectral_spatial_embedding(small_df, spatial_neighbors=5)
    
    def test_standardize_features(self):
        """Test feature standardization."""
        features = np.array([
            [1, 10],
            [2, 20],
            [3, 30]
        ])
        
        standardized = standardize_features(features)
        
        # Should have zero mean (approximately)
        assert np.allclose(standardized.mean(axis=0), 0, atol=1e-10)
        
        # Should have unit variance (approximately) 
        assert np.allclose(standardized.std(axis=0), 1, atol=1e-10)


class TestEmbeddingBasedAssignment:
    """Test cases for EmbeddingBasedAssignment (general embedding approach)."""
    
    def create_sample_data_with_spatial(self, n_geos=20, seed=42):
        """Create sample geo data with spatial coordinates."""
        np.random.seed(seed)
        
        geo_features = pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(n_geos)],
            'base_sales': np.random.normal(10000, 2000, n_geos),
            'base_spend': np.random.normal(5000, 1000, n_geos),
            'covariate': np.random.normal(0, 1, n_geos),
            'xy1': np.random.uniform(0, 100, n_geos),
            'xy2': np.random.uniform(0, 100, n_geos)
        })
        
        return geo_features
    
    def test_basic_assignment(self):
        """Test basic hybrid embedding assignment."""
        from assignment.methods import EmbeddingBasedAssignment
        
        geo_features = self.create_sample_data_with_spatial(n_geos=20)
        assignment_method = EmbeddingBasedAssignment(
            neural_epochs=10,  # Fast training for tests
            n_clusters=3
        )
        
        assignment_df = assignment_method.assign(geo_features, seed=42)
        
        # Check structure
        assert len(assignment_df) == 20
        assert set(assignment_df.columns) == {'geo', 'assignment', 'cluster'}
        
        # Check assignments are valid
        assignments = set(assignment_df['assignment'].unique())
        assert assignments == {'treatment', 'control'}
        
        # Check we have multiple clusters
        clusters = set(assignment_df['cluster'].unique())
        assert len(clusters) == 3
        
        # Check balance (should be roughly 50/50)
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        control_count = (assignment_df['assignment'] == 'control').sum()
        assert treatment_count > 0 and control_count > 0
    
    def test_custom_parameters(self):
        """Test with custom neural and spatial parameters."""
        from assignment.methods import EmbeddingBasedAssignment
        
        geo_features = self.create_sample_data_with_spatial(n_geos=15)
        
        assignment_method = EmbeddingBasedAssignment(
            feature_cols=['base_sales', 'base_spend'],
            spatial_cols=['xy1', 'xy2'],
            neural_embedding_dim=4,
            spatial_embedding_dim=3,
            n_clusters=2,
            spatial_neighbors=5,
            neural_epochs=5,
            neural_lr=0.1
        )
        
        assignment_df = assignment_method.assign(geo_features, treatment_ratio=0.3, seed=123)
        
        # Check basic structure
        assert len(assignment_df) == 15
        assert set(assignment_df['assignment'].unique()) == {'treatment', 'control'}
        assert len(set(assignment_df['cluster'].unique())) == 2
        
        # Check treatment ratio is approximately correct
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        treatment_ratio = treatment_count / len(assignment_df)
        assert 0.1 < treatment_ratio < 0.5  # Should be roughly 30% but with some variance due to clustering
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        from assignment.methods import EmbeddingBasedAssignment
        
        geo_features = self.create_sample_data_with_spatial(n_geos=15)
        assignment_method = EmbeddingBasedAssignment(neural_epochs=5)
        
        # Run twice with same seed
        assignment1 = assignment_method.assign(geo_features, seed=999)
        assignment2 = assignment_method.assign(geo_features, seed=999)
        
        # Should be identical
        pd.testing.assert_frame_equal(assignment1, assignment2)
    
    def test_error_handling(self):
        """Test error handling."""
        from assignment.methods import EmbeddingBasedAssignment
        
        assignment_method = EmbeddingBasedAssignment()
        
        # Missing geo column
        bad_features = pd.DataFrame({'bad_col': [1, 2, 3]})
        with pytest.raises(ValueError, match="must contain 'geo' column"):
            assignment_method.assign(bad_features)
        
        # Missing spatial columns
        features_no_spatial = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'base_sales': [1000, 2000]
        })
        with pytest.raises(ValueError, match="Missing spatial columns"):
            assignment_method.assign(features_no_spatial)
        
        # No suitable feature columns
        minimal_features = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'xy1': [1, 2],
            'xy2': [3, 4]
        })
        with pytest.raises(ValueError, match="No suitable feature columns found"):
            assignment_method.assign(minimal_features)
    
    def test_small_dataset(self):
        """Test with small dataset."""
        from assignment.methods import EmbeddingBasedAssignment
        
        # Small dataset with minimal neighbors
        geo_features = self.create_sample_data_with_spatial(n_geos=8, seed=456)
        
        assignment_method = EmbeddingBasedAssignment(
            n_clusters=2,
            spatial_neighbors=3,  # Reduced for small dataset
            neural_epochs=5
        )
        
        assignment_df = assignment_method.assign(geo_features, seed=456)
        
        # Should still work
        assert len(assignment_df) == 8
        assert set(assignment_df['assignment'].unique()) == {'treatment', 'control'}
        
        # Check both treatment and control are present
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        control_count = (assignment_df['assignment'] == 'control').sum()
        assert treatment_count > 0 and control_count > 0
    
    def test_embedding_integration(self):
        """Test that neural and spatial embeddings are properly integrated."""
        from assignment.methods import EmbeddingBasedAssignment
        
        # Create data where spatial and feature patterns differ
        np.random.seed(789)
        geo_features = pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(12)],
            'base_sales': [1000, 1100, 5000, 5100, 9000, 9100] * 2,  # 3 distinct groups
            'base_spend': [500, 550, 2500, 2550, 4500, 4550] * 2,
            'covariate': np.random.normal(0, 1, 12),
            'xy1': np.repeat([10, 50, 90], 4),  # 3 spatial clusters
            'xy2': np.repeat([20, 60, 80], 4)
        })
        
        assignment_method = EmbeddingBasedAssignment(
            n_clusters=3,
            neural_epochs=10,
            spatial_neighbors=3
        )
        
        assignment_df = assignment_method.assign(geo_features, seed=789)
        
        # Should create meaningful clusters that consider both feature similarity and spatial proximity
        assert len(assignment_df) == 12
        assert len(set(assignment_df['cluster'].unique())) == 3
        
        # Each cluster should have a mix of treatment and control (ideally)
        cluster_assignments = assignment_df.groupby('cluster')['assignment'].value_counts().unstack(fill_value=0)
        
        # At least 2 clusters should have both treatment and control
        clusters_with_both = (cluster_assignments['treatment'] > 0) & (cluster_assignments['control'] > 0)
        assert clusters_with_both.sum() >= 2


class TestHybridEmbeddingAssignment:
    """Test cases for HybridEmbeddingAssignment (semi-supervised approach)."""
    
    def create_sample_data_with_panel(self, n_geos=15, n_days=60, seed=42):
        """Create sample geo data with panel time series."""
        np.random.seed(seed)
        
        # Create geo features with spatial coordinates
        geo_features = pd.DataFrame({
            'geo': [f'geo_{i:03d}' for i in range(n_geos)],
            'xy1': np.random.uniform(0, 100, n_geos),
            'xy2': np.random.uniform(0, 100, n_geos)
        })
        
        # Create panel data with time series
        dates = pd.date_range('2024-01-01', periods=n_days)
        panel_data = []
        
        for _, geo_row in geo_features.iterrows():
            base_sales = np.random.normal(10000, 2000)
            for date in dates:
                panel_data.append({
                    'geo': geo_row['geo'],
                    'date': date,
                    'sales': base_sales + np.random.normal(0, 500),
                    'spend_dollars': np.random.normal(5000, 1000)
                })
        
        panel_df = pd.DataFrame(panel_data)
        return geo_features, panel_df
    
    def test_basic_assignment(self):
        """Test basic hybrid embedding assignment with panel data."""
        from assignment.methods import HybridEmbeddingAssignment
        
        geo_features, panel_data = self.create_sample_data_with_panel(n_geos=12, n_days=40)
        
        assignment_method = HybridEmbeddingAssignment(
            neural_epochs=5,  # Fast training for tests
            n_clusters=3
        )
        
        assignment_df = assignment_method.assign(
            geo_features, panel_data=panel_data, seed=42
        )
        
        # Check structure
        assert len(assignment_df) == 12
        assert set(assignment_df.columns) == {'geo', 'assignment', 'cluster'}
        
        # Check assignments are valid
        assignments = set(assignment_df['assignment'].unique())
        assert assignments == {'treatment', 'control'}
        
        # Check we have multiple clusters
        clusters = set(assignment_df['cluster'].unique())
        assert len(clusters) == 3
        
        # Check balance
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        control_count = (assignment_df['assignment'] == 'control').sum()
        assert treatment_count > 0 and control_count > 0
    
    def test_custom_parameters(self):
        """Test with custom hybrid parameters."""
        from assignment.methods import HybridEmbeddingAssignment
        
        geo_features, panel_data = self.create_sample_data_with_panel(n_geos=10, n_days=30)
        
        assignment_method = HybridEmbeddingAssignment(
            prediction_fraction=0.2,  # Use 20% for prediction
            reconstruction_weight=0.4,
            prediction_weight=0.4,
            regularization_weight=0.2,
            neural_embedding_dim=6,
            spatial_embedding_dim=3,
            n_clusters=2,
            neural_epochs=3
        )
        
        assignment_df = assignment_method.assign(
            geo_features, panel_data=panel_data, treatment_ratio=0.3, seed=123
        )
        
        # Check basic structure
        assert len(assignment_df) == 10
        assert set(assignment_df['assignment'].unique()) == {'treatment', 'control'}
        assert len(set(assignment_df['cluster'].unique())) == 2
        
        # Check treatment ratio is approximately correct
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        treatment_ratio = treatment_count / len(assignment_df)
        assert 0.1 < treatment_ratio < 0.5  # Should be roughly 30% but with clustering variance
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        from assignment.methods import HybridEmbeddingAssignment
        
        geo_features, panel_data = self.create_sample_data_with_panel(n_geos=10, n_days=25)
        assignment_method = HybridEmbeddingAssignment(
            neural_epochs=3,
            spatial_neighbors=5  # Reduced for smaller dataset
        )
        
        # Run twice with same seed
        assignment1 = assignment_method.assign(geo_features, panel_data=panel_data, seed=999)
        assignment2 = assignment_method.assign(geo_features, panel_data=panel_data, seed=999)
        
        # Should be identical
        pd.testing.assert_frame_equal(assignment1, assignment2)
    
    def test_error_handling(self):
        """Test error handling for hybrid assignment."""
        from assignment.methods import HybridEmbeddingAssignment
        
        assignment_method = HybridEmbeddingAssignment()
        
        # Missing panel_data
        geo_features = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'xy1': [1, 2],
            'xy2': [3, 4]
        })
        with pytest.raises(ValueError, match="panel_data is required"):
            assignment_method.assign(geo_features)
        
        # Missing geo column
        bad_features = pd.DataFrame({'bad_col': [1, 2, 3]})
        panel_data = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'date': ['2024-01-01', '2024-01-02'],
            'sales': [100, 200]
        })
        with pytest.raises(ValueError, match="must contain 'geo' column"):
            assignment_method.assign(bad_features, panel_data=panel_data)
        
        # Missing spatial columns
        features_no_spatial = pd.DataFrame({
            'geo': ['geo_1', 'geo_2']
        })
        with pytest.raises(ValueError, match="Missing spatial columns"):
            assignment_method.assign(features_no_spatial, panel_data=panel_data)
        
        # Missing panel data columns
        incomplete_panel = pd.DataFrame({
            'geo': ['geo_1', 'geo_2'],
            'date': ['2024-01-01', '2024-01-02']
            # Missing 'sales' column
        })
        with pytest.raises(ValueError, match="Missing panel data columns"):
            assignment_method.assign(geo_features, panel_data=incomplete_panel)
    
    def test_loss_weight_validation(self):
        """Test that loss weights must sum to 1.0."""
        from assignment.methods import HybridEmbeddingAssignment
        
        # Weights that don't sum to 1.0
        with pytest.raises(ValueError, match="Loss weights must sum to 1.0"):
            HybridEmbeddingAssignment(
                reconstruction_weight=0.5,
                prediction_weight=0.5,
                regularization_weight=0.5  # Total = 1.5, not 1.0
            )
        
        # Valid weights should work
        method = HybridEmbeddingAssignment(
            reconstruction_weight=0.6,
            prediction_weight=0.3,
            regularization_weight=0.1  # Total = 1.0
        )
        assert method.reconstruction_weight == 0.6
    
    def test_time_series_preparation(self):
        """Test time series data preparation logic."""
        from assignment.methods import HybridEmbeddingAssignment
        
        geo_features, panel_data = self.create_sample_data_with_panel(n_geos=5, n_days=20)
        
        assignment_method = HybridEmbeddingAssignment(
            pre_period_fraction=0.7,  # Use 70% for pre-period
            prediction_fraction=0.2   # Use 20% for prediction
        )
        
        # Test the internal time series preparation
        pre_period_data, prediction_targets, common_geos = assignment_method._prepare_time_series_data(panel_data)
        
        # Check dimensions
        assert len(common_geos) == 5  # All geos should be present
        assert pre_period_data.shape[0] == 5  # n_geos
        assert pre_period_data.shape[1] == 14  # 70% of 20 days = 14 days
        assert len(prediction_targets) == 5  # One target per geo
        
        # Check that targets are reasonable (not NaN or extreme)
        assert not np.any(np.isnan(prediction_targets))
        assert np.all(prediction_targets > 0)  # Sales should be positive
    
    def test_small_dataset(self):
        """Test with small dataset."""
        from assignment.methods import HybridEmbeddingAssignment
        
        geo_features, panel_data = self.create_sample_data_with_panel(n_geos=6, n_days=15)
        
        assignment_method = HybridEmbeddingAssignment(
            n_clusters=2,
            spatial_neighbors=3,  # Reduced for small dataset
            neural_epochs=3
        )
        
        assignment_df = assignment_method.assign(
            geo_features, panel_data=panel_data, seed=456
        )
        
        # Should still work
        assert len(assignment_df) == 6
        assert set(assignment_df['assignment'].unique()) == {'treatment', 'control'}
        
        # Check both treatment and control are present
        treatment_count = (assignment_df['assignment'] == 'treatment').sum()
        control_count = (assignment_df['assignment'] == 'control').sum()
        assert treatment_count > 0 and control_count > 0