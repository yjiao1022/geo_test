"""
Tests for assignment module.
"""

import pytest
import pandas as pd
import numpy as np

from assignment.methods import RandomAssignment


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
        
        # Check structure
        assert set(assignment_df.columns) == {'geo', 'assignment'}
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