"""Tests for feature transformation."""

import numpy as np
import pandas as pd
import pytest

from ml_framework.config.base import FeatureConfig
from ml_framework.features.transformer import FeatureTransformer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "numeric_1": [1.0, 2.0, np.nan, 4.0, 5.0],
        "numeric_2": [10, 20, 30, 40, 50],
        "categorical_1": ["a", "b", "a", "c", "b"],
        "categorical_2": ["x", "y", "x", np.nan, "y"],
    })


@pytest.fixture
def feature_config():
    """Default feature configuration."""
    return FeatureConfig(
        numeric_impute_strategy="median",
        numeric_scaling="standard",
        categorical_encoding="onehot",
    )


class TestFeatureTransformer:
    """Tests for FeatureTransformer."""

    def test_fit_transform(self, sample_data, feature_config):
        """Test fit_transform produces valid output."""
        transformer = FeatureTransformer(feature_config)
        result = transformer.fit_transform(sample_data)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_data)
        assert not np.isnan(result).any()

    def test_transform_after_fit(self, sample_data, feature_config):
        """Test transform works after fitting."""
        transformer = FeatureTransformer(feature_config)
        transformer.fit(sample_data)

        new_data = sample_data.iloc[:3].copy()
        result = transformer.transform(new_data)

        assert result.shape[0] == 3

    def test_transform_without_fit_raises(self, sample_data, feature_config):
        """Test that transform without fit raises error."""
        transformer = FeatureTransformer(feature_config)

        with pytest.raises(RuntimeError, match="not fitted"):
            transformer.transform(sample_data)

    def test_numeric_imputation(self, sample_data, feature_config):
        """Test that NaN values are imputed."""
        transformer = FeatureTransformer(feature_config)
        result = transformer.fit_transform(sample_data)

        # No NaN values should remain
        assert not np.isnan(result).any()

    def test_get_feature_names(self, sample_data, feature_config):
        """Test feature name extraction."""
        transformer = FeatureTransformer(feature_config)
        transformer.fit_transform(sample_data)

        names = transformer.get_feature_names()
        assert len(names) > 0
        assert isinstance(names, list)

    def test_no_scaling(self, sample_data):
        """Test transformer without scaling."""
        config = FeatureConfig(
            numeric_scaling=None,
            categorical_encoding="onehot",
        )
        transformer = FeatureTransformer(config)
        result = transformer.fit_transform(sample_data)

        # Values should not be standardized
        assert result.max() > 10  # Original numeric values preserved

    def test_column_types_detected(self, sample_data, feature_config):
        """Test column type detection."""
        transformer = FeatureTransformer(feature_config)
        transformer.fit(sample_data)

        col_types = transformer.get_column_types()
        assert "numeric" in col_types
        assert "categorical" in col_types
        assert len(col_types["numeric"]) == 2
        assert len(col_types["categorical"]) == 2
