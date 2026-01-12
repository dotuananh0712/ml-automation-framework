"""Tests for configuration loading and validation."""

import pytest
from pydantic import ValidationError

from ml_framework.config.base import (
    DataConfig,
    FeatureConfig,
    ModelConfig,
    ModelType,
    PipelineConfig,
    PipelineType,
    MLflowConfig,
)
from ml_framework.config.loader import validate_config


class TestDataConfig:
    """Tests for DataConfig validation."""

    def test_valid_config(self):
        """Test valid data configuration."""
        config = DataConfig(
            source="data/train.parquet",
            target_column="target",
            train_ratio=0.8,
        )
        assert config.source == "data/train.parquet"
        assert config.target_column == "target"
        assert config.train_ratio == 0.8

    def test_invalid_train_ratio(self):
        """Test that invalid train ratio raises error."""
        with pytest.raises(ValidationError):
            DataConfig(
                source="data.csv",
                target_column="target",
                train_ratio=1.5,  # Invalid: > 1
            )

    def test_auto_detect_features(self):
        """Test that feature_columns defaults to None (auto-detect)."""
        config = DataConfig(
            source="data.csv",
            target_column="target",
        )
        assert config.feature_columns is None


class TestModelConfig:
    """Tests for ModelConfig validation."""

    def test_valid_xgboost_config(self):
        """Test valid XGBoost configuration."""
        config = ModelConfig(
            model_type=ModelType.XGBOOST,
            hyperparameters={"n_estimators": 100, "max_depth": 6},
        )
        assert config.model_type == ModelType.XGBOOST
        assert config.hyperparameters["n_estimators"] == 100

    def test_default_values(self):
        """Test default model configuration values."""
        config = ModelConfig(model_type=ModelType.RANDOM_FOREST)
        assert config.cross_validation is True
        assert config.cv_folds == 5
        assert config.random_state == 42


class TestPipelineConfig:
    """Tests for complete pipeline configuration."""

    def test_valid_classification_config(self):
        """Test valid classification pipeline config."""
        config_dict = {
            "name": "test_pipeline",
            "pipeline_type": "classification",
            "data": {
                "source": "data.parquet",
                "target_column": "target",
            },
            "model": {
                "model_type": "xgboost",
            },
            "mlflow": {
                "experiment_name": "/test/experiment",
            },
        }
        config = validate_config(config_dict)
        assert config.name == "test_pipeline"
        assert config.pipeline_type == PipelineType.CLASSIFICATION

    def test_extra_fields_rejected(self):
        """Test that extra fields are rejected."""
        config_dict = {
            "name": "test",
            "pipeline_type": "classification",
            "data": {"source": "data.csv", "target_column": "target"},
            "model": {"model_type": "xgboost"},
            "mlflow": {"experiment_name": "/test"},
            "unknown_field": "should fail",
        }
        with pytest.raises(ValidationError):
            validate_config(config_dict)


class TestFeatureConfig:
    """Tests for feature configuration."""

    def test_default_feature_config(self):
        """Test default feature configuration."""
        config = FeatureConfig()
        assert config.numeric_impute_strategy == "median"
        assert config.categorical_encoding == "onehot"
        assert config.numeric_scaling == "standard"

    def test_no_scaling(self):
        """Test disabling numeric scaling."""
        config = FeatureConfig(numeric_scaling=None)
        assert config.numeric_scaling is None
