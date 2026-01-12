"""Tests for CatBoost model integration."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from ml_framework.config.base import ModelConfig, ModelType, PipelineType
from ml_framework.models.factory import create_model, get_model_type_for_task


class TestCatBoostFactory:
    """Tests for CatBoost model creation via factory."""

    def test_create_catboost_classifier(self) -> None:
        """Test CatBoostClassifier creation."""
        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 10, "depth": 3, "verbose": False},
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        assert model is not None
        assert "CatBoostClassifier" in type(model).__name__

    def test_create_catboost_regressor(self) -> None:
        """Test CatBoostRegressor creation."""
        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 10, "depth": 3, "verbose": False},
        )
        model = create_model(config, PipelineType.REGRESSION)

        assert model is not None
        assert "CatBoostRegressor" in type(model).__name__

    def test_catboost_default_params(self) -> None:
        """Test that default parameters are applied."""
        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"verbose": False},
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        params = model.get_params()
        assert params.get("depth") == 6
        assert params.get("learning_rate") == 0.1

    def test_catboost_random_state_mapping(self) -> None:
        """Test that random_state is mapped to random_seed."""
        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"verbose": False},
            random_state=123,
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        params = model.get_params()
        assert params.get("random_seed") == 123

    def test_catboost_early_stopping(self) -> None:
        """Test CatBoost with early stopping enabled."""
        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 100, "verbose": False},
            early_stopping=True,
            early_stopping_rounds=10,
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        params = model.get_params()
        assert params.get("early_stopping_rounds") == 10

    def test_catboost_in_classification_models_list(self) -> None:
        """Test that CatBoost is in classification models list."""
        models = get_model_type_for_task("classification")
        assert ModelType.CATBOOST in models

    def test_catboost_in_regression_models_list(self) -> None:
        """Test that CatBoost is in regression models list."""
        models = get_model_type_for_task("regression")
        assert ModelType.CATBOOST in models


class TestCatBoostFitPredict:
    """Tests for CatBoost fit and predict functionality."""

    def test_catboost_classifier_fit_predict(self) -> None:
        """Test CatBoost classifier can fit and predict."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 10, "verbose": False},
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})

    def test_catboost_classifier_predict_proba(self) -> None:
        """Test CatBoost classifier supports predict_proba."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 10, "verbose": False},
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        model.fit(X, y)
        proba = model.predict_proba(X)

        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_catboost_regressor_fit_predict(self) -> None:
        """Test CatBoost regressor can fit and predict."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 10, "verbose": False},
        )
        model = create_model(config, PipelineType.REGRESSION)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert predictions.dtype == np.float64

    def test_catboost_with_eval_set(self) -> None:
        """Test CatBoost with early stopping eval_set."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        config = ModelConfig(
            model_type=ModelType.CATBOOST,
            hyperparameters={"iterations": 100, "verbose": False},
            early_stopping=True,
            early_stopping_rounds=10,
        )
        model = create_model(config, PipelineType.CLASSIFICATION)

        # Train with eval_set for early stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        # Should stop before 100 iterations if early stopping works
        predictions = model.predict(X_val)
        assert len(predictions) == len(y_val)
