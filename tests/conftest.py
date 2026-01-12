"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame({
        "feature_1": np.random.randn(n_samples),
        "feature_2": np.random.randn(n_samples),
        "feature_3": np.random.choice(["a", "b", "c"], n_samples),
        "feature_4": np.random.uniform(0, 100, n_samples),
        "target": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    })

    return data


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 200

    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    noise = np.random.randn(n_samples) * 0.1

    data = pd.DataFrame({
        "feature_1": X1,
        "feature_2": X2,
        "feature_3": np.random.choice(["x", "y"], n_samples),
        "target": 2 * X1 + 3 * X2 + noise,
    })

    return data


@pytest.fixture
def sample_config_dict():
    """Sample pipeline configuration dictionary."""
    return {
        "name": "test_pipeline",
        "description": "Test pipeline for unit tests",
        "pipeline_type": "classification",
        "data": {
            "source": "test_data.parquet",
            "format": "parquet",
            "target_column": "target",
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
        },
        "features": {
            "numeric_impute_strategy": "median",
            "numeric_scaling": "standard",
            "categorical_encoding": "onehot",
        },
        "model": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10, "max_depth": 3},
            "cross_validation": False,
            "random_state": 42,
        },
        "mlflow": {
            "experiment_name": "/test/experiment",
            "log_model": False,
            "log_feature_importance": False,
        },
    }


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Mock MLflow to avoid actual tracking during tests."""
    # Set environment to disable MLflow tracking
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")
