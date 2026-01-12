"""Pytest fixtures for integration tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from ml_framework.cli import app


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def integration_temp_dir():
    """Create a temporary directory for integration test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_mlflow_tracking(integration_temp_dir, monkeypatch):
    """Configure MLflow to use temp directory for tracking."""
    tracking_dir = integration_temp_dir / "mlruns"
    tracking_dir.mkdir()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tracking_dir}")
    yield tracking_dir


@pytest.fixture
def sample_churn_data():
    """Generate sample churn classification dataset."""
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(1, n_samples + 1)],
        "age": np.random.randint(18, 75, n_samples),
        "tenure_months": np.random.randint(1, 72, n_samples),
        "monthly_charges": np.round(np.random.uniform(20, 150, n_samples), 2),
        "total_charges": np.round(np.random.uniform(100, 5000, n_samples), 2),
        "churn": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    })

    return data


@pytest.fixture
def sample_churn_data_path(integration_temp_dir, sample_churn_data):
    """Save sample churn data to parquet and return path."""
    data_path = integration_temp_dir / "churn_data.parquet"
    sample_churn_data.to_parquet(data_path, index=False)
    return data_path


@pytest.fixture
def classification_config_dict(sample_churn_data_path):
    """Sample classification pipeline configuration."""
    return {
        "name": "integration_test_classification",
        "description": "Integration test classification pipeline",
        "pipeline_type": "classification",
        "data": {
            "source": str(sample_churn_data_path),
            "format": "parquet",
            "target_column": "churn",
            "feature_columns": ["age", "tenure_months", "monthly_charges", "total_charges"],
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
        },
        "features": {
            "numeric_impute_strategy": "median",
            "numeric_scaling": "standard",
        },
        "model": {
            "model_type": "random_forest",
            "hyperparameters": {
                "n_estimators": 10,
                "max_depth": 3,
            },
            "cross_validation": False,
            "random_state": 42,
        },
        "mlflow": {
            "experiment_name": "/integration_tests/classification",
            "log_model": True,
            "log_feature_importance": True,
        },
    }


@pytest.fixture
def classification_config_path(integration_temp_dir, classification_config_dict):
    """Save classification config to YAML and return path."""
    config_path = integration_temp_dir / "test_classification.yaml"
    with open(config_path, "w") as f:
        yaml.dump(classification_config_dict, f, default_flow_style=False)
    return config_path


@pytest.fixture
def invalid_yaml_path(integration_temp_dir):
    """Create an invalid YAML file for error testing."""
    config_path = integration_temp_dir / "invalid.yaml"
    with open(config_path, "w") as f:
        f.write("name: test\n  invalid_indent: true")  # Invalid indentation
    return config_path


@pytest.fixture
def missing_column_config_path(integration_temp_dir, sample_churn_data_path):
    """Create config with non-existent column."""
    config = {
        "name": "missing_column_test",
        "pipeline_type": "classification",
        "data": {
            "source": str(sample_churn_data_path),
            "format": "parquet",
            "target_column": "nonexistent_column",
            "train_ratio": 0.8,
        },
        "model": {
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 10},
        },
        "mlflow": {
            "experiment_name": "/integration_tests/error_test",
        },
    }
    config_path = integration_temp_dir / "missing_column.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path
