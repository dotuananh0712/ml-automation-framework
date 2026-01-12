# ML Automation Framework

A Python-first automation framework for standardizing data science workflows.

## Overview

ML Automation Framework provides a config-driven approach to building, training, and deploying machine learning models. It integrates with MLflow for experiment tracking and supports both local and Databricks environments.

## Key Features

- **Config-Driven Pipelines**: Define your entire ML pipeline in YAML
- **Multiple Pipeline Types**: Classification, Regression, Forecasting, Many-Model
- **Automatic Feature Engineering**: Built-in transformers for numeric and categorical features
- **MLflow Integration**: Automatic logging of parameters, metrics, and models
- **Cross-Runtime Support**: Works locally and on Databricks
- **Data Validation**: Great Expectations integration for data quality checks
- **Model Explainability**: SHAP-based feature importance and explanations
- **Hyperparameter Tuning**: Optuna integration for automated optimization
- **Model Deployment**: Deploy to Databricks Model Serving

## Quick Start

### Installation

```bash
pip install ml-automation-framework

# With optional features
pip install "ml-automation-framework[all-features]"
```

### Basic Usage

1. **Create a configuration file** (`config.yaml`):

```yaml
name: churn_prediction
pipeline_type: classification

data:
  source: data/customers.parquet
  target_column: churn
  train_ratio: 0.8

model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100
    max_depth: 6

mlflow:
  experiment_name: /experiments/churn
  log_model: true
```

2. **Train the model**:

```bash
mlf train config.yaml
```

3. **Evaluate on new data**:

```bash
mlf evaluate <run_id> data/test.parquet
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `mlf train` | Train a model using YAML config |
| `mlf evaluate` | Evaluate a trained model on new data |
| `mlf validate` | Validate a configuration file |
| `mlf tune` | Run hyperparameter tuning |
| `mlf deploy` | Deploy model to serving endpoint |
| `mlf init` | Generate a starter config file |
| `mlf list-models` | List available model types |

## Next Steps

- [Getting Started Guide](user-guide/getting-started.md)
- [Configuration Reference](user-guide/configuration.md)
- [Developer Guide](developer-guide/architecture.md)
