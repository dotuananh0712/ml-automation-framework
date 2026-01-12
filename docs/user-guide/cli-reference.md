# CLI Reference

Complete reference for all CLI commands.

## Commands Overview

```bash
mlf --help
```

## train

Train a model using a YAML configuration file.

```bash
mlf train <config_path> [OPTIONS]
```

**Arguments:**

- `config_path`: Path to pipeline YAML config (required)

**Options:**

- `--dry-run`: Validate config without running

**Example:**

```bash
mlf train configs/classification/churn.yaml
mlf train configs/classification/churn.yaml --dry-run
```

## evaluate

Evaluate a trained model on new data.

```bash
mlf evaluate <run_id> <data_path> [OPTIONS]
```

**Arguments:**

- `run_id`: MLflow run ID of trained model (required)
- `data_path`: Path to evaluation data (required)

**Options:**

- `--format, -f`: Data format (parquet, csv). Default: parquet

**Example:**

```bash
mlf evaluate abc123def456 data/test.parquet
mlf evaluate abc123def456 data/test.csv --format csv
```

## validate

Validate a pipeline configuration file.

```bash
mlf validate <config_path>
```

**Arguments:**

- `config_path`: Path to pipeline YAML config (required)

**Example:**

```bash
mlf validate configs/classification/churn.yaml
```

## tune

Run hyperparameter tuning using Optuna.

```bash
mlf tune <config_path> [OPTIONS]
```

**Arguments:**

- `config_path`: Path to pipeline YAML config (required)

**Options:**

- `--trials, -n`: Override number of trials

**Example:**

```bash
mlf tune configs/classification/churn_tuning.yaml
mlf tune configs/classification/churn_tuning.yaml --trials 50
```

## deploy

Deploy a trained model to a serving endpoint.

```bash
mlf deploy <run_id> [OPTIONS]
```

**Arguments:**

- `run_id`: MLflow run ID of model to deploy (required)

**Options:**

- `--endpoint, -e`: Endpoint name (required)
- `--size, -s`: Workload size (Small, Medium, Large). Default: Small
- `--scale-to-zero/--no-scale-to-zero`: Enable scale to zero. Default: true

**Example:**

```bash
mlf deploy abc123def456 --endpoint my-model-endpoint
mlf deploy abc123def456 --endpoint my-model-endpoint --size Medium --no-scale-to-zero
```

## init

Generate a starter configuration file.

```bash
mlf init <name> [OPTIONS]
```

**Arguments:**

- `name`: Pipeline name (required)

**Options:**

- `--type, -t`: Pipeline type (classification, forecasting). Default: classification
- `--output, -o`: Output path. Default: configs/{type}/{name}.yaml

**Example:**

```bash
mlf init my_pipeline
mlf init my_forecast -t forecasting -o my_forecast.yaml
```

## list-models

List available model types.

```bash
mlf list-models
```

**Example output:**

```
        Available Models
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Model Type            ┃ Tasks                           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ logistic_regression   │ Classification                  │
│ linear_regression     │ Regression, Forecasting         │
│ random_forest         │ Classification, Regression, ... │
│ xgboost               │ Classification, Regression, ... │
│ lightgbm              │ Classification, Regression, ... │
└───────────────────────┴─────────────────────────────────┘
```

## Error Handling

All commands provide user-friendly error messages:

```bash
# Missing file
mlf train nonexistent.yaml
# Output: File not found: nonexistent.yaml
#         Suggestion: Check path and ensure file exists

# Invalid YAML
mlf train invalid.yaml
# Output: Invalid YAML: <parse error details>
#         Suggestion: Validate YAML syntax

# Config validation error
mlf train bad_config.yaml
# Output: Configuration validation failed:
#           data -> train_ratio: value must be between 0.1 and 0.95
```
