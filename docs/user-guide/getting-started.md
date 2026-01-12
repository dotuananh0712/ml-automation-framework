# Getting Started

This guide walks you through installing and using the ML Automation Framework.

## Prerequisites

- Python 3.10 or higher
- pip or uv package manager

## Installation

### Basic Installation

```bash
pip install ml-automation-framework
```

### With Optional Features

```bash
# All features
pip install "ml-automation-framework[all-features]"

# Individual features
pip install "ml-automation-framework[validation]"      # Great Expectations
pip install "ml-automation-framework[explainability]"  # SHAP
pip install "ml-automation-framework[tuning]"          # Optuna
```

### Development Installation

```bash
git clone https://github.com/org/ml-automation-framework.git
cd ml-automation-framework
pip install -e ".[dev]"
```

## Your First Pipeline

### 1. Prepare Your Data

The framework supports Parquet and CSV files. Ensure your data has:

- Feature columns (numeric and/or categorical)
- A target column for prediction

Example dataset structure:

| age | tenure | monthly_charges | churn |
|-----|--------|-----------------|-------|
| 25  | 12     | 50.5            | 0     |
| 45  | 24     | 75.0            | 0     |
| 32  | 3      | 89.99           | 1     |

### 2. Create Configuration

Create a YAML file (`my_pipeline.yaml`):

```yaml
name: my_first_pipeline
description: My first ML pipeline
pipeline_type: classification

data:
  source: path/to/data.parquet
  format: parquet
  target_column: churn
  feature_columns:
    - age
    - tenure
    - monthly_charges
  train_ratio: 0.8
  validation_ratio: 0.1

features:
  numeric_impute_strategy: median
  numeric_scaling: standard

model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
  cross_validation: true
  cv_folds: 5

mlflow:
  experiment_name: /experiments/my_first_pipeline
  log_model: true
  log_feature_importance: true
```

### 3. Validate Configuration

Before training, validate your config:

```bash
mlf validate my_pipeline.yaml
```

### 4. Train the Model

```bash
mlf train my_pipeline.yaml
```

Expected output:

```
Loading config: my_pipeline.yaml
Config validated: my_first_pipeline
  Pipeline type: classification
  Model: xgboost
  Data source: path/to/data.parquet

Training complete!
MLflow Run ID: abc123def456

         Metrics
┏━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric     ┃ Value  ┃
┡━━━━━━━━━━━━╇━━━━━━━━┩
│ accuracy   │ 0.8500 │
│ f1         │ 0.7200 │
│ precision  │ 0.7800 │
│ recall     │ 0.6700 │
│ roc_auc    │ 0.8900 │
└────────────┴────────┘
```

### 5. View Results in MLflow

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Navigate to `http://localhost:5000` to see your experiment.

### 6. Evaluate on New Data

```bash
mlf evaluate abc123def456 path/to/test_data.parquet
```

## Next Steps

- Learn about [Configuration Options](configuration.md)
- Explore [Classification Pipelines](classification.md)
- Try [Forecasting Pipelines](forecasting.md)
