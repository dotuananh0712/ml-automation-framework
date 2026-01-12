# Configuration Reference

Complete reference for YAML pipeline configuration.

## Configuration Structure

```yaml
name: string                    # Required: Pipeline name
description: string             # Optional: Pipeline description
pipeline_type: string           # Required: classification, regression, forecasting

data: DataConfig                # Required: Data configuration
features: FeatureConfig         # Optional: Feature engineering settings
model: ModelConfig              # Required: Model configuration
mlflow: MLflowConfig            # Required: MLflow settings

# Optional sections (new features)
data_validation: DataValidationConfig  # Great Expectations validation
explainability: ExplainabilityConfig   # SHAP explanations
tuning: TuningConfig                   # Optuna hyperparameter tuning
```

## Data Configuration

```yaml
data:
  source: string                # Path to data file or table name
  format: parquet | csv | delta | table
  target_column: string         # Target variable column
  feature_columns: list[string] # Optional: explicit feature list (auto-detect if omitted)
  date_column: string           # Optional: for time series
  id_column: string             # Optional: for tracking/grouping
  train_ratio: float            # 0.1-0.95, default 0.8
  validation_ratio: float       # 0.05-0.5, default 0.1
  stratify: bool                # Stratified split, default false
```

## Feature Configuration

```yaml
features:
  numeric_impute_strategy: median | mean | most_frequent
  numeric_scaling: standard | minmax | robust | null
  categorical_impute_strategy: most_frequent | constant
  categorical_encoding: onehot | label | target
  handle_unknown: ignore | error
  feature_selection: bool
  feature_selection_k: int      # Top K features if selection enabled
```

## Model Configuration

```yaml
model:
  model_type: string            # See available models below
  hyperparameters: dict         # Model-specific parameters
  cross_validation: bool        # Enable CV, default true
  cv_folds: int                 # 2-20, default 5
  early_stopping: bool          # For tree models
  early_stopping_rounds: int
  random_state: int             # Default 42
```

### Available Model Types

| Model Type | Tasks |
|------------|-------|
| `logistic_regression` | Classification |
| `linear_regression` | Regression, Forecasting |
| `random_forest` | Classification, Regression, Forecasting |
| `xgboost` | Classification, Regression, Forecasting |
| `lightgbm` | Classification, Regression, Forecasting |

## MLflow Configuration

```yaml
mlflow:
  experiment_name: string       # Required: Experiment path
  tracking_uri: string          # Optional: MLflow server URI
  run_name: string              # Optional: Custom run name
  tags: dict                    # Optional: Key-value tags
  log_model: bool               # Log model artifact
  log_feature_importance: bool  # Log feature importance
  register_model: bool          # Register in model registry
  model_name: string            # Name for registry
```

## Data Validation (Great Expectations)

```yaml
data_validation:
  enabled: bool
  fail_on_error: bool
  generate_data_docs: bool
  min_rows: int
  max_rows: int
  expectations:
    - column: string
      expectation: string       # GE expectation type
      kwargs: dict              # Expectation parameters
```

### Common Expectations

- `expect_column_values_to_not_be_null`
- `expect_column_values_to_be_between`
- `expect_column_values_to_be_in_set`
- `expect_column_values_to_match_regex`
- `expect_column_values_to_be_unique`

## Explainability (SHAP)

```yaml
explainability:
  enabled: bool
  explainer_type: auto | tree | linear | kernel
  max_samples: int              # Samples for SHAP calculation
  generate_summary_plot: bool
  generate_bar_plot: bool
  generate_dependence_plots: bool
  dependence_top_k: int
```

## Hyperparameter Tuning (Optuna)

```yaml
tuning:
  enabled: bool
  n_trials: int
  timeout: int                  # Seconds
  direction: maximize | minimize
  metric: string                # Metric to optimize
  sampler: tpe | random | cmaes
  register_best_model: bool
  search_space:
    - name: string              # Parameter name
      type: int | float | log_float | categorical
      low: float                # For numeric types
      high: float
      choices: list             # For categorical
```

## Environment Variables

Use `${VAR_NAME}` or `${VAR_NAME:-default}` syntax:

```yaml
data:
  source: ${DATA_PATH:-data/default.parquet}
mlflow:
  tracking_uri: ${MLFLOW_TRACKING_URI}
```
