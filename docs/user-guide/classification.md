
# Classification Pipelines

Guide to building classification models with ML Automation Framework.

## Overview

Classification pipelines predict categorical outcomes (binary or multi-class).

## Basic Example

```yaml
name: customer_churn
pipeline_type: classification

data:
  source: data/customers.parquet
  target_column: churn
  train_ratio: 0.8
  stratify: true

features:
  numeric_impute_strategy: median
  numeric_scaling: standard
  categorical_encoding: onehot

model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.1
    scale_pos_weight: 2.5  # Handle class imbalance

mlflow:
  experiment_name: /classification/churn
  log_model: true
```

## Metrics

Classification pipelines automatically compute:

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (binary only)

## Handling Class Imbalance

### Using Stratified Split

```yaml
data:
  stratify: true
```

### Using Class Weights (XGBoost)

```yaml
model:
  model_type: xgboost
  hyperparameters:
    scale_pos_weight: 3.0  # Ratio of negative to positive samples
```

### Using Class Weights (Random Forest)

```yaml
model:
  model_type: random_forest
  hyperparameters:
    class_weight: balanced
```

## Cross-Validation

Enable stratified cross-validation:

```yaml
model:
  cross_validation: true
  cv_folds: 5
```

## Binary vs Multi-Class

The framework automatically detects binary vs multi-class based on target column:

- **Binary**: 2 unique values → uses `average='binary'` for metrics
- **Multi-class**: 3+ unique values → uses `average='weighted'` for metrics

## Example with Data Validation

```yaml
name: validated_classification
pipeline_type: classification

data:
  source: data/customers.parquet
  target_column: churn

data_validation:
  enabled: true
  fail_on_error: true
  expectations:
    - column: churn
      expectation: expect_column_values_to_be_in_set
      kwargs:
        value_set: [0, 1]
    - column: age
      expectation: expect_column_values_to_be_between
      kwargs:
        min_value: 18
        max_value: 100

model:
  model_type: random_forest

mlflow:
  experiment_name: /classification/validated
```

## Example with SHAP Explainability

```yaml
name: explainable_classification
pipeline_type: classification

data:
  source: data/customers.parquet
  target_column: churn

model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100

explainability:
  enabled: true
  explainer_type: auto
  generate_summary_plot: true
  generate_bar_plot: true

mlflow:
  experiment_name: /classification/explainable
```
