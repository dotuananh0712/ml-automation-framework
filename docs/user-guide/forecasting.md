# Forecasting Pipelines

Guide to building time series forecasting models.

## Overview

Forecasting pipelines predict future values based on historical time series data. The framework supports:

- **Single-series forecasting**: One model for one time series
- **Many-model forecasting**: One model per group (e.g., per store)

## Basic Example

```yaml
name: sales_forecast
pipeline_type: forecasting

data:
  source: data/sales.parquet
  target_column: sales
  date_column: date
  train_ratio: 0.8

features:
  numeric_impute_strategy: median
  numeric_scaling: standard

model:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100
    max_depth: 6

mlflow:
  experiment_name: /forecasting/sales
```

## Automatic Feature Engineering

The framework automatically creates:

- **Lag features**: Previous values (1, 7, 14, 28 days)
- **Rolling statistics**: Mean, std for 7, 14, 28-day windows
- **Date features**: Day of week, month, year (if applicable)

## Metrics

Forecasting pipelines compute:

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## Many-Model Forecasting

Train separate models for each group:

```yaml
name: store_demand_forecast
pipeline_type: forecasting

data:
  source: data/store_sales.parquet
  target_column: demand
  date_column: date
  id_column: store_id  # Group column

# Each store gets its own model
model:
  model_type: xgboost

mlflow:
  experiment_name: /forecasting/many_model
```

## Walk-Forward Backtesting

Configure backtesting for robust evaluation:

```yaml
name: backtested_forecast
pipeline_type: forecasting

data:
  source: data/sales.parquet
  target_column: sales
  date_column: date

backtesting:
  enabled: true
  backtest_length: 30      # Evaluation window
  prediction_length: 7      # Forecast horizon
  stride: 7                 # Step between trials

model:
  model_type: xgboost

mlflow:
  experiment_name: /forecasting/backtested
```
