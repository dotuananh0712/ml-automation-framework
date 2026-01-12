---
name: ml-pipeline-patterns
description: ML pipeline design patterns for classification, regression, forecasting. Use when building pipelines, training models, evaluating metrics, or working with MLflow logging.
---

# ML Pipeline Patterns

## When to Use

- Building new ML pipelines
- Adding model training logic
- Implementing evaluation metrics
- Configuring MLflow logging
- Adding feature engineering steps

## Core Patterns

### 1. Config-Driven Pipeline

Always use YAML configs validated by Pydantic:

```python
# GOOD: Config-driven
config = load_config("configs/classification/churn.yaml")
pipeline = ClassificationPipeline(config)
pipeline.run()

# BAD: Hardcoded parameters
pipeline = ClassificationPipeline(
    data_path="data.csv",
    target="churn",
    model="xgboost",
    n_estimators=100,
)
```

### 2. Runtime Abstraction

Always check runtime for environment-specific code:

```python
# GOOD: Runtime-aware
from ml_framework.utils.runtime import get_runtime, Runtime

if get_runtime() == Runtime.DATABRICKS:
    df = spark.read.format("delta").load(path)
else:
    df = pd.read_parquet(path)

# BAD: Hardcoded environment
df = spark.read.format("delta").load(path)  # Fails locally
```

### 3. MLflow Logging

Always log within context manager:

```python
# GOOD: Context manager ensures run closes
with mlflow_logger.start_run():
    mlflow_logger.log_config(config)
    model = train(data)
    mlflow_logger.log_metrics(evaluate(model, data))
    mlflow_logger.log_model(model, "xgboost")

# BAD: Manual run management
mlflow.start_run()
# ... if exception, run never closes
mlflow.end_run()
```

### 4. Feature Transformer Pattern

Fit on train, transform on all splits:

```python
# GOOD: Fit once, transform consistently
transformer = FeatureTransformer(config.features)
X_train = transformer.fit_transform(train_df)
X_val = transformer.transform(val_df)
X_test = transformer.transform(test_df)

# BAD: Fitting on each split (data leakage)
X_train = transformer.fit_transform(train_df)
X_val = transformer.fit_transform(val_df)  # Wrong!
```

### 5. Metric Prefixing

Always prefix metrics by split:

```python
# GOOD: Clear metric naming
train_metrics = evaluate(train, prefix="train")  # train_accuracy, train_f1
val_metrics = evaluate(val, prefix="val")        # val_accuracy, val_f1
test_metrics = evaluate(test, prefix="test")     # test_accuracy, test_f1

# BAD: Ambiguous metrics
metrics = evaluate(test)  # "accuracy" - which split?
```

## Anti-Patterns

### 1. Data Leakage

```python
# BAD: Using test data before final evaluation
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# GOOD: Use validation for early stopping
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

### 2. Silent Failures

```python
# BAD: Silently continuing on error
try:
    mlflow.log_model(model, "model")
except Exception:
    pass  # Model not logged, no one knows

# GOOD: Log warning, fail loudly in critical paths
try:
    mlflow.log_model(model, "model")
except Exception as e:
    logger.error("Failed to log model", error=str(e))
    raise
```

### 3. Hardcoded Paths

```python
# BAD: Hardcoded paths
df = pd.read_parquet("/home/user/data/train.parquet")

# GOOD: Config-driven
df = pd.read_parquet(config.data.source)
```

## Integration

- Related: `databricks-patterns` for Databricks-specific code
- Related: `feature-engineering` for feature transformation
