# Adding New Models

Guide to adding new model types to the framework.

## Overview

To add a new model, you need to:

1. Add the model type to the `ModelType` enum
2. Register default hyperparameters
3. Update the model factory
4. Add tests

## Step 1: Add Model Type

Edit `src/ml_framework/config/base.py`:

```python
class ModelType(str, Enum):
    # Existing models
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    # Add your new model
    CATBOOST = "catboost"
```

## Step 2: Register Default Hyperparameters

Edit `src/ml_framework/models/factory.py`:

```python
DEFAULT_HYPERPARAMETERS = {
    # Existing models...

    ModelType.CATBOOST: {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "Logloss",
        "verbose": False,
    },
}
```

## Step 3: Update Model Factory

Edit `src/ml_framework/models/factory.py`:

```python
def create_model(config: ModelConfig, pipeline_type: PipelineType):
    # Get hyperparameters
    hyperparams = {
        **DEFAULT_HYPERPARAMETERS.get(config.model_type, {}),
        **config.hyperparameters,
    }

    # Add your model creation logic
    if config.model_type == ModelType.CATBOOST:
        if pipeline_type == PipelineType.CLASSIFICATION:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(**hyperparams)
        else:
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**hyperparams)

    # ... existing models
```

## Step 4: Update MLflow Logger (if needed)

If your model requires special MLflow logging, update `src/ml_framework/logging/mlflow_logger.py`:

```python
def log_model(self, model, model_name: str):
    model_class = type(model).__name__.lower()

    if "catboost" in model_class:
        import mlflow.catboost
        mlflow.catboost.log_model(model, model_name)
    # ... existing handlers
```

## Step 5: Add Tests

Create tests in `tests/unit/test_model_factory.py`:

```python
def test_create_catboost_classifier():
    config = ModelConfig(
        model_type=ModelType.CATBOOST,
        hyperparameters={"iterations": 50},
    )
    model = create_model(config, PipelineType.CLASSIFICATION)
    assert model is not None
    assert "CatBoost" in type(model).__name__
```

## Step 6: Update Documentation

Update `docs/user-guide/configuration.md` to include the new model type.

## Example: Adding CatBoost

Complete example:

```python
# config/base.py
class ModelType(str, Enum):
    CATBOOST = "catboost"

# models/factory.py
DEFAULT_HYPERPARAMETERS[ModelType.CATBOOST] = {
    "iterations": 100,
    "depth": 6,
    "learning_rate": 0.1,
}

def create_model(config, pipeline_type):
    if config.model_type == ModelType.CATBOOST:
        from catboost import CatBoostClassifier, CatBoostRegressor
        if pipeline_type == PipelineType.CLASSIFICATION:
            return CatBoostClassifier(**hyperparams)
        return CatBoostRegressor(**hyperparams)
```
