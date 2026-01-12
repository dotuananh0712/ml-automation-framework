"""Model factory for creating ML models from configuration."""

from typing import Any

import structlog
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml_framework.config.base import ModelConfig, ModelType, PipelineType

logger = structlog.get_logger(__name__)

# Default hyperparameters for each model type
DEFAULT_PARAMS: dict[ModelType, dict[str, Any]] = {
    ModelType.LOGISTIC_REGRESSION: {
        "max_iter": 1000,
        "solver": "lbfgs",
    },
    ModelType.LINEAR_REGRESSION: {},
    ModelType.RANDOM_FOREST: {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "n_jobs": -1,
    },
    ModelType.XGBOOST: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    },
    ModelType.LIGHTGBM: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "n_jobs": -1,
    },
    ModelType.CATBOOST: {
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "verbose": False,
        "thread_count": -1,
    },
}


def create_model(
    config: ModelConfig, pipeline_type: PipelineType | None = None
) -> Any:
    """Create a model instance from configuration.

    Args:
        config: Model configuration.
        pipeline_type: Pipeline type to determine classifier vs regressor.

    Returns:
        Configured model instance.
    """
    model_type = config.model_type

    # Merge default params with user-provided params
    params = {**DEFAULT_PARAMS.get(model_type, {}), **config.hyperparameters}
    params["random_state"] = config.random_state

    # Add early stopping for tree models
    if config.early_stopping and model_type in (ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST):
        params["early_stopping_rounds"] = config.early_stopping_rounds

    logger.info("Creating model", model_type=model_type.value, params=params)

    # Create appropriate model
    if model_type == ModelType.LOGISTIC_REGRESSION:
        return LogisticRegression(**params)

    elif model_type == ModelType.LINEAR_REGRESSION:
        params.pop("random_state", None)  # Linear regression doesn't use random_state
        return LinearRegression(**params)

    elif model_type == ModelType.RANDOM_FOREST:
        is_classifier = pipeline_type in (None, PipelineType.CLASSIFICATION)
        if is_classifier:
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    elif model_type == ModelType.XGBOOST:
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost") from e

        is_classifier = pipeline_type in (None, PipelineType.CLASSIFICATION)

        # for xgb remove sklearn params
        xgb_params = params.copy()

        xgb_params.pop("n_jobs", None)

        if is_classifier:
            return xgb.XGBClassifier(**xgb_params)

        # for regressor, remove eval_metric
        xgb_params.pop("eval_metric", None)

        return xgb.XGBRegressor(**xgb_params)

    elif model_type == ModelType.LIGHTGBM:
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM not installed. Install with: pip install lightgbm"
            ) from e

        is_classifier = pipeline_type in (None, PipelineType.CLASSIFICATION)

        # remove sklearn params for lgbm
        lgb_params = params.copy()
        lgb_params.pop("n_jobs", None)
        if is_classifier:
            return lgb.LGBMClassifier(**lgb_params)
        return lgb.LGBMRegressor(**lgb_params)

    elif model_type == ModelType.CATBOOST:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as e:
            raise ImportError(
                "CatBoost not installed. Install with: pip install catboost"
            ) from e

        is_classifier = pipeline_type in (None, PipelineType.CLASSIFICATION)

        # CatBoost-specific parameter adjustments
        cb_params = params.copy()

        # Map sklearn-style params to CatBoost params
        if "random_state" in cb_params:
            cb_params["random_seed"] = cb_params.pop("random_state")

        # Remove sklearn params not used by CatBoost
        cb_params.pop("n_jobs", None)
        cb_params.pop("colsample_bytree", None)

        if is_classifier:
            return CatBoostClassifier(**cb_params)
        return CatBoostRegressor(**cb_params)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_model_type_for_task(task: str) -> list[ModelType]:
    """Get recommended model types for a task.

    Args:
        task: Task type (classification, regression, forecasting).

    Returns:
        List of recommended ModelType values.
    """
    classification_models = [
        ModelType.LOGISTIC_REGRESSION,
        ModelType.RANDOM_FOREST,
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.CATBOOST,
    ]

    regression_models = [
        ModelType.LINEAR_REGRESSION,
        ModelType.RANDOM_FOREST,
        ModelType.XGBOOST,
        ModelType.LIGHTGBM,
        ModelType.CATBOOST,
    ]

    if task == "classification":
        return classification_models
    elif task in ("regression", "forecasting"):
        return regression_models
    else:
        return classification_models + regression_models
