"""Dynamic model registry with YAML-driven factory pattern.

Enables configuration-driven model selection without code changes:
- Models defined in YAML with module paths and hyperparameters
- Dynamic import and instantiation at runtime
- Support for local, global, and foundation model types
"""

import importlib
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml

from ml_framework.exceptions import ModelNotFoundError, UnsupportedModelError

logger = structlog.get_logger(__name__)


class ModelType(str, Enum):
    """Model type classification."""

    LOCAL = "local"  # Univariate, per-series models
    GLOBAL = "global"  # Cross-series learning models
    FOUNDATION = "foundation"  # Pre-trained zero-shot models


class ModelFramework(str, Enum):
    """Supported model frameworks."""

    SKLEARN = "sklearn"
    STATSFORECAST = "statsforecast"
    NEURALFORECAST = "neuralforecast"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    CHRONOS = "chronos"
    TIMESFM = "timesfm"
    MOIRAI = "moirai"


# Default model configurations
DEFAULT_MODELS_CONFIG = {
    "models": {
        # Local Models (Sklearn)
        "LogisticRegression": {
            "module": "sklearn.linear_model",
            "class": "LogisticRegression",
            "framework": "sklearn",
            "model_type": "local",
            "hyperparameters": {"max_iter": 1000},
        },
        "RandomForest": {
            "module": "sklearn.ensemble",
            "class": "RandomForestClassifier",
            "framework": "sklearn",
            "model_type": "local",
            "hyperparameters": {"n_estimators": 100, "max_depth": 10},
        },
        # Gradient Boosting
        "XGBoost": {
            "module": "xgboost",
            "class": "XGBClassifier",
            "framework": "xgboost",
            "model_type": "local",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
        },
        "LightGBM": {
            "module": "lightgbm",
            "class": "LGBMClassifier",
            "framework": "lightgbm",
            "model_type": "local",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
        },
        "CatBoost": {
            "module": "catboost",
            "class": "CatBoostClassifier",
            "framework": "catboost",
            "model_type": "local",
            "hyperparameters": {
                "iterations": 100,
                "depth": 6,
                "learning_rate": 0.1,
                "verbose": False,
            },
        },
        # Time Series - Local
        "StatsForecastAutoARIMA": {
            "module": "statsforecast.models",
            "class": "AutoARIMA",
            "framework": "statsforecast",
            "model_type": "local",
            "hyperparameters": {"season_length": 7},
        },
        "StatsForecastAutoETS": {
            "module": "statsforecast.models",
            "class": "AutoETS",
            "framework": "statsforecast",
            "model_type": "local",
            "hyperparameters": {"season_length": 7},
        },
        "StatsForecastAutoTheta": {
            "module": "statsforecast.models",
            "class": "AutoTheta",
            "framework": "statsforecast",
            "model_type": "local",
            "hyperparameters": {"season_length": 7},
        },
        # Time Series - Global (Neural)
        "NeuralForecastNBEATS": {
            "module": "neuralforecast.models",
            "class": "NBEATS",
            "framework": "neuralforecast",
            "model_type": "global",
            "hyperparameters": {
                "input_size": 30,
                "h": 7,
                "max_steps": 1000,
            },
        },
        "NeuralForecastTiDE": {
            "module": "neuralforecast.models",
            "class": "TiDE",
            "framework": "neuralforecast",
            "model_type": "global",
            "hyperparameters": {
                "input_size": 30,
                "h": 7,
                "max_steps": 1000,
            },
        },
        # Foundation Models
        "ChronosT5Small": {
            "module": "ml_framework.models.foundation",
            "class": "ChronosForecaster",
            "framework": "chronos",
            "model_type": "foundation",
            "hyperparameters": {
                "model_path": "amazon/chronos-t5-small",
                "num_samples": 20,
            },
        },
        "ChronosT5Base": {
            "module": "ml_framework.models.foundation",
            "class": "ChronosForecaster",
            "framework": "chronos",
            "model_type": "foundation",
            "hyperparameters": {
                "model_path": "amazon/chronos-t5-base",
                "num_samples": 20,
            },
        },
    }
}


class ModelRegistry:
    """Dynamic model registry with configuration-driven instantiation.

    Example:
        registry = ModelRegistry(user_config)
        model = registry.get_model("XGBoost")
        model.fit(X, y)

        # Or with custom hyperparameters
        model = registry.get_model("XGBoost", n_estimators=200)
    """

    def __init__(
        self,
        user_config: dict | None = None,
        config_path: str | Path | None = None,
    ):
        """Initialize model registry.

        Args:
            user_config: User-provided configuration to merge with defaults.
            config_path: Path to YAML config file (alternative to user_config).
        """
        # Load base configuration
        self.base_config = DEFAULT_MODELS_CONFIG.copy()

        # Load from file if provided
        if config_path:
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
                self._merge_config(file_config)

        # Merge user config
        if user_config:
            self._merge_config(user_config)

        self._model_cache: dict[str, Any] = {}

        logger.info(
            "Model registry initialized",
            available_models=list(self.base_config["models"].keys()),
        )

    def _merge_config(self, override: dict) -> None:
        """Merge override config into base config."""
        if "models" in override:
            for model_name, model_conf in override["models"].items():
                if model_name in self.base_config["models"]:
                    # Merge hyperparameters
                    base_params = self.base_config["models"][model_name].get(
                        "hyperparameters", {}
                    )
                    override_params = model_conf.get("hyperparameters", {})
                    merged_params = {**base_params, **override_params}

                    self.base_config["models"][model_name] = {
                        **self.base_config["models"][model_name],
                        **model_conf,
                        "hyperparameters": merged_params,
                    }
                else:
                    self.base_config["models"][model_name] = model_conf

    def list_models(self, model_type: ModelType | None = None) -> list[str]:
        """List available models.

        Args:
            model_type: Filter by model type (local, global, foundation).

        Returns:
            List of model names.
        """
        models = self.base_config["models"]

        if model_type:
            return [
                name
                for name, conf in models.items()
                if conf.get("model_type") == model_type.value
            ]
        return list(models.keys())

    def get_model_config(self, model_name: str) -> dict:
        """Get configuration for a model.

        Args:
            model_name: Name of the model.

        Returns:
            Model configuration dictionary.

        Raises:
            ModelNotFoundError: If model is not in registry.
        """
        if model_name not in self.base_config["models"]:
            available = list(self.base_config["models"].keys())
            raise ModelNotFoundError(
                f"Model '{model_name}' not found. Available: {available}"
            )
        return self.base_config["models"][model_name]

    def get_model(self, model_name: str, **override_params: Any) -> Any:
        """Instantiate a model from the registry.

        Args:
            model_name: Name of the model to instantiate.
            **override_params: Override hyperparameters.

        Returns:
            Instantiated model.

        Raises:
            ModelNotFoundError: If model is not found.
            UnsupportedModelError: If model module cannot be imported.
        """
        config = self.get_model_config(model_name)

        # Merge hyperparameters
        params = {**config.get("hyperparameters", {}), **override_params}

        # Dynamic import
        try:
            module = importlib.import_module(config["module"])
            model_class = getattr(module, config["class"])
        except ImportError as e:
            raise UnsupportedModelError(
                f"Cannot import model '{model_name}': {e}. "
                f"Install required package for {config['framework']}"
            )
        except AttributeError as e:
            raise UnsupportedModelError(
                f"Class '{config['class']}' not found in {config['module']}: {e}"
            )

        # Instantiate
        try:
            model = model_class(**params)
        except TypeError as e:
            logger.warning(
                "Some hyperparameters not accepted",
                model=model_name,
                error=str(e),
            )
            # Try with subset of params
            model = model_class()

        logger.info(
            "Model instantiated",
            model=model_name,
            framework=config["framework"],
            type=config["model_type"],
        )

        return model

    def get_model_type(self, model_name: str) -> ModelType:
        """Get the type of a model (local, global, foundation).

        Args:
            model_name: Name of the model.

        Returns:
            ModelType enum value.
        """
        config = self.get_model_config(model_name)
        return ModelType(config.get("model_type", "local"))

    def get_framework(self, model_name: str) -> ModelFramework:
        """Get the framework of a model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelFramework enum value.
        """
        config = self.get_model_config(model_name)
        return ModelFramework(config.get("framework", "sklearn"))


# Global registry instance
_default_registry: ModelRegistry | None = None


def get_registry(user_config: dict | None = None) -> ModelRegistry:
    """Get or create the default model registry.

    Args:
        user_config: Optional user configuration to merge.

    Returns:
        ModelRegistry instance.
    """
    global _default_registry

    if _default_registry is None or user_config:
        _default_registry = ModelRegistry(user_config)

    return _default_registry
