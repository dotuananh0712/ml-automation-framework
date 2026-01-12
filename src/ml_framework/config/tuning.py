"""Hyperparameter tuning configuration models.

Defines Pydantic models for Optuna-based hyperparameter optimization.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class SamplerType(str, Enum):
    """Supported Optuna sampler types."""

    TPE = "tpe"  # Tree-structured Parzen Estimator (default)
    RANDOM = "random"  # Random search
    CMAES = "cmaes"  # CMA-ES
    GRID = "grid"  # Grid search


class PrunerType(str, Enum):
    """Supported Optuna pruner types."""

    MEDIAN = "median"  # Median pruner
    HYPERBAND = "hyperband"  # Hyperband
    NONE = "none"  # No pruning


class ParamType(str, Enum):
    """Hyperparameter types for search space definition."""

    INT = "int"  # Integer parameter
    FLOAT = "float"  # Float parameter
    CATEGORICAL = "categorical"  # Categorical parameter
    LOG_FLOAT = "log_float"  # Log-uniform float parameter


class SearchParam(BaseModel):
    """Single hyperparameter search space definition.

    Example:
        ```yaml
        - name: n_estimators
          type: int
          low: 50
          high: 500
        - name: learning_rate
          type: log_float
          low: 0.001
          high: 0.3
        - name: max_features
          type: categorical
          choices: ["sqrt", "log2", null]
        ```
    """

    name: str = Field(..., description="Parameter name")
    type: ParamType = Field(..., description="Parameter type")
    low: float | None = Field(default=None, description="Lower bound (for int/float)")
    high: float | None = Field(default=None, description="Upper bound (for int/float)")
    choices: list[Any] | None = Field(default=None, description="Choices (for categorical)")
    step: float | None = Field(default=None, description="Step size (for int/float)")

    @model_validator(mode="after")
    def validate_param_config(self) -> "SearchParam":
        """Validate that required fields are set based on param type."""
        if self.type in (ParamType.INT, ParamType.FLOAT, ParamType.LOG_FLOAT):
            if self.low is None or self.high is None:
                raise ValueError(f"low and high are required for {self.type} parameters")
            if self.low >= self.high:
                raise ValueError("low must be less than high")
        elif self.type == ParamType.CATEGORICAL:
            if not self.choices or len(self.choices) < 2:
                raise ValueError("choices must have at least 2 values for categorical parameters")
        return self


class TuningConfig(BaseModel):
    """Configuration for Optuna hyperparameter tuning.

    Example:
        ```yaml
        tuning:
          enabled: true
          n_trials: 50
          direction: maximize
          metric: val_f1
          search_space:
            - name: n_estimators
              type: int
              low: 50
              high: 500
            - name: learning_rate
              type: log_float
              low: 0.001
              high: 0.3
        ```
    """

    enabled: bool = Field(default=False, description="Enable hyperparameter tuning")
    n_trials: int = Field(
        default=100,
        ge=1,
        description="Number of optimization trials",
    )
    timeout: int | None = Field(
        default=None,
        ge=60,
        description="Timeout in seconds (None = no timeout)",
    )
    direction: str = Field(
        default="maximize",
        pattern="^(minimize|maximize)$",
        description="Optimization direction: minimize or maximize",
    )
    metric: str = Field(
        default="val_f1",
        description="Metric to optimize (e.g., val_f1, val_accuracy, val_rmse)",
    )
    sampler: SamplerType = Field(
        default=SamplerType.TPE,
        description="Optuna sampler type",
    )
    pruner: PrunerType = Field(
        default=PrunerType.MEDIAN,
        description="Optuna pruner type for early stopping",
    )
    search_space: list[SearchParam] = Field(
        default_factory=list,
        description="List of hyperparameters to tune",
    )
    log_all_trials: bool = Field(
        default=True,
        description="Log all trials to MLflow",
    )
    register_best_model: bool = Field(
        default=True,
        description="Register best model in MLflow model registry",
    )
