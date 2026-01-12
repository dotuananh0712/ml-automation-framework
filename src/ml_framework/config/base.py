"""Base configuration models using Pydantic."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class PipelineType(str, Enum):
    """Supported pipeline types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    FEATURE_ENGINEERING = "feature_engineering"


class DataFormat(str, Enum):
    """Supported data formats."""

    CSV = "csv"
    PARQUET = "parquet"
    DELTA = "delta"
    TABLE = "table"  # Databricks table


class ModelType(str, Enum):
    """Supported model types."""

    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LINEAR_REGRESSION = "linear_regression"
    SPARK_RF = "spark_random_forest"
    SPARK_GBT = "spark_gbt"


class DataConfig(BaseModel):
    """Data source configuration."""

    source: str = Field(..., description="Path or table name for data source")
    format: DataFormat = Field(default=DataFormat.PARQUET)
    target_column: str = Field(..., description="Target variable column name")
    feature_columns: list[str] | None = Field(
        default=None, description="Feature columns (None = auto-detect)"
    )
    date_column: str | None = Field(default=None, description="Date column for time series")
    id_column: str | None = Field(default=None, description="ID column for tracking")
    train_ratio: float = Field(default=0.8, ge=0.1, le=0.95)
    validation_ratio: float = Field(default=0.1, ge=0.0, le=0.4)
    stratify: bool = Field(default=True, description="Stratified split for classification")

    @field_validator("train_ratio", "validation_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("Ratio must be between 0 and 1")
        return v


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    numeric_impute_strategy: str = Field(default="median")
    categorical_impute_strategy: str = Field(default="most_frequent")
    numeric_scaling: str | None = Field(default="standard")  # standard, minmax, robust, None
    categorical_encoding: str = Field(default="onehot")  # onehot, label, target
    handle_unknown: str = Field(default="ignore")  # ignore, error
    feature_selection: bool = Field(default=False)
    feature_selection_k: int = Field(default=20, ge=1)
    custom_transformers: list[dict[str, Any]] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Model training configuration."""

    model_type: ModelType = Field(...)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cross_validation: bool = Field(default=True)
    cv_folds: int = Field(default=5, ge=2, le=20)
    early_stopping: bool = Field(default=True)
    early_stopping_rounds: int = Field(default=50, ge=1)
    random_state: int = Field(default=42)


class MLflowConfig(BaseModel):
    """MLflow tracking configuration."""

    experiment_name: str = Field(...)
    tracking_uri: str | None = Field(default=None, description="None = auto-detect")
    run_name: str | None = Field(default=None)
    tags: dict[str, str] = Field(default_factory=dict)
    log_model: bool = Field(default=True)
    log_feature_importance: bool = Field(default=True)
    register_model: bool = Field(default=False)
    model_name: str | None = Field(default=None)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    name: str = Field(..., description="Pipeline name for identification")
    description: str = Field(default="", description="Pipeline description")
    pipeline_type: PipelineType = Field(...)
    data: DataConfig = Field(...)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(...)
    mlflow: MLflowConfig = Field(...)
    output_path: str | None = Field(default=None, description="Path to save outputs")

    # Optional new feature sections (backward compatible)
    data_validation: "DataValidationConfig | None" = Field(
        default=None,
        description="Data validation configuration (Great Expectations)",
    )
    explainability: "ExplainabilityConfig | None" = Field(
        default=None,
        description="SHAP explainability configuration",
    )
    tuning: "TuningConfig | None" = Field(
        default=None,
        description="Hyperparameter tuning configuration (Optuna)",
    )

    @field_validator("output_path", mode="before")
    @classmethod
    def validate_output_path(cls, v: str | None) -> str | None:
        if v is not None:
            Path(v)  # Validate path format
        return v

    model_config = {"extra": "forbid"}


# Import at end to avoid circular imports (forward references)
from ml_framework.config.validation import DataValidationConfig  # noqa: E402
from ml_framework.config.explainability import ExplainabilityConfig  # noqa: E402
from ml_framework.config.tuning import TuningConfig  # noqa: E402

# Update forward references
PipelineConfig.model_rebuild()
