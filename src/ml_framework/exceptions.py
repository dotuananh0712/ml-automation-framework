"""Domain-specific exception hierarchy for ML Automation Framework.

Provides granular error handling organized by domain:
- Configuration errors
- Model errors
- Data errors
- Forecasting errors
- Infrastructure errors
"""


class MLFrameworkError(Exception):
    """Base exception for all ML Framework errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(MLFrameworkError):
    """Base class for configuration-related errors."""


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration validation fails."""


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(MLFrameworkError):
    """Base class for model-related errors."""


class ModelNotFoundError(ModelError):
    """Raised when specified model is not found in registry."""


class ModelInitializationError(ModelError):
    """Raised when model fails to initialize."""


class ModelTrainingError(ModelError):
    """Raised when model training fails."""


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""


class UnsupportedModelError(ModelError):
    """Raised when model type is not supported."""


# =============================================================================
# Data Errors
# =============================================================================


class DataError(MLFrameworkError):
    """Base class for data-related errors."""


class DataQualityError(DataError):
    """Raised when data fails quality checks."""


class DataPreparationError(DataError):
    """Raised when data preparation fails."""


class MissingDataError(DataError):
    """Raised when required data is missing."""


class InvalidDataError(DataError):
    """Raised when data format or content is invalid."""


class EmptyDatasetError(DataError):
    """Raised when dataset is empty after filtering."""


class InsufficientDataError(DataError):
    """Raised when dataset has insufficient rows for training."""


# =============================================================================
# Feature Errors
# =============================================================================


class FeatureError(MLFrameworkError):
    """Base class for feature engineering errors."""


class FeatureTransformError(FeatureError):
    """Raised when feature transformation fails."""


class FeatureNotFittedError(FeatureError):
    """Raised when transformer is used before fitting."""


# =============================================================================
# Forecasting Errors
# =============================================================================


class ForecastingError(MLFrameworkError):
    """Base class for forecasting-specific errors."""


class BacktestError(ForecastingError):
    """Raised when backtesting fails."""


class EvaluationError(ForecastingError):
    """Raised when model evaluation fails."""


class ScoringError(ForecastingError):
    """Raised when scoring/inference fails."""


class UnsupportedMetricError(ForecastingError):
    """Raised when metric is not supported."""


class UnsupportedFrequencyError(ForecastingError):
    """Raised when time series frequency is not supported."""


# =============================================================================
# MLflow Errors
# =============================================================================


class MLflowError(MLFrameworkError):
    """Base class for MLflow-related errors."""


class ExperimentError(MLflowError):
    """Raised when experiment creation/access fails."""


class ModelRegistryError(MLflowError):
    """Raised when model registry operations fail."""


# =============================================================================
# Infrastructure Errors
# =============================================================================


class InfrastructureError(MLFrameworkError):
    """Base class for infrastructure-related errors."""


class SparkError(InfrastructureError):
    """Raised when Spark operations fail."""


class ResourceError(InfrastructureError):
    """Raised when required resources are unavailable."""


class AcceleratorError(InfrastructureError):
    """Raised when GPU/accelerator operations fail."""


# =============================================================================
# CLI Errors
# =============================================================================


class CLIError(MLFrameworkError):
    """Base class for CLI-specific errors."""


class FileNotFoundCLIError(CLIError):
    """Raised when a required file is not found.

    Example:
        raise FileNotFoundCLIError(
            "configs/missing.yaml",
            file_type="configuration file"
        )
    """

    def __init__(self, path: str, file_type: str = "file"):
        message = f"{file_type.capitalize()} not found: {path}"
        suggestions = [
            f"Check that the path exists: {path}",
            "Use absolute path or path relative to current directory",
        ]
        super().__init__(message, {"path": path, "suggestions": suggestions})


class InvalidYAMLError(CLIError):
    """Raised when YAML parsing fails.

    Example:
        raise InvalidYAMLError("config.yaml", "mapping values not allowed here")
    """

    def __init__(self, path: str, parse_error: str):
        message = f"Invalid YAML in {path}"
        super().__init__(
            message,
            {"path": path, "parse_error": parse_error, "suggestion": "Validate YAML syntax"},
        )


class ColumnNotFoundError(DataError):
    """Raised when a required column is missing from dataset.

    Example:
        raise ColumnNotFoundError("target", ["feature1", "feature2", "label"])
    """

    def __init__(self, column: str, available_columns: list[str]):
        message = f"Column '{column}' not found in dataset"
        # Show first 10 columns to avoid overwhelming output
        shown_columns = available_columns[:10]
        suffix = f"... ({len(available_columns) - 10} more)" if len(available_columns) > 10 else ""
        super().__init__(
            message,
            {
                "column": column,
                "available_columns": available_columns,
                "suggestion": f"Available columns: {shown_columns}{suffix}",
            },
        )


class DataValidationError(DataError):
    """Raised when data validation fails.

    Example:
        raise DataValidationError(
            "Data validation failed",
            {"errors": ["Column 'age' has null values", "Column 'id' not unique"]}
        )
    """
