# API Reference

Internal API documentation for ML Automation Framework.

## Configuration Models

### `ml_framework.config.base`

```python
class PipelineType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"

class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class DataConfig(BaseModel):
    source: str
    format: DataFormat = DataFormat.PARQUET
    target_column: str
    feature_columns: list[str] | None = None
    train_ratio: float = 0.8
    validation_ratio: float = 0.1

class PipelineConfig(BaseModel):
    name: str
    pipeline_type: PipelineType
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
```

## Pipeline Classes

### `ml_framework.pipelines.base`

```python
class BasePipeline(ABC):
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""

    @classmethod
    def from_config(cls, config_path: str) -> "BasePipeline":
        """Create pipeline from YAML config file."""

    def load_data(self) -> pd.DataFrame:
        """Load data from configured source."""

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """Split data into train/val/test sets."""

    @abstractmethod
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the model."""

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline."""
```

## Feature Transformer

### `ml_framework.features.transformer`

```python
class FeatureTransformer:
    def __init__(self, config: FeatureConfig):
        """Initialize with feature configuration."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureTransformer":
        """Fit transformer on training data."""

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features."""

    def get_feature_names(self) -> list[str]:
        """Get output feature names."""
```

## MLflow Logger

### `ml_framework.logging.mlflow_logger`

```python
class MLflowLogger:
    def __init__(self, config: MLflowConfig):
        """Initialize MLflow logger."""

    def start_run(self, run_name: str | None = None) -> ContextManager:
        """Context manager for MLflow run."""

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""

    def log_model(self, model: Any, model_name: str) -> None:
        """Log model artifact."""

    def log_feature_importance(self, importance: dict[str, float], top_k: int = 20) -> None:
        """Log feature importance."""
```

## Exceptions

### `ml_framework.exceptions`

```python
class MLFrameworkError(Exception):
    """Base exception."""
    message: str
    details: dict

class ConfigurationError(MLFrameworkError): ...
class ModelError(MLFrameworkError): ...
class DataError(MLFrameworkError): ...
class CLIError(MLFrameworkError): ...
```

## Utilities

### `ml_framework.utils.runtime`

```python
def get_runtime() -> str:
    """Get current runtime: 'local' or 'databricks'."""

def is_databricks() -> bool:
    """Check if running on Databricks."""

def get_spark_session() -> SparkSession:
    """Get or create Spark session."""
```
