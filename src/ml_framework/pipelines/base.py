"""Base pipeline class with common functionality."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from ml_framework.config.base import PipelineConfig
from ml_framework.config.loader import load_config
from ml_framework.logging.mlflow_logger import MLflowLogger
from ml_framework.utils.runtime import Runtime, get_runtime

logger = structlog.get_logger(__name__)


class BasePipeline(ABC):
    """Base class for all ML pipelines.

    Provides common functionality for data loading, logging, and execution.
    Subclasses implement specific training and prediction logic.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize pipeline with validated configuration.

        Args:
            config: Validated pipeline configuration.
        """
        self.config = config
        self.runtime = get_runtime()
        self.mlflow_logger = MLflowLogger(config.mlflow)
        self._data: pd.DataFrame | None = None
        self._model: Any = None
        self._feature_pipeline: Any = None

        logger.info(
            "Pipeline initialized",
            name=config.name,
            type=config.pipeline_type.value,
            runtime=self.runtime.value,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "BasePipeline":
        """Create pipeline from YAML configuration file.

        Args:
            config_path: Path to YAML configuration.

        Returns:
            Initialized pipeline instance.
        """
        config = load_config(config_path)
        return cls(config)

    def load_data(self) -> pd.DataFrame:
        """Load data from configured source.

        Automatically handles local vs Databricks runtime.

        Returns:
            Loaded DataFrame.
        """
        data_config = self.config.data
        source = data_config.source

        logger.info("Loading data", source=source, format=data_config.format.value)

        if self.runtime == Runtime.DATABRICKS:
            self._data = self._load_databricks(source, data_config.format.value)
        else:
            self._data = self._load_local(source, data_config.format.value)

        logger.info("Data loaded", rows=len(self._data), columns=len(self._data.columns))
        return self._data

    def _load_local(self, source: str, fmt: str) -> pd.DataFrame:
        """Load data locally using pandas."""
        path = Path(source)

        if fmt == "csv":
            return pd.read_csv(path)
        elif fmt == "parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported local format: {fmt}")

    def _load_databricks(self, source: str, fmt: str) -> pd.DataFrame:
        """Load data on Databricks using Spark."""
        from ml_framework.utils.runtime import get_spark_session

        spark = get_spark_session()

        if fmt == "delta":
            df = spark.read.format("delta").load(source)
        elif fmt == "table":
            df = spark.table(source)
        elif fmt == "parquet":
            df = spark.read.parquet(source)
        else:
            df = spark.read.format(fmt).load(source)

        return df.toPandas()

    def split_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.

        Args:
            data: Full dataset.

        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        from sklearn.model_selection import train_test_split

        data_config = self.config.data
        train_ratio = data_config.train_ratio
        val_ratio = data_config.validation_ratio
        test_ratio = 1.0 - train_ratio - val_ratio

        stratify_col = data[data_config.target_column] if data_config.stratify else None

        # First split: train vs (val + test)
        train, temp = train_test_split(
            data,
            train_size=train_ratio,
            random_state=self.config.model.random_state,
            stratify=stratify_col,
        )

        # Second split: val vs test
        if test_ratio > 0:
            val_of_remaining = val_ratio / (val_ratio + test_ratio)
            stratify_temp = temp[data_config.target_column] if data_config.stratify else None
            val, test = train_test_split(
                temp,
                train_size=val_of_remaining,
                random_state=self.config.model.random_state,
                stratify=stratify_temp,
            )
        else:
            val, test = temp, pd.DataFrame()

        logger.info(
            "Data split complete",
            train_size=len(train),
            val_size=len(val),
            test_size=len(test),
        )

        return train, val, test

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> Any:
        """Train the model.

        Args:
            train_data: Training dataset.
            val_data: Optional validation dataset.

        Returns:
            Trained model.
        """
        pass

    @abstractmethod
    def evaluate(self, data: pd.DataFrame, prefix: str = "test") -> dict[str, float]:
        """Evaluate model performance.

        Args:
            data: Evaluation dataset.
            prefix: Metric name prefix (e.g., "train", "val", "test").

        Returns:
            Dictionary of metric names to values.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions.

        Args:
            data: Input data for prediction.

        Returns:
            Predictions as a Series.
        """
        pass

    def _run_shap_explainability(self, train_data: pd.DataFrame) -> None:
        """Run SHAP explainability if configured.

        Args:
            train_data: Training data for computing SHAP values.
        """
        if self.config.explainability is None or not self.config.explainability.enabled:
            return

        if self._model is None:
            logger.warning("Cannot compute SHAP values: model not trained")
            return

        logger.info("Generating SHAP explanations")

        try:
            from ml_framework.explainability.shap_explainer import SHAPExplainer
            import tempfile

            # Prepare features
            target_col = self.config.data.target_column
            feature_cols = self.config.data.feature_columns
            if feature_cols is None:
                feature_cols = [c for c in train_data.columns if c != target_col]

            X_train = train_data[feature_cols]

            # Create explainer and compute SHAP values
            explainer = SHAPExplainer(self.config.explainability)
            explainer.fit(self._model, X_train.values, feature_names=feature_cols)

            # Log SHAP feature importance
            shap_importance = explainer.get_feature_importance()
            self.mlflow_logger.log_dict(shap_importance, "shap_feature_importance.json")

            # Generate and log plots
            with tempfile.TemporaryDirectory() as tmpdir:
                plots = explainer.generate_plots(tmpdir)
                for plot_path in plots:
                    self.mlflow_logger.log_artifact(str(plot_path), "shap_plots")

            logger.info("SHAP explanations complete", n_plots=len(plots) if plots else 0)

        except ImportError:
            logger.warning("SHAP not installed, skipping explainability")
        except Exception as e:
            logger.error("Error generating SHAP explanations", error=str(e))

    def _run_data_validation(self, data: pd.DataFrame) -> None:
        """Run data validation if configured.

        Args:
            data: DataFrame to validate.

        Raises:
            DataValidationError: If validation fails and fail_on_error is True.
        """
        if self.config.data_validation is None or not self.config.data_validation.enabled:
            return

        logger.info("Running data validation")

        try:
            from ml_framework.validation.ge_validator import GreatExpectationsValidator

            validator = GreatExpectationsValidator(self.config.data_validation)
            is_valid, results = validator.validate(data)

            if is_valid:
                logger.info("Data validation passed")
            else:
                logger.warning("Data validation completed with issues", results=results)
        except ImportError:
            # Fall back to simple validation if GE not installed
            logger.warning("Great Expectations not installed, using simple validation")
            from ml_framework.validation.ge_validator import GreatExpectationsValidator

            validator = GreatExpectationsValidator(self.config.data_validation)
            is_valid, errors = validator.validate_simple(data)

            if not is_valid:
                from ml_framework.exceptions import DataValidationError

                if self.config.data_validation.fail_on_error:
                    raise DataValidationError("Data validation failed", {"errors": errors})
                logger.warning("Data validation issues (continuing)", errors=errors)

    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline.

        Returns:
            Dictionary with run results and metrics.
        """
        logger.info("Starting pipeline run", name=self.config.name)

        # Load data
        data = self.load_data()

        # Run data validation if configured
        self._run_data_validation(data)

        # Split data
        train, val, test = self.split_data(data)

        # Start MLflow run
        with self.mlflow_logger.start_run():
            # Log configuration
            self.mlflow_logger.log_config(self.config)

            # Train model
            self._model = self.train(train, val)

            # Evaluate on all splits
            train_metrics = self.evaluate(train, prefix="train")
            val_metrics = self.evaluate(val, prefix="val") if len(val) > 0 else {}
            test_metrics = self.evaluate(test, prefix="test") if len(test) > 0 else {}

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics, **test_metrics}
            self.mlflow_logger.log_metrics(all_metrics)

            # Log model
            if self.config.mlflow.log_model:
                self.mlflow_logger.log_model(self._model, self.config.model.model_type.value)

            # Log feature importance if available
            if self.config.mlflow.log_feature_importance:
                importance = self._get_feature_importance()
                if importance:
                    self.mlflow_logger.log_feature_importance(importance)

            # Generate SHAP explanations if configured
            self._run_shap_explainability(train)

        logger.info("Pipeline run complete", metrics=all_metrics)

        return {
            "model": self._model,
            "metrics": all_metrics,
            "run_id": self.mlflow_logger.run_id,
        }

    def _get_feature_importance(self) -> dict[str, float] | None:
        """Extract feature importance from trained model.

        Returns:
            Dictionary of feature names to importance values, or None.
        """
        if self._model is None:
            return None

        # Try common attribute names
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            importances = abs(self._model.coef_.flatten())
        else:
            return None

        # Get feature names
        feature_cols = self.config.data.feature_columns
        if feature_cols and len(feature_cols) == len(importances):
            return dict(zip(feature_cols, importances.tolist()))

        return None
