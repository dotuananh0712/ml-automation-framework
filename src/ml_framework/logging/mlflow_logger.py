"""MLflow experiment tracking integration."""

import os
from contextlib import contextmanager
from typing import Any, Generator

import structlog

from ml_framework.config.base import MLflowConfig, PipelineConfig
from ml_framework.utils.runtime import is_databricks

logger = structlog.get_logger(__name__)


class MLflowLogger:
    """MLflow experiment tracking wrapper.

    Handles logging for both local and Databricks environments.
    """

    def __init__(self, config: MLflowConfig) -> None:
        """Initialize MLflow logger.

        Args:
            config: MLflow configuration.
        """
        self.config = config
        self._run_id: str | None = None
        self._active_run: Any = None

        # Lazy import mlflow
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError as e:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow"
            ) from e

        self._setup_tracking()

    def _setup_tracking(self) -> None:
        """Configure MLflow tracking URI and experiment."""
        # Set tracking URI
        if self.config.tracking_uri:
            self._mlflow.set_tracking_uri(self.config.tracking_uri)
        elif is_databricks():
            # On Databricks, use the workspace tracking
            self._mlflow.set_tracking_uri("databricks")
        else:
            # Local: use file-based tracking
            local_uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
            self._mlflow.set_tracking_uri(local_uri)

        # Set or create experiment
        experiment = self._mlflow.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            self._mlflow.create_experiment(self.config.experiment_name)
        self._mlflow.set_experiment(self.config.experiment_name)

        logger.info(
            "MLflow configured",
            experiment=self.config.experiment_name,
            tracking_uri=self._mlflow.get_tracking_uri(),
        )

    @property
    def run_id(self) -> str | None:
        """Get the current run ID."""
        return self._run_id

    @contextmanager
    def start_run(self, run_name: str | None = None) -> Generator[Any, None, None]:
        """Start an MLflow run as a context manager.

        Args:
            run_name: Optional run name (uses config if not provided).

        Yields:
            Active MLflow run.
        """
        name = run_name or self.config.run_name
        tags = {**self.config.tags}

        # Add runtime tag
        tags["runtime"] = "databricks" if is_databricks() else "local"

        self._active_run = self._mlflow.start_run(run_name=name, tags=tags)
        self._run_id = self._active_run.info.run_id

        logger.info("MLflow run started", run_id=self._run_id, run_name=name)

        try:
            yield self._active_run
        finally:
            self._mlflow.end_run()
            logger.info("MLflow run ended", run_id=self._run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameter names to values.
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)

        # MLflow has a limit of 500 chars per param value
        truncated_params = {
            k: str(v)[:500] for k, v in flat_params.items()
        }

        self._mlflow.log_params(truncated_params)
        logger.debug("Logged parameters", count=len(truncated_params))

    def log_config(self, config: PipelineConfig) -> None:
        """Log pipeline configuration as parameters.

        Args:
            config: Pipeline configuration.
        """
        config_dict = config.model_dump(mode="json")
        self.log_params(config_dict)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step number.
        """
        for name, value in metrics.items():
            self._mlflow.log_metric(name, value, step=step)

        logger.debug("Logged metrics", count=len(metrics))

    def log_model(self, model: Any, model_name: str) -> None:
        """Log model artifact to MLflow.

        Args:
            model: Trained model.
            model_name: Model type name.
        """
        # Detect model type and use appropriate logger
        model_class = type(model).__module__

        if "xgboost" in model_class:
            # Fix XGBoost/sklearn compatibility: ensure _estimator_type is set
            if not hasattr(model, "_estimator_type"):
                model_class_name = type(model).__name__.lower()
                if "classifier" in model_class_name:
                    model._estimator_type = "classifier"
                elif "regressor" in model_class_name:
                    model._estimator_type = "regressor"
                else:
                    model._estimator_type = "classifier"  # Default fallback
            self._mlflow.xgboost.log_model(model, artifact_path="model")
        elif "lightgbm" in model_class:
            self._mlflow.lightgbm.log_model(model, artifact_path="model")
        elif "catboost" in model_class:
            self._mlflow.catboost.log_model(model, artifact_path="model")
        else:
            self._mlflow.sklearn.log_model(model, artifact_path="model")

        logger.info("Model logged", model_type=model_name)

        # Register model if configured
        if self.config.register_model and self.config.model_name:
            model_uri = f"runs:/{self._run_id}/model"
            self._mlflow.register_model(model_uri, self.config.model_name)
            logger.info("Model registered", name=self.config.model_name)

    def log_feature_importance(
        self, importance: dict[str, float], top_k: int = 20
    ) -> None:
        """Log feature importance as artifact and metrics.

        Args:
            importance: Dictionary of feature names to importance values.
            top_k: Number of top features to log as metrics.
        """
        import json

        # Sort by importance
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        # Log full importance as artifact
        self._mlflow.log_dict(sorted_importance, "feature_importance.json")

        # Log top features as metrics
        for i, (name, value) in enumerate(list(sorted_importance.items())[:top_k]):
            safe_name = name.replace(" ", "_")[:200]
            self._mlflow.log_metric(f"feat_importance_{i:02d}_{safe_name}", value)

        logger.info("Feature importance logged", n_features=len(importance))

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log.
            artifact_file: Filename for the artifact (e.g., "data.json").
        """
        self._mlflow.log_dict(dictionary, artifact_file)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as artifact.

        Args:
            local_path: Path to local file.
            artifact_path: Optional artifact subdirectory.
        """
        self._mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, figure: Any, filename: str) -> None:
        """Log a matplotlib/plotly figure.

        Args:
            figure: Figure object.
            filename: Filename for the artifact.
        """
        self._mlflow.log_figure(figure, filename)

    def _flatten_dict(
        self, d: dict, parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten nested dictionary.

        Args:
            d: Nested dictionary.
            parent_key: Parent key prefix.
            sep: Separator between keys.

        Returns:
            Flattened dictionary.
        """
        items: list[tuple[str, Any]] = []

        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))

        return dict(items)
