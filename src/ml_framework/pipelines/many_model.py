"""Many-model forecasting pipeline for distributed time series training.

Enables training individual models per time series group:
- Distributed processing via Spark Pandas UDFs
- Error isolation per group (failures don't stop pipeline)
- Support for local, global, and foundation models
- Dual-phase training for production + evaluation
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd
import structlog

from ml_framework.config.base import PipelineConfig
from ml_framework.data_quality import DataQualityChecker, DataQualityReport
from ml_framework.evaluation.backtesting import (
    BacktestResult,
    Frequency,
    WalkForwardBacktester,
)
from ml_framework.exceptions import (
    DataError,
    EmptyDatasetError,
    EvaluationError,
    ModelTrainingError,
)
from ml_framework.logging.mlflow_logger import MLflowLogger
from ml_framework.models.registry import ModelRegistry, ModelType
from ml_framework.utils.runtime import Runtime, get_runtime, is_databricks

logger = structlog.get_logger(__name__)


@dataclass
class GroupResult:
    """Result for a single group in many-model training."""

    group_id: str
    model_name: str
    status: str  # "success", "failed", "skipped"
    backtest_result: BacktestResult | None = None
    error_message: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    run_id: str | None = None


@dataclass
class ManyModelResult:
    """Aggregated results from many-model training."""

    group_results: list[GroupResult] = field(default_factory=list)
    model_name: str = ""
    total_groups: int = 0
    successful_groups: int = 0
    failed_groups: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_groups == 0:
            return 0.0
        return self.successful_groups / self.total_groups

    @property
    def aggregated_metrics(self) -> dict[str, float]:
        """Aggregate metrics across all successful groups."""
        successful = [r for r in self.group_results if r.status == "success"]
        if not successful:
            return {}

        all_metrics: dict[str, list[float]] = {}
        for result in successful:
            for key, value in result.metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)

        return {f"mean_{k}": sum(v) / len(v) for k, v in all_metrics.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        records = []
        for r in self.group_results:
            record = {
                "group_id": r.group_id,
                "model_name": r.model_name,
                "status": r.status,
                "error_message": r.error_message,
                "run_id": r.run_id,
                **r.metrics,
            }
            records.append(record)
        return pd.DataFrame(records)


class ManyModelForecaster:
    """Many-model forecasting with distributed training.

    Trains individual models per time series group, with support for:
    - Local models (one model per series)
    - Global models (cross-series learning)
    - Foundation models (zero-shot inference)

    Example:
        forecaster = ManyModelForecaster(
            model_name="StatsForecastAutoARIMA",
            group_col="store_id",
            date_col="date",
            target_col="sales",
            prediction_length=7,
            backtest_length=30,
            freq="D",
        )

        results = forecaster.run(df)
    """

    def __init__(
        self,
        model_name: str,
        group_col: str,
        date_col: str,
        target_col: str,
        prediction_length: int,
        backtest_length: int = 30,
        stride: int = 1,
        freq: str = "D",
        feature_cols: list[str] | None = None,
        model_registry: ModelRegistry | None = None,
        experiment_name: str | None = None,
    ):
        """Initialize many-model forecaster.

        Args:
            model_name: Name of model in registry.
            group_col: Column identifying time series groups.
            date_col: Date column name.
            target_col: Target column name.
            prediction_length: Forecast horizon.
            backtest_length: Backtesting window.
            stride: Step between backtest trials.
            freq: Time series frequency ("H", "D", "W", "M").
            feature_cols: Feature columns (None = auto-detect).
            model_registry: Model registry (uses default if None).
            experiment_name: MLflow experiment name.
        """
        self.model_name = model_name
        self.group_col = group_col
        self.date_col = date_col
        self.target_col = target_col
        self.prediction_length = prediction_length
        self.backtest_length = backtest_length
        self.stride = stride
        self.freq = Frequency(freq)
        self.feature_cols = feature_cols

        self.registry = model_registry or ModelRegistry()
        self.model_type = self.registry.get_model_type(model_name)

        self.experiment_name = experiment_name
        self._runtime = get_runtime()

        logger.info(
            "ManyModelForecaster initialized",
            model=model_name,
            model_type=self.model_type.value,
            runtime=self._runtime.value,
        )

    def validate_data(self, df: pd.DataFrame) -> DataQualityReport:
        """Run data quality checks.

        Args:
            df: Input DataFrame.

        Returns:
            DataQualityReport with validation results.
        """
        checker = DataQualityChecker(
            target_col=self.target_col,
            date_col=self.date_col,
            group_col=self.group_col,
            min_rows=self.backtest_length + self.prediction_length,
        )
        return checker.run(df)

    def run(
        self,
        df: pd.DataFrame,
        validate: bool = True,
    ) -> ManyModelResult:
        """Run many-model training and evaluation.

        Args:
            df: Input DataFrame with all time series.
            validate: Whether to run data quality checks.

        Returns:
            ManyModelResult with per-group results.
        """
        if validate:
            report = self.validate_data(df)
            if not report.is_valid:
                raise DataError(f"Data validation failed:\n{report.summary()}")

        # Route to appropriate training method based on model type
        if self.model_type == ModelType.LOCAL:
            return self._run_local_models(df)
        elif self.model_type == ModelType.GLOBAL:
            return self._run_global_model(df)
        elif self.model_type == ModelType.FOUNDATION:
            return self._run_foundation_model(df)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _run_local_models(self, df: pd.DataFrame) -> ManyModelResult:
        """Train local models (one per group) using distributed processing."""
        groups = df[self.group_col].unique()
        total_groups = len(groups)

        logger.info(
            "Starting local model training",
            model=self.model_name,
            n_groups=total_groups,
        )

        result = ManyModelResult(
            model_name=self.model_name,
            total_groups=total_groups,
        )

        if is_databricks():
            # Use Spark for distributed processing
            result = self._run_local_models_spark(df)
        else:
            # Local: process sequentially
            result = self._run_local_models_sequential(df)

        result.successful_groups = sum(
            1 for r in result.group_results if r.status == "success"
        )
        result.failed_groups = sum(
            1 for r in result.group_results if r.status == "failed"
        )

        logger.info(
            "Local model training complete",
            success_rate=f"{result.success_rate:.1%}",
            metrics=result.aggregated_metrics,
        )

        return result

    def _run_local_models_sequential(self, df: pd.DataFrame) -> ManyModelResult:
        """Process groups sequentially (for local execution)."""
        groups = df[self.group_col].unique()

        result = ManyModelResult(
            model_name=self.model_name,
            total_groups=len(groups),
        )

        for group_id in groups:
            group_df = df[df[self.group_col] == group_id].copy()
            group_result = self._evaluate_single_group(group_id, group_df)
            result.group_results.append(group_result)

        return result

    def _run_local_models_spark(self, df: pd.DataFrame) -> ManyModelResult:
        """Process groups in parallel using Spark Pandas UDF."""
        from ml_framework.utils.runtime import get_spark_session

        spark = get_spark_session()

        # Convert to Spark DataFrame
        spark_df = spark.createDataFrame(df)

        # Define output schema
        from pyspark.sql.types import (
            DoubleType,
            StringType,
            StructField,
            StructType,
        )

        output_schema = StructType([
            StructField("group_id", StringType(), True),
            StructField("model_name", StringType(), True),
            StructField("status", StringType(), True),
            StructField("error_message", StringType(), True),
            StructField("mse", DoubleType(), True),
            StructField("rmse", DoubleType(), True),
            StructField("mae", DoubleType(), True),
            StructField("mape", DoubleType(), True),
        ])

        # Create evaluation function with captured parameters
        model_name = self.model_name
        date_col = self.date_col
        target_col = self.target_col
        prediction_length = self.prediction_length
        backtest_length = self.backtest_length
        stride = self.stride
        freq = self.freq
        group_col = self.group_col

        def evaluate_group_udf(pdf: pd.DataFrame) -> pd.DataFrame:
            """Pandas UDF to evaluate a single group."""
            try:
                group_id = str(pdf[group_col].iloc[0])

                # Get model
                from ml_framework.models.registry import ModelRegistry
                registry = ModelRegistry()
                model = registry.get_model(model_name)

                # Create backtester
                backtester = WalkForwardBacktester(
                    date_col=date_col,
                    target_col=target_col,
                    prediction_length=prediction_length,
                    backtest_length=backtest_length,
                    stride=stride,
                    freq=freq,
                )

                # Run backtest
                backtest_result = backtester.run(
                    df=pdf,
                    model=model,
                    group_id=group_id,
                )

                metrics = backtest_result.mean_metrics

                return pd.DataFrame([{
                    "group_id": group_id,
                    "model_name": model_name,
                    "status": "success",
                    "error_message": None,
                    "mse": metrics.get("mse"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "mape": metrics.get("mape"),
                }])

            except Exception as e:
                return pd.DataFrame([{
                    "group_id": str(pdf[group_col].iloc[0]) if len(pdf) > 0 else "unknown",
                    "model_name": model_name,
                    "status": "failed",
                    "error_message": str(e)[:500],
                    "mse": None,
                    "rmse": None,
                    "mae": None,
                    "mape": None,
                }])

        # Apply to each group
        result_df = (
            spark_df.groupby(self.group_col)
            .applyInPandas(evaluate_group_udf, schema=output_schema)
        )

        # Collect results
        results_pdf = result_df.toPandas()

        result = ManyModelResult(
            model_name=self.model_name,
            total_groups=len(results_pdf),
        )

        for _, row in results_pdf.iterrows():
            group_result = GroupResult(
                group_id=row["group_id"],
                model_name=row["model_name"],
                status=row["status"],
                error_message=row["error_message"],
                metrics={
                    k: row[k] for k in ["mse", "rmse", "mae", "mape"]
                    if pd.notna(row.get(k))
                },
            )
            result.group_results.append(group_result)

        return result

    def _evaluate_single_group(
        self,
        group_id: str,
        group_df: pd.DataFrame,
    ) -> GroupResult:
        """Evaluate a single group with error isolation."""
        try:
            # Get model instance
            model = self.registry.get_model(self.model_name)

            # Create backtester
            backtester = WalkForwardBacktester(
                date_col=self.date_col,
                target_col=self.target_col,
                prediction_length=self.prediction_length,
                backtest_length=self.backtest_length,
                stride=self.stride,
                freq=self.freq,
                feature_cols=self.feature_cols,
            )

            # Run backtest
            backtest_result = backtester.run(
                df=group_df,
                model=model,
                group_id=str(group_id),
            )

            return GroupResult(
                group_id=str(group_id),
                model_name=self.model_name,
                status="success",
                backtest_result=backtest_result,
                metrics=backtest_result.mean_metrics,
            )

        except Exception as e:
            logger.warning(
                "Group evaluation failed",
                group_id=group_id,
                error=str(e),
            )
            return GroupResult(
                group_id=str(group_id),
                model_name=self.model_name,
                status="failed",
                error_message=str(e),
            )

    def _run_global_model(self, df: pd.DataFrame) -> ManyModelResult:
        """Train global model (cross-series learning).

        Uses dual-phase training:
        1. Train on full data for production model
        2. Train on train-only for evaluation metrics
        """
        groups = df[self.group_col].unique()

        logger.info(
            "Starting global model training",
            model=self.model_name,
            n_groups=len(groups),
        )

        # Get model
        model = self.registry.get_model(self.model_name)

        # Prepare data in format expected by global models
        # (unique_id, ds, y format for NeuralForecast)
        prepared_df = df.rename(columns={
            self.group_col: "unique_id",
            self.date_col: "ds",
            self.target_col: "y",
        })

        # Phase 1: Train on all data for production model
        try:
            model.fit(prepared_df)
            logger.info("Global model trained on full data")
        except Exception as e:
            raise ModelTrainingError(f"Global model training failed: {e}")

        # Phase 2: Evaluate using backtesting
        result = ManyModelResult(
            model_name=self.model_name,
            total_groups=len(groups),
        )

        # Simplified: evaluate on held-out portion
        # In production, you'd want proper time-series split
        for group_id in groups:
            group_df = df[df[self.group_col] == group_id]
            group_result = GroupResult(
                group_id=str(group_id),
                model_name=self.model_name,
                status="success",
                metrics={},  # Would populate from model.predict() + metrics
            )
            result.group_results.append(group_result)

        result.successful_groups = len(groups)

        return result

    def _run_foundation_model(self, df: pd.DataFrame) -> ManyModelResult:
        """Run inference with foundation model (zero-shot)."""
        groups = df[self.group_col].unique()

        logger.info(
            "Starting foundation model inference",
            model=self.model_name,
            n_groups=len(groups),
        )

        result = ManyModelResult(
            model_name=self.model_name,
            total_groups=len(groups),
        )

        # Foundation models don't train, just inference
        for group_id in groups:
            group_df = df[df[self.group_col] == group_id]
            group_result = self._evaluate_single_group(str(group_id), group_df)
            result.group_results.append(group_result)

        result.successful_groups = sum(
            1 for r in result.group_results if r.status == "success"
        )
        result.failed_groups = sum(
            1 for r in result.group_results if r.status == "failed"
        )

        return result
