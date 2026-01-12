"""Walk-forward backtesting framework for time series.

Provides expanding window validation with configurable stride:
- backtest_length: Total historical window for evaluation
- prediction_length: Forecast horizon
- stride: Step size between backtest trials
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import structlog

from ml_framework.evaluation.metrics import calculate_regression_metrics
from ml_framework.exceptions import BacktestError

logger = structlog.get_logger(__name__)


class Frequency(str, Enum):
    """Supported time series frequencies."""

    HOURLY = "H"
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"


def get_date_offset(freq: Frequency, periods: int) -> pd.DateOffset:
    """Get DateOffset for a frequency and number of periods.

    Args:
        freq: Time series frequency.
        periods: Number of periods.

    Returns:
        pandas DateOffset.
    """
    offset_map = {
        Frequency.HOURLY: pd.DateOffset(hours=periods),
        Frequency.DAILY: pd.DateOffset(days=periods),
        Frequency.WEEKLY: pd.DateOffset(weeks=periods),
        Frequency.MONTHLY: pd.DateOffset(months=periods),
    }
    return offset_map[freq]


@dataclass
class BacktestTrial:
    """Result of a single backtest trial."""

    trial_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_size: int
    val_size: int
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.Series | None = None
    actuals: pd.Series | None = None


@dataclass
class BacktestResult:
    """Aggregated backtesting results."""

    trials: list[BacktestTrial] = field(default_factory=list)
    group_id: str | None = None

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Calculate mean metrics across all trials."""
        if not self.trials:
            return {}

        all_metric_names = set()
        for trial in self.trials:
            all_metric_names.update(trial.metrics.keys())

        means = {}
        for metric_name in all_metric_names:
            values = [t.metrics.get(metric_name) for t in self.trials if metric_name in t.metrics]
            if values:
                means[metric_name] = np.mean(values)

        return means

    @property
    def std_metrics(self) -> dict[str, float]:
        """Calculate std of metrics across all trials."""
        if not self.trials:
            return {}

        all_metric_names = set()
        for trial in self.trials:
            all_metric_names.update(trial.metrics.keys())

        stds = {}
        for metric_name in all_metric_names:
            values = [t.metrics.get(metric_name) for t in self.trials if metric_name in t.metrics]
            if values:
                stds[f"{metric_name}_std"] = np.std(values)

        return stds

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        records = []
        for trial in self.trials:
            record = {
                "trial_id": trial.trial_id,
                "train_start": trial.train_start,
                "train_end": trial.train_end,
                "val_start": trial.val_start,
                "val_end": trial.val_end,
                "train_size": trial.train_size,
                "val_size": trial.val_size,
                **trial.metrics,
            }
            if self.group_id:
                record["group_id"] = self.group_id
            records.append(record)

        return pd.DataFrame(records)


class WalkForwardBacktester:
    """Walk-forward backtesting with expanding window.

    Example:
        backtester = WalkForwardBacktester(
            date_col="date",
            target_col="sales",
            prediction_length=7,
            backtest_length=30,
            stride=7,
            freq=Frequency.DAILY,
        )

        results = backtester.run(
            df=data,
            model=model,
            fit_fn=lambda m, X, y: m.fit(X, y),
            predict_fn=lambda m, X: m.predict(X),
        )
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        prediction_length: int,
        backtest_length: int,
        stride: int = 1,
        freq: Frequency = Frequency.DAILY,
        feature_cols: list[str] | None = None,
    ):
        """Initialize backtester.

        Args:
            date_col: Name of date column.
            target_col: Name of target column.
            prediction_length: Forecast horizon.
            backtest_length: Total backtesting window.
            stride: Step size between trials.
            freq: Time series frequency.
            feature_cols: Feature columns to use (None = auto-detect).
        """
        self.date_col = date_col
        self.target_col = target_col
        self.prediction_length = prediction_length
        self.backtest_length = backtest_length
        self.stride = stride
        self.freq = freq
        self.feature_cols = feature_cols

        # Validate configuration
        if prediction_length > backtest_length:
            raise BacktestError(
                f"prediction_length ({prediction_length}) must be <= "
                f"backtest_length ({backtest_length})"
            )

        self._prediction_offset = get_date_offset(freq, prediction_length)
        self._stride_offset = get_date_offset(freq, stride)
        self._backtest_offset = get_date_offset(freq, backtest_length)

    def run(
        self,
        df: pd.DataFrame,
        model: Any,
        fit_fn: Callable[[Any, pd.DataFrame, pd.Series], None] | None = None,
        predict_fn: Callable[[Any, pd.DataFrame], np.ndarray] | None = None,
        group_id: str | None = None,
    ) -> BacktestResult:
        """Run walk-forward backtesting.

        Args:
            df: Input DataFrame with date and target columns.
            model: Model to evaluate.
            fit_fn: Function to fit model. Default: model.fit(X, y)
            predict_fn: Function to predict. Default: model.predict(X)
            group_id: Optional group identifier for many-model scenarios.

        Returns:
            BacktestResult with all trials.
        """
        # Ensure sorted by date
        df = df.sort_values(self.date_col).reset_index(drop=True)
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # Determine backtest window
        end_date = df[self.date_col].max()
        start_date = end_date - self._backtest_offset

        # Calculate number of trials
        num_trials = (self.backtest_length - self.prediction_length) // self.stride + 1

        logger.info(
            "Starting walk-forward backtest",
            n_trials=num_trials,
            start_date=start_date,
            end_date=end_date,
            group_id=group_id,
        )

        result = BacktestResult(group_id=group_id)
        curr_date = start_date

        trial_id = 0
        while curr_date + self._prediction_offset <= end_date:
            trial = self._run_single_trial(
                df=df,
                model=model,
                fit_fn=fit_fn,
                predict_fn=predict_fn,
                trial_id=trial_id,
                split_date=curr_date,
            )
            result.trials.append(trial)

            curr_date = curr_date + self._stride_offset
            trial_id += 1

        logger.info(
            "Backtest complete",
            n_trials=result.n_trials,
            mean_metrics=result.mean_metrics,
        )

        return result

    def _run_single_trial(
        self,
        df: pd.DataFrame,
        model: Any,
        fit_fn: Callable | None,
        predict_fn: Callable | None,
        trial_id: int,
        split_date: pd.Timestamp,
    ) -> BacktestTrial:
        """Run a single backtest trial."""
        # Split data
        train_df = df[df[self.date_col] < split_date]
        val_df = df[
            (df[self.date_col] >= split_date)
            & (df[self.date_col] < split_date + self._prediction_offset)
        ]

        if len(train_df) == 0 or len(val_df) == 0:
            return BacktestTrial(
                trial_id=trial_id,
                train_start=df[self.date_col].min(),
                train_end=split_date,
                val_start=split_date,
                val_end=split_date + self._prediction_offset,
                train_size=len(train_df),
                val_size=len(val_df),
                metrics={},
            )

        # Prepare features and target
        feature_cols = self.feature_cols
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in [self.date_col, self.target_col]]

        X_train = train_df[feature_cols] if feature_cols else train_df.drop(
            columns=[self.date_col, self.target_col]
        )
        y_train = train_df[self.target_col]
        X_val = val_df[feature_cols] if feature_cols else val_df.drop(
            columns=[self.date_col, self.target_col]
        )
        y_val = val_df[self.target_col]

        # Fit model
        if fit_fn:
            fit_fn(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Predict
        if predict_fn:
            predictions = predict_fn(model, X_val)
        else:
            predictions = model.predict(X_val)

        # Calculate metrics
        metrics = calculate_regression_metrics(
            y_true=y_val.values,
            y_pred=predictions,
            prefix="",
        )

        return BacktestTrial(
            trial_id=trial_id,
            train_start=train_df[self.date_col].min(),
            train_end=split_date,
            val_start=split_date,
            val_end=split_date + self._prediction_offset,
            train_size=len(train_df),
            val_size=len(val_df),
            metrics=metrics,
            predictions=pd.Series(predictions, index=val_df.index),
            actuals=y_val,
        )


def backtest_time_series(
    df: pd.DataFrame,
    model: Any,
    date_col: str,
    target_col: str,
    prediction_length: int,
    backtest_length: int,
    stride: int = 1,
    freq: str = "D",
    **kwargs,
) -> BacktestResult:
    """Convenience function for time series backtesting.

    Args:
        df: Input DataFrame.
        model: Model to evaluate.
        date_col: Date column name.
        target_col: Target column name.
        prediction_length: Forecast horizon.
        backtest_length: Total backtesting window.
        stride: Step between trials.
        freq: Frequency string ("H", "D", "W", "M").
        **kwargs: Additional arguments for WalkForwardBacktester.

    Returns:
        BacktestResult with all trials.
    """
    frequency = Frequency(freq)

    backtester = WalkForwardBacktester(
        date_col=date_col,
        target_col=target_col,
        prediction_length=prediction_length,
        backtest_length=backtest_length,
        stride=stride,
        freq=frequency,
        **kwargs,
    )

    return backtester.run(df=df, model=model)
