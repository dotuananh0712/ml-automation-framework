"""Evaluation and metrics modules."""

from ml_framework.evaluation.backtesting import (
    BacktestResult,
    BacktestTrial,
    Frequency,
    WalkForwardBacktester,
    backtest_time_series,
)
from ml_framework.evaluation.cross_validation import cross_validate_model
from ml_framework.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_regression_metrics,
)

__all__ = [
    "calculate_classification_metrics",
    "calculate_regression_metrics",
    "cross_validate_model",
    "WalkForwardBacktester",
    "BacktestResult",
    "BacktestTrial",
    "Frequency",
    "backtest_time_series",
]
