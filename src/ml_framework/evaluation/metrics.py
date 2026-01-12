"""Metric calculation utilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (optional, for ROC AUC).
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metric names to values.
    """
    unique_classes = np.unique(y_true)
    is_binary = len(unique_classes) == 2
    average = "binary" if is_binary else "weighted"

    prefix = f"{prefix}_" if prefix else ""

    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        f"{prefix}f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # ROC AUC for binary classification
    if is_binary and y_prob is not None:
        try:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass  # Skip if ROC AUC can't be calculated

    return metrics


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Calculate regression metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        prefix: Prefix for metric names.

    Returns:
        Dictionary of metric names to values.
    """
    prefix = f"{prefix}_" if prefix else ""

    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        f"{prefix}mse": mse,
        f"{prefix}rmse": np.sqrt(mse),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
    }

    # MAPE (Mean Absolute Percentage Error)
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics[f"{prefix}mape"] = mape

    return metrics
