"""Cross-validation utilities."""

from typing import Any

import numpy as np
import structlog
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

logger = structlog.get_logger(__name__)


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    stratified: bool = True,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> dict[str, float]:
    """Perform cross-validation on a model.

    Args:
        model: Sklearn-compatible model.
        X: Feature array.
        y: Target array.
        n_folds: Number of CV folds.
        stratified: Use stratified folds (for classification).
        scoring: Scoring metric.
        random_state: Random seed.

    Returns:
        Dictionary with mean and std of CV scores.
    """
    from sklearn.base import clone

    if stratified:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    scores = []

    try:
        # Try manual cross-validation to avoid sklearn/XGBoost compatibility issues
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            try:
                fold_model = clone(model)
            except (TypeError, AttributeError):
                # If clone fails (e.g., with XGBoost), create a new instance
                fold_model = model.__class__(**model.get_params())

            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_val)

            # Calculate metric
            if scoring == "accuracy":
                score = float(accuracy_score(y_val, y_pred))
            elif scoring == "f1":
                score = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
            elif scoring == "mse":
                score = float(mean_squared_error(y_val, y_pred))
            else:
                score = float(accuracy_score(y_val, y_pred))

            scores.append(score)

    except Exception as e:
        logger.warning(
            "Cross-validation failed, returning empty results",
            error=str(e),
        )
        return {
            f"cv_{scoring}_mean": 0.0,
            f"cv_{scoring}_std": 0.0,
        }

    scores_array = np.array(scores)
    results = {
        f"cv_{scoring}_mean": float(np.mean(scores_array)),
        f"cv_{scoring}_std": float(np.std(scores_array)),
    }

    logger.info(
        "Cross-validation complete",
        scoring=scoring,
        n_folds=n_folds,
        mean=results[f"cv_{scoring}_mean"],
        std=results[f"cv_{scoring}_std"],
    )

    return results


def get_cv_predictions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    stratified: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Get out-of-fold predictions using cross-validation.

    Args:
        model: Sklearn-compatible model.
        X: Feature array.
        y: Target array.
        n_folds: Number of CV folds.
        stratified: Use stratified folds.
        random_state: Random seed.

    Returns:
        Tuple of (oof_predictions, oof_indices).
    """
    from sklearn.base import clone

    if stratified:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_predictions = np.zeros(len(y))
    oof_indices = np.zeros(len(y), dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        try:
            fold_model = clone(model)
        except (TypeError, AttributeError):
            # If clone fails (e.g., with XGBoost), create a new instance
            fold_model = model.__class__(**model.get_params())

        fold_model.fit(X_train, y_train)

        oof_predictions[val_idx] = fold_model.predict(X_val)
        oof_indices[val_idx] = fold_idx

    return oof_predictions, oof_indices
