"""Classification pipeline implementation."""

from typing import Any

import pandas as pd
import structlog
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_framework.config.base import ModelType, PipelineConfig
from ml_framework.features.transformer import FeatureTransformer
from ml_framework.models.factory import create_model
from ml_framework.pipelines.base import BasePipeline

logger = structlog.get_logger(__name__)


class ClassificationPipeline(BasePipeline):
    """Pipeline for binary and multiclass classification tasks."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize classification pipeline."""
        super().__init__(config)
        self._feature_transformer: FeatureTransformer | None = None
        self._classes: list | None = None

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> Any:
        """Train classification model.

        Args:
            train_data: Training dataset.
            val_data: Optional validation dataset for early stopping.

        Returns:
            Trained classifier.
        """
        target_col = self.config.data.target_column
        feature_cols = self.config.data.feature_columns

        # Prepare features
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]

        if feature_cols:
            X_train = X_train[feature_cols]

        # Store classes for prediction
        self._classes = sorted(y_train.unique().tolist())

        # Create and fit feature transformer
        self._feature_transformer = FeatureTransformer(self.config.features)
        X_train_transformed = self._feature_transformer.fit_transform(X_train)

        # Store feature names after transformation
        if self.config.data.feature_columns is None:
            self.config.data.feature_columns = self._feature_transformer.get_feature_names()

        # Create model
        model = create_model(self.config.model)

        # Handle early stopping if validation data provided
        if val_data is not None and len(val_data) > 0 and self.config.model.early_stopping:
            X_val = val_data.drop(columns=[target_col])
            y_val = val_data[target_col]

            if feature_cols:
                X_val = X_val[feature_cols]

            X_val_transformed = self._feature_transformer.transform(X_val)

            # XGBoost / LightGBM / CatBoost early stopping
            if self.config.model.model_type in (ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST):
                model.fit(
                    X_train_transformed,
                    y_train,
                    eval_set=[(X_val_transformed, y_val)],
                )
            else:
                model.fit(X_train_transformed, y_train)
        else:
            model.fit(X_train_transformed, y_train)

        logger.info("Model training complete", model_type=self.config.model.model_type.value)
        return model

    def evaluate(self, data: pd.DataFrame, prefix: str = "test") -> dict[str, float]:
        """Evaluate classification model.

        Args:
            data: Evaluation dataset.
            prefix: Metric name prefix.

        Returns:
            Dictionary of metrics.
        """
        if len(data) == 0:
            return {}

        target_col = self.config.data.target_column
        feature_cols = self.config.data.feature_columns

        X = data.drop(columns=[target_col])
        y_true = data[target_col]

        if feature_cols:
            X = X[feature_cols]

        X_transformed = self._feature_transformer.transform(X)
        y_pred = self._model.predict(X_transformed)

        # Calculate metrics
        is_binary = len(self._classes) == 2
        average = "binary" if is_binary else "weighted"

        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}_precision": precision_score(y_true, y_pred, average=average),
            f"{prefix}_recall": recall_score(y_true, y_pred, average=average),
            f"{prefix}_f1": f1_score(y_true, y_pred, average=average),
        }

        # ROC AUC for binary classification with probability support
        if is_binary and hasattr(self._model, "predict_proba"):
            try:
                y_prob = self._model.predict_proba(X_transformed)[:, 1]
                metrics[f"{prefix}_roc_auc"] = roc_auc_score(y_true, y_prob)
            except Exception:
                pass  # Skip if ROC AUC calculation fails

        logger.info("Evaluation complete", prefix=prefix, metrics=metrics)
        return metrics

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate class predictions.

        Args:
            data: Input features.

        Returns:
            Predicted classes.
        """
        feature_cols = self.config.data.feature_columns

        X = data[feature_cols] if feature_cols else data
        X_transformed = self._feature_transformer.transform(X)

        return pd.Series(self._model.predict(X_transformed), index=data.index)

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities.

        Args:
            data: Input features.

        Returns:
            DataFrame with probability for each class.
        """
        if not hasattr(self._model, "predict_proba"):
            raise ValueError(f"Model {self.config.model.model_type} doesn't support predict_proba")

        feature_cols = self.config.data.feature_columns

        X = data[feature_cols] if feature_cols else data
        X_transformed = self._feature_transformer.transform(X)

        probs = self._model.predict_proba(X_transformed)
        return pd.DataFrame(probs, index=data.index, columns=self._classes)
