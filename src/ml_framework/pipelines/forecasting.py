"""Time series forecasting pipeline implementation."""

from typing import Any

import pandas as pd
import structlog
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_framework.config.base import PipelineConfig
from ml_framework.features.transformer import FeatureTransformer
from ml_framework.models.factory import create_model
from ml_framework.pipelines.base import BasePipeline

logger = structlog.get_logger(__name__)


class ForecastingPipeline(BasePipeline):
    """Pipeline for time series forecasting with lag features."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize forecasting pipeline."""
        super().__init__(config)
        self._feature_transformer: FeatureTransformer | None = None

    def split_data(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split time series data chronologically.

        Unlike random splitting, time series requires chronological order.

        Args:
            data: Full dataset (must be sorted by date).

        Returns:
            Tuple of (train, validation, test) DataFrames.
        """
        date_col = self.config.data.date_column

        if date_col is None:
            logger.warning("No date column specified, using index for time-based split")
            data = data.sort_index()
        else:
            data = data.sort_values(date_col)

        n = len(data)
        train_end = int(n * self.config.data.train_ratio)
        val_end = train_end + int(n * self.config.data.validation_ratio)

        train = data.iloc[:train_end]
        val = data.iloc[train_end:val_end]
        test = data.iloc[val_end:]

        logger.info(
            "Time series split complete",
            train_size=len(train),
            val_size=len(val),
            test_size=len(test),
        )

        return train, val, test

    def _create_lag_features(
        self, data: pd.DataFrame, lags: list[int] | None = None
    ) -> pd.DataFrame:
        """Create lag features for time series.

        Args:
            data: Input DataFrame.
            lags: List of lag periods. Default [1, 7, 14, 28].

        Returns:
            DataFrame with lag features added.
        """
        if lags is None:
            lags = [1, 7, 14, 28]

        target_col = self.config.data.target_column
        result = data.copy()

        for lag in lags:
            result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)

        # Add rolling statistics
        for window in [7, 14, 28]:
            result[f"{target_col}_rolling_mean_{window}"] = (
                result[target_col].shift(1).rolling(window=window).mean()
            )
            result[f"{target_col}_rolling_std_{window}"] = (
                result[target_col].shift(1).rolling(window=window).std()
            )

        # Drop rows with NaN from lag features
        result = result.dropna()

        return result

    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame | None = None) -> Any:
        """Train forecasting model with lag features.

        Args:
            train_data: Training dataset.
            val_data: Optional validation dataset.

        Returns:
            Trained model.
        """
        target_col = self.config.data.target_column

        # Create lag features
        train_with_lags = self._create_lag_features(train_data)

        X_train = train_with_lags.drop(columns=[target_col])
        y_train = train_with_lags[target_col]

        # Remove date column if present
        date_col = self.config.data.date_column
        if date_col and date_col in X_train.columns:
            X_train = X_train.drop(columns=[date_col])

        # Create and fit feature transformer
        self._feature_transformer = FeatureTransformer(self.config.features)
        X_train_transformed = self._feature_transformer.fit_transform(X_train)

        # Store feature names
        self.config.data.feature_columns = self._feature_transformer.get_feature_names()

        # Create and train model
        model = create_model(self.config.model)

        if val_data is not None and len(val_data) > 0:
            val_with_lags = self._create_lag_features(
                pd.concat([train_data, val_data]).tail(len(val_data) + 50)
            ).tail(len(val_data))

            if len(val_with_lags) > 0:
                X_val = val_with_lags.drop(columns=[target_col])
                y_val = val_with_lags[target_col]

                if date_col and date_col in X_val.columns:
                    X_val = X_val.drop(columns=[date_col])

                X_val_transformed = self._feature_transformer.transform(X_val)

                if hasattr(model, "fit") and self.config.model.early_stopping:
                    try:
                        model.fit(
                            X_train_transformed,
                            y_train,
                            eval_set=[(X_val_transformed, y_val)],
                        )
                    except TypeError:
                        model.fit(X_train_transformed, y_train)
                else:
                    model.fit(X_train_transformed, y_train)
            else:
                model.fit(X_train_transformed, y_train)
        else:
            model.fit(X_train_transformed, y_train)

        logger.info("Forecasting model training complete")
        return model

    def evaluate(self, data: pd.DataFrame, prefix: str = "test") -> dict[str, float]:
        """Evaluate forecasting model.

        Args:
            data: Evaluation dataset.
            prefix: Metric name prefix.

        Returns:
            Dictionary of regression metrics.
        """
        if len(data) == 0:
            return {}

        target_col = self.config.data.target_column
        date_col = self.config.data.date_column

        # Create lag features (need historical data)
        data_with_lags = self._create_lag_features(data)

        if len(data_with_lags) == 0:
            logger.warning("No valid rows after lag feature creation", prefix=prefix)
            return {}

        X = data_with_lags.drop(columns=[target_col])
        y_true = data_with_lags[target_col]

        if date_col and date_col in X.columns:
            X = X.drop(columns=[date_col])

        X_transformed = self._feature_transformer.transform(X)
        y_pred = self._model.predict(X_transformed)

        # Calculate regression metrics
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            f"{prefix}_mse": mse,
            f"{prefix}_rmse": mse ** 0.5,
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}_r2": r2_score(y_true, y_pred),
        }

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.any():
            mape = (abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
            metrics[f"{prefix}_mape"] = mape

        logger.info("Forecasting evaluation complete", prefix=prefix, metrics=metrics)
        return metrics

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate forecasts.

        Args:
            data: Input data with features.

        Returns:
            Forecasted values.
        """
        target_col = self.config.data.target_column
        date_col = self.config.data.date_column

        # Create lag features
        data_with_lags = self._create_lag_features(data)

        X = data_with_lags.drop(columns=[target_col], errors="ignore")

        if date_col and date_col in X.columns:
            X = X.drop(columns=[date_col])

        X_transformed = self._feature_transformer.transform(X)

        return pd.Series(self._model.predict(X_transformed), index=data_with_lags.index)
