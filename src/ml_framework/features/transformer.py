"""Feature transformation pipeline."""

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

from ml_framework.config.base import FeatureConfig

logger = structlog.get_logger(__name__)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """Unified feature transformation pipeline.

    Handles numeric and categorical features with configurable
    imputation, scaling, and encoding strategies.
    """

    def __init__(self, config: FeatureConfig) -> None:
        """Initialize transformer with configuration.

        Args:
            config: Feature engineering configuration.
        """
        self.config = config
        self._transformer: ColumnTransformer | None = None
        self._feature_names: list[str] = []
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []

    def _detect_column_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Detect numeric and categorical columns.

        Args:
            X: Input DataFrame.

        Returns:
            Tuple of (numeric_columns, categorical_columns).
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        logger.info(
            "Column types detected",
            numeric=len(numeric_cols),
            categorical=len(categorical_cols),
        )

        return numeric_cols, categorical_cols

    def _build_numeric_pipeline(self) -> Pipeline:
        """Build preprocessing pipeline for numeric features."""
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy=self.config.numeric_impute_strategy))
        ]

        if self.config.numeric_scaling:
            scaler_map = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }
            scaler = scaler_map.get(self.config.numeric_scaling)
            if scaler:
                steps.append(("scaler", scaler))

        return Pipeline(steps)

    def _build_categorical_pipeline(self) -> Pipeline:
        """Build preprocessing pipeline for categorical features."""
        steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=self.config.categorical_impute_strategy,
                    fill_value="missing",
                ),
            )
        ]

        if self.config.categorical_encoding == "onehot":
            steps.append(
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown=self.config.handle_unknown,
                        sparse_output=False,
                        drop="if_binary",
                    ),
                )
            )

        return Pipeline(steps)

    def fit(self, X: pd.DataFrame) -> "FeatureTransformer":
        """Fit the transformer on training data.

        Args:
            X: Training features.

        Returns:
            Self for chaining.
        """
        self._numeric_cols, self._categorical_cols = self._detect_column_types(X)

        transformers = []

        if self._numeric_cols:
            transformers.append(
                ("numeric", self._build_numeric_pipeline(), self._numeric_cols)
            )

        if self._categorical_cols:
            transformers.append(
                ("categorical", self._build_categorical_pipeline(), self._categorical_cols)
            )

        if not transformers:
            raise ValueError("No valid columns found for transformation")

        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self._transformer.fit(X)
        self._feature_names = self._get_transformed_feature_names()

        logger.info("Transformer fitted", n_features=len(self._feature_names))
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features.

        Args:
            X: Input features.

        Returns:
            Transformed feature array.
        """
        if self._transformer is None:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        return self._transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Training features.

        Returns:
            Transformed feature array.
        """
        self.fit(X)
        return self.transform(X)

    def _get_transformed_feature_names(self) -> list[str]:
        """Get feature names after transformation."""
        if self._transformer is None:
            return []

        try:
            return list(self._transformer.get_feature_names_out())
        except AttributeError:
            # Fallback for older sklearn versions
            names = []
            for name, transformer, cols in self._transformer.transformers_:
                if name == "remainder":
                    continue
                if hasattr(transformer, "get_feature_names_out"):
                    names.extend(transformer.get_feature_names_out(cols))
                else:
                    names.extend(cols)
            return names

    def get_feature_names(self) -> list[str]:
        """Get list of feature names after transformation.

        Returns:
            List of feature names.
        """
        return self._feature_names.copy()

    def get_column_types(self) -> dict[str, list[str]]:
        """Get detected column types.

        Returns:
            Dictionary with 'numeric' and 'categorical' column lists.
        """
        return {
            "numeric": self._numeric_cols.copy(),
            "categorical": self._categorical_cols.copy(),
        }
