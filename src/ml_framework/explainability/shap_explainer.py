"""SHAP-based model explainability.

Provides automated SHAP explanation generation and visualization.
"""

from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ml_framework.config.explainability import ExplainabilityConfig, ExplainerType

logger = structlog.get_logger(__name__)


class SHAPExplainer:
    """SHAP-based model explainability.

    Generates SHAP values and visualizations for model interpretability.

    Example:
        ```python
        explainer = SHAPExplainer(config)
        explainer.fit(model, X_train, feature_names)
        importance = explainer.get_feature_importance()
        plots = explainer.generate_plots(output_dir)
        ```
    """

    def __init__(self, config: ExplainabilityConfig):
        """Initialize SHAP explainer.

        Args:
            config: Explainability configuration.
        """
        self.config = config
        self._shap = self._import_shap()
        self._explainer: Any = None
        self._shap_values: Any = None
        self._X_sample: np.ndarray | None = None
        self._feature_names: list[str] | None = None

    def _import_shap(self) -> Any:
        """Import SHAP, raising helpful error if not installed."""
        try:
            import shap

            return shap
        except ImportError:
            raise ImportError(
                "SHAP not installed. "
                "Install with: pip install 'ml-automation-framework[explainability]'"
            )

    def _detect_explainer_type(self, model: Any) -> ExplainerType:
        """Auto-detect appropriate explainer for model type.

        Args:
            model: Trained model instance.

        Returns:
            Detected explainer type.
        """
        model_class = type(model).__name__.lower()

        tree_keywords = ["xgb", "lgb", "lightgbm", "randomforest", "gradientboosting", "tree", "forest"]
        linear_keywords = ["linear", "logistic", "ridge", "lasso", "elasticnet"]

        if any(kw in model_class for kw in tree_keywords):
            return ExplainerType.TREE
        elif any(kw in model_class for kw in linear_keywords):
            return ExplainerType.LINEAR
        else:
            return ExplainerType.KERNEL

    def fit(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "SHAPExplainer":
        """Create SHAP explainer and compute values.

        Args:
            model: Trained model.
            X: Feature matrix for computing SHAP values.
            feature_names: Optional list of feature names.

        Returns:
            Self for chaining.
        """
        if not self.config.enabled:
            logger.info("SHAP explainability disabled, skipping")
            return self

        explainer_type = self.config.explainer_type
        if explainer_type == ExplainerType.AUTO:
            explainer_type = self._detect_explainer_type(model)

        logger.info("Creating SHAP explainer", type=explainer_type.value)
        #DEBUG: Check input data type
        logger.info("Input data info",
            dtype=str(X.dtype if hasattr(X, 'dtype') else type(X)),
            shape=X.shape if hasattr(X, 'shape') else len(X))

        # Sample data if too large
        if len(X) > self.config.max_samples:
            indices = np.random.choice(len(X), self.config.max_samples, replace=False)
            X_sample = X[indices] if isinstance(X, np.ndarray) else X.iloc[indices].values
        else:
            X_sample = X if isinstance(X, np.ndarray) else X.values

        self._X_sample = X_sample
        self._feature_names = feature_names

        # Create explainer based on type
        try:
            if explainer_type == ExplainerType.TREE:
                logger.info("Creating TreeExplainer")
                self._explainer = self._shap.TreeExplainer(model)

            elif explainer_type == ExplainerType.LINEAR:
                logger.info("Creating LinearExplainer")
                self._explainer = self._shap.LinearExplainer(model, X_sample)
            else:
                # Kernel explainer (model-agnostic)
                logger.info("Creating KernelExplainer")

                background = self._shap.sample(X_sample, min(self.config.background_samples, len(X_sample)))

                predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
                self._explainer = self._shap.KernelExplainer(predict_fn, background)

            # Compute SHAP values
            logger.info("Computing SHAP values", n_samples=len(X_sample))
            self._shap_values = self._explainer.shap_values(X_sample)

        except Exception as e:
            logger.error("Failed to compute SHAP values", error=str(e))
            raise

        return self

    def get_feature_importance(self) -> dict[str, float]:
        """Get mean absolute SHAP values as feature importance.

        Returns:
            Dictionary of feature names to importance values.
        """
        if self._shap_values is None:
            raise RuntimeError("Must call fit() first")

        # Handle multi-class (take mean across classes)
        if isinstance(self._shap_values, list):
            shap_values = np.abs(np.array(self._shap_values)).mean(axis=0)
        else:
            shap_values = np.abs(self._shap_values)

        mean_importance = shap_values.mean(axis=0)

        if self._feature_names and len(self._feature_names) == len(mean_importance):
            return dict(zip(self._feature_names, mean_importance.tolist()))

        return {f"feature_{i}": float(v) for i, v in enumerate(mean_importance)}

    def generate_plots(self, output_dir: Path | str) -> list[Path]:
        """Generate SHAP visualization plots.

        Args:
            output_dir: Directory to save plots.

        Returns:
            List of paths to generated plot files.
        """
        if self._shap_values is None:
            raise RuntimeError("Must call fit() first")

        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plots: list[Path] = []

        # Handle multi-class - use positive class for binary or first class
        shap_values = self._shap_values
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        try:
            if self.config.generate_summary_plot:
                plt.figure(figsize=(10, 8))
                self._shap.summary_plot(
                    shap_values,
                    self._X_sample,
                    feature_names=self._feature_names,
                    show=False,
                )
                path = output_dir / "shap_summary.png"
                plt.savefig(path, bbox_inches="tight", dpi=150)
                plt.close()
                plots.append(path)
                logger.info("Generated SHAP summary plot", path=str(path))

            if self.config.generate_bar_plot:
                plt.figure(figsize=(10, 8))
                self._shap.summary_plot(
                    shap_values,
                    self._X_sample,
                    feature_names=self._feature_names,
                    plot_type="bar",
                    show=False,
                )
                path = output_dir / "shap_bar.png"
                plt.savefig(path, bbox_inches="tight", dpi=150)
                plt.close()
                plots.append(path)
                logger.info("Generated SHAP bar plot", path=str(path))

            if self.config.generate_dependence_plots and self._feature_names:
                importance = self.get_feature_importance()
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

                for feat_name, _ in top_features[: self.config.dependence_top_k]:
                    try:
                        feat_idx = self._feature_names.index(feat_name)
                        plt.figure(figsize=(8, 6))
                        self._shap.dependence_plot(
                            feat_idx,
                            shap_values,
                            self._X_sample,
                            feature_names=self._feature_names,
                            show=False,
                        )
                        safe_name = feat_name.replace("/", "_").replace(" ", "_")
                        path = output_dir / f"shap_dependence_{safe_name}.png"
                        plt.savefig(path, bbox_inches="tight", dpi=150)
                        plt.close()
                        plots.append(path)
                    except Exception as e:
                        logger.warning(f"Failed to generate dependence plot for {feat_name}", error=str(e))

        except Exception as e:
            logger.error("Error generating SHAP plots", error=str(e))

        return plots
