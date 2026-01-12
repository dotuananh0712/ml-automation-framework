"""Optuna-based hyperparameter tuning.

Provides automated hyperparameter optimization with MLflow integration.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

from ml_framework.config.base import ModelConfig, PipelineConfig, PipelineType
from ml_framework.config.tuning import ParamType, PrunerType, SamplerType, TuningConfig
from ml_framework.features.transformer import FeatureTransformer
from ml_framework.models.factory import create_model

logger = structlog.get_logger(__name__)


@dataclass
class TuningResult:
    """Result of hyperparameter tuning.

    Attributes:
        best_params: Best hyperparameters found.
        best_value: Best metric value achieved.
        best_trial_number: Trial number with best result.
        n_trials: Total number of trials run.
        study_name: Name of the Optuna study.
        all_trials: Summary of all trials.
    """

    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int
    n_trials: int
    study_name: str
    all_trials: list[dict[str, Any]] = field(default_factory=list)


class OptunaTuner:
    """Optuna-based hyperparameter tuner with MLflow integration.

    Example:
        ```python
        tuner = OptunaTuner(config)
        result = tuner.run(train_data, val_data)
        print(f"Best params: {result.best_params}")
        print(f"Best {config.tuning.metric}: {result.best_value:.4f}")
        ```
    """

    def __init__(self, config: PipelineConfig):
        """Initialize tuner with pipeline configuration.

        Args:
            config: Pipeline configuration with tuning settings.
        """
        self.config = config
        self.tuning_config = config.tuning or TuningConfig()
        self._optuna = self._import_optuna()
        self._feature_transformer: FeatureTransformer | None = None
        self._X_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._X_val: pd.DataFrame | None = None
        self._y_val: pd.Series | None = None

    def _import_optuna(self) -> Any:
        """Import Optuna, raising helpful error if not installed."""
        try:
            import optuna

            return optuna
        except ImportError:
            raise ImportError(
                "Optuna not installed. "
                "Install with: pip install 'ml-automation-framework[tuning]'"
            )

    def _create_sampler(self) -> Any:
        """Create Optuna sampler based on configuration."""
        sampler_type = self.tuning_config.sampler

        if sampler_type == SamplerType.TPE:
            return self._optuna.samplers.TPESampler()
        elif sampler_type == SamplerType.RANDOM:
            return self._optuna.samplers.RandomSampler()
        elif sampler_type == SamplerType.CMAES:
            return self._optuna.samplers.CmaEsSampler()
        elif sampler_type == SamplerType.GRID:
            # Grid sampler requires search space - build from config
            search_space = self._build_grid_search_space()
            return self._optuna.samplers.GridSampler(search_space)
        else:
            return self._optuna.samplers.TPESampler()

    def _build_grid_search_space(self) -> dict[str, list[Any]]:
        """Build grid search space from tuning config."""
        search_space: dict[str, list[Any]] = {}

        for param in self.tuning_config.search_space:
            if param.type == ParamType.CATEGORICAL:
                search_space[param.name] = param.choices or []
            elif param.type in (ParamType.INT, ParamType.FLOAT):
                # For grid search, create discrete values
                if param.low is not None and param.high is not None:
                    step = param.step or ((param.high - param.low) / 5)
                    if param.type == ParamType.INT:
                        search_space[param.name] = list(
                            range(int(param.low), int(param.high) + 1, int(step) or 1)
                        )
                    else:
                        import numpy as np

                        search_space[param.name] = list(
                            np.arange(param.low, param.high + step, step)
                        )

        return search_space

    def _create_pruner(self) -> Any:
        """Create Optuna pruner based on configuration."""
        pruner_type = self.tuning_config.pruner

        if pruner_type == PrunerType.MEDIAN:
            return self._optuna.pruners.MedianPruner()
        elif pruner_type == PrunerType.HYPERBAND:
            return self._optuna.pruners.HyperbandPruner()
        elif pruner_type == PrunerType.NONE:
            return self._optuna.pruners.NopPruner()
        else:
            return self._optuna.pruners.MedianPruner()

    def _suggest_params(self, trial: Any) -> dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Dictionary of suggested hyperparameters.
        """
        params: dict[str, Any] = {}

        for param in self.tuning_config.search_space:
            if param.type == ParamType.INT:
                params[param.name] = trial.suggest_int(
                    param.name,
                    int(param.low or 0),
                    int(param.high or 100),
                    step=int(param.step) if param.step else 1,
                )
            elif param.type == ParamType.FLOAT:
                params[param.name] = trial.suggest_float(
                    param.name,
                    param.low or 0.0,
                    param.high or 1.0,
                    step=param.step,
                )
            elif param.type == ParamType.LOG_FLOAT:
                params[param.name] = trial.suggest_float(
                    param.name,
                    param.low or 1e-6,
                    param.high or 1.0,
                    log=True,
                )
            elif param.type == ParamType.CATEGORICAL:
                params[param.name] = trial.suggest_categorical(
                    param.name,
                    param.choices or [],
                )

        return params

    def _create_objective(self) -> Any:
        """Create objective function for Optuna optimization."""

        def objective(trial: Any) -> float:
            # Suggest hyperparameters
            suggested_params = self._suggest_params(trial)

            # Create model config with suggested params
            model_config = ModelConfig(
                model_type=self.config.model.model_type,
                hyperparameters={**self.config.model.hyperparameters, **suggested_params},
                random_state=self.config.model.random_state,
                cross_validation=False,  # Use validation set instead of CV
                early_stopping=self.config.model.early_stopping,
                early_stopping_rounds=self.config.model.early_stopping_rounds,
            )

            # Create and train model
            model = create_model(model_config, self.config.pipeline_type)

            try:
                # Handle early stopping for tree-based models
                if self._X_val is not None and self.config.model.early_stopping:
                    if self.config.model.model_type.value in ("xgboost", "lightgbm"):
                        model.fit(
                            self._X_train,
                            self._y_train,
                            eval_set=[(self._X_val, self._y_val)],
                        )
                    else:
                        model.fit(self._X_train, self._y_train)
                else:
                    model.fit(self._X_train, self._y_train)

                # Evaluate on validation set
                metric_value = self._evaluate_model(model)

                # Log trial to MLflow if enabled
                if self.tuning_config.log_all_trials:
                    self._log_trial_to_mlflow(trial, suggested_params, metric_value)

                return metric_value

            except Exception as e:
                logger.warning(
                    "Trial failed",
                    trial_number=trial.number,
                    error=str(e),
                )
                # Return a very bad value so this trial is not selected
                if self.tuning_config.direction == "maximize":
                    return float("-inf")
                else:
                    return float("inf")

        return objective

    def _evaluate_model(self, model: Any) -> float:
        """Evaluate model on validation data.

        Args:
            model: Trained model.

        Returns:
            Metric value for optimization.
        """
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

        y_pred = model.predict(self._X_val)
        y_true = self._y_val

        metric_name = self.tuning_config.metric

        # Strip prefix if present (e.g., "val_f1" -> "f1")
        if metric_name.startswith("val_"):
            metric_name = metric_name[4:]

        # Classification metrics
        if self.config.pipeline_type == PipelineType.CLASSIFICATION:
            is_binary = len(np.unique(y_true)) == 2
            average = "binary" if is_binary else "weighted"

            if metric_name == "accuracy":
                return float(accuracy_score(y_true, y_pred))
            elif metric_name == "f1":
                return float(f1_score(y_true, y_pred, average=average, zero_division=0))
            elif metric_name == "precision":
                return float(precision_score(y_true, y_pred, average=average, zero_division=0))
            elif metric_name == "recall":
                return float(recall_score(y_true, y_pred, average=average, zero_division=0))
            elif metric_name == "roc_auc":
                if is_binary and hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(self._X_val)[:, 1]
                    return float(roc_auc_score(y_true, y_prob))
                return float(f1_score(y_true, y_pred, average=average, zero_division=0))
            else:
                # Default to F1
                return float(f1_score(y_true, y_pred, average=average, zero_division=0))

        # Regression/Forecasting metrics
        else:
            if metric_name == "mse":
                return float(mean_squared_error(y_true, y_pred))
            elif metric_name == "rmse":
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))
            elif metric_name == "mae":
                return float(mean_absolute_error(y_true, y_pred))
            elif metric_name == "r2":
                return float(r2_score(y_true, y_pred))
            else:
                # Default to RMSE (lower is better)
                return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def _log_trial_to_mlflow(
        self, trial: Any, params: dict[str, Any], metric_value: float
    ) -> None:
        """Log trial results to MLflow.

        Args:
            trial: Optuna trial.
            params: Hyperparameters used.
            metric_value: Metric value achieved.
        """
        try:
            import mlflow

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params(params)
                mlflow.log_metric(self.tuning_config.metric, metric_value)
                mlflow.set_tag("trial_number", trial.number)

        except Exception as e:
            logger.warning("Failed to log trial to MLflow", error=str(e))

    def run(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ) -> TuningResult:
        """Run hyperparameter optimization.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.

        Returns:
            TuningResult with best parameters and metrics.
        """
        if not self.tuning_config.enabled:
            logger.info("Tuning disabled, returning default config")
            return TuningResult(
                best_params=self.config.model.hyperparameters,
                best_value=0.0,
                best_trial_number=0,
                n_trials=0,
                study_name="",
            )

        logger.info(
            "Starting hyperparameter tuning",
            n_trials=self.tuning_config.n_trials,
            direction=self.tuning_config.direction,
            metric=self.tuning_config.metric,
            sampler=self.tuning_config.sampler.value,
        )

        # Prepare data
        target_col = self.config.data.target_column
        feature_cols = self.config.data.feature_columns

        X_train = train_data.drop(columns=[target_col])
        self._y_train = train_data[target_col]

        X_val = val_data.drop(columns=[target_col])
        self._y_val = val_data[target_col]

        if feature_cols:
            X_train = X_train[feature_cols]
            X_val = X_val[feature_cols]

        # Apply feature transformation
        self._feature_transformer = FeatureTransformer(self.config.features)
        self._X_train = self._feature_transformer.fit_transform(X_train)
        self._X_val = self._feature_transformer.transform(X_val)

        # Create study
        study_name = f"{self.config.name}_tuning"
        study = self._optuna.create_study(
            study_name=study_name,
            direction=self.tuning_config.direction,
            sampler=self._create_sampler(),
            pruner=self._create_pruner(),
        )

        # Run optimization
        try:
            import mlflow

            # Start parent MLflow run if logging enabled
            if self.tuning_config.log_all_trials:
                with mlflow.start_run(run_name=f"{self.config.name}_tuning"):
                    mlflow.log_param("n_trials", self.tuning_config.n_trials)
                    mlflow.log_param("direction", self.tuning_config.direction)
                    mlflow.log_param("metric", self.tuning_config.metric)
                    mlflow.log_param("sampler", self.tuning_config.sampler.value)

                    study.optimize(
                        self._create_objective(),
                        n_trials=self.tuning_config.n_trials,
                        timeout=self.tuning_config.timeout,
                        show_progress_bar=True,
                    )

                    # Log best results
                    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
                    mlflow.log_metric(f"best_{self.tuning_config.metric}", study.best_value)
            else:
                study.optimize(
                    self._create_objective(),
                    n_trials=self.tuning_config.n_trials,
                    timeout=self.tuning_config.timeout,
                    show_progress_bar=True,
                )
        except ImportError:
            # MLflow not available, run without logging
            study.optimize(
                self._create_objective(),
                n_trials=self.tuning_config.n_trials,
                timeout=self.tuning_config.timeout,
                show_progress_bar=True,
            )

        # Collect trial summaries
        all_trials = [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ]

        logger.info(
            "Tuning complete",
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial=study.best_trial.number,
        )

        return TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            best_trial_number=study.best_trial.number,
            n_trials=len(study.trials),
            study_name=study_name,
            all_trials=all_trials,
        )

    def run_with_best_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
    ) -> tuple[TuningResult, Any]:
        """Run tuning and return best model.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.

        Returns:
            Tuple of (TuningResult, trained best model).
        """
        result = self.run(train_data, val_data)

        # Train final model with best params
        model_config = ModelConfig(
            model_type=self.config.model.model_type,
            hyperparameters={**self.config.model.hyperparameters, **result.best_params},
            random_state=self.config.model.random_state,
            cross_validation=self.config.model.cross_validation,
            cv_folds=self.config.model.cv_folds,
            early_stopping=self.config.model.early_stopping,
            early_stopping_rounds=self.config.model.early_stopping_rounds,
        )

        model = create_model(model_config, self.config.pipeline_type)

        # Train on combined train + val data
        combined = pd.concat([train_data, val_data], ignore_index=True)
        target_col = self.config.data.target_column
        feature_cols = self.config.data.feature_columns

        X_combined = combined.drop(columns=[target_col])
        y_combined = combined[target_col]

        if feature_cols:
            X_combined = X_combined[feature_cols]

        X_transformed = self._feature_transformer.fit_transform(X_combined)
        model.fit(X_transformed, y_combined)

        return result, model
