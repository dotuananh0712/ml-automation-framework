"""Foundation model wrappers for zero-shot forecasting.

Supports pre-trained transformer models:
- Chronos (Amazon): T5-based time series foundation model
- TimesFM (Google): Time series foundation model
- Moirai (Salesforce): Mixture-of-experts time series model
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import structlog

from ml_framework.exceptions import ModelPredictionError

logger = structlog.get_logger(__name__)


class FoundationForecaster(ABC):
    """Base class for foundation model forecasters.

    Foundation models are pre-trained and require no fine-tuning.
    They generate forecasts directly from historical context.
    """

    def __init__(
        self,
        model_path: str,
        prediction_length: int = 7,
        num_samples: int = 20,
        device: str = "auto",
    ):
        """Initialize foundation forecaster.

        Args:
            model_path: HuggingFace model path or local path.
            prediction_length: Number of steps to forecast.
            num_samples: Number of samples for probabilistic forecast.
            device: Device to use ('auto', 'cuda', 'cpu').
        """
        self.model_path = model_path
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.device = device
        self._model = None

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the pre-trained model."""
        pass

    @abstractmethod
    def predict(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> np.ndarray:
        """Generate point forecast.

        Args:
            context: Historical time series values.
            prediction_length: Override forecast horizon.

        Returns:
            Array of forecasted values.
        """
        pass

    def predict_quantiles(
        self,
        context: np.ndarray | pd.Series,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        prediction_length: int | None = None,
    ) -> dict[float, np.ndarray]:
        """Generate quantile forecasts.

        Args:
            context: Historical time series values.
            quantiles: Quantile levels to predict.
            prediction_length: Override forecast horizon.

        Returns:
            Dictionary mapping quantile level to forecast array.
        """
        # Default implementation using samples
        samples = self._predict_samples(context, prediction_length)
        return {q: np.quantile(samples, q, axis=0) for q in quantiles}

    @abstractmethod
    def _predict_samples(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> np.ndarray:
        """Generate sample forecasts.

        Args:
            context: Historical time series values.
            prediction_length: Override forecast horizon.

        Returns:
            Array of shape (num_samples, prediction_length).
        """
        pass


class ChronosForecaster(FoundationForecaster):
    """Chronos foundation model forecaster (Amazon).

    Chronos is a T5-based model pre-trained on diverse time series.
    Supports: chronos-t5-tiny, chronos-t5-mini, chronos-t5-small,
              chronos-t5-base, chronos-t5-large
    """

    def _load_model(self) -> Any:
        """Load Chronos model from HuggingFace."""
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "Chronos not installed. Install with: "
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = ChronosPipeline.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype="auto",
        )

        logger.info("Chronos model loaded", model=self.model_path, device=device)
        return self._model

    def predict(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> np.ndarray:
        """Generate median forecast."""
        samples = self._predict_samples(context, prediction_length)
        return np.median(samples, axis=0)

    def _predict_samples(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> np.ndarray:
        """Generate sample forecasts using Chronos."""
        import torch

        if self._model is None:
            self._load_model()

        horizon = prediction_length or self.prediction_length

        # Convert to tensor
        if isinstance(context, pd.Series):
            context = context.values
        context_tensor = torch.tensor(context, dtype=torch.float32)

        try:
            forecast = self._model.predict(
                context=context_tensor.unsqueeze(0),  # Add batch dimension
                prediction_length=horizon,
                num_samples=self.num_samples,
            )
            # Shape: (1, num_samples, horizon) -> (num_samples, horizon)
            return forecast.numpy().squeeze(0)

        except Exception as e:
            raise ModelPredictionError(f"Chronos prediction failed: {e}")


class TimesFMForecaster(FoundationForecaster):
    """TimesFM foundation model forecaster (Google).

    TimesFM is Google's time series foundation model.
    """

    def _load_model(self) -> Any:
        """Load TimesFM model."""
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "TimesFM not installed. Install with: pip install timesfm"
            )

        self._model = timesfm.TimesFm(
            context_len=512,
            horizon_len=self.prediction_length,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        )
        self._model.load_from_checkpoint(self.model_path)

        logger.info("TimesFM model loaded", model=self.model_path)
        return self._model

    def predict(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> np.ndarray:
        """Generate point forecast using TimesFM."""
        if self._model is None:
            self._load_model()

        horizon = prediction_length or self.prediction_length

        if isinstance(context, pd.Series):
            context = context.values

        try:
            # TimesFM expects 2D input (batch, time)
            context_2d = context.reshape(1, -1)
            forecast, _ = self._model.forecast(
                context_2d,
                freq=[0],  # Default frequency
                horizon=horizon,
            )
            return forecast.flatten()

        except Exception as e:
            raise ModelPredictionError(f"TimesFM prediction failed: {e}")

    def _predict_samples(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> np.ndarray:
        """TimesFM returns point forecasts, so we return single sample."""
        forecast = self.predict(context, prediction_length)
        return forecast.reshape(1, -1)  # Shape: (1, horizon)


def get_foundation_model(model_name: str, **kwargs) -> FoundationForecaster:
    """Factory function for foundation models.

    Args:
        model_name: Name or path of the model.
        **kwargs: Additional arguments passed to the forecaster.

    Returns:
        Initialized foundation forecaster.
    """
    model_lower = model_name.lower()

    if "chronos" in model_lower:
        return ChronosForecaster(model_path=model_name, **kwargs)
    elif "timesfm" in model_lower:
        return TimesFMForecaster(model_path=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown foundation model: {model_name}")
