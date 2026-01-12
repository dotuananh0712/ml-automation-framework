"""ML Automation Framework - Config-driven pipelines for data science workflows."""

from ml_framework.config.base import PipelineConfig
from ml_framework.data_quality import DataQualityChecker, DataQualityReport
from ml_framework.pipelines.base import BasePipeline
from ml_framework.pipelines.many_model import ManyModelForecaster, ManyModelResult
from ml_framework.utils.runtime import Runtime, get_runtime

__version__ = "0.1.0"
__all__ = [
    "PipelineConfig",
    "BasePipeline",
    "ManyModelForecaster",
    "ManyModelResult",
    "DataQualityChecker",
    "DataQualityReport",
    "Runtime",
    "get_runtime",
]
