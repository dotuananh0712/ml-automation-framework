"""Model deployment module.

Provides deployment adapters for various targets including Databricks Model Serving.
"""

from ml_framework.deployment.base import BaseDeployer, DeploymentResult
from ml_framework.deployment.databricks import DatabricksDeployer

__all__ = ["BaseDeployer", "DeploymentResult", "DatabricksDeployer"]
