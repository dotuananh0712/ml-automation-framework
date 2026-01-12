"""Deployment configuration models.

Defines Pydantic models for model deployment to various targets.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeploymentTarget(str, Enum):
    """Supported deployment targets."""

    DATABRICKS_MODEL_SERVING = "databricks-model-serving"


class WorkloadSize(str, Enum):
    """Databricks Model Serving workload sizes."""

    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class DatabricksServingConfig(BaseModel):
    """Configuration for Databricks Model Serving deployment.

    Example:
        ```yaml
        deployment:
          target: databricks-model-serving
          databricks:
            endpoint_name: my-model-endpoint
            workload_size: Small
            scale_to_zero: true
        ```
    """

    endpoint_name: str = Field(..., description="Name of the serving endpoint")
    workload_size: WorkloadSize = Field(
        default=WorkloadSize.SMALL,
        description="Workload size: Small, Medium, or Large",
    )
    scale_to_zero: bool = Field(
        default=True,
        description="Scale to zero when no traffic",
    )
    environment_vars: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for the endpoint",
    )
    served_model_name: str | None = Field(
        default=None,
        description="Name for the served model (defaults to endpoint_name)",
    )
    model_version: int | None = Field(
        default=None,
        description="Specific model version to deploy (None = latest)",
    )


class DeploymentConfig(BaseModel):
    """Configuration for model deployment.

    Example:
        ```yaml
        deployment:
          enabled: true
          target: databricks-model-serving
          auto_deploy_best: true
          databricks:
            endpoint_name: churn-predictor
            workload_size: Small
            scale_to_zero: true
        ```
    """

    enabled: bool = Field(
        default=False,
        description="Enable deployment after training",
    )
    target: DeploymentTarget = Field(
        default=DeploymentTarget.DATABRICKS_MODEL_SERVING,
        description="Deployment target",
    )
    auto_deploy_best: bool = Field(
        default=False,
        description="Automatically deploy best model after tuning",
    )
    databricks: DatabricksServingConfig | None = Field(
        default=None,
        description="Databricks Model Serving configuration",
    )

    def get_target_config(self) -> Any:
        """Get configuration for the selected deployment target.

        Returns:
            Target-specific configuration.

        Raises:
            ValueError: If target configuration is missing.
        """
        if self.target == DeploymentTarget.DATABRICKS_MODEL_SERVING:
            if self.databricks is None:
                raise ValueError("Databricks configuration is required for databricks-model-serving target")
            return self.databricks
        else:
            raise ValueError(f"Unsupported deployment target: {self.target}")
