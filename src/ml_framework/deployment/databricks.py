"""Databricks Model Serving deployment.

Provides deployment to Databricks Model Serving endpoints via the Databricks SDK.
"""

import os
from typing import Any

import structlog

from ml_framework.config.deployment import DatabricksServingConfig, DeploymentConfig, DeploymentTarget
from ml_framework.deployment.base import BaseDeployer, DeploymentResult
from ml_framework.exceptions import ConfigurationError, InfrastructureError

logger = structlog.get_logger(__name__)


class DatabricksDeployer(BaseDeployer):
    """Deploy models to Databricks Model Serving.

    Requires Databricks SDK and proper authentication:
    - DATABRICKS_HOST environment variable
    - DATABRICKS_TOKEN environment variable (or other auth method)

    Example:
        ```python
        config = DeploymentConfig(
            enabled=True,
            target=DeploymentTarget.DATABRICKS_MODEL_SERVING,
            databricks=DatabricksServingConfig(
                endpoint_name="my-endpoint",
                workload_size="Small",
            ),
        )
        deployer = DatabricksDeployer(config)
        result = deployer.deploy("run_abc123")
        ```
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize Databricks deployer.

        Args:
            config: Deployment configuration with Databricks settings.
        """
        super().__init__(config)
        self._workspace_client: Any = None
        self._databricks_config: DatabricksServingConfig = config.get_target_config()

    def _validate_config(self) -> None:
        """Validate Databricks deployment configuration."""
        if self.config.target != DeploymentTarget.DATABRICKS_MODEL_SERVING:
            raise ConfigurationError(
                f"DatabricksDeployer requires target {DeploymentTarget.DATABRICKS_MODEL_SERVING.value}, "
                f"got {self.config.target.value}"
            )

        if self.config.databricks is None:
            raise ConfigurationError(
                "Databricks configuration is required for Databricks Model Serving"
            )

        # Check for Databricks credentials
        if not os.environ.get("DATABRICKS_HOST"):
            logger.warning(
                "DATABRICKS_HOST not set - deployment will fail unless running on Databricks"
            )

    def _get_workspace_client(self) -> Any:
        """Get or create Databricks workspace client.

        Returns:
            Databricks WorkspaceClient instance.

        Raises:
            InfrastructureError: If SDK not installed or auth fails.
        """
        if self._workspace_client is None:
            try:
                from databricks.sdk import WorkspaceClient

                self._workspace_client = WorkspaceClient()
                logger.info(
                    "Databricks client initialized",
                    host=self._workspace_client.config.host,
                )
            except ImportError:
                raise InfrastructureError(
                    "Databricks SDK not installed",
                    {"suggestion": "pip install databricks-sdk"},
                )
            except Exception as e:
                raise InfrastructureError(
                    f"Failed to initialize Databricks client: {e}",
                    {"error": str(e)},
                )

        return self._workspace_client

    def _register_model(self, run_id: str, model_name: str) -> str:
        """Register model from MLflow run to Unity Catalog.

        Args:
            run_id: MLflow run ID.
            model_name: Name for the registered model.

        Returns:
            Model version number as string.
        """
        import mlflow

        model_uri = f"runs:/{run_id}/model"

        # Register model
        result = mlflow.register_model(model_uri, model_name)
        version = result.version

        logger.info(
            "Model registered",
            model_name=model_name,
            version=version,
            run_id=run_id,
        )

        return str(version)

    def deploy(self, run_id: str, model_name: str | None = None) -> DeploymentResult:
        """Deploy model to Databricks Model Serving.

        Args:
            run_id: MLflow run ID containing the model.
            model_name: Optional model name for registration.

        Returns:
            DeploymentResult with endpoint information.
        """
        endpoint_name = self._databricks_config.endpoint_name
        served_model_name = self._databricks_config.served_model_name or endpoint_name

        # Use provided name or derive from endpoint
        if model_name is None:
            model_name = endpoint_name.replace("-", "_")

        logger.info(
            "Starting deployment",
            endpoint=endpoint_name,
            model_name=model_name,
            run_id=run_id,
        )

        try:
            ws = self._get_workspace_client()

            # Register model if not already registered
            if self._databricks_config.model_version:
                version = str(self._databricks_config.model_version)
            else:
                version = self._register_model(run_id, model_name)

            # Build serving config
            from databricks.sdk.service.serving import (
                EndpointCoreConfigInput,
                ServedModelInput,
                ServedModelInputWorkloadSize,
            )

            workload_size = ServedModelInputWorkloadSize(
                self._databricks_config.workload_size.value
            )

            served_model = ServedModelInput(
                model_name=model_name,
                model_version=version,
                workload_size=workload_size,
                scale_to_zero_enabled=self._databricks_config.scale_to_zero,
                environment_vars=self._databricks_config.environment_vars or None,
            )

            config_input = EndpointCoreConfigInput(served_models=[served_model])

            # Check if endpoint exists
            try:
                existing = ws.serving_endpoints.get(endpoint_name)
                # Update existing endpoint
                logger.info("Updating existing endpoint", endpoint=endpoint_name)
                ws.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_models=[served_model],
                )
            except Exception:
                # Create new endpoint
                logger.info("Creating new endpoint", endpoint=endpoint_name)
                ws.serving_endpoints.create(name=endpoint_name, config=config_input)

            # Get endpoint URL
            endpoint_url = f"https://{ws.config.host}/serving-endpoints/{endpoint_name}"

            logger.info(
                "Deployment initiated",
                endpoint=endpoint_name,
                endpoint_url=endpoint_url,
            )

            return DeploymentResult(
                success=True,
                endpoint_name=endpoint_name,
                endpoint_url=endpoint_url,
                status="PENDING",
                message="Deployment initiated successfully",
                metadata={
                    "model_name": model_name,
                    "model_version": version,
                    "run_id": run_id,
                    "workload_size": self._databricks_config.workload_size.value,
                    "scale_to_zero": self._databricks_config.scale_to_zero,
                },
            )

        except Exception as e:
            error_msg = str(e)
            logger.error("Deployment failed", error=error_msg, endpoint=endpoint_name)

            return DeploymentResult(
                success=False,
                endpoint_name=endpoint_name,
                status="FAILED",
                message=f"Deployment failed: {error_msg}",
                metadata={"error": error_msg, "run_id": run_id},
            )

    def get_endpoint_status(self, endpoint_name: str) -> str:
        """Get status of a serving endpoint.

        Args:
            endpoint_name: Name of the endpoint.

        Returns:
            Status string: READY, PENDING, UPDATING, FAILED, NOT_FOUND, or UNKNOWN.
        """
        try:
            ws = self._get_workspace_client()
            endpoint = ws.serving_endpoints.get(endpoint_name)

            # Extract state from endpoint
            if endpoint.state:
                ready = endpoint.state.ready
                if ready == "READY":
                    return "READY"
                elif ready == "NOT_READY":
                    # Check config update state
                    config_update = endpoint.state.config_update
                    if config_update in ("IN_PROGRESS", "PENDING"):
                        return "UPDATING"
                    return "PENDING"
                else:
                    return str(ready)

            return "UNKNOWN"

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "does not exist" in error_str:
                return "NOT_FOUND"
            logger.warning("Failed to get endpoint status", error=str(e))
            return "UNKNOWN"

    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a serving endpoint.

        Args:
            endpoint_name: Name of the endpoint to delete.

        Returns:
            True if deletion succeeded.
        """
        try:
            ws = self._get_workspace_client()
            ws.serving_endpoints.delete(endpoint_name)
            logger.info("Endpoint deleted", endpoint=endpoint_name)
            return True
        except Exception as e:
            logger.error("Failed to delete endpoint", error=str(e))
            return False

    def list_endpoints(self) -> list[dict[str, Any]]:
        """List all serving endpoints.

        Returns:
            List of endpoint information dictionaries.
        """
        try:
            ws = self._get_workspace_client()
            endpoints = ws.serving_endpoints.list()

            return [
                {
                    "name": ep.name,
                    "state": ep.state.ready if ep.state else "UNKNOWN",
                    "creation_timestamp": ep.creation_timestamp,
                }
                for ep in endpoints
            ]
        except Exception as e:
            logger.error("Failed to list endpoints", error=str(e))
            return []
