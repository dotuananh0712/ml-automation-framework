"""Base deployer interface for model deployment.

Defines abstract base class and common types for deployment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from ml_framework.config.deployment import DeploymentConfig

logger = structlog.get_logger(__name__)


@dataclass
class DeploymentResult:
    """Result of a deployment operation.

    Attributes:
        success: Whether deployment succeeded.
        endpoint_name: Name of the deployed endpoint.
        endpoint_url: URL of the deployed endpoint.
        status: Current status of the endpoint.
        message: Status message or error description.
        metadata: Additional deployment metadata.
    """

    success: bool
    endpoint_name: str
    endpoint_url: str | None = None
    status: str = "UNKNOWN"
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDeployer(ABC):
    """Abstract base class for model deployers.

    Subclasses implement deployment to specific targets (Databricks, etc.).

    Example:
        ```python
        deployer = DatabricksDeployer(config)
        result = deployer.deploy(run_id="abc123")
        if result.success:
            print(f"Deployed to: {result.endpoint_url}")
        ```
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize deployer.

        Args:
            config: Deployment configuration.
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate deployment configuration.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        pass

    @abstractmethod
    def deploy(self, run_id: str, model_name: str | None = None) -> DeploymentResult:
        """Deploy a model from MLflow run.

        Args:
            run_id: MLflow run ID containing the model.
            model_name: Optional model name for registration.

        Returns:
            DeploymentResult with status and endpoint info.
        """
        pass

    @abstractmethod
    def get_endpoint_status(self, endpoint_name: str) -> str:
        """Get current status of an endpoint.

        Args:
            endpoint_name: Name of the endpoint.

        Returns:
            Status string (e.g., "READY", "PENDING", "FAILED").
        """
        pass

    @abstractmethod
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete a deployed endpoint.

        Args:
            endpoint_name: Name of the endpoint to delete.

        Returns:
            True if deletion succeeded.
        """
        pass

    def wait_for_ready(
        self, endpoint_name: str, timeout_seconds: int = 600, poll_interval: int = 30
    ) -> bool:
        """Wait for endpoint to become ready.

        Args:
            endpoint_name: Name of the endpoint.
            timeout_seconds: Maximum time to wait.
            poll_interval: Seconds between status checks.

        Returns:
            True if endpoint is ready, False if timeout.
        """
        import time

        start = time.time()
        while time.time() - start < timeout_seconds:
            status = self.get_endpoint_status(endpoint_name)
            if status == "READY":
                return True
            if status in ("FAILED", "DELETED"):
                logger.error("Endpoint entered failed state", status=status)
                return False
            logger.info("Waiting for endpoint", status=status, endpoint=endpoint_name)
            time.sleep(poll_interval)

        logger.warning("Timeout waiting for endpoint", endpoint=endpoint_name)
        return False
