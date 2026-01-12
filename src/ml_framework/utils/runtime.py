"""Runtime detection and environment utilities."""

import os
from enum import Enum
from functools import lru_cache


class Runtime(str, Enum):
    """Execution runtime environment."""

    LOCAL = "local"
    DATABRICKS = "databricks"


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    """Detect the current runtime environment.

    Detection is cached for performance.

    Returns:
        Runtime.DATABRICKS if running on Databricks, Runtime.LOCAL otherwise.
    """
    # Check for Databricks-specific environment variables
    databricks_indicators = [
        "DATABRICKS_RUNTIME_VERSION",
        "DB_HOME",
        "SPARK_HOME",
    ]

    for indicator in databricks_indicators:
        if os.environ.get(indicator):
            return Runtime.DATABRICKS

    # Check for dbutils availability
    try:
        from pyspark.dbutils import DBUtils  # noqa: F401
        return Runtime.DATABRICKS
    except ImportError:
        pass

    return Runtime.LOCAL


def is_databricks() -> bool:
    """Check if running on Databricks."""
    return get_runtime() == Runtime.DATABRICKS


def is_local() -> bool:
    """Check if running locally."""
    return get_runtime() == Runtime.LOCAL


def get_spark_session():
    """Get or create a Spark session.

    On Databricks: Returns the active session.
    Locally: Creates a new local session.

    Returns:
        SparkSession instance.

    Raises:
        ImportError: If PySpark is not installed.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise ImportError(
            "PySpark is required. Install with: pip install 'ml-automation-framework[databricks]'"
        ) from e

    if is_databricks():
        # On Databricks, get the active session
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    else:
        # Locally, create a session with sensible defaults
        return (
            SparkSession.builder
            .appName("ml-automation-framework")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.driver.memory", "4g")
            .master("local[*]")
            .getOrCreate()
        )
