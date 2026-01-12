"""Spark data loading utilities for Databricks environments.

Provides convenient methods for querying Delta tables and running Spark SQL.
"""

from typing import Any

import pandas as pd
import structlog

from ml_framework.utils.runtime import get_spark_session, is_databricks

logger = structlog.get_logger(__name__)


class SparkDataLoader:
    """Utility class for loading data via Spark on Databricks.

    Provides convenient methods for querying Delta tables, running Spark SQL,
    and loading data from Unity Catalog.

    Example:
        ```python
        loader = SparkDataLoader()

        # Run SQL query
        df = loader.query("SELECT * FROM catalog.schema.table WHERE date > '2024-01-01'")

        # Load table directly
        df = loader.load_table("catalog.schema.table")

        # Load Delta table by path
        df = loader.load_delta("dbfs:/mnt/data/my_table")
        ```
    """

    def __init__(self) -> None:
        """Initialize SparkDataLoader.

        Gets or creates a Spark session appropriate for the runtime.
        """
        self._spark = get_spark_session()
        logger.info(
            "SparkDataLoader initialized",
            runtime="databricks" if is_databricks() else "local",
        )

    @property
    def spark(self) -> Any:
        """Get the underlying Spark session."""
        return self._spark

    def query(self, sql: str, to_pandas: bool = True) -> pd.DataFrame | Any:
        """Execute a Spark SQL query.

        Args:
            sql: SQL query string.
            to_pandas: If True, convert result to pandas DataFrame.

        Returns:
            Query result as pandas DataFrame or Spark DataFrame.

        Example:
            ```python
            df = loader.query('''
                SELECT customer_id, SUM(amount) as total
                FROM sales.transactions
                WHERE date >= '2024-01-01'
                GROUP BY customer_id
            ''')
            ```
        """
        logger.info("Executing Spark SQL query", query_preview=sql[:100])
        result = self._spark.sql(sql)

        if to_pandas:
            return result.toPandas()
        return result

    def load_table(
        self,
        table_name: str,
        columns: list[str] | None = None,
        limit: int | None = None,
        to_pandas: bool = True,
    ) -> pd.DataFrame | Any:
        """Load a table from Unity Catalog or Hive metastore.

        Args:
            table_name: Fully qualified table name (catalog.schema.table).
            columns: Optional list of columns to select.
            limit: Optional row limit.
            to_pandas: If True, convert result to pandas DataFrame.

        Returns:
            Table data as pandas DataFrame or Spark DataFrame.

        Example:
            ```python
            # Load full table
            df = loader.load_table("ml_catalog.features.customer_features")

            # Load specific columns with limit
            df = loader.load_table(
                "ml_catalog.features.customer_features",
                columns=["customer_id", "feature_1", "feature_2"],
                limit=10000
            )
            ```
        """
        logger.info("Loading table", table=table_name)

        df = self._spark.table(table_name)

        if columns:
            df = df.select(*columns)

        if limit:
            df = df.limit(limit)

        if to_pandas:
            return df.toPandas()
        return df

    def load_delta(
        self,
        path: str,
        version: int | None = None,
        timestamp: str | None = None,
        columns: list[str] | None = None,
        to_pandas: bool = True,
    ) -> pd.DataFrame | Any:
        """Load a Delta table by path with optional time travel.

        Args:
            path: Path to Delta table (e.g., "dbfs:/mnt/data/table").
            version: Optional version number for time travel.
            timestamp: Optional timestamp for time travel (ISO format).
            columns: Optional list of columns to select.
            to_pandas: If True, convert result to pandas DataFrame.

        Returns:
            Delta table data as pandas DataFrame or Spark DataFrame.

        Example:
            ```python
            # Load current version
            df = loader.load_delta("dbfs:/mnt/data/sales")

            # Load specific version
            df = loader.load_delta("dbfs:/mnt/data/sales", version=42)

            # Load as of timestamp
            df = loader.load_delta(
                "dbfs:/mnt/data/sales",
                timestamp="2024-01-15T00:00:00Z"
            )
            ```
        """
        logger.info("Loading Delta table", path=path, version=version, timestamp=timestamp)

        reader = self._spark.read.format("delta")

        if version is not None:
            reader = reader.option("versionAsOf", version)
        elif timestamp is not None:
            reader = reader.option("timestampAsOf", timestamp)

        df = reader.load(path)

        if columns:
            df = df.select(*columns)

        if to_pandas:
            return df.toPandas()
        return df

    def load_parquet(
        self,
        path: str,
        columns: list[str] | None = None,
        to_pandas: bool = True,
    ) -> pd.DataFrame | Any:
        """Load a Parquet file or directory.

        Args:
            path: Path to Parquet file(s).
            columns: Optional list of columns to select.
            to_pandas: If True, convert result to pandas DataFrame.

        Returns:
            Parquet data as pandas DataFrame or Spark DataFrame.
        """
        logger.info("Loading Parquet", path=path)

        df = self._spark.read.parquet(path)

        if columns:
            df = df.select(*columns)

        if to_pandas:
            return df.toPandas()
        return df

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the catalog.

        Args:
            table_name: Fully qualified table name.

        Returns:
            True if table exists, False otherwise.
        """
        try:
            self._spark.table(table_name).limit(0)
            return True
        except Exception:
            return False

    def get_table_schema(self, table_name: str) -> list[tuple[str, str]]:
        """Get schema of a table as list of (name, type) tuples.

        Args:
            table_name: Fully qualified table name.

        Returns:
            List of (column_name, data_type) tuples.
        """
        df = self._spark.table(table_name)
        return [(f.name, f.dataType.simpleString()) for f in df.schema.fields]
