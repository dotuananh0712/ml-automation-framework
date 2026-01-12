---
name: databricks-patterns
description: Databricks-specific patterns for Spark, Delta Lake, MLflow on Databricks, Unity Catalog. Use when writing code that runs on Databricks clusters.
---

# Databricks Patterns

## When to Use

- Writing code that runs on Databricks
- Working with Delta Lake tables
- Using Unity Catalog
- Configuring Databricks MLflow
- Spark DataFrame operations

## Core Patterns

### 1. Runtime Detection

```python
from ml_framework.utils.runtime import is_databricks, get_spark_session

if is_databricks():
    spark = get_spark_session()  # Gets active session
    df = spark.table("catalog.schema.table")
else:
    # Local fallback
    df = pd.read_parquet("data/sample.parquet")
```

### 2. Delta Lake Reading

```python
# GOOD: Use Delta format for time travel and ACID
df = spark.read.format("delta").load("/mnt/data/features")

# With time travel
df = (spark.read
      .format("delta")
      .option("versionAsOf", 5)
      .load("/mnt/data/features"))

# From Unity Catalog
df = spark.table("ml_catalog.features.customer_features")
```

### 3. MLflow on Databricks

```python
import mlflow

# On Databricks, tracking URI is auto-configured
# Just set experiment path
mlflow.set_experiment("/Users/user@company.com/experiment")

# Or use workspace path
mlflow.set_experiment("/Shared/ml-experiments/churn")
```

### 4. Efficient Pandas Conversion

```python
# GOOD: Use Arrow for large DataFrames
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
pdf = spark_df.toPandas()

# GOOD: Sample if too large
if spark_df.count() > 1_000_000:
    spark_df = spark_df.sample(fraction=0.1, seed=42)
pdf = spark_df.toPandas()
```

### 5. Feature Store Integration

```python
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Create feature table
fs.create_table(
    name="ml_catalog.features.customer_features",
    primary_keys=["customer_id"],
    df=feature_df,
    description="Customer features for churn prediction"
)

# Read features
training_set = fs.create_training_set(
    df=labels_df,
    feature_lookups=[
        FeatureLookup(
            table_name="ml_catalog.features.customer_features",
            lookup_key="customer_id"
        )
    ],
    label="churn"
)
```

## Anti-Patterns

### 1. Collect on Large Data

```python
# BAD: Crashes driver on large data
all_data = spark_df.collect()

# GOOD: Use toPandas with Arrow, or sample
pdf = spark_df.limit(100000).toPandas()
```

### 2. Ignoring Partitioning

```python
# BAD: Full table scan
df = spark.table("events").filter(col("date") == "2024-01-01")

# GOOD: Push down partition filter
df = spark.table("events").where(col("date") == "2024-01-01")
```

### 3. Local File Paths

```python
# BAD: Local paths don't work on cluster
df = pd.read_csv("/local/path/data.csv")

# GOOD: Use DBFS or cloud storage
df = spark.read.csv("dbfs:/mnt/data/data.csv")
```

## Configuration

### Cluster Libraries

Required libraries for the framework:
```
mlflow
xgboost
lightgbm
structlog
pydantic>=2.0
```

### Init Script (optional)

```bash
#!/bin/bash
pip install ml-automation-framework
```

## Integration

- Related: `ml-pipeline-patterns` for general pipeline patterns
