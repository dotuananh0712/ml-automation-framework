---
name: feature-engineering
description: Feature engineering patterns for numeric, categorical, temporal features. Use when building feature pipelines, handling missing values, encoding, scaling.
---

# Feature Engineering Patterns

## When to Use

- Building feature transformation pipelines
- Handling missing values
- Encoding categorical features
- Scaling numeric features
- Creating time-based features
- Feature selection

## Core Patterns

### 1. FeatureTransformer Usage

```python
from ml_framework.features.transformer import FeatureTransformer
from ml_framework.config.base import FeatureConfig

config = FeatureConfig(
    numeric_impute_strategy="median",
    numeric_scaling="standard",
    categorical_encoding="onehot",
)

transformer = FeatureTransformer(config)

# Fit on training data only
X_train = transformer.fit_transform(train_df)

# Transform validation and test
X_val = transformer.transform(val_df)
X_test = transformer.transform(test_df)
```

### 2. Numeric Feature Handling

```python
# Imputation strategies
config = FeatureConfig(
    numeric_impute_strategy="median",  # or "mean", "most_frequent"
)

# Scaling options
config = FeatureConfig(
    numeric_scaling="standard",  # z-score normalization
    # numeric_scaling="minmax",  # 0-1 scaling
    # numeric_scaling="robust",  # handles outliers
    # numeric_scaling=None,      # no scaling (tree models)
)
```

### 3. Categorical Encoding

```python
# One-hot encoding (default, good for linear models)
config = FeatureConfig(
    categorical_encoding="onehot",
    handle_unknown="ignore",  # handle new categories at inference
)

# For high cardinality, consider target encoding
# (requires custom transformer)
```

### 4. Time Series Features

```python
# Lag features
def create_lag_features(df: pd.DataFrame, target: str, lags: list[int]):
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    return df

# Rolling statistics
def create_rolling_features(df: pd.DataFrame, target: str, windows: list[int]):
    for window in windows:
        df[f"{target}_rolling_mean_{window}"] = (
            df[target].shift(1).rolling(window).mean()
        )
        df[f"{target}_rolling_std_{window}"] = (
            df[target].shift(1).rolling(window).std()
        )
    return df
```

### 5. Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X_train, y_train)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = [f for f, m in zip(feature_names, selected_mask) if m]
```

## Anti-Patterns

### 1. Fitting on Test Data

```python
# BAD: Data leakage!
transformer.fit_transform(test_df)

# GOOD: Only transform test data
transformer.transform(test_df)
```

### 2. Target Leakage

```python
# BAD: Using future information
df["next_day_sales"] = df["sales"].shift(-1)
df["is_churn"] = df["churned_date"] < df["prediction_date"]  # wrong

# GOOD: Only use past information
df["prev_day_sales"] = df["sales"].shift(1)
```

### 3. Ignoring Missing Values

```python
# BAD: NaN values crash models
model.fit(X_train, y_train)  # May fail or produce NaN predictions

# GOOD: Handle missing values explicitly
X_train = transformer.fit_transform(train_df)  # Imputes NaN
```

### 4. Scaling for Tree Models

```python
# UNNECESSARY: Trees don't need scaling
config = FeatureConfig(
    numeric_scaling="standard",  # Not needed for XGBoost/LightGBM
)

# GOOD: Skip scaling for tree models
config = FeatureConfig(
    numeric_scaling=None,
)
```

## Common Feature Types

| Feature Type | Imputation | Encoding | Scaling |
|-------------|------------|----------|---------|
| Numeric continuous | median | N/A | standard |
| Numeric discrete | most_frequent | N/A | None |
| Categorical low-card | most_frequent | onehot | N/A |
| Categorical high-card | constant | target/frequency | N/A |
| Date/Time | N/A | extract components | None |
| Text | N/A | TF-IDF/embeddings | N/A |

## Integration

- Related: `ml-pipeline-patterns` for pipeline structure
