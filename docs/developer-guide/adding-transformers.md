# Adding Feature Transformers

Guide to adding custom feature transformers.

## Overview

The framework uses scikit-learn's `ColumnTransformer` for feature engineering. To add custom transformers:

1. Create a transformer class following sklearn's API
2. Register it in the feature configuration
3. Add tests

## Sklearn Transformer API

Your transformer must implement:

```python
class CustomTransformer:
    def fit(self, X, y=None):
        """Learn parameters from data."""
        return self

    def transform(self, X):
        """Apply transformation."""
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
```

## Example: Log Transformer

```python
# src/ml_framework/features/custom.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to numeric features."""

    def __init__(self, offset: float = 1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)
```

## Registering Custom Transformers

Edit `src/ml_framework/features/transformer.py`:

```python
from ml_framework.features.custom import LogTransformer

CUSTOM_TRANSFORMERS = {
    "log": LogTransformer,
}

def get_transformer(name: str, **kwargs):
    if name in CUSTOM_TRANSFORMERS:
        return CUSTOM_TRANSFORMERS[name](**kwargs)
    raise ValueError(f"Unknown transformer: {name}")
```

## Using in Config

```yaml
features:
  custom_transformers:
    - name: log
      columns: [revenue, sales]
      params:
        offset: 1.0
```

## Testing

```python
# tests/unit/test_transformers.py
def test_log_transformer():
    from ml_framework.features.custom import LogTransformer

    transformer = LogTransformer(offset=1.0)
    X = np.array([[0], [1], [10]])
    X_transformed = transformer.fit_transform(X)

    expected = np.log(X + 1.0)
    np.testing.assert_array_almost_equal(X_transformed, expected)
```
