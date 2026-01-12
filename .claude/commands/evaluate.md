---
description: Evaluate a trained model on test data
allowed-tools: Bash, Read, Glob
---

# Evaluate Model

Evaluate a trained model using test data.

## Usage

```
/evaluate <config_path> [--data <data_path>]
```

## Process

1. **Load Configuration**
   - Parse the YAML config file
   - Locate the trained model in MLflow

2. **Run Evaluation**
   - Execute: `python -m ml_framework.cli evaluate $ARGUMENTS`
   - Calculate metrics on test data

3. **Report Results**
   - Display evaluation metrics
   - Compare with training metrics
   - Highlight any performance degradation

## Arguments

- `$1`: Path to YAML configuration file (required)
- `--data`: Optional override for test data path

## Example

```
/evaluate configs/classification/churn.yaml
/evaluate configs/classification/churn.yaml --data data/test.parquet
```
