---
description: Train an ML model using a YAML config file
allowed-tools: Bash, Read, Write, Glob, Grep
---

# Train Model

Train an ML model using the specified configuration file.

## Usage

```
/train <config_path>
```

## Process

1. **Validate Configuration**
   - Load and validate the YAML config file
   - Check that data source exists
   - Verify model type is supported

2. **Run Training Pipeline**
   - Execute: `python -m ml_framework.cli train $ARGUMENTS`
   - Monitor for errors

3. **Report Results**
   - Display training metrics
   - Show MLflow run ID
   - Link to MLflow UI if available

## Arguments

- `$1`: Path to YAML configuration file (required)

## Example

```
/train configs/classification/churn.yaml
```
