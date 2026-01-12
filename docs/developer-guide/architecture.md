# Architecture

Overview of the ML Automation Framework architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                           CLI Layer                              │
│  ┌──────┐ ┌──────────┐ ┌────────┐ ┌──────┐ ┌────────┐           │
│  │train │ │ evaluate │ │validate│ │ tune │ │ deploy │           │
│  └──────┘ └──────────┘ └────────┘ └──────┘ └────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Config Layer                               │
│  ┌─────────────────┐ ┌───────────────┐ ┌──────────────────────┐ │
│  │ PipelineConfig  │ │ ModelConfig   │ │ DataValidationConfig │ │
│  │ DataConfig      │ │ MLflowConfig  │ │ ExplainabilityConfig │ │
│  │ FeatureConfig   │ │ TuningConfig  │ │ DeploymentConfig     │ │
│  └─────────────────┘ └───────────────┘ └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline Layer                              │
│  ┌────────────────┐ ┌──────────────────┐ ┌──────────────────┐   │
│  │ BasePipeline   │ │ Classification   │ │ Forecasting      │   │
│  │                │ │ Pipeline         │ │ Pipeline         │   │
│  └────────────────┘ └──────────────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Components                             │
│  ┌────────────┐ ┌─────────────┐ ┌────────────┐ ┌─────────────┐  │
│  │ Features   │ │ Models      │ │ Evaluation │ │ MLflow      │  │
│  │ Transformer│ │ Factory     │ │ Metrics    │ │ Logger      │  │
│  └────────────┘ └─────────────┘ └────────────┘ └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Config Layer (`ml_framework.config`)

Pydantic models for configuration validation:

- `PipelineConfig`: Root configuration
- `DataConfig`: Data source and split settings
- `FeatureConfig`: Feature engineering settings
- `ModelConfig`: Model type and hyperparameters
- `MLflowConfig`: Experiment tracking settings

### Pipeline Layer (`ml_framework.pipelines`)

Pipeline orchestration:

- `BasePipeline`: Abstract base with common functionality
- `ClassificationPipeline`: Binary/multiclass classification
- `ForecastingPipeline`: Time series forecasting

### Features (`ml_framework.features`)

Feature engineering:

- `FeatureTransformer`: Unified numeric/categorical preprocessing

### Models (`ml_framework.models`)

Model management:

- `ModelFactory`: Creates models from config
- `ModelRegistry`: Dynamic model loading

### Evaluation (`ml_framework.evaluation`)

Metrics and validation:

- Classification/regression metrics
- Walk-forward backtesting

### Logging (`ml_framework.logging`)

MLflow integration:

- `MLflowLogger`: Parameter, metric, model logging

## Design Principles

1. **Config-Driven**: YAML drives all behavior
2. **Runtime Abstraction**: Works locally and on Databricks
3. **Fail Fast**: Validate early with clear errors
4. **Type Safety**: Strict typing throughout
5. **Domain Exceptions**: Granular, actionable errors

## Directory Structure

```
src/ml_framework/
├── cli.py              # CLI entry point
├── config/             # Pydantic config models
│   ├── base.py
│   ├── loader.py
│   ├── validation.py
│   ├── explainability.py
│   └── tuning.py
├── pipelines/          # Pipeline implementations
│   ├── base.py
│   ├── classification.py
│   └── forecasting.py
├── features/           # Feature engineering
│   └── transformer.py
├── models/             # Model factory and registry
│   ├── factory.py
│   └── registry.py
├── evaluation/         # Metrics and backtesting
│   ├── metrics.py
│   └── backtesting.py
├── logging/            # MLflow integration
│   └── mlflow_logger.py
├── validation/         # Data validation
│   └── ge_validator.py
├── explainability/     # SHAP integration
│   └── shap_explainer.py
├── tuning/             # Optuna integration
│   └── optuna_tuner.py
├── deployment/         # Model deployment
│   └── databricks.py
├── exceptions.py       # Domain exceptions
└── utils/              # Utilities
    └── runtime.py
```
