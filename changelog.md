# Changelog

All notable changes to ML Automation Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-12

### Added

#### CatBoost Model Support
- **CatBoost integration**: Full support for CatBoost classification and regression
  - Added `CATBOOST` to `ModelType` enum
  - Added `CATBOOST` to `ModelFramework` enum in model registry
  - Factory support with automatic parameter mapping (random_state -> random_seed)
  - Early stopping support matching XGBoost/LightGBM pattern
  - MLflow integration with native `mlflow.catboost.log_model()`
- Default hyperparameters: iterations=100, depth=6, learning_rate=0.1
- Example configuration: `configs/classification/catboost_example.yaml`

#### Spark Data Utilities
- **SparkDataLoader**: New utility class for convenient Spark/Delta data access
  - `query(sql)`: Execute Spark SQL queries
  - `load_table(table_name)`: Load Unity Catalog tables
  - `load_delta(path)`: Load Delta tables with optional time travel (version/timestamp)
  - `load_parquet(path)`: Load Parquet files
  - `table_exists(table_name)`: Check if table exists
  - `get_table_schema(table_name)`: Get table schema
- Located in `ml_framework.data` module

### Changed
- Added `catboost>=1.2,<2.0` to core dependencies
- Added `catboost.*` to mypy ignore list

### New Files
- `src/ml_framework/data/__init__.py`
- `src/ml_framework/data/spark_loader.py`
- `tests/unit/test_catboost.py`
- `configs/classification/catboost_example.yaml`

## [0.2.0] - 2026-01-11

### Added

#### CLI Enhancements
- **Evaluate command**: Load pre-trained models from MLflow run ID and evaluate on new data
  - Supports both classification and regression metrics
  - Logs evaluation metrics to new MLflow run linked to original
- **Tune command**: Run Optuna hyperparameter optimization via CLI
  - Configurable trials, timeout, and metric overrides
  - Progress bar and best hyperparameters table output
- **Deploy command**: Deploy models to Databricks Model Serving
  - Supports workload size and scale-to-zero configuration
  - Wait for endpoint readiness option
- **Endpoint-status command**: Check status of deployed endpoints
- **Error handling decorator**: User-friendly CLI error messages with suggestions

#### Data Validation (Great Expectations)
- Config-driven validation via `data_validation:` section in YAML
- Support for common expectations:
  - `expect_column_values_to_not_be_null`
  - `expect_column_values_to_be_unique`
  - `expect_column_values_to_be_in_set`
  - `expect_column_values_to_be_between`
  - `expect_column_values_to_match_regex`
  - `expect_column_values_to_be_of_type`
- Fallback simple validation when Great Expectations not installed
- Integration into pipeline execution (validates before training)

#### Model Explainability (SHAP)
- Config-driven SHAP explanations via `explainability:` section in YAML
- Auto-detection of explainer type (tree, linear, kernel)
- Generated artifacts:
  - SHAP summary plot
  - SHAP bar plot
  - SHAP feature importance JSON
  - Optional dependence plots for top-K features
- Automatic logging to MLflow artifacts

#### Hyperparameter Tuning (Optuna)
- Config-driven tuning via `tuning:` section in YAML
- Supported parameter types: int, float, log_float, categorical
- Sampler options: TPE, random, CMA-ES, grid
- Pruner options: median, hyperband, none
- MLflow integration: logs all trials as nested runs
- Best model registration option

#### Deployment (Databricks Model Serving)
- Deploy models to Databricks Model Serving endpoints
- Configurable workload size (Small, Medium, Large)
- Scale-to-zero support
- Endpoint status monitoring
- Wait for deployment readiness

#### Testing & Documentation
- Integration test infrastructure with MLflow mocking
- mkdocs documentation structure with material theme
- Manual test data generator and example configs

### Changed
- **Breaking**: `evaluate` command now takes `run_id` instead of `config_path`
- Updated `PipelineConfig` to include optional new feature sections
- Updated `pyproject.toml` with new optional dependencies

### New Dependencies (Optional)
- `validation`: great-expectations
- `explainability`: shap, matplotlib
- `tuning`: optuna, optuna-integration
- `docs`: mkdocs, mkdocs-material
- `all-features`: combines validation, explainability, and tuning

## [0.1.0] - 2025-12-01

### Added
- Initial release
- Config-driven ML pipelines (classification, forecasting)
- YAML configuration with Pydantic validation
- MLflow experiment tracking integration
- Model registry with dynamic instantiation
- Feature engineering transformers
- Cross-validation support
- Databricks runtime detection
- CLI with train, validate, init, list-models commands
