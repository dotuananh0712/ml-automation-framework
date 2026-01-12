"""Command-line interface for ML Automation Framework."""

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import ast
import structlog
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from ml_framework.config.base import PipelineType
from ml_framework.config.loader import load_config
from ml_framework.exceptions import (
    CLIError,
    ColumnNotFoundError,
    DataValidationError,
    InvalidYAMLError,
    MLFrameworkError,
)

app = typer.Typer(
    name="mlf",
    help="ML Automation Framework - Config-driven ML pipelines",
    add_completion=False,
)
console = Console()

# Configure structlog for CLI
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)


def handle_cli_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to catch and format errors for CLI output.

    Provides user-friendly error messages instead of raw stack traces.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            console.print(f"[bold red]File not found:[/] {e}")
            console.print("[dim]Suggestion: Check path and ensure file exists[/]")
            raise typer.Exit(1)
        except yaml.YAMLError as e:
            console.print(f"[bold red]Invalid YAML:[/] {e}")
            console.print("[dim]Suggestion: Validate YAML syntax[/]")
            raise typer.Exit(1)
        except ValidationError as e:
            console.print("[bold red]Configuration validation failed:[/]")
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                console.print(f"  [yellow]{loc}:[/] {error['msg']}")
            raise typer.Exit(1)
        except InvalidYAMLError as e:
            console.print(f"[bold red]Invalid YAML:[/] {e.message}")
            if "parse_error" in e.details:
                console.print(f"[dim]Parse error: {e.details['parse_error']}[/]")
            raise typer.Exit(1)
        except ColumnNotFoundError as e:
            console.print(f"[bold red]Column error:[/] {e.message}")
            if "suggestion" in e.details:
                console.print(f"[dim]{e.details['suggestion']}[/]")
            raise typer.Exit(1)
        except DataValidationError as e:
            console.print(f"[bold red]Data validation failed:[/] {e.message}")
            if "errors" in e.details:
                for err in e.details["errors"]:
                    console.print(f"  [yellow]-[/] {err}")
            raise typer.Exit(1)
        except CLIError as e:
            console.print(f"[bold red]CLI Error:[/] {e.message}")
            if "suggestions" in e.details:
                for suggestion in e.details["suggestions"]:
                    console.print(f"[dim]  - {suggestion}[/]")
            raise typer.Exit(1)
        except MLFrameworkError as e:
            console.print(f"[bold red]Error:[/] {e.message}")
            if e.details:
                console.print(f"[dim]Details: {e.details}[/]")
            raise typer.Exit(1)

    return wrapper


@app.command()
@handle_cli_errors
def train(
    config_path: Path = typer.Argument(..., help="Path to pipeline YAML config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without running"),
) -> None:
    """Train a model using the specified configuration."""
    console.print(f"[bold blue]Loading config:[/] {config_path}")

    config = load_config(config_path)

    console.print(f"[green]Config validated:[/] {config.name}")
    console.print(f"  Pipeline type: {config.pipeline_type.value}")
    console.print(f"  Model: {config.model.model_type.value}")
    console.print(f"  Data source: {config.data.source}")

    if dry_run:
        console.print("[yellow]Dry run - skipping execution[/]")
        return

    # Run pipeline
    from ml_framework.pipelines.classification import ClassificationPipeline
    from ml_framework.pipelines.forecasting import ForecastingPipeline

    if config.pipeline_type == PipelineType.CLASSIFICATION:
        pipeline = ClassificationPipeline(config)
    elif config.pipeline_type == PipelineType.FORECASTING:
        pipeline = ForecastingPipeline(config)
    else:
        console.print(f"[red]Unsupported pipeline type:[/] {config.pipeline_type}")
        raise typer.Exit(1)

    result = pipeline.run()

    # Display results
    console.print("\n[bold green]Training complete![/]")
    console.print(f"MLflow Run ID: {result['run_id']}")

    table = Table(title="Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for name, value in sorted(result["metrics"].items()):
        table.add_row(name, f"{value:.4f}")

    console.print(table)


@app.command()
@handle_cli_errors
def evaluate(
    run_id: str = typer.Argument(..., help="MLflow run ID of trained model"),
    data_path: Path = typer.Argument(..., help="Path to evaluation data"),
    data_format: str = typer.Option("parquet", "--format", "-f", help="Data format"),
) -> None:
    """Evaluate a trained model on new data.

    Loads a model from MLflow and evaluates it on new data, displaying
    metrics and optionally logging them to a new MLflow run.

    Example:
        mlf evaluate abc123def456 data/test.parquet
    """
    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )

    console.print(f"[bold blue]Loading model from MLflow run:[/] {run_id}")

    # Load the model from MLflow
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        console.print(f"[bold red]Failed to load model:[/] {e}")
        console.print(f"[dim]Model URI: {model_uri}[/]")
        raise typer.Exit(1)

        console.print(f"[green]Model loaded successfully[/]")

    # Load evaluation data    console.print(f"[bold blue]Loading evaluation data:[/] {data_path}")

    if not data_path.exists():
        console.print(f"[bold red]Data file not found:[/] {data_path}")
        raise typer.Exit(1)

    if data_format.lower() == "parquet":
        df = pd.read_parquet(data_path)
    elif data_format.lower() == "csv":
        df = pd.read_csv(data_path)
    else:
        console.print(f"[bold red]Unsupported data format:[/] {data_format}")
        raise typer.Exit(1)

    console.print(f"[green]Data loaded:[/] {len(df)} rows, {len(df.columns)} columns")

    # Get run info to determine pipeline type and target column
    run = mlflow.get_run(run_id)
    run_params = run.data.params

    target_column = run_params.get("data.target_column", "target")
    feature_columns = ast.literal_eval(run_params["data.feature_columns"]) ##feature columns is a list

    pipeline_type = run_params.get("pipeline_type", "classification")

    if target_column not in df.columns:
        console.print(f"[bold red]Target column not found:[/] {target_column}")
        console.print(f"[dim]Available columns: {list(df.columns)[:10]}[/]")
        raise typer.Exit(1)

    # Split features and target
    y_true = df[target_column]
    X = df[feature_columns]

    # Generate predictions
    console.print("[bold blue]Generating predictions...[/]")
    console.print(target_column)

    y_pred = model.predict(X)

    # Calculate metrics based on pipeline type
    metrics: dict[str, float] = {}

    if pipeline_type.lower() == "classification":
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # ROC-AUC for binary classification
        unique_classes = np.unique(y_true)

        if len(unique_classes) == 2:
            try:
                # Try to get probabilities for ROC-AUC
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            except Exception:
                pass  # Skip ROC-AUC if not available
    else:
        # Regression/Forecasting metrics
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

        # MAPE (handle zero values)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics["mape"] = float(mape)

    # Display results
    console.print("\n[bold green]Evaluation complete![/]")

    table = Table(title=f"Evaluation Metrics (Run: {run_id[:8]}...)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for name, value in sorted(metrics.items()):
        table.add_row(name, f"{value:.4f}")

    console.print(table)

    # Log to new MLflow run
    console.print("\n[bold blue]Logging metrics to MLflow...[/]")
    with mlflow.start_run(run_name=f"eval_{run_id[:8]}"):
        mlflow.log_param("source_run_id", run_id)
        mlflow.log_param("eval_data_path", str(data_path))
        mlflow.log_param("eval_data_rows", len(df))
        mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
        eval_run_id = mlflow.active_run().info.run_id

    console.print(f"[green]Metrics logged to run:[/] {eval_run_id}")


@app.command()
@handle_cli_errors
def validate(
    config_path: Path = typer.Argument(..., help="Path to pipeline YAML config"),
) -> None:
    """Validate a pipeline configuration file."""
    console.print(f"[bold blue]Validating:[/] {config_path}")

    config = load_config(config_path)
    console.print("[bold green]Valid configuration![/]")
    console.print(f"  Name: {config.name}")
    console.print(f"  Type: {config.pipeline_type.value}")
    console.print(f"  Model: {config.model.model_type.value}")
    console.print(f"  Experiment: {config.mlflow.experiment_name}")


@app.command()
@handle_cli_errors
def tune(
    config_path: Path = typer.Argument(..., help="Path to pipeline YAML config"),
    n_trials: Optional[int] = typer.Option(
        None, "--trials", "-n", help="Override number of trials"
    ),
    timeout: Optional[int] = typer.Option(
        None, "--timeout", "-t", help="Timeout in seconds"
    ),
    metric: Optional[str] = typer.Option(
        None, "--metric", "-m", help="Override optimization metric"
    ),
) -> None:
    """Run hyperparameter tuning using Optuna.

    Optimizes model hyperparameters based on the tuning configuration
    in the YAML file. Logs all trials to MLflow if configured.

    Example:
        mlf tune configs/classification/churn.yaml --trials 50
    """
    import pandas as pd

    from ml_framework.tuning.optuna_tuner import OptunaTuner

    console.print(f"[bold blue]Loading config:[/] {config_path}")

    config = load_config(config_path)

    # Check that tuning is configured
    if config.tuning is None:
        console.print("[bold red]Error:[/] No tuning configuration found in config")
        console.print("[dim]Add a 'tuning:' section to your config file[/]")
        raise typer.Exit(1)

    # Override tuning settings if provided
    if n_trials:
        config.tuning.n_trials = n_trials
    if timeout:
        config.tuning.timeout = timeout
    if metric:
        config.tuning.metric = metric

    # Ensure tuning is enabled
    config.tuning.enabled = True

    console.print(f"[green]Config validated:[/] {config.name}")
    console.print(f"  Pipeline type: {config.pipeline_type.value}")
    console.print(f"  Model: {config.model.model_type.value}")
    console.print(f"  Trials: {config.tuning.n_trials}")
    console.print(f"  Metric: {config.tuning.metric}")
    console.print(f"  Direction: {config.tuning.direction}")

    # Load data
    console.print(f"\n[bold blue]Loading data:[/] {config.data.source}")

    data_path = Path(config.data.source)
    if not data_path.exists():
        console.print(f"[bold red]Data file not found:[/] {data_path}")
        raise typer.Exit(1)

    if config.data.format.value == "parquet":
        data = pd.read_parquet(data_path)
    elif config.data.format.value == "csv":
        data = pd.read_csv(data_path)
    else:
        console.print(f"[bold red]Unsupported format:[/] {config.data.format.value}")
        raise typer.Exit(1)

    console.print(f"[green]Data loaded:[/] {len(data)} rows, {len(data.columns)} columns")

    # Split data
    from sklearn.model_selection import train_test_split

    train_ratio = config.data.train_ratio
    val_ratio = config.data.validation_ratio
    target_col = config.data.target_column

    stratify = data[target_col] if config.data.stratify else None

    train_data, temp_data = train_test_split(
        data,
        train_size=train_ratio,
        random_state=config.model.random_state,
        stratify=stratify,
    )

    # Use remaining data as validation
    val_data = temp_data

    console.print(f"  Train: {len(train_data)} rows")
    console.print(f"  Validation: {len(val_data)} rows")

    # Run tuning
    console.print("\n[bold blue]Starting hyperparameter tuning...[/]\n")

    tuner = OptunaTuner(config)
    result = tuner.run(train_data, val_data)

    # Display results
    console.print("\n[bold green]Tuning complete![/]")

    # Best hyperparameters table
    table = Table(title="Best Hyperparameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for param, value in sorted(result.best_params.items()):
        if isinstance(value, float):
            table.add_row(param, f"{value:.6g}")
        else:
            table.add_row(param, str(value))

    console.print(table)

    # Summary
    console.print(f"\n[bold]Best {config.tuning.metric}:[/] {result.best_value:.4f}")
    console.print(f"[bold]Best trial:[/] #{result.best_trial_number}")
    console.print(f"[bold]Total trials:[/] {result.n_trials}")

    # Print search space summary
    console.print("\n[dim]Search space:[/]")
    for param in config.tuning.search_space:
        if param.type.value in ("int", "float", "log_float"):
            console.print(f"  {param.name}: [{param.low}, {param.high}] ({param.type.value})")
        else:
            console.print(f"  {param.name}: {param.choices} (categorical)")


@app.command()
@handle_cli_errors
def init(
    name: str = typer.Argument(..., help="Pipeline name"),
    pipeline_type: str = typer.Option(
        "classification", "--type", "-t", help="Pipeline type"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
) -> None:
    """Generate a starter configuration file."""
    import yaml

    template = {
        "name": name,
        "description": f"{name} pipeline",
        "pipeline_type": pipeline_type,
        "data": {
            "source": "path/to/data.parquet",
            "format": "parquet",
            "target_column": "target",
            "train_ratio": 0.8,
            "validation_ratio": 0.1,
        },
        "features": {
            "numeric_impute_strategy": "median",
            "categorical_impute_strategy": "most_frequent",
            "numeric_scaling": "standard",
            "categorical_encoding": "onehot",
        },
        "model": {
            "model_type": "xgboost",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            "cross_validation": True,
            "cv_folds": 5,
        },
        "mlflow": {
            "experiment_name": f"/Experiments/{name}",
            "log_model": True,
            "log_feature_importance": True,
        },
    }

    output_path = output or Path(f"configs/{pipeline_type}/{name}.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    console.print(f"[bold green]Config created:[/] {output_path}")


@app.command()
@handle_cli_errors
def list_models() -> None:
    """List available model types."""
    from ml_framework.config.base import ModelType

    table = Table(title="Available Models")
    table.add_column("Model Type", style="cyan")
    table.add_column("Tasks", style="green")

    models = {
        ModelType.LOGISTIC_REGRESSION: "Classification",
        ModelType.LINEAR_REGRESSION: "Regression, Forecasting",
        ModelType.RANDOM_FOREST: "Classification, Regression, Forecasting",
        ModelType.XGBOOST: "Classification, Regression, Forecasting",
        ModelType.LIGHTGBM: "Classification, Regression, Forecasting",
    }

    for model, tasks in models.items():
        table.add_row(model.value, tasks)

    console.print(table)


@app.command()
@handle_cli_errors
def deploy(
    run_id: str = typer.Argument(..., help="MLflow run ID of trained model"),
    endpoint_name: str = typer.Option(..., "--endpoint", "-e", help="Endpoint name"),
    workload_size: str = typer.Option(
        "Small", "--size", "-s", help="Workload size: Small, Medium, Large"
    ),
    scale_to_zero: bool = typer.Option(
        True, "--scale-to-zero/--no-scale-to-zero", help="Scale to zero when idle"
    ),
    wait: bool = typer.Option(False, "--wait", "-w", help="Wait for endpoint to be ready"),
    timeout: int = typer.Option(600, "--timeout", help="Timeout for wait in seconds"),
) -> None:
    """Deploy a trained model to Databricks Model Serving.

    Requires DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.

    Example:
        mlf deploy abc123def456 --endpoint my-model-endpoint --size Small
    """
    from ml_framework.config.deployment import (
        DatabricksServingConfig,
        DeploymentConfig,
        DeploymentTarget,
        WorkloadSize,
    )
    from ml_framework.deployment.databricks import DatabricksDeployer

    console.print(f"[bold blue]Deploying model[/]")
    console.print(f"  Run ID: {run_id}")
    console.print(f"  Target: {DeploymentTarget.DATABRICKS_MODEL_SERVING.value}")
    console.print(f"  Endpoint: {endpoint_name}")

    # Validate workload size
    try:
        size = WorkloadSize(workload_size)
    except ValueError:
        console.print(f"[bold red]Invalid workload size:[/] {workload_size}")
        console.print("[dim]Valid options: Small, Medium, Large[/]")
        raise typer.Exit(1)

    # Create deployment config
    databricks_config = DatabricksServingConfig(
        endpoint_name=endpoint_name,
        workload_size=size,
        scale_to_zero=scale_to_zero,
    )

    config = DeploymentConfig(
        enabled=True,
        target=DeploymentTarget.DATABRICKS_MODEL_SERVING,
        databricks=databricks_config,
    )

    # Deploy
    deployer = DatabricksDeployer(config)
    result = deployer.deploy(run_id)

    if result.success:
        console.print("\n[bold green]Deployment initiated![/]")
        console.print(f"  Endpoint: {result.endpoint_name}")
        if result.endpoint_url:
            console.print(f"  URL: {result.endpoint_url}")
        console.print(f"  Status: {result.status}")

        # Display metadata
        if result.metadata:
            table = Table(title="Deployment Details")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            for key, value in result.metadata.items():
                table.add_row(key, str(value))

            console.print(table)

        # Wait for ready if requested
        if wait:
            console.print(f"\n[bold blue]Waiting for endpoint to be ready (timeout: {timeout}s)...[/]")
            is_ready = deployer.wait_for_ready(
                endpoint_name, timeout_seconds=timeout, poll_interval=30
            )

            if is_ready:
                console.print("[bold green]Endpoint is ready![/]")
            else:
                console.print("[bold yellow]Endpoint not ready within timeout[/]")
                console.print("[dim]Check status with: databricks serving-endpoints get <name>[/]")

    else:
        console.print(f"\n[bold red]Deployment failed:[/] {result.message}")
        raise typer.Exit(1)


@app.command()
@handle_cli_errors
def endpoint_status(
    endpoint_name: str = typer.Argument(..., help="Name of the endpoint"),
) -> None:
    """Check status of a Databricks Model Serving endpoint."""
    from ml_framework.config.deployment import (
        DatabricksServingConfig,
        DeploymentConfig,
        DeploymentTarget,
    )
    from ml_framework.deployment.databricks import DatabricksDeployer

    # Create minimal config for deployer
    config = DeploymentConfig(
        enabled=True,
        target=DeploymentTarget.DATABRICKS_MODEL_SERVING,
        databricks=DatabricksServingConfig(endpoint_name=endpoint_name),
    )

    deployer = DatabricksDeployer(config)
    status = deployer.get_endpoint_status(endpoint_name)

    status_colors = {
        "READY": "green",
        "PENDING": "yellow",
        "UPDATING": "yellow",
        "FAILED": "red",
        "NOT_FOUND": "red",
        "UNKNOWN": "dim",
    }

    color = status_colors.get(status, "white")
    console.print(f"Endpoint [bold]{endpoint_name}[/]: [{color}]{status}[/{color}]")


if __name__ == "__main__":
    app()
