"""Integration tests for CLI train command."""

import pytest
from ml_framework.cli import app


class TestTrainCommand:
    """Integration tests for the train CLI command."""

    def test_train_classification_pipeline(
        self, cli_runner, classification_config_path, mock_mlflow_tracking
    ):
        """Test complete classification pipeline via CLI."""
        result = cli_runner.invoke(app, ["train", str(classification_config_path)])

        assert result.exit_code == 0, f"Train failed: {result.output}"
        assert "Training complete!" in result.output
        assert "MLflow Run ID" in result.output
        assert "accuracy" in result.output.lower() or "Metric" in result.output

    def test_train_dry_run(self, cli_runner, classification_config_path):
        """Test --dry-run validates config without execution."""
        result = cli_runner.invoke(
            app, ["train", str(classification_config_path), "--dry-run"]
        )

        assert result.exit_code == 0
        assert "Dry run" in result.output
        assert "Training complete" not in result.output

    def test_train_missing_config_file(self, cli_runner):
        """Test error handling for missing config file."""
        result = cli_runner.invoke(app, ["train", "nonexistent_config.yaml"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_train_invalid_yaml(self, cli_runner, invalid_yaml_path):
        """Test error handling for invalid YAML."""
        result = cli_runner.invoke(app, ["train", str(invalid_yaml_path)])

        assert result.exit_code == 1
        assert "yaml" in result.output.lower() or "error" in result.output.lower()


class TestValidateCommand:
    """Integration tests for the validate CLI command."""

    def test_validate_valid_config(self, cli_runner, classification_config_path):
        """Test validation of a valid config file."""
        result = cli_runner.invoke(app, ["validate", str(classification_config_path)])

        assert result.exit_code == 0
        assert "Valid configuration" in result.output

    def test_validate_missing_file(self, cli_runner):
        """Test validation error for missing file."""
        result = cli_runner.invoke(app, ["validate", "nonexistent.yaml"])

        assert result.exit_code == 1

    def test_validate_invalid_yaml(self, cli_runner, invalid_yaml_path):
        """Test validation error for invalid YAML."""
        result = cli_runner.invoke(app, ["validate", str(invalid_yaml_path)])

        assert result.exit_code == 1


class TestInitCommand:
    """Integration tests for the init CLI command."""

    def test_init_creates_config(self, cli_runner, integration_temp_dir):
        """Test init creates a config file."""
        output_path = integration_temp_dir / "new_pipeline.yaml"
        result = cli_runner.invoke(
            app, ["init", "test_pipeline", "-o", str(output_path)]
        )

        assert result.exit_code == 0
        assert "Config created" in result.output
        assert output_path.exists()

    def test_init_with_pipeline_type(self, cli_runner, integration_temp_dir):
        """Test init with specific pipeline type."""
        output_path = integration_temp_dir / "forecasting_pipeline.yaml"
        result = cli_runner.invoke(
            app, ["init", "forecast_test", "-t", "forecasting", "-o", str(output_path)]
        )

        assert result.exit_code == 0
        assert output_path.exists()


class TestListModelsCommand:
    """Integration tests for the list_models CLI command."""

    def test_list_models_displays_table(self, cli_runner):
        """Test list_models displays available models."""
        result = cli_runner.invoke(app, ["list-models"])

        assert result.exit_code == 0
        assert "Available Models" in result.output or "Model Type" in result.output
        assert "xgboost" in result.output.lower() or "random_forest" in result.output.lower()
