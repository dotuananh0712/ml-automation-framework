"""Integration tests for CLI evaluate command."""

import pytest
import mlflow
from ml_framework.cli import app


class TestEvaluateCommand:
    """Integration tests for the evaluate CLI command."""

    @pytest.fixture
    def trained_model_run_id(
        self, cli_runner, classification_config_path, mock_mlflow_tracking
    ):
        """Train a model and return the run ID for evaluation tests."""
        result = cli_runner.invoke(app, ["train", str(classification_config_path)])

        assert result.exit_code == 0, f"Training failed: {result.output}"

        # Extract run ID from output
        for line in result.output.split("\n"):
            if "MLflow Run ID" in line:
                run_id = line.split(":")[-1].strip()
                return run_id

        pytest.fail("Could not extract run ID from training output")

    def test_evaluate_trained_model(
        self,
        cli_runner,
        trained_model_run_id,
        sample_churn_data_path,
        mock_mlflow_tracking,
    ):
        """Test evaluation of a trained model on new data."""
        result = cli_runner.invoke(
            app, ["evaluate", trained_model_run_id, str(sample_churn_data_path)]
        )

        assert result.exit_code == 0, f"Evaluate failed: {result.output}"
        assert "Evaluation complete" in result.output
        assert "accuracy" in result.output.lower() or "Metric" in result.output

    def test_evaluate_missing_data_file(
        self, cli_runner, trained_model_run_id, mock_mlflow_tracking
    ):
        """Test error handling for missing data file."""
        result = cli_runner.invoke(
            app, ["evaluate", trained_model_run_id, "nonexistent_data.parquet"]
        )

        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_evaluate_invalid_run_id(self, cli_runner, sample_churn_data_path, mock_mlflow_tracking):
        """Test error handling for invalid run ID."""
        result = cli_runner.invoke(
            app, ["evaluate", "invalid_run_id_12345", str(sample_churn_data_path)]
        )

        assert result.exit_code == 1

    def test_evaluate_with_csv_format(
        self,
        cli_runner,
        trained_model_run_id,
        integration_temp_dir,
        sample_churn_data,
        mock_mlflow_tracking,
    ):
        """Test evaluation with CSV data format."""
        csv_path = integration_temp_dir / "test_data.csv"
        sample_churn_data.to_csv(csv_path, index=False)

        result = cli_runner.invoke(
            app, ["evaluate", trained_model_run_id, str(csv_path), "--format", "csv"]
        )

        assert result.exit_code == 0, f"Evaluate with CSV failed: {result.output}"
        assert "Evaluation complete" in result.output

    def test_evaluate_logs_to_mlflow(
        self,
        cli_runner,
        trained_model_run_id,
        sample_churn_data_path,
        mock_mlflow_tracking,
    ):
        """Test that evaluation metrics are logged to MLflow."""
        result = cli_runner.invoke(
            app, ["evaluate", trained_model_run_id, str(sample_churn_data_path)]
        )

        assert result.exit_code == 0
        assert "Metrics logged to run" in result.output

        # Verify a new run was created
        eval_run_lines = [
            line for line in result.output.split("\n") if "Metrics logged" in line
        ]
        assert len(eval_run_lines) > 0
