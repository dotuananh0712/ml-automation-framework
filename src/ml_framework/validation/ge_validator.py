"""Great Expectations-based data validation.

Provides config-driven data validation using Great Expectations.
"""

from typing import Any

import pandas as pd
import structlog

from ml_framework.config.validation import DataValidationConfig, ExpectationType
from ml_framework.exceptions import DataValidationError

logger = structlog.get_logger(__name__)


class GreatExpectationsValidator:
    """Config-driven data validation using Great Expectations.

    Example:
        ```python
        validator = GreatExpectationsValidator(config)
        is_valid, results = validator.validate(df)
        if not is_valid and config.fail_on_error:
            raise DataValidationError("Validation failed", results)
        ```
    """

    def __init__(self, config: DataValidationConfig):
        """Initialize validator with configuration.

        Args:
            config: Data validation configuration with expectations.
        """
        self.config = config
        self._gx = self._import_great_expectations()

    def _import_great_expectations(self) -> Any:
        """Import Great Expectations, raising helpful error if not installed."""
        try:
            import great_expectations as gx

            return gx
        except ImportError:
            raise ImportError(
                "Great Expectations not installed. "
                "Install with: pip install 'ml-automation-framework[validation]'"
            )

    def validate(self, df: pd.DataFrame) -> tuple[bool, dict[str, Any]]:
        """Validate DataFrame against configured expectations.

        Args:
            df: DataFrame to validate.

        Returns:
            Tuple of (is_valid, results_dict) where results_dict contains
            validation details including any failed expectations.

        Raises:
            DataValidationError: If validation fails and fail_on_error is True.
        """
        if not self.config.enabled:
            logger.info("Data validation disabled, skipping")
            return True, {"status": "skipped", "reason": "disabled"}

        logger.info("Starting data validation", n_rows=len(df), n_expectations=len(self.config.expectations))

        results: dict[str, Any] = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "expectations_checked": 0,
            "expectations_passed": 0,
            "failures": [],
        }

        # Check row count constraints
        if self.config.min_rows is not None and len(df) < self.config.min_rows:
            failure = {
                "check": "min_rows",
                "expected": self.config.min_rows,
                "actual": len(df),
                "message": f"Dataset has {len(df)} rows, minimum required: {self.config.min_rows}",
            }
            results["failures"].append(failure)
            logger.warning("Row count validation failed", **failure)

        if self.config.max_rows is not None and len(df) > self.config.max_rows:
            failure = {
                "check": "max_rows",
                "expected": self.config.max_rows,
                "actual": len(df),
                "message": f"Dataset has {len(df)} rows, maximum allowed: {self.config.max_rows}",
            }
            results["failures"].append(failure)
            logger.warning("Row count validation failed", **failure)

        # Create Great Expectations context and run expectations
        try:
            context = self._gx.get_context()

            # Create a pandas datasource
            datasource = context.sources.add_or_update_pandas(name="runtime_datasource")
            data_asset = datasource.add_dataframe_asset(name="validation_data")

            # Build batch request
            batch_request = data_asset.build_batch_request(dataframe=df)

            # Create expectation suite
            suite_name = "config_driven_validation"
            try:
                context.delete_expectation_suite(suite_name)
            except Exception:
                pass  # Suite doesn't exist yet

            suite = context.add_expectation_suite(expectation_suite_name=suite_name)

            # Add expectations from config
            for exp_config in self.config.expectations:
                results["expectations_checked"] += 1

                # Check if column exists
                if exp_config.column not in df.columns:
                    failure = {
                        "check": "column_exists",
                        "column": exp_config.column,
                        "message": f"Column '{exp_config.column}' not found in dataset",
                    }
                    results["failures"].append(failure)
                    logger.warning("Column not found", column=exp_config.column)
                    continue

                # Build expectation kwargs
                expectation_kwargs = {"column": exp_config.column, **exp_config.kwargs}

                # Add expectation to suite
                suite.add_expectation(
                    expectation_configuration=self._gx.core.ExpectationConfiguration(
                        expectation_type=exp_config.expectation.value,
                        kwargs=expectation_kwargs,
                    )
                )

            # Save suite
            context.save_expectation_suite(suite)

            # Create checkpoint and run validation
            checkpoint_name = "validation_checkpoint"
            checkpoint = context.add_or_update_checkpoint(
                name=checkpoint_name,
                validations=[
                    {
                        "batch_request": batch_request,
                        "expectation_suite_name": suite_name,
                    }
                ],
            )

            checkpoint_result = checkpoint.run()

            # Process results
            validation_result = checkpoint_result.list_validation_results()[0]

            for exp_result in validation_result.results:
                if exp_result.success:
                    results["expectations_passed"] += 1
                else:
                    failure = {
                        "check": exp_result.expectation_config.expectation_type,
                        "column": exp_result.expectation_config.kwargs.get("column"),
                        "message": f"Expectation failed: {exp_result.expectation_config.expectation_type}",
                        "details": exp_result.result,
                    }
                    results["failures"].append(failure)
                    logger.warning(
                        "Expectation failed",
                        expectation=exp_result.expectation_config.expectation_type,
                        column=exp_result.expectation_config.kwargs.get("column"),
                    )

            # Generate data docs if configured
            if self.config.generate_data_docs:
                try:
                    context.build_data_docs()
                    logger.info("Generated data docs")
                except Exception as e:
                    logger.warning("Failed to generate data docs", error=str(e))

        except Exception as e:
            logger.error("Great Expectations validation error", error=str(e))
            results["failures"].append({
                "check": "ge_error",
                "message": f"Great Expectations error: {str(e)}",
            })

        # Determine overall validity
        is_valid = len(results["failures"]) == 0

        logger.info(
            "Data validation complete",
            is_valid=is_valid,
            expectations_passed=results["expectations_passed"],
            expectations_checked=results["expectations_checked"],
            n_failures=len(results["failures"]),
        )

        # Raise error if configured and validation failed
        if not is_valid and self.config.fail_on_error:
            error_messages = [f["message"] for f in results["failures"]]
            raise DataValidationError(
                "Data validation failed",
                {"errors": error_messages, "full_results": results},
            )

        return is_valid, results

    def validate_simple(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Simple validation without Great Expectations (fallback).

        Used when Great Expectations is not needed or as a fallback.

        Args:
            df: DataFrame to validate.

        Returns:
            Tuple of (is_valid, list_of_error_messages).
        """
        errors: list[str] = []

        # Row count checks
        if self.config.min_rows is not None and len(df) < self.config.min_rows:
            errors.append(f"Dataset has {len(df)} rows, minimum required: {self.config.min_rows}")

        if self.config.max_rows is not None and len(df) > self.config.max_rows:
            errors.append(f"Dataset has {len(df)} rows, maximum allowed: {self.config.max_rows}")

        # Column expectations
        for exp in self.config.expectations:
            if exp.column not in df.columns:
                errors.append(f"Column '{exp.column}' not found")
                continue

            col = df[exp.column]

            if exp.expectation == ExpectationType.NOT_NULL:
                null_count = col.isna().sum()
                if null_count > 0:
                    errors.append(f"Column '{exp.column}' has {null_count} null values")

            elif exp.expectation == ExpectationType.UNIQUE:
                duplicate_count = col.duplicated().sum()
                if duplicate_count > 0:
                    errors.append(f"Column '{exp.column}' has {duplicate_count} duplicate values")

            elif exp.expectation == ExpectationType.IN_SET:
                value_set = set(exp.kwargs.get("value_set", []))
                invalid = set(col.dropna().unique()) - value_set
                if invalid:
                    errors.append(f"Column '{exp.column}' has invalid values: {list(invalid)[:5]}")

            elif exp.expectation == ExpectationType.BETWEEN:
                min_val = exp.kwargs.get("min_value")
                max_val = exp.kwargs.get("max_value")
                if min_val is not None:
                    below = (col < min_val).sum()
                    if below > 0:
                        errors.append(f"Column '{exp.column}' has {below} values below {min_val}")
                if max_val is not None:
                    above = (col > max_val).sum()
                    if above > 0:
                        errors.append(f"Column '{exp.column}' has {above} values above {max_val}")

        is_valid = len(errors) == 0
        return is_valid, errors
