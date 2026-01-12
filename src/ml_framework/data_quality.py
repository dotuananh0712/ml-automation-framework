"""Data quality validation framework.

Provides comprehensive data validation before pipeline execution:
- Missing data detection
- Training period validation
- Negative value handling
- Group-level validation for many-model scenarios
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd
import structlog

from ml_framework.exceptions import DataQualityError, InsufficientDataError

logger = structlog.get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Stops pipeline
    WARNING = "warning"  # Logs warning, continues
    INFO = "info"  # Informational only


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    affected_groups: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @classmethod
    def success(cls, check_name: str, message: str = "Passed") -> "ValidationResult":
        return cls(
            check_name=check_name,
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=message,
        )

    @classmethod
    def warning(
        cls, check_name: str, message: str, affected_groups: list[str] | None = None
    ) -> "ValidationResult":
        return cls(
            check_name=check_name,
            is_valid=True,  # Warnings don't fail validation
            severity=ValidationSeverity.WARNING,
            message=message,
            affected_groups=affected_groups or [],
        )

    @classmethod
    def error(
        cls, check_name: str, message: str, details: dict | None = None
    ) -> "ValidationResult":
        return cls(
            check_name=check_name,
            is_valid=False,
            severity=ValidationSeverity.ERROR,
            message=message,
            details=details or {},
        )


@dataclass
class DataQualityReport:
    """Aggregated report of all validation checks."""

    results: list[ValidationResult] = field(default_factory=list)
    removed_groups: list[str] = field(default_factory=list)
    removal_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return all(r.is_valid for r in self.results)

    @property
    def errors(self) -> list[ValidationResult]:
        """Get all error-level results."""
        return [r for r in self.results if r.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationResult]:
        """Get all warning-level results."""
        return [r for r in self.results if r.severity == ValidationSeverity.WARNING]

    def add(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Data Quality Report", "=" * 40]
        lines.append(f"Total checks: {len(self.results)}")
        lines.append(f"Passed: {sum(1 for r in self.results if r.is_valid)}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")

        if self.removed_groups:
            lines.append(f"\nRemoved groups: {len(self.removed_groups)}")

        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  - {e.check_name}: {e.message}")

        return "\n".join(lines)


class DataQualityChecker:
    """Comprehensive data quality validation.

    Example:
        checker = DataQualityChecker(
            target_col="sales",
            date_col="date",
            group_col="store_id",
        )
        report = checker.run(df)
        if not report.is_valid:
            raise DataQualityError(report.summary())
    """

    def __init__(
        self,
        target_col: str,
        date_col: str | None = None,
        group_col: str | None = None,
        min_rows: int = 10,
        max_missing_ratio: float = 0.2,
        max_negative_ratio: float = 0.2,
    ):
        """Initialize data quality checker.

        Args:
            target_col: Name of target column.
            date_col: Name of date column (for time series).
            group_col: Name of group column (for many-model).
            min_rows: Minimum rows required per group.
            max_missing_ratio: Maximum allowed missing value ratio.
            max_negative_ratio: Maximum allowed negative value ratio.
        """
        self.target_col = target_col
        self.date_col = date_col
        self.group_col = group_col
        self.min_rows = min_rows
        self.max_missing_ratio = max_missing_ratio
        self.max_negative_ratio = max_negative_ratio

    def run(self, df: pd.DataFrame, verbose: bool = True) -> DataQualityReport:
        """Run all validation checks.

        Args:
            df: Input DataFrame to validate.
            verbose: Whether to log results.

        Returns:
            DataQualityReport with all results.
        """
        report = DataQualityReport()

        # Core validations
        report.add(self._check_required_columns(df))
        report.add(self._check_empty_data(df))
        report.add(self._check_missing_values(df))
        report.add(self._check_negative_values(df))

        # Time series validations
        if self.date_col:
            report.add(self._check_date_column(df))

        # Many-model validations
        if self.group_col:
            group_results = self._check_groups(df)
            for result in group_results:
                report.add(result)

        if verbose:
            logger.info("Data quality check complete", is_valid=report.is_valid)
            if report.warnings:
                for w in report.warnings:
                    logger.warning(w.check_name, message=w.message)
            if report.errors:
                for e in report.errors:
                    logger.error(e.check_name, message=e.message)

        return report

    def _check_required_columns(self, df: pd.DataFrame) -> ValidationResult:
        """Check that required columns exist."""
        required = [self.target_col]
        if self.date_col:
            required.append(self.date_col)
        if self.group_col:
            required.append(self.group_col)

        missing = [c for c in required if c not in df.columns]

        if missing:
            return ValidationResult.error(
                "required_columns",
                f"Missing required columns: {missing}",
                {"missing": missing, "available": list(df.columns)},
            )
        return ValidationResult.success("required_columns")

    def _check_empty_data(self, df: pd.DataFrame) -> ValidationResult:
        """Check that DataFrame is not empty."""
        if len(df) == 0:
            return ValidationResult.error("empty_data", "DataFrame is empty")
        return ValidationResult.success("empty_data", f"Found {len(df)} rows")

    def _check_missing_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check missing value ratio in target column."""
        if self.target_col not in df.columns:
            return ValidationResult.success("missing_values", "Skipped - target column not found")

        missing_ratio = df[self.target_col].isna().mean()

        if missing_ratio > self.max_missing_ratio:
            return ValidationResult.error(
                "missing_values",
                f"Target has {missing_ratio:.1%} missing values (max: {self.max_missing_ratio:.1%})",
                {"missing_ratio": missing_ratio},
            )
        elif missing_ratio > 0:
            return ValidationResult.warning(
                "missing_values",
                f"Target has {missing_ratio:.1%} missing values",
            )
        return ValidationResult.success("missing_values")

    def _check_negative_values(self, df: pd.DataFrame) -> ValidationResult:
        """Check negative value ratio in target column."""
        if self.target_col not in df.columns:
            return ValidationResult.success("negative_values", "Skipped")

        if not pd.api.types.is_numeric_dtype(df[self.target_col]):
            return ValidationResult.success("negative_values", "Skipped - non-numeric target")

        negative_ratio = (df[self.target_col] < 0).mean()

        if negative_ratio > self.max_negative_ratio:
            return ValidationResult.warning(
                "negative_values",
                f"Target has {negative_ratio:.1%} negative values",
            )
        return ValidationResult.success("negative_values")

    def _check_date_column(self, df: pd.DataFrame) -> ValidationResult:
        """Validate date column format and ordering."""
        if self.date_col not in df.columns:
            return ValidationResult.success("date_column", "Skipped")

        try:
            dates = pd.to_datetime(df[self.date_col])

            # Check for duplicates within groups
            if self.group_col:
                duplicates = df.groupby(self.group_col)[self.date_col].apply(
                    lambda x: x.duplicated().any()
                )
                if duplicates.any():
                    affected = duplicates[duplicates].index.tolist()
                    return ValidationResult.warning(
                        "date_column",
                        f"Duplicate dates found in {len(affected)} groups",
                        affected_groups=affected[:10],  # Limit to first 10
                    )

            return ValidationResult.success("date_column")

        except Exception as e:
            return ValidationResult.error(
                "date_column",
                f"Invalid date format: {e}",
            )

    def _check_groups(self, df: pd.DataFrame) -> list[ValidationResult]:
        """Validate each group for many-model scenarios."""
        results = []

        if self.group_col not in df.columns:
            return results

        group_sizes = df.groupby(self.group_col).size()
        total_groups = len(group_sizes)

        # Check minimum rows per group
        small_groups = group_sizes[group_sizes < self.min_rows]
        if len(small_groups) > 0:
            ratio = len(small_groups) / total_groups
            results.append(
                ValidationResult.warning(
                    "group_size",
                    f"{len(small_groups)}/{total_groups} groups have < {self.min_rows} rows",
                    affected_groups=small_groups.index.tolist()[:20],
                )
            )
        else:
            results.append(
                ValidationResult.success(
                    "group_size",
                    f"All {total_groups} groups have >= {self.min_rows} rows",
                )
            )

        # Check for groups with all missing target
        if self.target_col in df.columns:
            all_missing = df.groupby(self.group_col)[self.target_col].apply(
                lambda x: x.isna().all()
            )
            if all_missing.any():
                affected = all_missing[all_missing].index.tolist()
                results.append(
                    ValidationResult.error(
                        "group_missing_target",
                        f"{len(affected)} groups have all missing target values",
                        {"affected_groups": affected},
                    )
                )

        return results


def validate_training_period(
    backtest_length: int,
    prediction_length: int,
    stride: int = 1,
) -> ValidationResult:
    """Validate that training period configuration is valid.

    Args:
        backtest_length: Total backtesting window.
        prediction_length: Forecast horizon.
        stride: Step size between backtest trials.

    Returns:
        ValidationResult indicating if configuration is valid.
    """
    if prediction_length > backtest_length:
        return ValidationResult.error(
            "training_period",
            f"prediction_length ({prediction_length}) > backtest_length ({backtest_length})",
        )

    num_trials = (backtest_length - prediction_length) // stride + 1
    if num_trials < 1:
        return ValidationResult.error(
            "training_period",
            f"Configuration yields {num_trials} backtest trials (need at least 1)",
        )

    return ValidationResult.success(
        "training_period",
        f"Configuration yields {num_trials} backtest trials",
    )
