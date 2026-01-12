"""Data validation configuration models.

Defines Pydantic models for Great Expectations-based data validation.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ExpectationType(str, Enum):
    """Supported Great Expectations expectation types."""

    NOT_NULL = "expect_column_values_to_not_be_null"
    UNIQUE = "expect_column_values_to_be_unique"
    IN_SET = "expect_column_values_to_be_in_set"
    BETWEEN = "expect_column_values_to_be_between"
    REGEX = "expect_column_values_to_match_regex"
    TYPE = "expect_column_values_to_be_of_type"
    NOT_NULL_SET = "expect_column_values_to_not_be_in_set"
    LENGTH_BETWEEN = "expect_column_value_lengths_to_be_between"


class ColumnExpectation(BaseModel):
    """Configuration for a single column expectation.

    Example:
        ```yaml
        - column: age
          expectation: expect_column_values_to_be_between
          kwargs:
            min_value: 0
            max_value: 120
        ```
    """

    column: str = Field(..., description="Column name to validate")
    expectation: ExpectationType = Field(..., description="Type of expectation to apply")
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments for the expectation (e.g., min_value, max_value)",
    )


class DataValidationConfig(BaseModel):
    """Configuration for data validation using Great Expectations.

    Example:
        ```yaml
        data_validation:
          enabled: true
          fail_on_error: true
          generate_data_docs: true
          min_rows: 100
          expectations:
            - column: age
              expectation: expect_column_values_to_not_be_null
            - column: age
              expectation: expect_column_values_to_be_between
              kwargs:
                min_value: 0
                max_value: 120
        ```
    """

    enabled: bool = Field(default=False, description="Enable data validation")
    fail_on_error: bool = Field(
        default=True,
        description="Fail pipeline if validation fails. If false, log warning and continue.",
    )
    generate_data_docs: bool = Field(
        default=True,
        description="Generate Great Expectations data docs",
    )
    min_rows: int | None = Field(
        default=None,
        ge=1,
        description="Minimum number of rows required in dataset",
    )
    max_rows: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of rows allowed in dataset",
    )
    expectations: list[ColumnExpectation] = Field(
        default_factory=list,
        description="List of column-level expectations to validate",
    )
