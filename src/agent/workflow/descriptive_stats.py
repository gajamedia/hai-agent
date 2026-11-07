"""Descriptive statistics for validated workflow datasets.

Dependency-free, in-memory calculations. No database access, SQL execution,
implicit numeric coercion, or mutation of source rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class NumericColumnStatistics:
    column: str
    count: int
    null_count: int
    mean: float | None
    stddev: float | None
    minimum: int | float | None
    percentile_25: float | None
    median: float | None
    percentile_75: float | None
    maximum: int | float | None
    total: int | float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "count": self.count,
            "null_count": self.null_count,
            "mean": self.mean,
            "stddev": self.stddev,
            "minimum": self.minimum,
            "percentile_25": self.percentile_25,
            "median": self.median,
            "percentile_75": self.percentile_75,
            "maximum": self.maximum,
            "total": self.total,
        }


@dataclass(frozen=True)
class CategoricalColumnStatistics:
    column: str
    count: int
    null_count: int
    distinct_count: int
    top_value: Any
    top_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "count": self.count,
            "null_count": self.null_count,
            "distinct_count": self.distinct_count,
            "top_value": self.top_value,
            "top_count": self.top_count,
        }


@dataclass(frozen=True)
class DescriptiveStatisticsReport:
    row_count: int
    numeric: tuple[NumericColumnStatistics, ...]
    categorical: tuple[CategoricalColumnStatistics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "numeric": [item.to_dict() for item in self.numeric],
            "categorical": [item.to_dict() for item in self.categorical],
        }


def _validate_columns(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> None:
    for label, columns in (
        ("numeric_columns", numeric_columns),
        ("categorical_columns", categorical_columns),
    ):
        if isinstance(columns, (str, bytes)):
            raise TypeError(f"{label} must be a sequence of column names")
        if any(not isinstance(name, str) or not name for name in columns):
            raise ValueError(f"{label} must contain non-empty column names")
        if len(set(columns)) != len(columns):
            raise ValueError(f"{label} contains duplicate column names")

    overlap = set(numeric_columns).intersection(categorical_columns)
    if overlap:
        raise ValueError("a column cannot be both numeric and categorical")


def _quantile(sorted_values: list[int | float], probability: float) -> float | None:
    """Linear interpolation using the conventional (n - 1) position."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return (
        float(sorted_values[lower]) * (1.0 - fraction)
        + float(sorted_values[upper]) * fraction
    )


def _numeric_summary(
    rows: list[Mapping[str, Any]], column: str
) -> NumericColumnStatistics:
    values: list[int | float] = []
    null_count = 0
    for index, row in enumerate(rows):
        if column not in row:
            raise ValueError(f"row at index {index} is missing column {column!r}")
        value = row[column]
        if value is None:
            null_count += 1
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"column {column!r} contains a non-numeric value")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"column {column!r} contains a non-finite numeric value")
        values.append(value)

    count = len(values)
    if not values:
        return NumericColumnStatistics(
            column=column, count=0, null_count=null_count, mean=None, stddev=None,
            minimum=None, percentile_25=None, median=None, percentile_75=None,
            maximum=None, total=None,
        )

    total = sum(values)
    mean = float(total / count)
    if count < 2:
        stddev = None
    else:
        # Sample standard deviation (n - 1 denominator).
        variance = sum((float(value) - mean) ** 2 for value in values) / (count - 1)
        stddev = math.sqrt(variance)

    ordered = sorted(values)
    return NumericColumnStatistics(
        column=column,
        count=count,
        null_count=null_count,
        mean=mean,
        stddev=stddev,
        minimum=min(values),
        percentile_25=_quantile(ordered, 0.25),
        median=_quantile(ordered, 0.50),
        percentile_75=_quantile(ordered, 0.75),
        maximum=max(values),
        total=total,
    )


def _categorical_summary(
    rows: list[Mapping[str, Any]], column: str
) -> CategoricalColumnStatistics:
    counts: dict[Any, tuple[Any, int]] = {}
    null_count = 0
    count = 0
    for index, row in enumerate(rows):
        if column not in row:
            raise ValueError(f"row at index {index} is missing column {column!r}")
        value = row[column]
        if value is None:
            null_count += 1
            continue
        try:
            key = value
            hash(key)
        except TypeError as exc:
            raise ValueError(
                f"categorical column {column!r} contains a non-scalar value"
            ) from exc
        count += 1
        if key in counts:
            original, current_count = counts[key]
            counts[key] = (original, current_count + 1)
        else:
            counts[key] = (value, 1)

    if not counts:
        top_value = None
        top_count = 0
    else:
        # max() preserves first insertion on ties because dict order is stable.
        top_value, top_count = max(counts.values(), key=lambda item: item[1])

    return CategoricalColumnStatistics(
        column=column,
        count=count,
        null_count=null_count,
        distinct_count=len(counts),
        top_value=top_value,
        top_count=top_count,
    )


def describe_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    numeric_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
) -> DescriptiveStatisticsReport:
    """Calculate summaries for explicitly selected columns.

    Numeric summaries ignore nulls and report sample standard deviation.
    Quartiles use linear interpolation at positions based on (n - 1).
    Categorical summaries ignore nulls; top-value ties resolve to first-seen.
    All selected columns must exist in every row.
    """
    _validate_columns(numeric_columns, categorical_columns)
    if not numeric_columns and not categorical_columns:
        raise ValueError("at least one numeric or categorical column is required")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mapping objects")

    materialized = list(rows)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"row at index {index} must be a mapping")

    numeric = tuple(_numeric_summary(materialized, column) for column in numeric_columns)
    categorical = tuple(
        _categorical_summary(materialized, column) for column in categorical_columns
    )
    return DescriptiveStatisticsReport(
        row_count=len(materialized),
        numeric=numeric,
        categorical=categorical,
    )
