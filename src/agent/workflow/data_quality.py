"""Dependency-free data quality checks for workflow datasets.

This module does not access databases, execute SQL, or mutate input rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence


_ALLOWED_COMPARISONS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


@dataclass(frozen=True)
class ColumnComparisonRule:
    """Compare two values in each row; rows with either value None are skipped."""

    left_column: str
    operator: str
    right_column: str


@dataclass(frozen=True)
class DataQualityRules:
    """Declarative rules applied to an in-memory sequence of mapping rows."""

    required_non_null_columns: tuple[str, ...] = ()
    max_null_fraction_by_column: Mapping[str, float] = field(default_factory=dict)
    unique_key_columns: tuple[str, ...] = ()
    allowed_values_by_column: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)
    column_comparisons: tuple[ColumnComparisonRule, ...] = ()


@dataclass(frozen=True)
class DataQualityIssue:
    code: str
    severity: str
    message: str
    row_index: int | None = None
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ColumnQualityMetric:
    column: str
    null_count: int
    null_fraction: float


@dataclass(frozen=True)
class DataQualityReport:
    valid: bool
    row_count: int
    checked_columns: tuple[str, ...]
    metrics: tuple[ColumnQualityMetric, ...]
    issues: tuple[DataQualityIssue, ...]

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "row_count": self.row_count,
            "checked_columns": list(self.checked_columns),
            "metrics": [
                {
                    "column": item.column,
                    "null_count": item.null_count,
                    "null_fraction": item.null_fraction,
                }
                for item in self.metrics
            ],
            "issues": [
                {
                    "code": item.code,
                    "severity": item.severity,
                    "message": item.message,
                    "row_index": item.row_index,
                    "columns": list(item.columns),
                }
                for item in self.issues
            ],
            "issue_count": self.issue_count,
        }


def _validate_rules(rules: DataQualityRules) -> None:
    if not isinstance(rules, DataQualityRules):
        raise TypeError("rules must be a DataQualityRules instance")

    for name in rules.required_non_null_columns:
        if not isinstance(name, str) or not name:
            raise ValueError("required_non_null_columns must contain non-empty names")

    for name, threshold in rules.max_null_fraction_by_column.items():
        if not isinstance(name, str) or not name:
            raise ValueError("null-fraction rule column names must be non-empty strings")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"null-fraction threshold for {name!r} must be numeric")
        if not math.isfinite(float(threshold)) or not 0 <= threshold <= 1:
            raise ValueError(f"null-fraction threshold for {name!r} must be between 0 and 1")

    if not rules.unique_key_columns:
        pass
    elif any(not isinstance(name, str) or not name for name in rules.unique_key_columns):
        raise ValueError("unique_key_columns must contain non-empty names")

    for name, values in rules.allowed_values_by_column.items():
        if not isinstance(name, str) or not name:
            raise ValueError("allowed-value rule column names must be non-empty strings")
        if not isinstance(values, tuple):
            raise ValueError(f"allowed values for {name!r} must be a tuple")

    for rule in rules.column_comparisons:
        if not isinstance(rule, ColumnComparisonRule):
            raise TypeError("column_comparisons must contain ColumnComparisonRule instances")
        if not rule.left_column or not rule.right_column:
            raise ValueError("comparison rule column names must be non-empty")
        if rule.operator not in _ALLOWED_COMPARISONS:
            raise ValueError(f"unsupported comparison operator: {rule.operator!r}")


def _hashable(value: Any) -> Any:
    """Convert common JSON-like values to deterministic hashable structures."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_hashable(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("unique-key values must be scalar or JSON-like") from exc
    return value


def _compare(left: Any, operator: str, right: Any) -> bool:
    if operator == "eq":
        return left == right
    if operator == "ne":
        return left != right
    if operator == "lt":
        return left < right
    if operator == "le":
        return left <= right
    if operator == "gt":
        return left > right
    if operator == "ge":
        return left >= right
    raise ValueError(f"unsupported comparison operator: {operator!r}")


def assess_data_quality(
    rows: Sequence[Mapping[str, Any]],
    rules: DataQualityRules,
) -> DataQualityReport:
    """Evaluate null, uniqueness, allowed-value, and cross-column consistency rules.

    Issue messages intentionally do not include actual cell values.
    `row_index` is zero-based and refers to the supplied sequence.
    """
    _validate_rules(rules)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mapping objects")

    materialized = list(rows)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"row at index {index} must be a mapping")

    all_rule_columns: list[str] = []
    candidates = (
        list(rules.required_non_null_columns)
        + list(rules.max_null_fraction_by_column.keys())
        + list(rules.unique_key_columns)
        + list(rules.allowed_values_by_column.keys())
        + [c for rule in rules.column_comparisons for c in (rule.left_column, rule.right_column)]
    )
    for column in candidates:
        if column not in all_rule_columns:
            all_rule_columns.append(column)

    issues: list[DataQualityIssue] = []
    metrics: list[ColumnQualityMetric] = []
    row_count = len(materialized)

    for column in all_rule_columns:
        missing_column = any(column not in row for row in materialized)
        if missing_column:
            issues.append(DataQualityIssue(
                code="column_missing",
                severity="error",
                message=f"Required quality-check column {column!r} is missing from one or more rows.",
                columns=(column,),
            ))
        null_count = sum(1 for row in materialized if column not in row or row[column] is None)
        null_fraction = (null_count / row_count) if row_count else 0.0
        metrics.append(ColumnQualityMetric(column, null_count, null_fraction))

    for column in rules.required_non_null_columns:
        for index, row in enumerate(materialized):
            if column not in row or row[column] is None:
                issues.append(DataQualityIssue(
                    code="null_not_allowed",
                    severity="error",
                    message=f"Column {column!r} must not be null.",
                    row_index=index,
                    columns=(column,),
                ))

    for column, threshold in rules.max_null_fraction_by_column.items():
        null_count = sum(1 for row in materialized if column not in row or row[column] is None)
        fraction = (null_count / row_count) if row_count else 0.0
        if fraction > float(threshold):
            issues.append(DataQualityIssue(
                code="null_fraction_exceeded",
                severity="error",
                message=f"Column {column!r} exceeds its allowed null fraction.",
                columns=(column,),
            ))

    if rules.unique_key_columns:
        seen: dict[Any, int] = {}
        for index, row in enumerate(materialized):
            values = tuple(row.get(column) for column in rules.unique_key_columns)
            # Incomplete keys are handled by required_non_null_columns when configured;
            # they are not considered duplicate keys here.
            if any(value is None or column not in row
                   for column, value in zip(rules.unique_key_columns, values)):
                continue
            key = tuple(_hashable(value) for value in values)
            if key in seen:
                issues.append(DataQualityIssue(
                    code="duplicate_key",
                    severity="error",
                    message="Duplicate key detected.",
                    row_index=index,
                    columns=rules.unique_key_columns,
                ))
            else:
                seen[key] = index

    for column, allowed_values in rules.allowed_values_by_column.items():
        for index, row in enumerate(materialized):
            if column not in row or row[column] is None:
                continue
            if not any(row[column] == allowed for allowed in allowed_values):
                issues.append(DataQualityIssue(
                    code="value_not_allowed",
                    severity="error",
                    message=f"Column {column!r} contains a value outside its allowed set.",
                    row_index=index,
                    columns=(column,),
                ))

    for rule in rules.column_comparisons:
        for index, row in enumerate(materialized):
            if (rule.left_column not in row or rule.right_column not in row
                    or row[rule.left_column] is None or row[rule.right_column] is None):
                continue
            try:
                consistent = _compare(
                    row[rule.left_column], rule.operator, row[rule.right_column]
                )
            except (TypeError, ValueError):
                consistent = False
            if not consistent:
                issues.append(DataQualityIssue(
                    code="consistency_check_failed",
                    severity="error",
                    message=(
                        f"Consistency rule {rule.left_column!r} "
                        f"{rule.operator} {rule.right_column!r} failed."
                    ),
                    row_index=index,
                    columns=(rule.left_column, rule.right_column),
                ))

    checked_columns = tuple(all_rule_columns)
    return DataQualityReport(
        valid=not any(issue.severity == "error" for issue in issues),
        row_count=row_count,
        checked_columns=checked_columns,
        metrics=tuple(metrics),
        issues=tuple(issues),
    )
