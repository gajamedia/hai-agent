"""Deterministic in-memory grouping and aggregation for workflow datasets.

This module is dependency-free. It does not access databases, execute SQL,
mutate source rows, or perform implicit type coercion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


_ALLOWED_OPERATIONS = frozenset({
    "count", "count_distinct", "sum", "avg", "min", "max",
})


@dataclass(frozen=True)
class AggregationSpec:
    """One aggregate output.

    `count` may omit source_column to count rows in each group. If supplied,
    it counts non-null values. All other operations require source_column.
    """

    output_column: str
    operation: str
    source_column: str | None = None


@dataclass(frozen=True)
class GroupedAggregationResult:
    source_row_count: int
    group_count: int
    group_by: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_row_count": self.source_row_count,
            "group_count": self.group_count,
            "group_by": list(self.group_by),
            "rows": [dict(row) for row in self.rows],
        }


def _validate_spec(group_by: Sequence[str], aggregations: Sequence[AggregationSpec]) -> None:
    if isinstance(group_by, (str, bytes)):
        raise TypeError("group_by must be a sequence of column names, not a string")
    if any(not isinstance(name, str) or not name for name in group_by):
        raise ValueError("group_by must contain non-empty column names")
    if len(set(group_by)) != len(group_by):
        raise ValueError("group_by contains duplicate column names")
    if not aggregations:
        raise ValueError("at least one aggregation is required")

    outputs: set[str] = set(group_by)
    for spec in aggregations:
        if not isinstance(spec, AggregationSpec):
            raise TypeError("aggregations must contain AggregationSpec instances")
        if not isinstance(spec.output_column, str) or not spec.output_column:
            raise ValueError("aggregation output_column must be a non-empty string")
        if spec.output_column in outputs:
            raise ValueError("aggregation output names must be unique and distinct from group columns")
        outputs.add(spec.output_column)
        if spec.operation not in _ALLOWED_OPERATIONS:
            raise ValueError(f"unsupported aggregation operation: {spec.operation!r}")
        if spec.operation == "count" and spec.source_column is None:
            continue
        if not isinstance(spec.source_column, str) or not spec.source_column:
            raise ValueError(f"{spec.operation!r} requires a non-empty source_column")


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_hashable(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError("grouping and distinct values must be scalar or JSON-like") from exc
    return value


def _numeric_values(values: list[Any], operation: str, column: str) -> list[int | float]:
    result: list[int | float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"operation {operation!r} requires numeric values in column {column!r}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"operation {operation!r} requires finite numeric values in column {column!r}"
            )
        result.append(value)
    return result


def _aggregate(values: list[Any], spec: AggregationSpec, group_size: int) -> Any:
    operation = spec.operation
    if operation == "count":
        return group_size if spec.source_column is None else sum(
            value is not None for value in values
        )
    if operation == "count_distinct":
        return len({_hashable(value) for value in values if value is not None})
    if operation in {"sum", "avg"}:
        numeric = _numeric_values(values, operation, spec.source_column or "")
        if not numeric:
            return None
        total = sum(numeric)
        return total if operation == "sum" else total / len(numeric)
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    try:
        return min(non_null) if operation == "min" else max(non_null)
    except TypeError as exc:
        raise ValueError(
            f"operation {operation!r} requires mutually comparable values "
            f"in column {spec.source_column!r}"
        ) from exc


def aggregate_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    aggregations: Sequence[AggregationSpec],
) -> GroupedAggregationResult:
    """Group rows in first-seen order and calculate requested aggregates.

    Null group-key values form a normal group. Numeric aggregates ignore nulls;
    if a group has no non-null numeric values, sum/avg return None. `count`
    without a source counts rows; `count` with a source counts non-null values.
    `count_distinct` ignores null. An ungrouped aggregation over an empty input
    returns one aggregate row (count=0; other empty aggregates=None).
    """
    _validate_spec(group_by, aggregations)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of mapping objects")

    materialized = list(rows)
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            raise TypeError(f"row at index {index} must be a mapping")

    required_columns = list(group_by)
    for spec in aggregations:
        if spec.source_column is not None and spec.source_column not in required_columns:
            required_columns.append(spec.source_column)
    for index, row in enumerate(materialized):
        missing = [column for column in required_columns if column not in row]
        if missing:
            # Column names are safe schema metadata; cell values are never included.
            raise ValueError(
                f"row at index {index} is missing required columns: {', '.join(missing)}"
            )

    groups: dict[Any, tuple[tuple[Any, ...], list[Mapping[str, Any]]]] = {}
    for row in materialized:
        key_values = tuple(row[column] for column in group_by)
        key = _hashable(key_values)
        if key not in groups:
            groups[key] = (key_values, [])
        groups[key][1].append(row)

    if not group_by and not materialized:
        groups[()] = ((), [])

    output_rows: list[Mapping[str, Any]] = []
    for key_values, group_rows in groups.values():
        result: dict[str, Any] = {
            column: value for column, value in zip(group_by, key_values)
        }
        for spec in aggregations:
            values = (
                [row[spec.source_column] for row in group_rows]
                if spec.source_column is not None
                else []
            )
            result[spec.output_column] = _aggregate(values, spec, len(group_rows))
        output_rows.append(result)

    return GroupedAggregationResult(
        source_row_count=len(materialized),
        group_count=len(output_rows),
        group_by=tuple(group_by),
        rows=tuple(output_rows),
    )
