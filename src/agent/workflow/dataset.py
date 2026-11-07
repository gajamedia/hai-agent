from __future__ import annotations

"""Dataset contracts and deterministic structural/type validation.

This module is intentionally dependency-free. It validates already retrieved
rows; it does not open database connections or execute workflow tasks.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence


SUPPORTED_DATA_TYPES = frozenset({
    "any", "string", "integer", "number", "boolean", "date", "datetime",
})


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: str = "any"
    nullable: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ColumnSpec.name must be non-empty text.")
        if self.data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError(
                f"Unsupported data_type for column {self.name!r}: {self.data_type!r}."
            )
        if type(self.nullable) is not bool or type(self.required) is not bool:
            raise TypeError("nullable and required must be bool values.")


@dataclass(frozen=True)
class DatasetSchema:
    columns: tuple[ColumnSpec, ...]
    allow_extra_columns: bool = False
    min_rows: int = 0
    max_rows: int = 10000

    def __post_init__(self) -> None:
        if not isinstance(self.columns, tuple) or not self.columns:
            raise ValueError("DatasetSchema.columns must be a non-empty tuple.")
        if any(not isinstance(item, ColumnSpec) for item in self.columns):
            raise TypeError("Every dataset column must be a ColumnSpec.")
        names = [item.name for item in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("DatasetSchema column names must be unique.")
        if type(self.allow_extra_columns) is not bool:
            raise TypeError("allow_extra_columns must be bool.")
        if (isinstance(self.min_rows, bool) or not isinstance(self.min_rows, int)
                or isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int)):
            raise TypeError("min_rows and max_rows must be integers.")
        if self.min_rows < 0 or self.max_rows < 1 or self.min_rows > self.max_rows:
            raise ValueError("Row limits must satisfy 0 <= min_rows <= max_rows.")


@dataclass(frozen=True)
class DatasetValidationIssue:
    code: str
    message: str
    row_index: int | None = None
    column: str | None = None


@dataclass(frozen=True)
class DatasetValidationReport:
    valid: bool
    row_count: int
    columns: tuple[str, ...]
    issues: tuple[DatasetValidationIssue, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "row_index": issue.row_index,
                    "column": issue.column,
                }
                for issue in self.issues
            ],
        }


def _matches_type(value: Any, data_type: str) -> bool:
    if data_type == "any":
        return True
    if data_type == "string":
        return isinstance(value, str)
    if data_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if data_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            return False
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    if data_type == "boolean":
        return isinstance(value, bool)
    if data_type == "date":
        if isinstance(value, datetime):
            return False
        if isinstance(value, date):
            return True
        if isinstance(value, str):
            try:
                date.fromisoformat(value)
                return True
            except ValueError:
                return False
        return False
    if data_type == "datetime":
        if isinstance(value, datetime):
            return True
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
                return True
            except ValueError:
                return False
        return False
    return False


def validate_dataset(
    rows: Iterable[Mapping[str, Any]],
    schema: DatasetSchema,
) -> DatasetValidationReport:
    """Validate row shape and declared value types without mutating input.

    Error messages intentionally omit cell values to avoid leaking business data
    into workflow diagnostics. row_index is zero-based.
    """
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be a DatasetSchema.")
    if isinstance(rows, (str, bytes, Mapping)) or rows is None:
        raise TypeError("rows must be an iterable of row mappings.")

    materialized = list(rows)
    issues: list[DatasetValidationIssue] = []
    if len(materialized) < schema.min_rows:
        issues.append(DatasetValidationIssue(
            "row_count_below_minimum",
            f"Dataset contains fewer than {schema.min_rows} required rows.",
        ))
    if len(materialized) > schema.max_rows:
        issues.append(DatasetValidationIssue(
            "row_count_above_maximum",
            f"Dataset exceeds the maximum of {schema.max_rows} rows.",
        ))

    expected = {column.name: column for column in schema.columns}
    observed: list[str] = []
    for index, row in enumerate(materialized):
        if not isinstance(row, Mapping):
            issues.append(DatasetValidationIssue(
                "row_not_mapping", "Each dataset row must be a mapping.",
                row_index=index,
            ))
            continue

        for key in row:
            if not isinstance(key, str) or not key:
                issues.append(DatasetValidationIssue(
                    "invalid_column_name",
                    "Dataset column names must be non-empty strings.",
                    row_index=index,
                ))
                continue
            if key not in observed:
                observed.append(key)

        actual = set(row.keys())
        for name, spec in expected.items():
            if name not in actual:
                if spec.required:
                    issues.append(DatasetValidationIssue(
                        "missing_required_column",
                        "A required column is missing from the row.",
                        row_index=index, column=name,
                    ))
                continue
            value = row[name]
            if value is None:
                if not spec.nullable:
                    issues.append(DatasetValidationIssue(
                        "null_not_allowed", "Null value is not allowed.",
                        row_index=index, column=name,
                    ))
                continue
            if not _matches_type(value, spec.data_type):
                issues.append(DatasetValidationIssue(
                    "type_mismatch",
                    f"Value does not match declared type {spec.data_type!r}.",
                    row_index=index, column=name,
                ))

        if not schema.allow_extra_columns:
            for name in actual - set(expected):
                issues.append(DatasetValidationIssue(
                    "unexpected_column", "Dataset contains an undeclared column.",
                    row_index=index,
                    column=name if isinstance(name, str) else None,
                ))

    columns = tuple(observed)
    return DatasetValidationReport(
        valid=not issues,
        row_count=len(materialized),
        columns=columns,
        issues=tuple(issues),
    )


__all__ = [
    "SUPPORTED_DATA_TYPES",
    "ColumnSpec",
    "DatasetSchema",
    "DatasetValidationIssue",
    "DatasetValidationReport",
    "validate_dataset",
]
