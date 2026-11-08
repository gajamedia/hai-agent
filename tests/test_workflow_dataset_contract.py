from datetime import date, datetime
from decimal import Decimal

import pytest

from agent.workflow.dataset import (
    ColumnSpec,
    DatasetSchema,
    validate_dataset,
)


def schema():
    return DatasetSchema(columns=(
        ColumnSpec("id", "integer", nullable=False),
        ColumnSpec("name", "string", nullable=False),
        ColumnSpec("amount", "number", nullable=True),
        ColumnSpec("active", "boolean", nullable=False),
    ))


def test_valid_dataset_accepts_declared_types_and_decimal():
    report = validate_dataset([
        {"id": 1, "name": "Bread", "amount": Decimal("12.50"), "active": True},
        {"id": 2, "name": "Cake", "amount": 20.0, "active": False},
    ], schema())
    assert report.valid is True
    assert report.row_count == 2
    assert report.columns == ("id", "name", "amount", "active")
    assert report.issues == ()


def test_integer_rejects_boolean_and_number_rejects_non_finite():
    report = validate_dataset([
        {"id": True, "name": "A", "amount": float("nan"), "active": True},
    ], schema())
    assert report.valid is False
    assert {item.code for item in report.issues} == {"type_mismatch"}


def test_missing_required_and_null_not_allowed_are_reported():
    report = validate_dataset([
        {"id": None, "amount": 1, "active": True},
    ], schema())
    assert {item.code for item in report.issues} == {
        "null_not_allowed", "missing_required_column"
    }


def test_unexpected_column_rejected_by_default():
    report = validate_dataset([
        {"id": 1, "name": "A", "amount": None, "active": True, "secret": "x"},
    ], schema())
    assert any(item.code == "unexpected_column" for item in report.issues)
    assert "secret" in report.columns


def test_optional_missing_column_is_allowed():
    contract = DatasetSchema(columns=(
        ColumnSpec("id", "integer", nullable=False),
        ColumnSpec("note", "string", required=False),
    ))
    assert validate_dataset([{"id": 1}], contract).valid


def test_extra_columns_can_be_allowed_explicitly():
    contract = DatasetSchema(
        columns=(ColumnSpec("id", "integer", nullable=False),),
        allow_extra_columns=True,
    )
    assert validate_dataset([{"id": 1, "source": "test"}], contract).valid


def test_row_bounds_are_enforced():
    contract = DatasetSchema(
        columns=(ColumnSpec("id", "integer"),), min_rows=2, max_rows=3,
    )
    report = validate_dataset([{"id": 1}], contract)
    assert not report.valid
    assert report.issues[0].code == "row_count_below_minimum"


def test_non_mapping_row_is_reported_without_crashing():
    report = validate_dataset([{"id": 1}, "invalid"], DatasetSchema(
        columns=(ColumnSpec("id", "integer"),)
    ))
    assert any(item.code == "row_not_mapping" for item in report.issues)


def test_schema_rejects_duplicate_column_names_and_invalid_bounds():
    with pytest.raises(ValueError):
        DatasetSchema(columns=(ColumnSpec("id"), ColumnSpec("id")))
    with pytest.raises(ValueError):
        DatasetSchema(columns=(ColumnSpec("id"),), min_rows=4, max_rows=2)


def test_date_and_datetime_validation():
    contract = DatasetSchema(columns=(
        ColumnSpec("day", "date", nullable=False),
        ColumnSpec("created_at", "datetime", nullable=False),
    ))
    report = validate_dataset([{
        "day": date(2026, 11, 7),
        "created_at": datetime.fromisoformat("2026-11-07T10:30:00+07:00"),
    }], contract)
    assert report.valid


def test_validation_report_is_json_friendly_and_does_not_include_cell_values():
    report = validate_dataset(
        [{"id": "sensitive-value"}],
        DatasetSchema(columns=(ColumnSpec("id", "integer"),)),
    )
    payload = report.to_dict()
    assert payload["valid"] is False
    assert "sensitive-value" not in repr(payload)
