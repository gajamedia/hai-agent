import pytest

from agent.workflow.data_quality import (
    ColumnComparisonRule,
    DataQualityRules,
    assess_data_quality,
)


def test_clean_dataset_passes():
    report = assess_data_quality(
        [{"id": 1, "status": "paid", "amount": 100, "total": 100}],
        DataQualityRules(
            required_non_null_columns=("id", "status"),
            unique_key_columns=("id",),
            allowed_values_by_column={"status": ("paid", "pending")},
            column_comparisons=(ColumnComparisonRule("amount", "le", "total"),),
        ),
    )
    assert report.valid is True
    assert report.issue_count == 0


def test_required_non_null_detects_null_and_missing():
    report = assess_data_quality(
        [{"id": None}, {}],
        DataQualityRules(required_non_null_columns=("id",)),
    )
    assert report.valid is False
    assert [issue.code for issue in report.issues].count("null_not_allowed") == 2
    assert any(issue.code == "column_missing" for issue in report.issues)


def test_null_fraction_threshold_is_inclusive():
    report = assess_data_quality(
        [{"email": "a"}, {"email": None}],
        DataQualityRules(max_null_fraction_by_column={"email": 0.5}),
    )
    assert report.valid is True
    assert report.metrics[0].null_fraction == 0.5


def test_null_fraction_above_threshold_fails():
    report = assess_data_quality(
        [{"email": "a"}, {"email": None}, {"email": None}],
        DataQualityRules(max_null_fraction_by_column={"email": 0.5}),
    )
    assert report.valid is False
    assert any(issue.code == "null_fraction_exceeded" for issue in report.issues)


def test_duplicate_composite_key_detected():
    report = assess_data_quality(
        [{"order": 1, "line": 1}, {"order": 1, "line": 2}, {"order": 1, "line": 1}],
        DataQualityRules(unique_key_columns=("order", "line")),
    )
    duplicates = [issue for issue in report.issues if issue.code == "duplicate_key"]
    assert len(duplicates) == 1
    assert duplicates[0].row_index == 2


def test_incomplete_unique_key_is_skipped():
    report = assess_data_quality(
        [{"id": None}, {"id": None}],
        DataQualityRules(unique_key_columns=("id",)),
    )
    assert report.valid is True


def test_allowed_values_detect_invalid_value_without_echoing_it():
    report = assess_data_quality(
        [{"status": "paid"}, {"status": "secret-value"}],
        DataQualityRules(allowed_values_by_column={"status": ("paid", "pending")}),
    )
    assert report.valid is False
    issue = next(issue for issue in report.issues if issue.code == "value_not_allowed")
    assert issue.row_index == 1
    assert "secret-value" not in issue.message


def test_column_comparison_detects_inconsistency():
    report = assess_data_quality(
        [{"subtotal": 100, "grand_total": 90}],
        DataQualityRules(
            column_comparisons=(ColumnComparisonRule("subtotal", "le", "grand_total"),)
        ),
    )
    assert report.valid is False
    assert any(issue.code == "consistency_check_failed" for issue in report.issues)


def test_comparison_skips_null_operands():
    report = assess_data_quality(
        [{"start": None, "end": 10}],
        DataQualityRules(
            column_comparisons=(ColumnComparisonRule("start", "le", "end"),)
        ),
    )
    assert report.valid is True


def test_empty_dataset_has_zero_null_fraction():
    report = assess_data_quality(
        [],
        DataQualityRules(max_null_fraction_by_column={"email": 0.0}),
    )
    assert report.valid is True
    assert report.metrics[0].null_fraction == 0.0


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        assess_data_quality([], DataQualityRules(max_null_fraction_by_column={"x": 1.5}))


def test_non_mapping_row_rejected():
    with pytest.raises(TypeError):
        assess_data_quality([{"id": 1}, "not-a-row"], DataQualityRules())


def test_invalid_input_rows_type_rejected():
    with pytest.raises(TypeError):
        assess_data_quality("not-rows", DataQualityRules())


def test_report_serialization_is_json_compatible():
    report = assess_data_quality(
        [{"id": 1}, {"id": 1}],
        DataQualityRules(unique_key_columns=("id",)),
    )
    data = report.to_dict()
    assert data["valid"] is False
    assert data["issue_count"] == 1
    assert data["issues"][0]["code"] == "duplicate_key"
