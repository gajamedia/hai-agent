import math
import pytest

from agent.workflow.descriptive_stats import describe_dataset


def test_numeric_summary_basic_statistics():
    report = describe_dataset(
        [{"amount": 1}, {"amount": 2}, {"amount": 3}, {"amount": 4}],
        numeric_columns=("amount",),
    )
    item = report.numeric[0]
    assert item.count == 4
    assert item.null_count == 0
    assert item.mean == 2.5
    assert item.stddev == pytest.approx(math.sqrt(5 / 3))
    assert item.minimum == 1
    assert item.percentile_25 == 1.75
    assert item.median == 2.5
    assert item.percentile_75 == 3.25
    assert item.maximum == 4
    assert item.total == 10


def test_numeric_summary_ignores_nulls():
    item = describe_dataset(
        [{"v": 2}, {"v": None}, {"v": 4}], numeric_columns=("v",)
    ).numeric[0]
    assert item.count == 2
    assert item.null_count == 1
    assert item.mean == 3.0
    assert item.total == 6


def test_single_numeric_value_has_no_sample_stddev():
    item = describe_dataset([{"v": 7}], numeric_columns=("v",)).numeric[0]
    assert item.mean == 7.0
    assert item.stddev is None
    assert item.median == 7.0


def test_all_null_numeric_column_has_empty_statistics():
    item = describe_dataset(
        [{"v": None}, {"v": None}], numeric_columns=("v",)
    ).numeric[0]
    assert item.count == 0
    assert item.null_count == 2
    assert item.mean is None
    assert item.stddev is None
    assert item.minimum is None
    assert item.total is None


def test_categorical_summary_counts_distinct_and_top_value():
    item = describe_dataset(
        [{"status": "paid"}, {"status": "pending"}, {"status": "paid"},
         {"status": None}],
        categorical_columns=("status",),
    ).categorical[0]
    assert item.count == 3
    assert item.null_count == 1
    assert item.distinct_count == 2
    assert item.top_value == "paid"
    assert item.top_count == 2


def test_categorical_top_tie_uses_first_seen_value():
    item = describe_dataset(
        [{"kind": "b"}, {"kind": "a"}, {"kind": "a"}, {"kind": "b"}],
        categorical_columns=("kind",),
    ).categorical[0]
    assert item.top_value == "b"
    assert item.top_count == 2


def test_all_null_categorical_column():
    item = describe_dataset(
        [{"kind": None}], categorical_columns=("kind",)
    ).categorical[0]
    assert item.count == 0
    assert item.null_count == 1
    assert item.distinct_count == 0
    assert item.top_value is None
    assert item.top_count == 0


def test_multiple_numeric_and_categorical_columns():
    report = describe_dataset(
        [{"qty": 2, "price": 10, "category": "A"},
         {"qty": 3, "price": 20, "category": "B"}],
        numeric_columns=("qty", "price"),
        categorical_columns=("category",),
    )
    assert len(report.numeric) == 2
    assert len(report.categorical) == 1
    assert report.row_count == 2


def test_boolean_is_not_numeric():
    with pytest.raises(ValueError, match="non-numeric"):
        describe_dataset([{"v": True}], numeric_columns=("v",))


def test_non_finite_numeric_is_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        describe_dataset([{"v": float("inf")}], numeric_columns=("v",))


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="missing column"):
        describe_dataset([{"other": 1}], numeric_columns=("v",))


def test_empty_dataset_returns_empty_numeric_and_categorical_metrics():
    report = describe_dataset(
        [], numeric_columns=("amount",), categorical_columns=("status",)
    )
    assert report.row_count == 0
    assert report.numeric[0].count == 0
    assert report.categorical[0].distinct_count == 0


def test_empty_rows_with_no_selected_columns_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        describe_dataset([])


def test_duplicate_column_selection_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        describe_dataset([{"v": 1}], numeric_columns=("v", "v"))


def test_column_cannot_be_both_numeric_and_categorical():
    with pytest.raises(ValueError, match="both numeric and categorical"):
        describe_dataset(
            [{"v": 1}], numeric_columns=("v",), categorical_columns=("v",)
        )


def test_non_mapping_row_is_rejected():
    with pytest.raises(TypeError, match="mapping"):
        describe_dataset([{"v": 1}, 5], numeric_columns=("v",))


def test_non_scalar_categorical_value_is_rejected():
    with pytest.raises(ValueError, match="non-scalar"):
        describe_dataset([{"kind": ["a", "b"]}], categorical_columns=("kind",))


def test_report_to_dict_has_expected_structure():
    report = describe_dataset(
        [{"v": 2, "status": "ok"}],
        numeric_columns=("v",),
        categorical_columns=("status",),
    )
    data = report.to_dict()
    assert data["row_count"] == 1
    assert data["numeric"][0]["mean"] == 2.0
    assert data["categorical"][0]["top_value"] == "ok"


def test_quantiles_for_two_values_are_interpolated():
    item = describe_dataset(
        [{"v": 10}, {"v": 20}], numeric_columns=("v",)
    ).numeric[0]
    assert item.percentile_25 == 12.5
    assert item.median == 15.0
    assert item.percentile_75 == 17.5


def test_null_numeric_count_is_separate_from_non_null_count():
    item = describe_dataset(
        [{"v": 1}, {"v": None}, {"v": 3}, {"v": None}],
        numeric_columns=("v",),
    ).numeric[0]
    assert item.count == 2
    assert item.null_count == 2
