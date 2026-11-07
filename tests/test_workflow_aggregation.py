import pytest

from agent.workflow.aggregation import (
    AggregationSpec,
    aggregate_dataset,
)


def test_grouped_count_sum_and_average():
    result = aggregate_dataset(
        [
            {"region": "west", "amount": 10},
            {"region": "east", "amount": 20},
            {"region": "west", "amount": 30},
        ],
        group_by=("region",),
        aggregations=(
            AggregationSpec("orders", "count"),
            AggregationSpec("revenue", "sum", "amount"),
            AggregationSpec("average", "avg", "amount"),
        ),
    )
    assert result.rows == (
        {"region": "west", "orders": 2, "revenue": 40, "average": 20.0},
        {"region": "east", "orders": 1, "revenue": 20, "average": 20.0},
    )


def test_first_seen_group_order_is_deterministic():
    result = aggregate_dataset(
        [{"g": "b", "v": 1}, {"g": "a", "v": 2}, {"g": "b", "v": 3}],
        group_by=("g",),
        aggregations=(AggregationSpec("total", "sum", "v"),),
    )
    assert [row["g"] for row in result.rows] == ["b", "a"]


def test_null_group_key_forms_a_group():
    result = aggregate_dataset(
        [{"g": None, "v": 2}, {"g": "x", "v": 3}],
        group_by=("g",),
        aggregations=(AggregationSpec("n", "count"),),
    )
    assert result.rows[0] == {"g": None, "n": 1}


def test_numeric_aggregations_ignore_null_and_return_none_when_empty():
    result = aggregate_dataset(
        [{"g": "a", "v": None}, {"g": "a", "v": None}],
        group_by=("g",),
        aggregations=(
            AggregationSpec("total", "sum", "v"),
            AggregationSpec("average", "avg", "v"),
            AggregationSpec("minimum", "min", "v"),
            AggregationSpec("maximum", "max", "v"),
        ),
    )
    assert result.rows[0] == {
        "g": "a", "total": None, "average": None,
        "minimum": None, "maximum": None,
    }


def test_count_with_source_counts_non_null_values():
    result = aggregate_dataset(
        [{"g": "a", "v": 1}, {"g": "a", "v": None}],
        group_by=("g",),
        aggregations=(AggregationSpec("row_count", "count"),
                      AggregationSpec("value_count", "count", "v")),
    )
    assert result.rows[0]["row_count"] == 2
    assert result.rows[0]["value_count"] == 1


def test_distinct_count_ignores_null_and_supports_json_like_values():
    result = aggregate_dataset(
        [{"g": "a", "v": [1, 2]}, {"g": "a", "v": [1, 2]},
         {"g": "a", "v": [2, 3]}, {"g": "a", "v": None}],
        group_by=("g",),
        aggregations=(AggregationSpec("unique", "count_distinct", "v"),),
    )
    assert result.rows[0]["unique"] == 2


def test_min_max_for_strings():
    result = aggregate_dataset(
        [{"g": "a", "v": "pear"}, {"g": "a", "v": "apple"}],
        group_by=("g",),
        aggregations=(AggregationSpec("low", "min", "v"),
                      AggregationSpec("high", "max", "v")),
    )
    assert result.rows[0]["low"] == "apple"
    assert result.rows[0]["high"] == "pear"


def test_empty_ungrouped_input_returns_single_aggregate_row():
    result = aggregate_dataset(
        [], group_by=(),
        aggregations=(AggregationSpec("n", "count"),
                      AggregationSpec("total", "sum", "amount")),
    )
    assert result.group_count == 1
    assert result.rows == ({"n": 0, "total": None},)


def test_empty_grouped_input_returns_no_groups():
    result = aggregate_dataset(
        [], group_by=("region",),
        aggregations=(AggregationSpec("n", "count"),),
    )
    assert result.group_count == 0
    assert result.rows == ()


def test_composite_grouping():
    result = aggregate_dataset(
        [{"region": "x", "year": 2025, "v": 2},
         {"region": "x", "year": 2026, "v": 3}],
        group_by=("region", "year"),
        aggregations=(AggregationSpec("total", "sum", "v"),),
    )
    assert result.group_count == 2
    assert result.rows[1]["year"] == 2026


def test_missing_source_column_is_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_dataset([{"g": "a"}], group_by=("g",),
                          aggregations=(AggregationSpec("total", "sum", "amount"),))


def test_boolean_is_not_accepted_as_numeric():
    with pytest.raises(ValueError, match="requires numeric"):
        aggregate_dataset([{"v": True}], group_by=(),
                          aggregations=(AggregationSpec("total", "sum", "v"),))


def test_non_finite_number_is_rejected():
    with pytest.raises(ValueError, match="finite numeric"):
        aggregate_dataset([{"v": float("nan")}], group_by=(),
                          aggregations=(AggregationSpec("total", "avg", "v"),))


def test_invalid_aggregation_name_is_rejected():
    with pytest.raises(ValueError, match="unsupported aggregation"):
        aggregate_dataset([{"v": 1}], group_by=(),
                          aggregations=(AggregationSpec("x", "median", "v"),))


def test_duplicate_output_name_is_rejected():
    with pytest.raises(ValueError, match="output names"):
        aggregate_dataset([{"g": "a", "v": 1}], group_by=("g",),
                          aggregations=(AggregationSpec("g", "sum", "v"),))


def test_result_to_dict_is_serializable_shape():
    result = aggregate_dataset(
        [{"g": "a", "v": 2}], group_by=("g",),
        aggregations=(AggregationSpec("total", "sum", "v"),),
    )
    assert result.to_dict() == {
        "source_row_count": 1,
        "group_count": 1,
        "group_by": ["g"],
        "rows": [{"g": "a", "total": 2}],
    }


def test_unhashable_json_group_key_is_supported():
    result = aggregate_dataset(
        [{"g": ["a", "b"], "v": 1}, {"g": ["a", "b"], "v": 2}],
        group_by=("g",),
        aggregations=(AggregationSpec("total", "sum", "v"),),
    )
    assert result.group_count == 1
    assert result.rows[0]["total"] == 3


def test_mixed_uncomparable_min_values_raise_clear_error():
    with pytest.raises(ValueError, match="mutually comparable"):
        aggregate_dataset(
            [{"v": 1}, {"v": "two"}], group_by=(),
            aggregations=(AggregationSpec("minimum", "min", "v"),),
        )
