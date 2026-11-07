"""Listing rows ("tampilkan data"), not only aggregates: real components, independent SQL as the oracle."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from zipfile import ZipFile

import pytest

import tests.test_table_report_job_e2e as base
from agent.llm.business_database_adapter import ReadOnlyTableSpec
from agent.workflow.analytics import _build_detail
from agent.workflow.table_report_job import TableReportRequest
from tests.test_table_report_job_e2e import COLUMNS, _context, _make_db, _sample_rows, env  # noqa: F401

LIST_SPEC = ReadOnlyTableSpec(
    table="sales", columns=tuple(COLUMNS), filterable_columns=("region", "product"),
    sortable_columns=tuple(COLUMNS), max_limit=101)


@pytest.fixture
def lenv(monkeypatch, env):  # noqa: F811
    """The real-component environment, but with a table that may be sorted and lists up to 100 rows."""
    monkeypatch.setattr(base, "SALES_SPEC", LIST_SPEC)
    return env


def _list(**overrides):
    values = dict(table="sales", columns=COLUMNS, group_by=(), metrics=(), kind="list",
                  select_columns=("id", "region", "amount"), row_limit=101, limit=100,
                  title="Sales rows")
    values.update(overrides)
    return TableReportRequest(**values)


def _run(e, request, key="k1", context=None):
    return e.runner.run(goal="List sales", idempotency_key=key, request=request,
                        execution_context=context or _context())


def _sql(db_path, query, *params):
    with sqlite3.connect(db_path) as db:
        return [tuple(r) for r in db.execute(query, params).fetchall()]


def _report_text(e, outcome, encoding="utf-8"):
    return (e.artifacts_root / outcome.artifact["relative_path"]).read_bytes().decode(encoding)


def _tuples(outcome, columns):
    return [tuple(row[c] for c in columns) for row in outcome.rows]


# ------------------------------------------------------------------ the rows themselves
def test_rows_come_back_in_the_requested_order_and_match_independent_sql(lenv):
    e = lenv(_sample_rows(40))
    out = _run(e, _list(order_by=(("amount", "desc"),), limit=7))
    assert out.succeeded, (out.failed_task_key, out.error_code, out.message)
    assert out.kind == "list" and out.row_count == 7 and out.limit_reached is True   # 40 rows exist
    expected = _sql(e.db_path, "SELECT id, region, amount FROM sales ORDER BY amount DESC LIMIT 7")
    assert _tuples(out, ("id", "region", "amount")) == expected


def test_a_listing_is_not_summarized_each_row_is_kept(lenv):
    e = lenv(_sample_rows(40))
    out = _run(e, _list(limit=100))
    assert out.row_count == 40 and len(out.rows) == 40 and out.limit_reached is False
    assert {r["id"] for r in out.rows} == set(range(1, 41))          # every row, none merged


def test_the_limit_is_exact_and_more_rows_are_detected_only_when_they_exist(lenv):
    e = lenv(_sample_rows(10))
    exactly = _run(e, _list(limit=10), key="a")
    assert exactly.row_count == 10 and exactly.limit_reached is False   # 10 rows, limit 10: nothing is hidden
    fewer = _run(e, _list(limit=9), key="b")
    assert fewer.row_count == 9 and fewer.limit_reached is True         # one row is hidden


def test_two_sort_keys_and_directions_match_sql(lenv):
    e = lenv(_sample_rows(40))
    out = _run(e, _list(select_columns=("id", "region", "product"), order_by=(("region", "asc"), ("id", "desc")),
                        limit=25))
    expected = _sql(e.db_path, "SELECT id, region, product FROM sales ORDER BY region ASC, id DESC LIMIT 25")
    assert _tuples(out, ("id", "region", "product")) == expected


def test_filters_apply_before_the_limit(lenv):
    e = lenv(_sample_rows(40))
    out = _run(e, _list(filters={"product": "widget"}, order_by=(("id", "asc"),), limit=5))
    expected = _sql(e.db_path, "SELECT id, region, amount FROM sales WHERE product='widget' ORDER BY id ASC LIMIT 5")
    assert _tuples(out, ("id", "region", "amount")) == expected and out.limit_reached is True
    assert "[filter: product=widget]" in _report_text(e, out)


def test_the_columns_that_were_not_selected_never_reach_the_result(lenv):
    e = lenv(_sample_rows(10))
    out = _run(e, _list(select_columns=("region",), limit=3))
    assert out.succeeded and all(set(row) == {"region"} for row in out.rows)
    assert "amount" not in _report_text(e, out) and "product" not in _report_text(e, out)


def test_sorting_by_a_column_that_is_not_selected_is_allowed(lenv):
    e = lenv(_sample_rows(40))
    out = _run(e, _list(select_columns=("region",), order_by=(("amount", "desc"),), limit=4))
    expected = _sql(e.db_path, "SELECT region FROM sales ORDER BY amount DESC LIMIT 4")
    assert _tuples(out, ("region",)) == expected


def test_nulls_are_shown_as_they_are_instead_of_failing_validation(lenv):
    e = lenv(_sample_rows(10))
    with sqlite3.connect(e.db_path) as db:
        db.execute("UPDATE sales SET amount = NULL WHERE id IN (2, 5)")
    out = _run(e, _list(order_by=(("id", "asc"),)))
    assert out.succeeded, (out.error_code, out.message)
    assert [r["amount"] for r in out.rows if r["id"] in (2, 5)] == [None, None]


def test_no_matching_rows_is_reported_clearly_and_writes_no_file(lenv):
    e = lenv(_sample_rows(10))
    out = _run(e, _list(filters={"region": "atlantis"}))
    assert not out.succeeded and out.error_code == "no_matching_rows" and out.artifact is None


def test_the_results_screen_is_capped_but_the_count_and_the_file_hold_every_row(monkeypatch, env):  # noqa: F811
    from agent.workflow.table_report_job import MAX_RESULT_ROWS
    wide = ReadOnlyTableSpec(table="sales", columns=tuple(COLUMNS), filterable_columns=("region",),
                             sortable_columns=tuple(COLUMNS), max_limit=401)
    monkeypatch.setattr(base, "SALES_SPEC", wide)
    e = env(_sample_rows(300))
    out = _run(e, _list(report_format="csv", row_limit=401, limit=250, order_by=(("id", "asc"),)))
    assert out.succeeded, (out.error_code, out.message)
    assert MAX_RESULT_ROWS == 200
    assert len(out.rows) == 200                 # what the page shows
    assert out.row_count == 250                 # what the report holds
    assert out.limit_reached is True            # 300 rows exist
    rows = list(csv.DictReader(_report_text(e, out, "utf-8-sig").splitlines()))
    assert len(rows) == 250 and rows[-1]["id"] == "250"


# --------------------------------------------------------------- files for a listing
@pytest.mark.parametrize("fmt", ["csv", "html", "json", "xlsx", "pdf"])
def test_every_report_format_works_for_a_listing(lenv, fmt):
    e = lenv(_sample_rows(40))
    out = _run(e, _list(report_format=fmt, order_by=(("id", "asc"),), limit=6))
    assert out.succeeded, (out.failed_task_key, out.error_code, out.message)
    assert out.artifact["filename"] == f"report.{fmt}"
    path = e.artifacts_root / out.artifact["relative_path"]
    if fmt == "csv":
        rows = list(csv.DictReader(path.read_text("utf-8-sig").splitlines()))
        assert [r["id"] for r in rows] == ["1", "2", "3", "4", "5", "6"]
    if fmt == "json":
        data = json.loads(path.read_text("utf-8"))
        assert data["aggregation"]["kind"] == "detail" and data["aggregation"]["truncated"] is True
        assert data["aggregation"]["row_count"] == 6 and len(data["aggregation"]["rows"]) == 6
    if fmt == "xlsx":
        with ZipFile(path) as z:
            sheet = z.read("xl/worksheets/sheet1.xml").decode()
        assert sheet.count("<row ") == 7                              # header + 6 rows
    if fmt == "pdf":
        assert path.read_bytes().startswith(b"%PDF")


def test_a_cut_off_listing_says_so_in_the_file_itself(lenv):
    e = lenv(_sample_rows(40))
    cut = _run(e, _list(report_format="html", limit=5), key="cut")
    whole = _run(e, _list(report_format="html", limit=100), key="whole")
    assert "(first 5 rows)" in _report_text(e, cut)
    assert "first" not in _report_text(e, whole).split("<table")[0]


# ------------------------------------------------------------------------- safety
def test_a_filter_value_that_looks_like_sql_is_only_data(lenv):
    e = lenv(_sample_rows(10))
    out = _run(e, _list(filters={"region": "west' OR '1'='1"}))
    assert not out.succeeded and out.error_code == "no_matching_rows"
    assert _sql(e.db_path, "SELECT COUNT(*) FROM sales") == [(10,)]


def test_sorting_on_a_column_the_table_does_not_allow_is_refused_by_the_adapter(monkeypatch, env):  # noqa: F811
    unsortable = ReadOnlyTableSpec(table="sales", columns=tuple(COLUMNS), filterable_columns=("region",),
                                   max_limit=101)                       # no sortable_columns
    monkeypatch.setattr(base, "SALES_SPEC", unsortable)
    e = env(_sample_rows(10))
    out = _run(e, _list(order_by=(("amount", "desc"),)))
    assert not out.succeeded and out.failed_task_key == "read" and out.artifact is None


def test_a_principal_without_the_read_permission_gets_no_rows(lenv):
    e = lenv(_sample_rows(10))
    out = _run(e, _list(), context=_context(permissions=()))
    assert not out.succeeded and out.failed_task_key == "read" and out.rows == ()
    assert out.error_code == "capability_authorization_denied"


def test_the_row_cap_of_the_table_is_never_exceeded(lenv):
    e = lenv(_sample_rows(10))
    with pytest.raises(ValueError):
        _list(limit=101)                      # the table lists at most row_limit - 1 = 100
    with pytest.raises(ValueError):
        _list(limit=0)


# ------------------------------------------------------------------- request rules
@pytest.mark.parametrize("change", [
    {"kind": "pivot"}, {"select_columns": ()}, {"select_columns": ("id", "id")},
    {"select_columns": tuple(f"c{i}" for i in range(31)), "columns": {f"c{i}": "string" for i in range(31)}},
    {"limit": True}, {"limit": "5"}, {"limit": 2.5},
    {"order_by": (("id", "sideways"),)}, {"order_by": (("id", "asc"),) * 4},
    {"select_columns": ("not_declared",)}, {"order_by": (("not_declared", "asc"),)},
])
def test_invalid_list_requests_are_rejected(change):
    with pytest.raises(ValueError):
        _list(**change)


def test_aggregate_requests_still_need_group_by_and_metrics():
    with pytest.raises(ValueError):
        TableReportRequest(table="sales", columns=COLUMNS, group_by=(), metrics=())


# ------------------------------------------------------------ the analysis step itself
def test_detail_configuration_is_validated():
    assert _build_detail(None) is None
    assert _build_detail({"columns": ["a", "b"], "limit": 5}) == (("a", "b"), 5)
    for bad in ({"columns": [], "limit": 5}, {"columns": ["a", "a"], "limit": 5}, {"columns": ["a"], "limit": 0},
                {"columns": ["a"], "limit": True}, {"columns": ["a"], "limit": 100001},
                {"columns": ["a"], "limit": 5, "extra": 1}, ["a"]):
        with pytest.raises((ValueError, TypeError)):
            _build_detail(bad)


# ------------------------------------------------------------------------ column order
def test_a_listing_keeps_the_column_order_that_was_asked_for_on_screen_and_in_files(lenv):
    e = lenv(_sample_rows(10))
    out = _run(e, _list(select_columns=("amount", "id", "region"), report_format="csv",
                        order_by=(("id", "asc"),), limit=3))
    assert out.succeeded
    assert list(out.rows[0]) == ["amount", "id", "region"]
    text = _report_text(e, out, "utf-8-sig")
    assert text.splitlines()[0] == "amount,id,region"

    other = _run(e, _list(select_columns=("region", "id", "amount"), report_format="csv", limit=3), key="k2")
    assert list(other.rows[0]) == ["region", "id", "amount"]
    assert _report_text(e, other, "utf-8-sig").splitlines()[0] == "region,id,amount"


def test_a_summary_puts_the_group_columns_first_not_in_alphabetical_order(env):  # noqa: F811
    """Stored results sort their keys, which used to put 'orders' before 'region'."""
    e = env(_sample_rows(30))
    request = TableReportRequest(
        table="sales", columns=COLUMNS, group_by=("region",),
        metrics=(("orders", "count", "id"), ("revenue", "sum", "amount")), report_format="csv",
        required_non_null_columns=("id", "amount"), unique_key_columns=("id",), numeric_columns=("amount",))
    out = _run(e, request)
    assert out.succeeded, (out.error_code, out.message)
    assert list(out.rows[0]) == ["region", "orders", "revenue"]
    assert _report_text(e, out, "utf-8-sig").splitlines()[0] == "region,orders,revenue"


def test_the_order_helpers_are_safe_with_missing_or_extra_keys():
    from agent.workflow.analytics import analytical_column_order, order_columns
    assert analytical_column_order({}) == () and analytical_column_order({"detail": {"columns": ["b", "a"]}}) == ("b", "a")
    assert analytical_column_order({"aggregation": {"group_by": ["g"], "aggregations": [
        {"output_column": "z", "operation": "sum", "source_column": "x"},
        {"output_column": "a", "operation": "count", "source_column": "x"}]}}) == ("g", "z", "a")
    assert order_columns([{"a": 1, "b": 2, "extra": 3}], ["b", "a", "missing"]) == [{"b": 2, "a": 1, "extra": 3}]
    assert list(order_columns([{"a": 1, "b": 2}], ["b", "a"])[0]) == ["b", "a"]
    assert order_columns([{"a": 1}], []) == [{"a": 1}]


def test_the_pdf_header_follows_the_requested_order(lenv):
    pypdf = pytest.importorskip("pypdf")
    e = lenv(_sample_rows(10))
    out = _run(e, _list(select_columns=("region", "id"), report_format="pdf", limit=3,
                        order_by=(("id", "asc"),)))
    text = pypdf.PdfReader(io.BytesIO((e.artifacts_root / out.artifact["relative_path"]).read_bytes())
                           ).pages[0].extract_text()
    header = next(line for line in text.splitlines() if "region" in line and "id" in line)
    assert header.index("region") < header.index("id")
