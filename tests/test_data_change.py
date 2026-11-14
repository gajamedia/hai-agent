"""Data changes: validated plans, parameterized SQL, preview, all-or-nothing execution, audit.

Runs on SQLite (the builder is engine independent); the MySQL engine is covered by the live suite.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from agent.llm.contracts import LLMResponse
from agent.llm.service import LLMService
from agent.workflow.data_change import (
    ChangeRefused, ChangeSqlBuilder, DataChangePlanner, DataChangeService, WritableTable,
)
from agent.workflow.table_report_planner import TableCatalogEntry

COLUMNS = {"id": "integer", "nama": "string", "harga_beli": "number", "tanggal": "date"}
ENTRY = TableCatalogEntry(table="tb_bahan", columns=COLUMNS, key_column="id",
                          filterable_columns=("id", "nama", "harga_beli"))
TABLE = WritableTable(entry=ENTRY, operations=("insert", "update", "delete"),
                      writable_columns=("nama", "harga_beli", "tanggal"))


class Scripted:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "w.sqlite3"
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE tb_bahan (id INTEGER PRIMARY KEY, nama TEXT NOT NULL, "
                  "harga_beli REAL, tanggal TEXT)")
        c.executemany("INSERT INTO tb_bahan VALUES (?,?,?,?)",
                      [(i, f"bahan{i}", 10000.0 + i * 1000, "2026-01-01") for i in range(1, 11)])
    return path


def _service(db, tmp_path, *answers, detail="full"):
    planner = DataChangePlanner(LLMService(Scripted(*answers)), [TABLE], today=lambda: date(2026, 11, 7))
    return DataChangeService(planner=planner, connect=lambda: sqlite3.connect(db), quote="`",
                             paramstyle="qmark", audit_path=tmp_path / "audit.jsonl", lock_clause="", audit_detail=detail)


def _rows(db, sql="SELECT id, nama, harga_beli FROM tb_bahan ORDER BY id"):
    with sqlite3.connect(db) as c:
        return c.execute(sql).fetchall()


def _plan(svc, payload, max_rows=100):
    return svc.planner.validate(payload, max_rows=max_rows)


UPDATE = {"operation": "update", "table": "tb_bahan", "set": {"harga_beli": 5000},
          "conditions": [{"column": "harga_beli", "operator": "lt", "value": 13000}]}


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize("payload,needle", [
    ({**UPDATE, "conditions": []}, "at least one condition"),
    ({**UPDATE, "conditions": None}, "at least one condition"),
    ({**UPDATE, "set": {"id": 5}}, "key column"),
    ({**UPDATE, "set": {"zzz": 5}}, "not writable"),
    ({**UPDATE, "set": {}}, "non-empty"),
    ({**UPDATE, "set": {"harga_beli": "banyak"}}, "number"),
    ({**UPDATE, "conditions": [{"column": "tanggal", "operator": "eq", "value": "2026-01-01"}]},
     "Cannot filter"),                                            # tanggal is not filterable
    ({**UPDATE, "table": "other"}, "Unknown table"),
    ({**UPDATE, "operation": "drop"}, "operation must be"),
    ({"operation": "delete", "table": "tb_bahan", "conditions": UPDATE["conditions"], "set": {"nama": "x"}},
     "delete does not take set"),
    ({"operation": "insert", "table": "tb_bahan", "rows": []}, "rows must be"),
    ({"operation": "insert", "table": "tb_bahan", "rows": [{"nama": "a"}, {"nama": "b", "harga_beli": 1}]},
     "same columns"),
    ({"operation": "insert", "table": "tb_bahan", "rows": [{"id": 99, "nama": "a"}]}, "key column"),
])
def test_bad_plans_are_refused(db, tmp_path, payload, needle):
    with pytest.raises(ValueError) as err:
        _plan(_service(db, tmp_path), payload)
    assert needle in str(err.value)


def test_an_operation_the_user_did_not_approve_is_refused():
    only_update = WritableTable(entry=ENTRY, operations=("update",), writable_columns=("nama",))
    planner = DataChangePlanner(LLMService(Scripted()), [only_update])
    with pytest.raises(ValueError, match="not allowed"):
        planner.validate({"operation": "delete", "table": "tb_bahan",
                          "conditions": UPDATE["conditions"]}, max_rows=10)


def test_insert_is_limited_by_the_users_row_limit(db, tmp_path):
    rows = [{"nama": f"n{i}"} for i in range(5)]
    svc = _service(db, tmp_path)
    assert len(_plan(svc, {"operation": "insert", "table": "tb_bahan", "rows": rows}, 5).rows) == 5
    with pytest.raises(ValueError, match="rows must be"):
        _plan(svc, {"operation": "insert", "table": "tb_bahan", "rows": rows}, 4)


def test_the_planner_asks_the_llm_again_after_an_invalid_answer(db, tmp_path):
    bad = json.dumps({**UPDATE, "conditions": []})
    good = json.dumps(UPDATE)
    result = _service(db, tmp_path, bad, good).planner.plan("turunkan harga", max_rows=50)
    assert result.request and result.attempts == 2 and result.request.max_rows == 50


def test_the_llm_cannot_choose_the_row_limit(db, tmp_path):
    answer = json.dumps({**UPDATE, "max_rows": 100000})
    result = _service(db, tmp_path, answer).planner.plan("x", max_rows=7)
    assert result.request.max_rows == 7


# ------------------------------------------------------------------ sql
def test_values_are_bound_never_written_into_the_sql():
    b = ChangeSqlBuilder(quote="`", paramstyle="qmark")
    sql, params = b.update("tb_bahan", (("nama", "x'; DROP TABLE tb_bahan;--"),),
                           (("id", "eq", 1),))
    assert sql == "UPDATE `tb_bahan` SET `nama` = ? WHERE `id` = ?"
    assert params == ["x'; DROP TABLE tb_bahan;--", 1]
    sql, params = b.delete("tb_bahan", (("id", "in", (1, 2)), ("nama", "contains", "a%")))
    assert "TABLE" not in sql and sql.count("?") == len(params) == 3
    fmt = ChangeSqlBuilder(quote="`", paramstyle="format").delete("t", (("id", "eq", 1),))
    assert fmt[0] == "DELETE FROM `t` WHERE `id` = %s"


# ------------------------------------------------------------------ preview & execute
def test_preview_shows_before_and_after_and_changes_nothing(db, tmp_path):
    svc = _service(db, tmp_path)
    before = _rows(db)
    preview = svc.preview(_plan(svc, UPDATE))
    assert preview.expected_rows == 2                      # 11000 and 12000
    assert [r["id"] for r in preview.before] == [1, 2]
    assert {r["harga_beli"] for r in preview.after} == {5000}
    assert _rows(db) == before


def test_preview_refuses_when_more_rows_match_than_the_users_limit(db, tmp_path):
    svc = _service(db, tmp_path)
    with pytest.raises(ChangeRefused, match="more than 1 rows"):
        svc.preview(_plan(svc, UPDATE, max_rows=1))


def test_preview_refuses_when_nothing_matches(db, tmp_path):
    svc = _service(db, tmp_path)
    payload = {**UPDATE, "conditions": [{"column": "id", "operator": "eq", "value": 999}]}
    with pytest.raises(ChangeRefused, match="No rows match"):
        svc.preview(_plan(svc, payload))


def _run(svc, request, rows, approver=None):
    return svc.execute(request, expected_rows=rows, actor="alice", approver=approver,
                       plan_id="p1", question="q")


def test_update_changes_exactly_the_shown_rows_and_is_audited_with_the_old_values(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, UPDATE)
    result = _run(svc, request, svc.preview(request).expected_rows)
    assert result.rows_affected == 2
    assert [r[2] for r in _rows(db)[:3]] == [5000.0, 5000.0, 13000.0]
    (line,) = (tmp_path / "audit.jsonl").read_text("utf-8").splitlines()
    audit = json.loads(line)
    assert audit["status"] == "committed" and audit["actor"] == "alice" and audit["rows_affected"] == 2
    assert [r["harga_beli"] for r in audit["before_image"]] == [11000.0, 12000.0]
    assert audit["sql"].startswith("UPDATE") and audit["params"] == [5000, 13000]


def test_a_change_to_the_data_after_the_preview_is_not_applied(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, UPDATE)
    shown = svc.preview(request).expected_rows
    with sqlite3.connect(db) as c:
        c.execute("UPDATE tb_bahan SET harga_beli = 100 WHERE id = 5")   # a third row now matches
    before = _rows(db)
    with pytest.raises(ChangeRefused, match="changed since the preview"):
        _run(svc, request, shown)
    assert _rows(db) == before
    assert json.loads((tmp_path / "audit.jsonl").read_text("utf-8").splitlines()[-1])["status"] == "rolled_back"


def test_delete_removes_the_rows_and_keeps_them_in_the_audit(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, {"operation": "delete", "table": "tb_bahan",
                          "conditions": [{"column": "id", "operator": "in", "value": [3, 4]}]})
    assert _run(svc, request, 2, approver="bob").rows_affected == 2
    assert [r[0] for r in _rows(db)] == [1, 2, 5, 6, 7, 8, 9, 10]
    audit = json.loads((tmp_path / "audit.jsonl").read_text("utf-8").splitlines()[-1])
    assert audit["approver"] == "bob" and [r["nama"] for r in audit["before_image"]] == ["bahan3", "bahan4"]


def test_insert_adds_rows_and_a_failure_adds_none(db, tmp_path):
    svc = _service(db, tmp_path)
    ok = _plan(svc, {"operation": "insert", "table": "tb_bahan",
                     "rows": [{"nama": "gula", "harga_beli": 1}, {"nama": "teh", "harga_beli": 2}]})
    assert svc.preview(ok).expected_rows == 2
    assert _run(svc, ok, 2).rows_affected == 2
    assert len(_rows(db)) == 12
    bad = _plan(svc, {"operation": "insert", "table": "tb_bahan",
                      "rows": [{"nama": "ok", "harga_beli": 1}, {"nama": None, "harga_beli": 2}]})
    with pytest.raises(ChangeRefused, match="nama wajib diisi"):   # NOT NULL, the second row: said in words
        _run(svc, bad, 2)
    assert len(_rows(db)) == 12                                # the first row was rolled back too


def test_an_unexpected_row_count_rolls_everything_back(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, UPDATE)
    with pytest.raises(ChangeRefused):
        _run(svc, request, 1)                                  # claims 1 row, 2 match
    assert [r[2] for r in _rows(db)[:2]] == [11000.0, 12000.0]


def test_the_confirmation_phrase_names_the_action_the_rows_and_the_table(db, tmp_path):
    svc = _service(db, tmp_path)
    assert svc.confirm_phrase(_plan(svc, UPDATE), 2) == "UBAH 2 BARIS tb_bahan"
    delete = _plan(svc, {"operation": "delete", "table": "tb_bahan", "conditions": UPDATE["conditions"]})
    assert svc.confirm_phrase(delete, 2) == "HAPUS 2 BARIS tb_bahan"


# ------------------------------------------------------------------ verifying after the commit
def _committed(svc, request):
    preview = svc.preview(request)
    svc.execute(request, expected_rows=preview.expected_rows, actor="a", approver=None,
                plan_id="p", question="q")
    return preview.expected_rows


def test_verify_a_delete_by_reading_the_table_again(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, {"operation": "delete", "table": "tb_bahan",
                          "conditions": [{"column": "id", "operator": "in", "value": [1, 2]}]})
    rows = _committed(svc, request)
    assert svc.verify(request, rows) == {"method": "reread_after_commit", "expected_rows": 0,
                                         "observed_rows": 0, "confirmed": True}


def test_verify_notices_a_delete_that_did_not_hold(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, {"operation": "delete", "table": "tb_bahan",
                          "conditions": [{"column": "id", "operator": "in", "value": [1, 2]}]})
    rows = _committed(svc, request)
    with sqlite3.connect(db) as c:                       # somebody put a matching row back
        c.execute("INSERT INTO tb_bahan VALUES (1, 'back', 1.0, '2026-01-01')")
    seen = svc.verify(request, rows)
    assert seen["observed_rows"] == 1 and seen["confirmed"] is False


def test_verify_an_update_finds_the_rows_with_their_new_values(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, {"operation": "update", "table": "tb_bahan", "set": {"nama": "baru"},
                          "conditions": [{"column": "harga_beli", "operator": "lt", "value": 13000}]})
    rows = _committed(svc, request)
    seen = svc.verify(request, rows)
    assert seen == {"method": "reread_after_commit", "expected_rows": rows, "observed_rows": rows,
                    "confirmed": True}
    with sqlite3.connect(db) as c:                       # one of them was changed again afterwards
        c.execute("UPDATE tb_bahan SET nama='lain' WHERE id=1")
    assert svc.verify(request, rows)["confirmed"] is False


def test_verify_an_update_that_changes_what_it_selected_by_can_only_use_the_reported_count(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, UPDATE)                          # sets harga_beli where harga_beli < 13000
    rows = _committed(svc, request)
    assert svc.verify(request, rows)["method"] == "database_reported_rowcount"


def test_verify_an_insert_uses_the_reported_count(db, tmp_path):
    svc = _service(db, tmp_path)
    request = _plan(svc, {"operation": "insert", "table": "tb_bahan", "rows": [{"nama": "x"}, {"nama": "y"}]})
    rows = _committed(svc, request)
    assert svc.verify(request, rows) == {"method": "database_reported_rowcount", "expected_rows": 2,
                                         "observed_rows": 2, "confirmed": True}


def test_the_default_audit_keeps_no_business_data_only_keys_and_counts(db, tmp_path):
    svc = _service(db, tmp_path, detail="minimal")
    request = _plan(svc, UPDATE)
    result = _run(svc, request, svc.preview(request).expected_rows)
    assert result.rows_affected == 2
    text = (tmp_path / "audit.jsonl").read_text("utf-8")
    audit = json.loads(text.splitlines()[0])
    assert audit["status"] == "committed" and audit["audit_detail"] == "minimal"
    assert audit["before_image"] == [{"id": 2}, {"id": 3}] or all(set(r) == {"id"} for r in audit["before_image"])
    assert audit["before_image_rows"] == 2 and audit["params"] == [] and audit["params_count"] == 2
    for value in ("11000", "12000", "5000", "13000"):
        assert value not in text


def test_the_audit_detail_can_be_changed_without_a_restart(db, tmp_path):
    mode = {"value": "minimal"}
    planner = DataChangePlanner(LLMService(Scripted()), [TABLE], today=lambda: date(2026, 11, 7))
    svc = DataChangeService(planner=planner, connect=lambda: sqlite3.connect(db), quote="`", paramstyle="qmark",
                            audit_path=tmp_path / "audit.jsonl", lock_clause="", audit_detail=lambda: mode["value"])
    assert svc._detail_mode() == "minimal"
    mode["value"] = "full"
    assert svc._detail_mode() == "full"
