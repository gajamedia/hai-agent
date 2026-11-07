from __future__ import annotations

import json
import sqlite3

import pytest

from agent.llm.contracts import LLMResponse
from api.database_runtime_registry import DatabaseRuntimeRegistry, RegistryError
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import _context, _make_db, _sample_rows

SELECTION = {"sales": {"columns": ["id", "region", "product", "amount"],
                       "filterable_columns": ["region"], "key_column": "id",
                       "required_non_null_columns": ["amount"]}}
PLAN = json.dumps({"supported": True, "table": "sales", "group_by": ["region"],
                   "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}]})


class _Provider:
    name = "scripted"

    def __init__(self):
        self.answers = [PLAN] * 5

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_db(data / "sales.sqlite3", _sample_rows(30))
    made = []

    def registry():
        r = DatabaseRuntimeRegistry(
            data_root=data, work_dir=tmp_path / "work",
            authorization_policy=AuthorizationPolicy(), llm_provider_factory=_Provider)
        made.append(r)
        return r
    yield registry, data
    for r in made:
        r.shutdown()


def _approve(r, who="alice"):
    r.connect_sqlite("sales", "sales.sqlite3", actor_id=who)
    return r.activate("sales", SELECTION, actor_id="bob")


def test_approved_connection_survives_restart_and_still_answers(setup):
    make, _ = setup
    first = make()
    _approve(first)
    first.shutdown()
    second = make()
    (item,) = second.list()
    assert item["status"] == "active" and item["tables"] == ["sales"]
    assert item["approved_by"] == "alice" and item["tables_approved_by"] == "bob"
    outcome = second.service("sales").ask(
        "total per wilayah", execution_context=_context())
    assert outcome.status == "succeeded" and {r["region"] for r in outcome.rows} == {"west", "east", "north"}


def test_unapproved_and_disconnected_connections_are_not_restored(setup):
    make, _ = setup
    first = make()
    first.connect_sqlite("pending", "sales.sqlite3", actor_id="alice")
    _approve(first)
    first.disconnect("sales")
    first.shutdown()
    assert make().list() == []


def test_missing_file_is_listed_as_unavailable_and_can_be_removed(setup):
    make, data = setup
    first = make()
    _approve(first)
    first.shutdown()
    (data / "sales.sqlite3").unlink()
    second = make()
    (item,) = second.list()
    assert item["status"] == "unavailable" and item["tables"] == []
    with pytest.raises(RegistryError):
        second.service("sales")
    second.disconnect("sales")
    assert make().list() == []


def test_changed_schema_requires_reapproval_and_is_never_silently_active(setup):
    make, data = setup
    first = make()
    _approve(first)
    first.shutdown()
    with sqlite3.connect(data / "sales.sqlite3") as db:  # approved column "amount" is dropped
        db.execute("ALTER TABLE sales DROP COLUMN amount")
    second = make()
    (item,) = second.list()
    assert item["status"] == "needs_reapproval" and "changed" in item["reason"]
    with pytest.raises(RegistryError):
        second.service("sales")
    # the user can re-approve a valid selection; that is persisted again
    second.activate("sales", {"sales": {"columns": ["id", "region"], "key_column": "id"}},
                    actor_id="carol")
    third_item = second.list()[0]
    assert third_item["status"] == "active" and third_item["tables_approved_by"] == "carol"
    second.shutdown()
    assert make().list()[0]["status"] == "active"


def test_changed_column_type_requires_reapproval(setup):
    make, data = setup
    first = make()
    _approve(first)
    first.shutdown()
    with sqlite3.connect(data / "sales.sqlite3") as db:
        db.execute("ALTER TABLE sales RENAME TO old_sales")
        db.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, product TEXT, amount TEXT)")
    assert make().list()[0]["status"] == "needs_reapproval"


def test_unrelated_new_table_or_column_does_not_invalidate_approval(setup):
    make, data = setup
    first = make()
    _approve(first)
    first.shutdown()
    with sqlite3.connect(data / "sales.sqlite3") as db:
        db.execute("CREATE TABLE audit (id INTEGER)")
        db.execute("ALTER TABLE sales ADD COLUMN note TEXT")
    assert make().list()[0]["status"] == "active"


def test_name_of_a_degraded_connection_cannot_be_reused(setup):
    make, data = setup
    first = make()
    _approve(first)
    first.shutdown()
    (data / "sales.sqlite3").rename(data / "moved.sqlite3")
    second = make()
    with pytest.raises(RegistryError, match="already exists"):
        second.connect_sqlite("sales", "moved.sqlite3", actor_id="alice")


def test_store_contains_only_metadata_and_the_approved_selection(setup):
    make, _ = setup
    r = make()
    _approve(r)
    with sqlite3.connect(r._db_path) as db:
        cols = [c[1] for c in db.execute("PRAGMA table_info(connections)")]
        (selection,) = db.execute("SELECT selection_json FROM connections").fetchone()
    assert set(cols) == {"name", "filename", "engine", "connected_by", "connected_at", "approved_by",
                         "approved_at", "selection_json", "fingerprint"}
    assert json.loads(selection) == SELECTION


def test_tampered_store_row_degrades_instead_of_crashing(setup):
    make, _ = setup
    r = make()
    _approve(r)
    r.shutdown()
    with sqlite3.connect(r._db_path) as db:
        db.execute("UPDATE connections SET filename = '../escape.sqlite3'")
    item = make().list()[0]
    assert item["status"] == "unavailable"
