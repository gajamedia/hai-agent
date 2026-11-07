from __future__ import annotations

import json
import sqlite3

import pytest

from agent.llm.business_database_adapter import BusinessDatabaseAdapter, ReadOnlyTableSpec
from agent.llm.contracts import LLMResponse
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import _context

SPEC = ReadOnlyTableSpec(table="order", columns=("id", "group", "select"),
                         filterable_columns=("group",), sortable_columns=("select",))


class _Capture:
    """Records the SQL the adapter builds, without a database."""

    def __init__(self):
        self.queries = []

    def __call__(self):
        outer = self

        class Cursor:
            description = [("id",)]

            def execute(self, query, params):
                outer.queries.append((query, params))

            def fetchall(self):
                return []

            def close(self):
                pass

        class Conn:
            def cursor(self):
                return Cursor()

            def close(self):
                pass
        return Conn()


def _sql(quote, **arguments):
    capture = _Capture()
    BusinessDatabaseAdapter(connection_factory=capture, table_specs={"order": SPEC},
                            identifier_quote=quote).read(
        "business_database.select", {"table": "order", **arguments})
    return capture.queries[0]


def test_default_is_unquoted_for_backward_compatibility():
    sql, params = _sql(None, columns=["id", "group"], filters={"group": "a"})
    assert sql == "SELECT id, group FROM order WHERE group = ? LIMIT ?" and params == ("a", 100)


@pytest.mark.parametrize("quote", ['"', "`"])
def test_every_identifier_is_quoted_and_values_stay_bound(quote):
    sql, params = _sql(quote, columns=["id", "group"], filters={"group": "a'; DROP TABLE x;--"},
                       order_by=["-select"], limit=5)
    q = quote
    assert sql == (f"SELECT {q}id{q}, {q}group{q} FROM {q}order{q} WHERE {q}group{q} = ? "
                   f"ORDER BY {q}select{q} DESC LIMIT ?")
    assert params == ("a'; DROP TABLE x;--", 5)


def test_null_filter_is_quoted_too():
    sql, _ = _sql("`", filters={"group": None})
    assert "`group` IS NULL" in sql


@pytest.mark.parametrize("quote", ["'", "(", "``", "x", ""])
def test_unknown_quote_characters_are_rejected(quote):
    with pytest.raises(ValueError):
        BusinessDatabaseAdapter(connection_factory=lambda: None, table_specs={"order": SPEC},
                                identifier_quote=quote)


def test_a_malicious_identifier_still_cannot_get_through_the_allowlist():
    with pytest.raises(Exception):
        ReadOnlyTableSpec(table="order", columns=('id"; DROP TABLE x; --',))
    with pytest.raises(Exception):
        _sql('"', columns=['id", "x'])


class _Provider:
    name = "scripted"

    def complete(self, request):
        return LLMResponse(content=json.dumps({
            "supported": True, "table": "order", "group_by": ["group"],
            "metrics": [{"output": "total", "operation": "sum", "column": "select"}],
            "filters": {"group": "a"}}), provider=self.name)


def test_sqlite_report_over_reserved_word_table_and_columns_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    with sqlite3.connect(data / "shop.sqlite3") as db:
        db.execute('CREATE TABLE "order" (id INTEGER PRIMARY KEY, "group" TEXT, "select" REAL)')
        db.executemany('INSERT INTO "order" VALUES (?,?,?)',
                       [(i, ["a", "b"][i % 2], i * 1.5) for i in range(1, 11)])
    reg = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=_Provider)
    try:
        info = reg.connect_sqlite("shop", "shop.sqlite3", actor_id="alice")
        assert {t["name"] for t in info["schema"]["tables"]} == {"order"}
        reg.activate("shop", {"order": {"columns": ["id", "group", "select"],
                                        "filterable_columns": ["group"], "key_column": "id",
                                        "required_non_null_columns": ["select"]}}, actor_id="alice")
        outcome = reg.service("shop").ask("total per group", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        with sqlite3.connect(data / "shop.sqlite3") as db:
            (expected,) = db.execute('SELECT SUM("select") FROM "order" WHERE "group"=\'a\'').fetchone()
        assert list(outcome.rows) == [{"group": "a", "total": pytest.approx(expected)}]
    finally:
        reg.shutdown()
