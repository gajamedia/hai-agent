"""Conformance suite every database engine must pass.

A new engine (written by a person or drafted by an agent) is accepted only when it
passes these checks against a *disposable* server. They encode the guarantees the
platform promises users, independent of the database product:

1. a SELECT-only account connects and its schema is discovered (types, primary key);
2. the session is read-only and cannot change the database;
3. a question becomes a report whose numbers equal an independent SQL query;
4. privileged (superuser/all-privileges) and write-capable accounts are refused;
5. connection errors never leak host, user, database or password;
6. an approved connection survives a restart and answers again;
7. tables/columns named with reserved words or mixed case can be reported;
8. a "show the rows" listing returns the rows in the requested order and stops at its limit;
9. listings and sorting work on reserved-word names, with two sort keys;
10. comparison conditions (<, between, in, contains) return what an independent query returns.

To add an engine, write a fixture that creates the standard dataset (see the
``Harness`` contract below) and subclass ``ConformanceSuite``:

    class TestMyEngineConformance(ConformanceSuite):
        @pytest.fixture
        def harness(self, my_fixture):
            return Harness(...)

Engine-specific behaviour (TLS, grants syntax, type names) stays in the engine's own
test file; this module must not contain any product-specific code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import pytest

from api.database_runtime_registry import RegistryError
from tests.test_table_report_job_e2e import _context


@dataclass
class Harness:
    engine: str
    registry: Callable[..., Any]            # registry(*scripted_llm_answers)
    reader: str                             # profile of a SELECT-only account
    writer: str                             # profile of an account that can INSERT
    privileged: str                         # profile of a superuser / ALL PRIVILEGES account
    wrong_password: str                     # profile that authenticates incorrectly
    sql: Callable[[str], list]              # run a query as an administrator
    q_sales: str                            # fully qualified, quoted name of the sales table
    q_order: str                            # same for the reserved-word table
    order_name: str                         # catalog name of the reserved-word table
    quote: Callable[[str], str]             # quote a column name for this engine
    assert_read_only: Callable[[Any], None]  # probe a raw connection; raise if it can write
    leak_tokens: list[str]                  # strings that must never appear in an error


def _plan(table: str, group_by: str, **extra: Any) -> str:
    return json.dumps({
        "supported": True, "table": table, "group_by": [group_by],
        "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"},
                    {"output": "orders", "operation": "count"}], **extra})


SALES_SELECTION = {"sales": {
    "columns": ["id", "region", "product", "amount"],
    "filterable_columns": ["region", "product"], "key_column": "id",
    "required_non_null_columns": ["amount"]}}


class ConformanceSuite:
    """Base class: subclasses supply a ``harness`` fixture."""

    def _connect(self, h: Harness, name="conf", source=None):
        reg = h.registry(_plan("sales", "region", filters={"product": "gadget"}))
        info = reg.connect(name, engine=h.engine, source=source or h.reader, actor_id="alice")
        return reg, info

    def test_1_select_only_account_connects_and_schema_is_discovered(self, harness):
        reg, info = self._connect(harness)
        tables = {t["name"]: t for t in info["schema"]["tables"]}
        assert "sales" in tables and harness.order_name in tables
        cols = {c["name"]: c for c in tables["sales"]["columns"]}
        assert cols["id"]["primary_key"] is True
        assert {"id", "region", "product", "amount"} <= set(cols)

    def test_2_session_is_read_only(self, harness):
        reg, _ = self._connect(harness)
        with reg._connections["conf"].manager.connection("conf") as connection:
            harness.assert_read_only(connection)

    def test_3_report_equals_independent_sql(self, harness):
        reg, _ = self._connect(harness)
        reg.activate("conf", SALES_SELECTION, actor_id="bob")
        outcome = reg.service("conf").ask("gadget per region", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        rows = harness.sql(f"SELECT region, SUM(amount), COUNT(id) FROM {harness.q_sales} "
                           f"WHERE product = 'gadget' GROUP BY region")
        expected = {r: (float(s), int(n)) for r, s, n in rows}
        got = {r["region"]: (float(r["revenue"]), r["orders"]) for r in outcome.rows}
        assert got.keys() == expected.keys() and expected
        for region, (revenue, orders) in expected.items():
            assert got[region][0] == pytest.approx(revenue) and got[region][1] == orders

    def test_4_privileged_and_write_capable_accounts_are_refused(self, harness):
        reg = harness.registry()
        for name, source in (("p1", harness.privileged), ("p2", harness.writer)):
            with pytest.raises(RegistryError):
                reg.connect(name, engine=harness.engine, source=source, actor_id="alice")
        assert reg.list() == []

    def test_5_connection_errors_do_not_leak_secrets(self, harness):
        with pytest.raises(RegistryError) as info:
            harness.registry().connect("bad", engine=harness.engine,
                                       source=harness.wrong_password, actor_id="alice")
        text = str(info.value)
        for token in harness.leak_tokens:
            assert token and token not in text

    def test_6_connection_survives_restart(self, harness):
        reg, _ = self._connect(harness)
        reg.activate("conf", SALES_SELECTION, actor_id="bob")
        reg.shutdown()
        again = harness.registry(_plan("sales", "region"))
        (item,) = again.list()
        assert item["status"] == "active" and item["database"] == harness.engine
        outcome = again.service("conf").ask("per region", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)

    def test_7_reserved_word_and_mixed_case_names_can_be_reported(self, harness):
        plan = _plan(harness.order_name, "group", filters={"user": "u1"})
        reg = harness.registry(plan)
        reg.connect("conf", engine=harness.engine, source=harness.reader, actor_id="alice")
        reg.activate("conf", {harness.order_name: {
            "columns": ["id", "group", "user", "amount"], "filterable_columns": ["user"],
            "key_column": "id", "required_non_null_columns": ["amount"]}}, actor_id="bob")
        outcome = reg.service("conf").ask("per group", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        g, u = harness.quote("group"), harness.quote("user")
        rows = harness.sql(f"SELECT {g}, SUM(amount) FROM {harness.q_order} "
                           f"WHERE {u} = 'u1' GROUP BY {g}")
        expected = {k: float(v) for k, v in rows}
        assert {r["group"]: float(r["revenue"]) for r in outcome.rows} == pytest.approx(expected)

    def test_8_a_listing_returns_the_rows_in_the_requested_order_and_stops_at_its_limit(self, harness):
        answer = json.dumps({
            "supported": True, "kind": "list", "table": "sales", "columns": ["id", "region", "amount"],
            "order_by": [{"column": "id", "direction": "desc"}], "limit": 5, "format": "csv"})
        reg = harness.registry(answer)
        reg.connect("conf", engine=harness.engine, source=harness.reader, actor_id="alice")
        reg.activate("conf", SALES_SELECTION, actor_id="bob")
        outcome = reg.service("conf").ask("show five sales", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        assert outcome.kind == "list" and outcome.row_count == 5 and outcome.limit_reached is True
        expected = harness.sql(f"SELECT id, region, amount FROM {harness.q_sales} ORDER BY id DESC")[:5]
        got = [(r["id"], r["region"], float(r["amount"])) for r in outcome.rows]
        assert got == [(i, region, float(amount)) for i, region, amount in expected]
        assert list(outcome.rows[0]) == ["id", "region", "amount"]            # the requested column order

    def test_9_listings_and_sorting_work_on_reserved_word_names_with_two_sort_keys(self, harness):
        answer = json.dumps({
            "supported": True, "kind": "list", "table": harness.order_name, "columns": ["id", "group", "user"],
            "order_by": [{"column": "group", "direction": "desc"}, {"column": "id", "direction": "asc"}],
            "limit": 6})
        reg = harness.registry(answer)
        reg.connect("conf", engine=harness.engine, source=harness.reader, actor_id="alice")
        reg.activate("conf", {harness.order_name: {
            "columns": ["id", "group", "user", "amount"], "filterable_columns": ["user"],
            "key_column": "id"}}, actor_id="bob")
        outcome = reg.service("conf").ask("show the orders", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        g, u = harness.quote("group"), harness.quote("user")
        expected = harness.sql(f"SELECT id, {g}, {u} FROM {harness.q_order} ORDER BY {g} DESC, id ASC")[:6]
        got = [(r["id"], r["group"], r["user"]) for r in outcome.rows]
        assert got == [tuple(row) for row in expected] and len(got) == 6


    def test_10_conditions_return_what_an_independent_query_returns(self, harness):
        answer = json.dumps({
            "supported": True, "kind": "list", "table": "sales", "columns": ["id", "region", "amount"],
            "order_by": [{"column": "id", "direction": "asc"}], "limit": 100,
            "conditions": [{"column": "amount", "operator": "lt", "value": 40},
                           {"column": "id", "operator": "between", "value": [2, 30]},
                           {"column": "region", "operator": "in", "value": ["west", "east"]},
                           {"column": "product", "operator": "contains", "value": "get"}]})
        reg = harness.registry(answer)
        reg.connect("conf", engine=harness.engine, source=harness.reader, actor_id="alice")
        selection = {"sales": {**SALES_SELECTION["sales"],
                               "filterable_columns": ["region", "product", "amount", "id"]}}
        reg.activate("conf", selection, actor_id="bob")
        outcome = reg.service("conf").ask("cheap west/east widgets", execution_context=_context())
        assert outcome.status == "succeeded", (outcome.error_code, outcome.reason)
        expected = harness.sql(
            f"SELECT id FROM {harness.q_sales} WHERE amount < 40 AND id BETWEEN 2 AND 30 "
            f"AND region IN ('west', 'east') AND product LIKE '%get%' ORDER BY id")
        assert expected and [r["id"] for r in outcome.rows] == [row[0] for row in expected]
