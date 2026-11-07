from __future__ import annotations

import json
import sqlite3

import pytest

from agent.llm.contracts import LLMResponse
from agent.llm.service import LLMService
from agent.workflow.job_experience import JobExperienceStore
from agent.workflow.table_report_planner import TableCatalogEntry, TableReportPlanner
from agent.workflow.table_report_service import TableReportService
from tests.test_table_report_job_e2e import (  # noqa: F401
    COLUMNS, SALES_SPEC, _context, _sample_rows, env,
)


class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.systems = list(answers), []

    def complete(self, request):
        self.systems.append(request.messages[0].content)
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _plan(**filters):
    body = {"supported": True, "table": "sales", "group_by": ["region"],
            "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}]}
    if filters:
        body["filters"] = filters
    return json.dumps(body)


@pytest.fixture
def store(tmp_path):
    return JobExperienceStore(tmp_path / "experience.sqlite3")


def _service(e, store, *answers, connection="sales"):
    provider = _Provider(*answers)
    entry = TableCatalogEntry.from_table_spec(
        SALES_SPEC, COLUMNS, key_column="id", required_non_null_columns=("amount",))
    experience = store.bind(connection)
    planner = TableReportPlanner(LLMService(provider), [entry], experience=experience)
    return TableReportService(planner=planner, runner=e.runner, experience=experience), provider


def test_agent_improves_after_a_failure_and_a_success(env, store):  # noqa: F811
    """The same wrong-case question fails, then succeeds once the spelling is learned."""
    e = env(_sample_rows(40))
    svc, _ = _service(e, store, _plan(product="Widget"), _plan(product="widget"), _plan(product="Widget"))
    first = svc.ask("penjualan Widget per wilayah", execution_context=_context())
    assert first.status == "failed" and first.error_code == "no_matching_rows"
    second = svc.ask("penjualan widget per wilayah", execution_context=_context())
    assert second.status == "succeeded"
    third = svc.ask("penjualan Widget per wilayah", execution_context=_context())
    assert third.status == "succeeded" and third.plan["filters"] == {"product": "widget"}
    assert third.rows == second.rows


def test_failed_runs_do_not_teach_values_but_are_remembered_as_empty(env, store):  # noqa: F811
    e = env(_sample_rows(10))
    svc, _ = _service(e, store, _plan(region="Atlantis"))
    svc.ask("x", execution_context=_context())
    exp = store.bind("sales")
    assert exp.canonical_value("sales", "region", "atlantis") == "atlantis"  # nothing verified
    assert exp.empty_filter_notes() == [{"table": "sales", "filters": {"region": "Atlantis"}}]


def test_prompt_contains_examples_and_empty_filters_as_labelled_data(env, store):  # noqa: F811
    e = env(_sample_rows(30))
    svc, provider = _service(e, store, _plan(region="Atlantis"), _plan(region="west"), _plan())
    svc.ask("a", execution_context=_context())
    svc.ask("Penjualan west", execution_context=_context())
    svc.ask("c", execution_context=_context())
    assert "PAST RUNS" not in provider.systems[0]
    last = provider.systems[-1]
    assert "PAST RUNS ON THIS DATABASE" in last and "never override" in last
    assert '"region": "west"' in last and "Atlantis" in last and "Penjualan west" in last


def test_experience_is_isolated_per_connection(env, store):  # noqa: F811
    e = env(_sample_rows(30))
    svc, _ = _service(e, store, _plan(region="west"), connection="one")
    svc.ask("q", execution_context=_context())
    other = store.bind("two")
    assert other.canonical_value("sales", "region", "WEST") == "WEST"
    assert other.examples() == [] and other.prompt_block() == ""


def test_corrects_case_and_surrounding_spaces_only(store):
    exp = store.bind("c")
    exp.record("q", {"table": "sales", "filters": {"region": "west"}}, "succeeded")
    assert exp.canonical_value("sales", "region", "  WEST ") == "west"
    assert exp.canonical_value("sales", "region", "westside") == "westside"  # never fuzzy
    assert exp.canonical_value("sales", "product", "west") == "west"          # per column


def test_only_string_filters_are_stored_and_only_after_success(store):
    exp = store.bind("c")
    exp.record("q", {"table": "t", "filters": {"n": 5, "ok": True}}, "succeeded")
    exp.record("q", {"table": "t", "filters": {"s": "x"}}, "failed", "no_matching_rows")
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM known_values").fetchone() == (0,)


def test_store_never_contains_rows_or_results(env, store):  # noqa: F811
    e = env(_sample_rows(20))
    svc, _ = _service(e, store, _plan(region="west"))
    svc.ask("q", execution_context=_context())
    with sqlite3.connect(store.path) as db:
        dump = json.dumps([list(r) for r in db.execute("SELECT * FROM job_runs")])
    assert "revenue" in dump  # plan metadata (metric names) only
    assert '"rows"' not in dump and "10.0" not in dump


def test_history_is_capped_per_connection(tmp_path):
    small = JobExperienceStore(tmp_path / "e.sqlite3", max_runs_per_connection=5, max_values_per_column=3)
    exp = small.bind("c")
    for i in range(12):
        exp.record(f"q{i}", {"table": "t", "filters": {"s": f"v{i}"}}, "succeeded")
    with sqlite3.connect(small.path) as db:
        assert db.execute("SELECT COUNT(*) FROM job_runs").fetchone() == (5,)
        assert db.execute("SELECT COUNT(*) FROM known_values").fetchone() == (3,)


def test_forget_removes_everything_for_one_connection_only(store):
    a, b = store.bind("a"), store.bind("b")
    for exp in (a, b):
        exp.record("q", {"table": "t", "filters": {"s": "x"}}, "succeeded")
    a.forget()
    assert a.examples() == [] and a.canonical_value("t", "s", "X") == "X"
    assert b.canonical_value("t", "s", "X") == "x"


def test_a_broken_store_never_breaks_the_answer(env, store):  # noqa: F811
    e = env(_sample_rows(20))
    svc, _ = _service(e, store, _plan(region="west"))

    class Broken:
        def record(self, *a, **k):
            raise RuntimeError("disk full")
    svc.experience = Broken()
    assert svc.ask("q", execution_context=_context()).status == "succeeded"


def test_question_text_is_truncated(store):
    store.bind("c").record("x" * 5000, {"table": "t"}, "succeeded")
    with sqlite3.connect(store.path) as db:
        (q,) = db.execute("SELECT question FROM job_runs").fetchone()
    assert len(q) == 300


def _registry(tmp_path):
    from api.database_runtime_registry import DatabaseRuntimeRegistry
    from api.security.authorization import AuthorizationPolicy
    return DatabaseRuntimeRegistry(
        data_root=tmp_path / "data", work_dir=tmp_path / "work",
        authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: _Provider(*[_plan(product="widget")] * 5))


def test_registry_keeps_experience_across_restart_and_deletes_it_on_disconnect(tmp_path):
    from tests.test_table_report_job_e2e import _make_db
    (tmp_path / "data").mkdir()
    _make_db(tmp_path / "data" / "sales.sqlite3", _sample_rows(30))
    selection = {"sales": {"columns": ["id", "region", "product", "amount"],
                           "filterable_columns": ["product"], "key_column": "id"}}
    first = _registry(tmp_path)
    first.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
    first.activate("sales", selection, actor_id="alice")
    assert first.service("sales").ask("q", execution_context=_context()).status == "succeeded"
    first.shutdown()

    second = _registry(tmp_path)
    exp = second._experience.bind("sales")
    assert exp.canonical_value("sales", "product", "WIDGET") == "widget"
    assert len(exp.examples()) == 1
    second.disconnect("sales")
    assert second._experience.bind("sales").examples() == []
    assert second._experience.bind("sales").canonical_value("sales", "product", "WIDGET") == "WIDGET"
    second.shutdown()
