from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from api.platform_runtime import get_pipeline
from tests.test_table_report_job_e2e import _make_db, _sample_rows
from tests.test_table_reports_api import PERMISSION, READ, _Attach

BASE = "/api/v1/table-reports"
SELECTION = {"sales": {"columns": ["id", "region", "product", "amount"],
                       "filterable_columns": ["region", "product"], "key_column": "id",
                       "required_non_null_columns": ["amount"]}}


def _plan(**extra):
    return json.dumps({"supported": True, "table": "sales", "group_by": ["region"],
                       "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"}],
                       **extra})


class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _who(actor, *permissions):
    return AuthenticatedPrincipal(actor_id=actor, roles=(), permissions=permissions,
                                  issuer="test", subject=actor, is_authenticated=True)


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _make_db(data / "sales.sqlite3", _sample_rows(30))
    made = []

    def build(*answers):
        provider = _Provider(*answers)
        registry = DatabaseRuntimeRegistry(
            data_root=data, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
            llm_provider_factory=lambda: provider)
        registry.connect_sqlite("sales", "sales.sqlite3", actor_id="alice")
        registry.activate("sales", SELECTION, actor_id="alice")
        shared = {}

        def client(actor="alice", *permissions, pipeline=None):
            app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                             database_runtime_registry_factory=lambda _a: registry)
            app.add_middleware(_Attach, principal=_who(actor, *(permissions or (PERMISSION, READ))))
            c = TestClient(app)
            c.__enter__()
            made.append(c)
            # every client shares one pipeline, as one running server would
            if pipeline is not None:
                app.state.platform_pipeline = pipeline
            else:
                app.state.platform_pipeline = shared.setdefault("pipeline", get_pipeline(app))
            return c
        return client, registry, provider
    yield build, tmp_path
    for c in made:
        c.__exit__(None, None, None)


def _create(c, question="widget per region", **extra):
    return c.post(f"{BASE}/plans", json={"question": question, **extra})


def test_plan_reads_no_data_and_run_executes_exactly_that_plan(world):
    build, tmp = world
    client, registry, provider = build(_plan(filters={"product": "widget"}))
    c = client()
    created = _create(c)
    assert created.status_code == 201
    plan = created.json()
    assert plan["status"] == "planned" and plan["database"] == "sales"
    assert plan["plan"]["filters"] == {"product": "widget"} and plan["expires_in_seconds"] > 0
    assert provider.calls == 1
    assert registry._store.list_workflows() == ()  # planning created no workflow and read nothing

    ran = c.post(f"{BASE}/plans/{plan['plan_id']}/run")
    assert ran.status_code == 200, ran.text
    body = ran.json()
    assert body["status"] == "succeeded" and body["replayed"] is False and body["plan"] == plan["plan"]
    assert provider.calls == 1  # running does not ask the LLM again
    assert len(registry._store.list_workflows()) == 1
    with sqlite3.connect(tmp / "data" / "sales.sqlite3") as db:
        expected = dict(db.execute(
            "SELECT region, SUM(amount) FROM sales WHERE product='widget' GROUP BY region"))
    assert {r["region"]: r["revenue"] for r in body["rows"]} == pytest.approx(expected)


def test_a_plan_runs_once_and_a_second_run_is_refused(world):
    client, *_ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 200
    again = c.post(f"{BASE}/plans/{pid}/run")
    assert again.status_code == 409 and again.json()["detail"]["error"] == "plan_not_runnable"


def test_idempotency_key_replays_the_same_result_without_running_again(world):
    client, *_ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    first = c.post(f"{BASE}/plans/{pid}/run", headers={"Idempotency-Key": "k-1"}).json()
    second = c.post(f"{BASE}/plans/{pid}/run", headers={"Idempotency-Key": "k-1"})
    assert second.status_code == 200 and second.json()["replayed"] is True
    assert second.json()["workflow_id"] == first["workflow_id"] and second.json()["rows"] == first["rows"]
    assert c.post(f"{BASE}/plans/{pid}/run", headers={"Idempotency-Key": "other"}).status_code == 409
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 409


@pytest.mark.parametrize("key", ["", "   ", "x" * 256])
def test_invalid_idempotency_keys_are_rejected(world, key):
    client, *_ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    assert c.post(f"{BASE}/plans/{pid}/run", headers={"Idempotency-Key": key}).status_code == 400


def test_the_plan_cannot_be_edited_at_run_time(world):
    client, *_ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    for body in ({"filters": {"region": "west"}}, {"plan": {"table": "x"}}, {"question": "other"}):
        assert c.post(f"{BASE}/plans/{pid}/run", json=body).status_code == 422
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 200  # still runnable, unchanged


def test_plans_belong_to_their_owner(world):
    client, *_ = world[0](_plan())
    alice, bob = client("alice"), client("bob")
    pid = _create(alice).json()["plan_id"]
    assert bob.post(f"{BASE}/plans/{pid}/run").status_code == 404   # same answer as "unknown"
    assert bob.delete(f"{BASE}/plans/{pid}").status_code == 404
    assert alice.post(f"{BASE}/plans/{pid}/run").status_code == 200


def test_cancelled_and_unknown_plans_cannot_run(world):
    client, *_ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    assert c.delete(f"{BASE}/plans/{pid}").json()["status"] == "cancelled"
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 404
    assert c.post(f"{BASE}/plans/nope/run").status_code == 404


def test_unsupported_questions_get_no_plan_id(world):
    reason = json.dumps({"supported": False, "reason": "no date filters"})
    client, *_ = world[0](reason)
    body = _create(client(), "bulan lalu").json()
    assert body["status"] == "unsupported" and body["plan_id"] is None and "date" in body["reason"]
    assert body["needs_join"] is False


def test_a_missing_join_is_flagged_so_the_page_can_point_to_the_relation_form(world):
    reason = json.dumps({"supported": False, "reason": "perlu join", "needs_join": True})
    client, *_ = world[0](reason)
    body = _create(client(), "bahan dan jenisnya").json()
    assert body["status"] == "unsupported" and body["needs_join"] is True


def test_permissions_are_required_for_every_phase(world):
    client, *_ = world[0](_plan())
    nobody = client("nobody", "some:other:permission")
    assert _create(nobody).status_code == 403
    assert nobody.post(f"{BASE}/plans/x/run").status_code == 403
    assert nobody.delete(f"{BASE}/plans/x").status_code == 403


def test_running_still_needs_the_data_read_permission(world):
    client, *_ = world[0](_plan())
    planner_only = client("alice", PERMISSION)
    pid = _create(planner_only).json()["plan_id"]
    body = planner_only.post(f"{BASE}/plans/{pid}/run").json()
    assert body["status"] == "failed" and body["error_code"] == "capability_authorization_denied"
    assert body["rows"] == []


def test_changing_the_connection_after_planning_blocks_the_run(world):
    client, registry, _ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    registry.activate("sales", {"sales": {"columns": ["id", "region"], "key_column": "id"}},
                      actor_id="mallory")  # re-approval builds a different service
    blocked = c.post(f"{BASE}/plans/{pid}/run")
    assert blocked.status_code == 409 and blocked.json()["detail"]["error"] == "connection_changed"


def test_disconnecting_after_planning_blocks_the_run(world):
    client, registry, _ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    registry.disconnect("sales")
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 404


def test_plans_expire(world):
    from agent.platform import PlatformPipeline, SQLiteTaskStore
    now = [1000.0]
    client, *_ = world[0](_plan())
    probe = client()
    base = get_pipeline(probe.app)
    clocked = PlatformPipeline(
        registry=base.registry, store=SQLiteTaskStore(":memory:", clock=lambda: now[0]),
        authorize=base._authorize, clock=lambda: now[0], ttl_seconds=60)
    c = client(pipeline=clocked)
    pid = _create(c).json()["plan_id"]
    now[0] += 61
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 404


def test_the_combined_endpoint_still_works(world):
    client, *_ = world[0](_plan())
    body = client().post(BASE, json={"question": "q"}).json()
    assert body["status"] == "succeeded"


# ------------------------------------------------------- platform behaviour of the report plans
def test_too_many_pending_plans_answer_429(world):
    client, *_ = world[0](_plan(), _plan(), _plan())
    c = client()
    object.__setattr__(get_pipeline(c.app).registry.get("database.report"), "max_active", 2)
    assert _create(c).status_code == 201
    assert _create(c).status_code == 201
    third = _create(c)
    assert third.status_code == 429 and third.json()["detail"]["error"] == "too_many_plans"


def test_plans_survive_a_restart_and_run_once(world):
    """A new pipeline over the same task file (a restarted server) still knows the plan."""
    from agent.platform import PlatformPipeline, SQLiteTaskStore
    client, registry, _ = world[0](_plan())
    first = client()
    pid = _create(first).json()["plan_id"]
    path = registry.work_dir / "platform.sqlite3"
    assert path.exists()
    old = get_pipeline(first.app)
    reborn = PlatformPipeline(registry=old.registry, store=SQLiteTaskStore(path, clock=old._clock),
                              authorize=old._authorize, clock=old._clock)
    second = client(pipeline=reborn)
    ran = second.post(f"{BASE}/plans/{pid}/run")
    assert ran.status_code == 200 and ran.json()["status"] == "succeeded"
    assert second.post(f"{BASE}/plans/{pid}/run").status_code == 409


def test_report_rows_are_never_written_to_the_task_store(world):
    client, registry, _ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    body = c.post(f"{BASE}/plans/{pid}/run").json()
    assert body["rows"]
    stored = get_pipeline(c.app).store.get(pid)
    assert stored.status.value == "completed" and stored.verification["verdict"] == "verified"
    assert "rows" not in stored.result and "rows" not in stored.plan
    import json as _json
    blob = _json.dumps(get_pipeline(c.app).store.audit_records(pid))
    assert "rows" not in blob and "amount" not in blob


def test_a_replay_after_a_restart_has_the_facts_but_no_rows(world):
    from agent.platform import PlatformPipeline, SQLiteTaskStore
    client, registry, _ = world[0](_plan())
    first = client()
    pid = _create(first).json()["plan_id"]
    original = first.post(f"{BASE}/plans/{pid}/run", headers={"Idempotency-Key": "k"}).json()
    old = get_pipeline(first.app)
    reborn = PlatformPipeline(registry=old.registry,
                              store=SQLiteTaskStore(registry.work_dir / "platform.sqlite3", clock=old._clock),
                              authorize=old._authorize, clock=old._clock)
    second = client(pipeline=reborn)
    old.registry.report_outcomes.clear()                       # a new process has an empty cache
    replay = second.post(f"{BASE}/plans/{pid}/run", headers={"Idempotency-Key": "k"}).json()
    assert replay["replayed"] is True and replay["status"] == "succeeded"
    assert replay["workflow_id"] == original["workflow_id"] and replay["plan"] == original["plan"]
    assert replay["rows"] == []


def test_the_run_is_audited_and_verified(world):
    client, *_ = world[0](_plan())
    c = client()
    pid = _create(c).json()["plan_id"]
    c.post(f"{BASE}/plans/{pid}/run")
    events = [r["event"] for r in get_pipeline(c.app).store.audit_records(pid)]
    assert events == ["task_planned", "task_executing", "task_executed", "task_completed"]


def test_a_failed_report_is_recorded_as_failed_not_completed(world):
    client, *_ = world[0](_plan())
    planner_only = client("alice", PERMISSION)
    pid = _create(planner_only).json()["plan_id"]
    planner_only.post(f"{BASE}/plans/{pid}/run")
    stored = get_pipeline(planner_only.app).store.get(pid)
    assert stored.status.value == "failed" and stored.verification["verdict"] == "failed"
