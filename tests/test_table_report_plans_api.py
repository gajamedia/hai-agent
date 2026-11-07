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
from api.table_report_plans import PlanConflict, PlanNotFound, PlanStore
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

        def client(actor="alice", *permissions, store=None):
            app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                             database_runtime_registry_factory=lambda _a: registry)
            if store is not None:
                app.state.table_report_plans = store
            elif "store" in shared:
                app.state.table_report_plans = shared["store"]
            app.add_middleware(_Attach, principal=_who(actor, *(permissions or (PERMISSION, READ))))
            c = TestClient(app)
            c.__enter__()
            made.append(c)
            shared.setdefault("store", getattr(app.state, "table_report_plans", None))
            return c
        # make every client share one PlanStore, as one running server would
        shared["store"] = PlanStore()
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
    now = [1000.0]
    store = PlanStore(ttl_seconds=60, clock=lambda: now[0])
    client, *_ = world[0](_plan())
    c = client(store=store)
    pid = _create(c).json()["plan_id"]
    now[0] += 61
    assert c.post(f"{BASE}/plans/{pid}/run").status_code == 404


def test_the_combined_endpoint_still_works(world):
    client, *_ = world[0](_plan())
    body = client().post(BASE, json={"question": "q"}).json()
    assert body["status"] == "succeeded"


# ---------------------------------------------------------------- store unit
def _store_plan(store, owner="a"):
    return store.create(owner=owner, database="d", question="q", plan_result=object(),
                        summary={}, service=object())


def test_store_state_machine_and_caps():
    store = PlanStore(max_plans=2)
    p = _store_plan(store)
    assert store.begin_run(p.plan_id, "a", "k")[0] == "run"
    with pytest.raises(PlanConflict):
        store.begin_run(p.plan_id, "a", "k")        # still running
    store.finish(p, {"ok": 1})
    assert store.begin_run(p.plan_id, "a", "k")[0] == "replay"
    with pytest.raises(PlanConflict):
        store.begin_run(p.plan_id, "a", None)
    with pytest.raises(PlanNotFound):
        store.begin_run(p.plan_id, "intruder", "k")
    _store_plan(store)
    with pytest.raises(PlanConflict):
        _store_plan(store)                           # cap of 2 pending plans


def test_an_interrupted_run_is_not_silently_repeated():
    store = PlanStore()
    p = _store_plan(store)
    store.begin_run(p.plan_id, "a", "k")
    store.mark_unknown(p)
    with pytest.raises(PlanConflict):
        store.begin_run(p.plan_id, "a", "k")


def test_release_allows_a_retry_only_when_nothing_started():
    store = PlanStore()
    p = _store_plan(store)
    store.begin_run(p.plan_id, "a", None)
    store.release(p)
    assert store.begin_run(p.plan_id, "a", None)[0] == "run"
