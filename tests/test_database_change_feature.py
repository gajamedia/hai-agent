"""database.change on the platform: what the pipeline adds to the data-change flow."""
from __future__ import annotations

import json
import sqlite3

import pytest

from agent.platform import PlatformPipeline, SQLiteTaskStore, TaskStatus
from agent.workflow.data_change import ChangeRequest
from api.features.database_changes import change_request_from_json, change_request_to_json
from api.platform_runtime import get_pipeline
from tests.test_data_changes_api import (  # noqa: F401
    DELETE, SELECTION, UPDATE, USERS, _audit, _call, _count, _plan, world,
)


def _pipe(client):
    return get_pipeline(client.app)


def _everything_stored(client):
    store = _pipe(client).store
    tasks = [store.get(t.task_id).__dict__ for t in store.list_by_status(list(TaskStatus))]
    return json.dumps(tasks, default=str) + json.dumps(store.audit_records())


# ---- codec -------------------------------------------------------------------

@pytest.mark.parametrize("request_", [
    ChangeRequest("update", "sales", set_values=(("amount", 1), ("product", None)),
                  conditions=(("region", "eq", "west"), ("id", "in", (1, 2)), ("amount", "between", (1, 9))),
                  max_rows=7, title="t"),
    ChangeRequest("delete", "sales", conditions=(("id", "in", (3, 4)),)),
    ChangeRequest("insert", "sales", rows=((("product", "w"), ("amount", 3)), (("product", "x"), ("amount", 4)))),
])
def test_a_change_request_survives_json_unchanged(request_):
    assert change_request_from_json(json.loads(json.dumps(change_request_to_json(request_)))) == request_


# ---- the run is audited and verified ----------------------------------------

def test_an_update_is_verified_by_reading_the_table_again(world):
    client, provider, db, tmp = world
    plan = _plan(client, provider, UPDATE).json()
    done = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]}).json()
    assert done["status"] == "committed" and done["verification"] == "verified" and done["replayed"] is False
    task = _pipe(client).store.get(plan["plan_id"])
    assert task.status is TaskStatus.COMPLETED
    assert task.verification["verdict"] == "verified"
    assert task.verification["evidence"]["method"] == "reread_after_commit"
    events = [r["event"] for r in _pipe(client).store.audit_records(plan["plan_id"])]
    assert events == ["task_planned", "task_executing", "task_executed", "task_completed"]


def test_the_platform_audit_records_who_approved_and_the_old_audit_is_still_written(world):
    client, provider, db, tmp = world
    plan = _plan(client, provider, DELETE).json()
    _call(client, "POST", f"/approvals/{plan['plan_id']}/approve", "bob")
    _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    records = _pipe(client).store.audit_records(plan["plan_id"])
    assert [r["event"] for r in records] == [
        "task_planned", "task_approved", "task_executing", "task_executed", "task_completed"]
    assert [r["actor"] for r in records][:2] == ["alice", "bob"]
    assert [e.get("event") or e.get("status") for e in _audit(tmp)] == ["planned", "approved", "committed"]


def test_a_change_that_no_longer_matches_what_was_shown_is_refused_and_recorded(world):
    client, provider, db, tmp = world
    plan = _plan(client, provider, UPDATE).json()
    with sqlite3.connect(db) as c:                                   # the data moves on after the preview
        c.execute("UPDATE sales SET region='north' WHERE id IN (SELECT id FROM sales WHERE region='west' LIMIT 1)")
    r = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "change_refused"
    assert _count(db, "amount = 1") == 0
    assert _pipe(client).store.get(plan["plan_id"]).status is TaskStatus.FAILED
    assert _audit(tmp)[-1]["status"] == "rolled_back"
    again = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert again.status_code == 409 and again.json()["detail"]["error"] == "plan_not_runnable"


# ---- business data does not stay in the task store --------------------------

def test_the_values_are_erased_from_the_store_when_the_change_ends(world):
    client, provider, db, _ = world
    secret = {"supported": True, "operation": "update", "table": "sales", "set": {"product": "rahasia-42"},
              "conditions": [{"column": "region", "operator": "eq", "value": "west"}]}
    plan = _plan(client, provider, secret).json()
    assert "rahasia-42" in _everything_stored(client)                # needed while it waits to be run
    assert "rahasia-42" not in json.dumps(_pipe(client).store.audit_records())     # but never in the audit
    _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert "rahasia-42" not in _everything_stored(client)
    task = _pipe(client).store.get(plan["plan_id"])
    assert task.plan["scrubbed"] is True and task.result["rows_affected"] > 0


def test_a_cancelled_or_rejected_change_leaves_no_values_behind(world):
    client, provider, db, _ = world
    secret = {**DELETE, "conditions": [{"column": "product", "operator": "eq", "value": "rahasia-77"}]}
    with sqlite3.connect(db) as c:
        c.execute("UPDATE sales SET product='rahasia-77' WHERE id IN (1, 2)")
    cancelled = _plan(client, provider, secret).json()
    assert _call(client, "DELETE", f"/plans/{cancelled['plan_id']}").json()["status"] == "cancelled"
    rejected = _plan(client, provider, secret).json()
    assert _call(client, "POST", f"/approvals/{rejected['plan_id']}/reject", "bob").status_code == 200
    assert "rahasia-77" not in _everything_stored(client)
    assert _call(client, "GET", f"/plans/{cancelled['plan_id']}").status_code == 404


# ---- durable plans and approvals ---------------------------------------------

def test_a_plan_waiting_for_approval_survives_a_restart(world):
    client, provider, db, tmp = world
    plan = _plan(client, provider, DELETE).json()
    old = _pipe(client)
    reborn = PlatformPipeline(
        registry=old.registry, store=SQLiteTaskStore(tmp / "work" / "platform.sqlite3", clock=old._clock),
        authorize=old._authorize, clock=old._clock, allow_self_approval=old.allow_self_approval,
        step_up_max_age_seconds=old.step_up_max_age)
    client.app.state.platform_pipeline = reborn
    assert [a["plan_id"] for a in _call(client, "GET", "/approvals", "bob").json()["approvals"]] == [plan["plan_id"]]
    assert _call(client, "POST", f"/approvals/{plan['plan_id']}/approve", "bob").status_code == 200
    done = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert done.status_code == 200 and done.json()["approved_by"] == "bob" and _count(db) == 28


def test_a_change_interrupted_by_a_restart_is_marked_not_repeated(world):
    client, provider, db, tmp = world
    plan = _plan(client, provider, UPDATE).json()
    store = _pipe(client).store
    task = store.get(plan["plan_id"])
    store.transition(task.task_id, expected_version=task.version, new_status=TaskStatus.EXECUTING,
                     actor="alice", audit_event="task_executing", audit_detail={})
    recovered = _pipe(client).recover_interrupted()
    assert [t.status for t in recovered] == [TaskStatus.FAILED] and "unknown" in recovered[0].error
    assert _count(db, "amount = 1") == 0
    r = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert r.status_code == 409


# ---- the approvals a plan was made under --------------------------------------

def test_changing_what_may_be_written_after_planning_blocks_the_run(world):
    client, provider, db, tmp = world
    plan = _plan(client, provider, UPDATE).json()
    narrower = {"sales": {**SELECTION["tables"]["sales"],
                          "writable": {"operations": ["update"], "columns": ["amount"]}}}
    registry = client.app.state.database_runtime_registry
    registry.activate("sales", narrower, actor_id="alice")
    r = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "connection_changed"
    assert _count(db, "amount = 1") == 0


def test_the_token_is_the_same_for_the_same_approvals_and_differs_when_they_change(world):
    client, *_ = world
    registry = client.app.state.database_runtime_registry
    first = registry.change_token("sales")
    registry.activate("sales", SELECTION["tables"], actor_id="alice")        # the same approvals again
    assert registry.change_token("sales") == first
    registry.activate("sales", {"sales": {**SELECTION["tables"]["sales"],
                                          "writable": {"operations": ["update"], "columns": ["amount"]}}},
                      actor_id="alice")
    assert registry.change_token("sales") != first


# ---- limits -------------------------------------------------------------------

def test_too_many_waiting_changes_answer_429(world):
    client, provider, *_ = world
    _plan(client, provider, UPDATE)
    object.__setattr__(_pipe(client).registry.get("database.change"), "max_active", 1)
    r = _plan(client, provider, UPDATE)
    assert r.status_code == 429 and r.json()["detail"]["error"] == "too_many_plans"
