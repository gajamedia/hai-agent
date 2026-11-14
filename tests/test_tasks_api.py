"""The task history: only your own tasks, for operations you may still use, with nothing erased shown."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import time

import pytest

from api.security.authentication import AuthenticatedPrincipal
from tests.test_data_changes_api import (  # noqa: F401
    DELETE, UPDATE, USERS, _call, _plan, world,
)
from tests.test_table_report_plans_api import BASE as REPORTS, _create, _plan as report_answer  # noqa: F401
from tests.test_table_report_plans_api import world as report_world  # noqa: F401

TASKS = "/api/v1/tasks"
HISTORY = "task:history:read"
# every person in these tests may read their history, except where a test says otherwise
for _name in ("alice", "bob"):
    USERS[_name] = dataclasses.replace(USERS[_name], permissions=USERS[_name].permissions + (HISTORY,))
USERS["alice-readonly"] = AuthenticatedPrincipal(
    actor_id="alice", permissions=("report:table:run", HISTORY), auth_time=int(time.time()))
USERS["alice-nohistory"] = AuthenticatedPrincipal(
    actor_id="alice", permissions=("data:change:plan", "data:change:execute"), auth_time=int(time.time()))


def _get(client, path="", user="alice", **params):
    return client.get(f"{TASKS}{path}", headers={"X-Test-User": user}, params=params)


def _run(client, plan, user="alice"):
    return _call(client, "POST", f"/plans/{plan['plan_id']}/run", user, json={"confirm": plan["confirm_phrase"]})


def test_it_needs_a_signed_in_person(world):
    client, *_ = world
    assert client.get(TASKS).status_code == 401


def test_your_tasks_are_listed_newest_first_with_a_label_and_a_line_about_each(world):
    client, provider, *_ = world
    first = _plan(client, provider, UPDATE).json()
    _run(client, first)
    second = _plan(client, provider, DELETE).json()                       # waits for a second person
    body = _get(client).json()
    assert [t["task_id"] for t in body["tasks"]] == [second["plan_id"], first["plan_id"]]
    waiting, done = body["tasks"]
    assert waiting["status"] == "awaiting_approval" and waiting["label"] == "Ubah data"
    assert waiting["title"] == "Hapus data pada tabel sales" and waiting["expires_in_seconds"] > 0
    assert done["status"] == "completed" and done["verification"] == "verified"
    assert done["title"].startswith("Ubah data: ") and done["title"].endswith(" baris di sales")
    assert done["expires_in_seconds"] is None and body["has_more"] is False


def test_the_history_permission_is_required(world):
    client, *_ = world
    for path in ("", "/anything"):
        r = _get(client, path, user="alice-nohistory")
        assert r.status_code == 403 and r.json()["detail"]["error"] == "permission_denied"


def test_the_roles_that_run_features_carry_the_history_permission():
    from api.security.authorization_config import _role_map_for_keycloak
    roles = _role_map_for_keycloak("hai-api")
    for role in ("data-reports-user", "data-writer", "data-schema-creator"):
        assert HISTORY in roles[f"client:hai-api:{role}"], role
    # an approver only approves; having no task history of their own is correct
    assert HISTORY not in roles["client:hai-api:data-change-approver"]


def test_nobody_sees_someone_elses_tasks(world):
    client, provider, *_ = world
    mine = _plan(client, provider, UPDATE).json()
    assert [t["task_id"] for t in _get(client).json()["tasks"]] == [mine["plan_id"]]
    assert _get(client, user="bob").json()["tasks"] == []                  # an approver is not the owner
    assert _get(client, f"/{mine['plan_id']}", user="bob").status_code == 404
    assert _get(client, "/nope").status_code == 404


def test_the_filters_group_the_statuses(world):
    client, provider, *_ = world
    done = _plan(client, provider, UPDATE).json()
    _run(client, done)
    waiting = _plan(client, provider, DELETE).json()
    cancelled = _plan(client, provider, UPDATE).json()
    _call(client, "DELETE", f"/plans/{cancelled['plan_id']}")
    ids = lambda group: {t["task_id"] for t in _get(client, status=group).json()["tasks"]}  # noqa: E731
    assert ids("done") == {done["plan_id"]}
    assert ids("waiting") == {waiting["plan_id"]}
    assert ids("problem") == {cancelled["plan_id"]}
    assert ids("running") == set()
    assert len(_get(client).json()["tasks"]) == 3


@pytest.mark.parametrize("params", [{"status": "everything"}, {"limit": 0}, {"limit": 201}])
def test_bad_filters_are_refused(world, params):
    client, *_ = world
    assert _get(client, **params).status_code == 422


def test_the_list_is_paged(world):
    client, provider, *_ = world
    for _i in range(3):
        _plan(client, provider, UPDATE)
    body = _get(client, limit=2).json()
    assert len(body["tasks"]) == 2 and body["has_more"] is True
    assert _get(client, limit=3).json()["has_more"] is False


def test_the_detail_has_a_timeline_and_only_safe_fields(world):
    client, provider, *_ = world
    plan = _plan(client, provider, DELETE).json()
    _call(client, "POST", f"/approvals/{plan['plan_id']}/approve", "bob")
    _run(client, plan)
    detail = _get(client, f"/{plan['plan_id']}").json()
    assert [e["event"] for e in detail["timeline"]] == [
        "task_planned", "task_approved", "task_executing", "task_executed", "task_completed"]
    approved = detail["timeline"][1]
    assert approved["actor"] == "bob" and approved["self_approved"] is False
    assert detail["timeline"][-1]["verdict"] == "verified" and detail["timeline"][2]["to"] == "executing"
    assert detail["approved_by"] == "bob"
    assert detail["result"] == {"rows_affected": 2, "operation": "delete", "table": "sales"}
    assert detail["download"] is None
    assert all("plan_sha256" not in e and "approval" not in e and "risk" not in e for e in detail["timeline"])


def test_what_the_pipeline_erased_is_not_shown_and_not_missed(world):
    client, provider, db, _ = world
    secret = {**UPDATE, "set": {"product": "rahasia-99"}}
    plan = _plan(client, provider, secret).json()
    assert "rahasia-99" not in json.dumps(_get(client).json())             # never in the list, even while waiting
    _run(client, plan)
    for body in (_get(client).json(), _get(client, f"/{plan['plan_id']}").json()):
        assert "rahasia-99" not in json.dumps(body)
    cancelled = _plan(client, provider, secret).json()
    _call(client, "DELETE", f"/plans/{cancelled['plan_id']}")
    shown = _get(client, f"/{cancelled['plan_id']}").json()
    assert shown["status"] == "cancelled" and "rincian sudah dihapus" in shown["title"]


def test_losing_the_permission_hides_the_tasks_too(world):
    client, provider, *_ = world
    plan = _plan(client, provider, UPDATE).json()
    assert len(_get(client).json()["tasks"]) == 1
    assert _get(client, user="alice-readonly").json()["tasks"] == []       # same person, no change-plan right
    assert _get(client, f"/{plan['plan_id']}", user="alice-readonly").status_code == 404


def test_a_failed_change_shows_why_it_is_failed_not_how(world):
    client, provider, db, _ = world
    plan = _plan(client, provider, UPDATE).json()
    with sqlite3.connect(db) as c:
        c.execute("UPDATE sales SET region='north' WHERE id IN (SELECT id FROM sales WHERE region='west' LIMIT 1)")
    _run(client, plan)
    task = _get(client, f"/{plan['plan_id']}").json()
    assert task["status"] == "failed" and "check the target" in task["error"]
    assert task["result"] is None and task["verification"] is None          # nothing was claimed
    assert task["timeline"][-1]["event"] == "task_failed" and "error" not in task["timeline"][-1]


# ---- a report: the file can be downloaded from the task ------------------------------------------

def test_a_report_task_shows_its_question_and_links_to_the_file(report_world):
    build, _tmp = report_world
    make_client, *_ = build(report_answer())
    client = make_client("alice", "report:table:run", "business:database:read", HISTORY)
    created = _create(client, "omzet per wilayah").json()
    ran = client.post(f"{REPORTS}/plans/{created['plan_id']}/run").json()
    assert ran["status"] == "succeeded"
    listed = client.get(TASKS).json()["tasks"]
    assert listed[0]["label"] == "Laporan tabel" and listed[0]["title"].startswith("omzet per wilayah")
    assert listed[0]["status"] == "completed"
    detail = client.get(f"{TASKS}/{created['plan_id']}").json()
    assert detail["download"]["url"].endswith(f"/reports/{ran['workflow_id']}/download")
    assert "rows" not in json.dumps(detail["result"]) and "path" not in json.dumps(detail["download"])
    assert client.get(detail["download"]["url"]).status_code == 200
