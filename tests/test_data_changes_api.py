"""Data changes over HTTP: permissions, fresh login, exact phrase, second approver, the user's row limit."""
from __future__ import annotations

import dataclasses
import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from api.database_engines import SQLiteEngine
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_table_report_job_e2e import _make_db, _sample_rows

DB, CHANGES = "/api/v1/database-connections", "/api/v1/data-changes"
MANAGE, PLAN, EXECUTE, APPROVE = ("database:connection:manage", "data:change:plan",
                                  "data:change:execute", "data:change:approve")
SELECTION = {"tables": {"sales": {
    "columns": ["id", "region", "product", "amount"], "filterable_columns": ["id", "region", "product"],
    "key_column": "id",
    "writable": {"operations": ["insert", "update", "delete"], "columns": ["product", "amount"]}}}}


class WritableSQLite(SQLiteEngine):
    """The SQLite engine plus a writer, so the whole HTTP flow can run without a MySQL server."""

    lock_clause = ""                      # SQLite has no SELECT ... FOR UPDATE

    def has_writer(self, source):
        return True

    def open_writer(self, source):
        path = self.data_root / source
        return lambda: sqlite3.connect(path)


class Provider:
    name = "scripted"

    def __init__(self):
        self.answers = []

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


USERS = {
    "alice": AuthenticatedPrincipal(actor_id="alice", permissions=(MANAGE, PLAN, EXECUTE),
                                    auth_time=int(time.time())),
    "alice-stale": AuthenticatedPrincipal(actor_id="alice", permissions=(PLAN, EXECUTE),
                                          auth_time=int(time.time()) - 3600),
    "alice-notime": AuthenticatedPrincipal(actor_id="alice", permissions=(PLAN, EXECUTE)),
    "bob": AuthenticatedPrincipal(actor_id="bob", permissions=(APPROVE,), auth_time=int(time.time())),
    "bob-stale": AuthenticatedPrincipal(actor_id="bob", permissions=(APPROVE,),
                                        auth_time=int(time.time()) - 3600),
    "solo": AuthenticatedPrincipal(actor_id="solo", permissions=(MANAGE, PLAN, EXECUTE, APPROVE),
                                   auth_time=int(time.time())),
    "solo-stale": AuthenticatedPrincipal(actor_id="solo", permissions=(PLAN, EXECUTE, APPROVE),
                                         auth_time=int(time.time()) - 3600),
    "reader": AuthenticatedPrincipal(actor_id="carol", permissions=("report:table:run",),
                                     auth_time=int(time.time())),
}


class _AsUser:
    """Pretend the token belonged to the user named in the X-Test-User header."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            name = headers.get(b"x-test-user", b"").decode()
            if name:
                principal = USERS[name]
                if principal.auth_time is not None and "stale" not in name:
                    # a fresh login at the moment of the request, however long the test run has lasted
                    principal = dataclasses.replace(principal, auth_time=int(time.time()))
                scope.setdefault("state", {})["authenticated_principal"] = principal
        await self.app(scope, receive, send)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.delenv("HAI_DATA_CHANGE_APPROVAL", raising=False)
    monkeypatch.delenv("HAI_DATA_CHANGE_MAX_ROWS", raising=False)
    data = tmp_path / "data"
    data.mkdir()
    _make_db(data / "sales.sqlite3", _sample_rows(30))
    provider = Provider()
    app = create_app(
        authorization_policy_factory=lambda _a: AuthorizationPolicy(),
        database_runtime_registry_factory=lambda a: DatabaseRuntimeRegistry(
            data_root=data, work_dir=tmp_path / "work",
            authorization_policy=a.state.authorization_policy,
            llm_provider_factory=lambda: provider, engines={"sqlite": WritableSQLite(data)}))
    app.add_middleware(_AsUser)
    with TestClient(app) as client:
        assert client.post(DB, json={"name": "sales", "filename": "sales.sqlite3"},
                           headers={"X-Test-User": "alice"}).status_code == 200
        r = client.post(f"{DB}/sales/approve", json=SELECTION, headers={"X-Test-User": "alice"})
        assert r.status_code == 200, r.text
        yield client, provider, data / "sales.sqlite3", tmp_path


def _call(client, method, path, user="alice", **kw):
    return client.request(method, f"{CHANGES}{path}", headers={"X-Test-User": user, **kw.pop("headers", {})}, **kw)


def _plan(client, provider, answer, rows=100, user="alice"):
    provider.answers.append(json.dumps(answer))
    return _call(client, "POST", "/plans", user, json={"question": "q", "max_rows": rows})


UPDATE = {"supported": True, "operation": "update", "table": "sales", "set": {"amount": 1},
          "conditions": [{"column": "region", "operator": "eq", "value": "west"}]}
DELETE = {"supported": True, "operation": "delete", "table": "sales",
          "conditions": [{"column": "id", "operator": "in", "value": [1, 2]}]}


def _count(path, where="1=1"):
    with sqlite3.connect(path) as c:
        return c.execute(f"SELECT COUNT(*) FROM sales WHERE {where}").fetchone()[0]


def _audit(tmp_path):
    return [json.loads(x) for x in (tmp_path / "work" / "audit" / "changes-sales.jsonl")
            .read_text("utf-8").splitlines()]


# ------------------------------------------------------------------ the happy path
def test_update_needs_the_exact_phrase_and_changes_only_the_previewed_rows(world):
    client, provider, db, tmp = world
    west = _count(db, "region = 'west'")
    body = _plan(client, provider, UPDATE).json()
    assert body["status"] == "planned" and body["expected_rows"] == west
    assert body["confirm_phrase"] == f"UBAH {west} BARIS sales" and body["requires_approval"] is False
    assert {r["amount"] for r in body["preview"]["after"]} == {1}
    assert _count(db, "amount = 1") == 0                                   # planning changed nothing

    wrong = _call(client, "POST", f"/plans/{body['plan_id']}/run", json={"confirm": "ya"})
    assert wrong.status_code == 422 and wrong.json()["detail"]["error"] == "confirmation_mismatch"
    assert _count(db, "amount = 1") == 0

    done = _call(client, "POST", f"/plans/{body['plan_id']}/run", json={"confirm": body["confirm_phrase"]},
                 headers={"Idempotency-Key": "k1"})
    assert done.status_code == 200 and done.json()["rows_affected"] == west
    assert _count(db, "amount = 1") == west
    again = _call(client, "POST", f"/plans/{body['plan_id']}/run", json={"confirm": body["confirm_phrase"]},
                  headers={"Idempotency-Key": "k1"})
    assert again.json()["replayed"] is True and _count(db, "amount = 1") == west
    events = [e.get("event") or e.get("status") for e in _audit(tmp)]
    assert events == ["planned", "committed"]


def test_when_no_llm_provider_answers_the_user_gets_a_clear_retry_message(world, caplog):
    client, provider, _db, _tmp = world

    def timeout(_request):
        from agent.llm.fallback_provider import AllProvidersFailedError
        raise AllProvidersFailedError([("opencode", TimeoutError("timed out after 60 s")),
                                       ("claude-code", TimeoutError("timed out after 120 s"))])
    provider.complete = timeout
    with caplog.at_level("WARNING"):
        response = _call(client, "POST", "/plans", "alice", json={"question": "q", "max_rows": 1})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "llm_unavailable" and "did not answer in time" in detail["message"]
    assert "Nothing was changed" in detail["message"] and "opencode" not in response.text
    assert "timed out after 60 s" in caplog.text                       # the operator still sees why
    assert "Traceback" not in caplog.text


def test_a_plan_runs_once(world):
    client, provider, db, _ = world
    body = _plan(client, provider, UPDATE).json()
    run = {"confirm": body["confirm_phrase"]}
    assert _call(client, "POST", f"/plans/{body['plan_id']}/run", json=run).status_code == 200
    second = _call(client, "POST", f"/plans/{body['plan_id']}/run", json=run)
    assert second.status_code == 409


def test_someone_else_cannot_run_my_plan(world):
    client, provider, _, _ = world
    body = _plan(client, provider, UPDATE).json()
    other = AuthenticatedPrincipal(actor_id="mallory", permissions=(PLAN, EXECUTE), auth_time=int(time.time()))
    USERS["mallory"] = other
    r = _call(client, "POST", f"/plans/{body['plan_id']}/run", "mallory", json={"confirm": body["confirm_phrase"]})
    assert r.status_code == 404


# ------------------------------------------------------------------ fresh login
@pytest.mark.parametrize("user", ["alice-stale", "alice-notime"])
def test_an_old_or_unknown_login_cannot_execute(world, user):
    client, provider, db, _ = world
    body = _plan(client, provider, UPDATE).json()
    r = _call(client, "POST", f"/plans/{body['plan_id']}/run", user, json={"confirm": body["confirm_phrase"]})
    assert r.status_code == 403 and r.json()["detail"]["error"] == "step_up_required"
    assert _count(db, "amount = 1") == 0


# ------------------------------------------------------------------ the user's row limit
def test_a_limit_lower_than_the_matching_rows_refuses_the_plan(world):
    client, provider, db, _ = world
    body = _plan(client, provider, UPDATE, rows=2).json()
    assert body["status"] == "refused" and "more than 2 rows" in body["reason"] and body["plan_id"] is None


def test_the_server_ceiling_caps_what_a_user_may_ask_for(world, monkeypatch):
    client, provider, _, _ = world
    monkeypatch.setenv("HAI_DATA_CHANGE_MAX_ROWS", "50")
    assert _call(client, "POST", "/plans", json={"question": "q", "max_rows": 51}).status_code == 422
    assert _call(client, "GET", "/config").json()["max_rows_ceiling"] == 50


@pytest.mark.parametrize("rows", [0, -1, True, "5", 1.5, None])
def test_the_row_limit_must_be_a_positive_integer(world, rows):
    client, _, _, _ = world
    assert _call(client, "POST", "/plans", json={"question": "q", "max_rows": rows}).status_code == 422


# ------------------------------------------------------------------ approval by a second person
def test_delete_waits_for_a_different_person(world):
    client, provider, db, tmp = world
    body = _plan(client, provider, DELETE).json()
    assert body["requires_approval"] is True
    run = {"confirm": body["confirm_phrase"]}
    assert _call(client, "POST", f"/plans/{body['plan_id']}/run", json=run).status_code == 409
    assert _count(db) == 30
    # the owner can never approve their own change (even if they also held the approver permission)
    USERS["alice-approver"] = AuthenticatedPrincipal(
        actor_id="alice", permissions=(APPROVE,), auth_time=int(time.time()))
    own = _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "alice-approver")
    assert own.status_code == 409
    assert _call(client, "GET", "/approvals", "alice-approver").json()["approvals"] == []
    # a stale login cannot approve; a fresh one can
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "bob-stale").status_code == 403
    listed = _call(client, "GET", "/approvals", "bob").json()["approvals"]
    assert [a["plan_id"] for a in listed] == [body["plan_id"]] and "confirm_phrase" not in listed[0]
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "bob").status_code == 200
    done = _call(client, "POST", f"/plans/{body['plan_id']}/run", json=run)
    assert done.status_code == 200 and done.json()["approved_by"] == "bob"
    assert _count(db) == 28
    assert [e.get("event") or e.get("status") for e in _audit(tmp)] == ["planned", "approved", "committed"]
    assert _audit(tmp)[-1]["approver"] == "bob" and len(_audit(tmp)[-1]["before_image"]) == 2


def test_one_person_cannot_approve_their_own_delete_unless_the_server_allows_it(world, monkeypatch):
    client, provider, db, _ = world
    monkeypatch.delenv("HAI_DATA_CHANGE_SELF_APPROVAL", raising=False)
    body = _plan(client, provider, DELETE, user="solo").json()
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "solo").status_code == 409
    assert _call(client, "GET", "/approvals", "solo").json()["approvals"] == []
    assert _call(client, "GET", "/config", "solo").json()["self_approval"] is False
    for value in ("", "false", "0", "no", "maybe"):                                  # anything else stays off
        monkeypatch.setenv("HAI_DATA_CHANGE_SELF_APPROVAL", value)
        assert _call(client, "GET", "/config", "solo").json()["self_approval"] is False
    assert _count(db) == 30


def test_with_self_approval_on_one_person_plans_approves_and_runs_a_delete_with_a_fresh_login_each_time(world, monkeypatch):
    client, provider, db, tmp = world
    monkeypatch.setenv("HAI_DATA_CHANGE_SELF_APPROVAL", "true")
    assert _call(client, "GET", "/config", "solo").json()["self_approval"] is True
    body = _plan(client, provider, DELETE, user="solo").json()
    run = {"confirm": body["confirm_phrase"]}
    assert _call(client, "POST", f"/plans/{body['plan_id']}/run", "solo", json=run).status_code == 409   # not approved yet
    listed = _call(client, "GET", "/approvals", "solo").json()["approvals"]
    assert [(a["plan_id"], a["own"]) for a in listed] == [(body["plan_id"], True)]
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "solo-stale").status_code == 403   # fresh login
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "alice").status_code == 403        # needs the role
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "solo").status_code == 200
    done = _call(client, "POST", f"/plans/{body['plan_id']}/run", "solo", json=run)
    assert done.status_code == 200 and done.json()["approved_by"] == "solo" and _count(db) == 28
    events = _audit(tmp)
    approved = [e for e in events if e.get("event") == "approved"][0]
    assert approved["approver"] == "solo" and approved["owner"] == "solo" and approved["self_approval"] is True
    assert events[-1]["status"] == "committed" and len(events[-1]["before_image"]) == 2


def test_with_self_approval_on_a_second_person_is_still_listed_and_still_works(world, monkeypatch):
    client, provider, db, _ = world
    monkeypatch.setenv("HAI_DATA_CHANGE_SELF_APPROVAL", "true")
    body = _plan(client, provider, DELETE).json()                                   # alice's plan
    listed = _call(client, "GET", "/approvals", "bob").json()["approvals"]
    assert [(a["plan_id"], a["own"]) for a in listed] == [(body["plan_id"], False)]
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/approve", "bob").status_code == 200


def test_a_rejected_change_cannot_run(world):
    client, provider, db, _ = world
    body = _plan(client, provider, DELETE).json()
    assert _call(client, "POST", f"/approvals/{body['plan_id']}/reject", "bob").status_code == 200
    r = _call(client, "POST", f"/plans/{body['plan_id']}/run", json={"confirm": body["confirm_phrase"]})
    assert r.status_code == 409 and _count(db) == 30


@pytest.mark.parametrize("mode,operation,needs", [
    ("none", DELETE, False), ("all", UPDATE, True), ("delete", UPDATE, False), ("bogus", DELETE, True)])
def test_the_approval_rule_is_configurable_and_fails_safe(world, monkeypatch, mode, operation, needs):
    client, provider, _, _ = world
    monkeypatch.setenv("HAI_DATA_CHANGE_APPROVAL", mode)
    assert _plan(client, provider, operation).json()["requires_approval"] is needs


# ------------------------------------------------------------------ who may do what
def test_reading_permissions_do_not_allow_changing(world):
    client, _, _, _ = world
    assert _call(client, "POST", "/plans", "reader", json={"question": "q", "max_rows": 5}).status_code == 403
    assert _call(client, "GET", "/approvals", "alice").status_code == 403    # a writer is not an approver
    assert _call(client, "POST", "/plans/x/run", "bob", json={"confirm": "x"}).status_code == 403


def test_a_change_outside_what_was_approved_is_unsupported_before_anything_runs(world):
    client, provider, db, _ = world
    east = _count(db, "region = 'east'")
    bad = {**UPDATE, "set": {"region": "east"}}                       # region is not writable
    provider.answers += [json.dumps(bad), json.dumps(bad)]
    r = _call(client, "POST", "/plans", json={"question": "q", "max_rows": 100})
    assert r.json()["status"] == "unsupported" and "not writable" in r.json()["reason"]
    assert _count(db, "region = 'east'") == east                        # nothing executed


def test_a_connection_without_a_writer_cannot_be_approved_for_writing(tmp_path):
    data = tmp_path / "d"
    data.mkdir()
    _make_db(data / "sales.sqlite3", _sample_rows(5))
    registry = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: Provider())               # the plain, read-only SQLite engine
    registry.connect("sales", engine="sqlite", source="sales.sqlite3", actor_id="alice")
    with pytest.raises(ValueError, match="no writer account"):
        registry.activate("sales", SELECTION["tables"], actor_id="alice")
    assert registry.list()[0]["status"] == "connected_pending_approval"


def test_a_plan_can_be_shown_again_but_only_to_its_owner(world):
    client, provider, _, _ = world
    body = _plan(client, provider, UPDATE).json()
    again = _call(client, "GET", f"/plans/{body['plan_id']}")
    assert again.status_code == 200 and again.json()["confirm_phrase"] == body["confirm_phrase"]
    USERS["mallory2"] = AuthenticatedPrincipal(actor_id="mallory2", permissions=(PLAN,), auth_time=int(time.time()))
    assert _call(client, "GET", f"/plans/{body['plan_id']}", "mallory2").status_code == 404
    assert _call(client, "GET", "/plans/unknown").status_code == 404
