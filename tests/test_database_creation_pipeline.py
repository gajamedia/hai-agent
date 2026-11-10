"""Creating a database through the platform pipeline, with a stand-in server (no MySQL needed).

The live MySQL tests in test_database_creation.py are the real proof; these cover the pipeline wiring
(durable task, step-up, exact phrase, replay, verification, history) on every run.
"""
from __future__ import annotations

import dataclasses
import json
import re
import time

import pytest
from fastapi.testclient import TestClient

from agent.llm.service import LLMService
from agent.workflow.database_creation import DatabaseCreationService, DatabaseDesigner
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_data_changes_api import USERS, _AsUser
from tests.test_database_creation import GOOD, Scripted

CREATE, HISTORY = "database:create:manage", "task:history:read"
BASE = "/api/v1/database-creation"
USERS["creator"] = AuthenticatedPrincipal(actor_id="creator", permissions=(CREATE, HISTORY),
                                          auth_time=int(time.time()))
USERS["creator-stale"] = dataclasses.replace(USERS["creator"], auth_time=int(time.time()) - 3600)
USERS["nocreate"] = AuthenticatedPrincipal(actor_id="nocreate", permissions=(), auth_time=int(time.time()))
USERS["other"] = AuthenticatedPrincipal(actor_id="other", permissions=(CREATE,), auth_time=int(time.time()))


class Server:
    """A stand-in database server: remembers what was created, answers the questions the service asks."""

    def __init__(self):
        self.databases: dict[str, set[str]] = {}
        self.fail_on = None
        self.hide_tables = False
        self.refuse_questions = False
        self.statements: list[str] = []

    def connection(self):
        return _Connection(self)


class _Connection:
    def __init__(self, server):
        self.s = server

    def cursor(self):
        return _Cursor(self.s)

    def close(self):
        pass


class _Cursor:
    def __init__(self, server):
        self.s, self.rows = server, []

    def execute(self, sql, params=()):
        if sql.startswith("SELECT"):
            if self.s.refuse_questions and self.s.statements:
                raise RuntimeError("cannot ask")
            if sql.startswith("SELECT SCHEMA_NAME"):
                self.rows = [(params[0],)] if params[0] in self.s.databases else []
            else:
                tables = [] if self.s.hide_tables else sorted(self.s.databases.get(params[0], ()))
                self.rows = [(t,) for t in tables]
            return
        if self.s.fail_on and self.s.fail_on in sql:
            raise RuntimeError("boom")
        self.s.statements.append(sql)
        made = re.match(r"CREATE DATABASE `(\w+)`", sql)
        if made:
            self.s.databases[made.group(1)] = set()
        table = re.match(r"CREATE TABLE `(\w+)`\.`(\w+)`", sql)
        if table:
            self.s.databases[table.group(1)].add(table.group(2))

    def fetchall(self):
        return self.rows

    def close(self):
        pass


@pytest.fixture
def world(tmp_path):
    server = Server()
    designs = Scripted()
    service = DatabaseCreationService(
        designer=DatabaseDesigner(LLMService(designs), prefix="hai_"), connect=server.connection,
        profile="loc", audit_path=tmp_path / "creations.jsonl")
    data = tmp_path / "data"
    data.mkdir()
    registry = DatabaseRuntimeRegistry(
        data_root=data, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: designs, engines={})
    registry.database_creator = lambda profile=None: service
    registry.creatable_profiles = lambda: [{"name": "loc", "prefix": "hai_", "type": "mysql"}]
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda _a: registry)
    app.add_middleware(_AsUser)
    with TestClient(app) as client:
        yield client, designs, server, tmp_path


def _call(client, method, path, user="creator", **kw):
    return client.request(method, f"{BASE}{path}", headers={"X-Test-User": user, **kw.pop("headers", {})}, **kw)


def _plan(client, designs, answer=GOOD, user="creator"):
    designs.answers.append(json.dumps(answer))
    return _call(client, "POST", "/plans", user, json={"question": "buat toko"})


def test_planning_creates_nothing_and_shows_the_statements_and_the_phrase(world):
    client, designs, server, _ = world
    body = _plan(client, designs).json()
    assert body["status"] == "planned" and body["state"] == "planned"
    assert body["confirm_phrase"] == "BUAT DATABASE hai_toko" and body["mode"] == "create_database"
    assert len(body["statements"]) == 2 and body["profile"] == "loc" and body["expires_in_seconds"] > 0
    assert server.databases == {}


def test_running_needs_a_fresh_login_and_the_exact_phrase_then_creates_and_verifies(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    run = f"/plans/{plan['plan_id']}/run"
    stale = _call(client, "POST", run, "creator-stale", json={"confirm": plan["confirm_phrase"]})
    assert stale.status_code == 403 and stale.json()["detail"]["error"] == "step_up_required"
    wrong = _call(client, "POST", run, json={"confirm": "ya"})
    assert wrong.status_code == 422 and wrong.json()["detail"]["error"] == "confirmation_mismatch"
    assert server.databases == {}
    done = _call(client, "POST", run, json={"confirm": plan["confirm_phrase"]})
    assert done.status_code == 200, done.text
    assert done.json() == {"status": "created", "database": "hai_toko", "tables": ["tb_bahan"],
                           "profile": "loc", "mode": "create_database", "verification": "verified",
                           "replayed": False}
    assert server.databases == {"hai_toko": {"tb_bahan"}}


def test_the_same_key_replays_without_creating_twice(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    run = f"/plans/{plan['plan_id']}/run"
    first = _call(client, "POST", run, json={"confirm": plan["confirm_phrase"]}, headers={"Idempotency-Key": "k1"})
    again = _call(client, "POST", run, json={"confirm": plan["confirm_phrase"]}, headers={"Idempotency-Key": "k1"})
    assert first.json()["replayed"] is False and again.json()["replayed"] is True
    assert len([s for s in server.statements if s.startswith("CREATE DATABASE")]) == 1
    # without the key a finished plan is never run again
    assert _call(client, "POST", run, json={"confirm": plan["confirm_phrase"]}).status_code == 409


def test_a_creation_the_server_cannot_confirm_is_not_called_done(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    server.hide_tables = True
    done = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert done.status_code == 200 and done.json()["verification"] == "unverified"
    task = client.get("/api/v1/tasks/" + plan["plan_id"], headers={"X-Test-User": "creator"}).json()
    assert task["status"] == "unverified"


def test_a_server_that_cannot_be_asked_is_unverified_not_failed_or_verified(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    server.refuse_questions = True
    # questions are refused only once statements have run, so planning and the pre-run check still work
    done = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert done.status_code == 200 and done.json()["verification"] == "unverified"
    assert server.databases == {"hai_toko": {"tb_bahan"}}      # it was created; we just could not confirm it


def test_a_failure_part_way_is_reported_and_the_plan_is_not_run_again(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    server.fail_on = "CREATE TABLE"
    run = f"/plans/{plan['plan_id']}/run"
    failed = _call(client, "POST", run, json={"confirm": plan["confirm_phrase"]})
    assert failed.status_code == 503 and failed.json()["detail"]["error"] == "creation_failed"
    assert "nothing is dropped" in failed.json()["detail"]["message"]
    server.fail_on = None
    assert _call(client, "POST", run, json={"confirm": plan["confirm_phrase"]}).status_code == 409
    assert not any("DROP" in s for s in server.statements)
    assert _call(client, "GET", f"/plans/{plan['plan_id']}").json()["state"] == "unknown"


def test_a_name_that_exists_by_the_time_it_runs_is_refused_and_nothing_is_overwritten(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    server.databases["hai_toko"] = {"tb_bahan"}                 # someone made it meanwhile
    refused = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    assert refused.status_code == 409 and refused.json()["detail"]["error"] == "creation_refused"
    assert server.statements == []


def test_a_design_that_would_overwrite_is_refused_at_plan_time(world):
    client, designs, server, _ = world
    server.databases["hai_toko"] = {"tb_bahan"}
    body = _plan(client, designs).json()
    assert body["status"] == "refused" and "already has the table" in body["reason"] and body["plan_id"] is None


def test_an_unsupported_request_is_a_reason_not_an_error(world):
    client, designs, *_ = world
    body = _plan(client, designs, {"supported": False, "reason": "bukan database"}).json()
    assert body["status"] == "unsupported" and body["plan_id"] is None


def test_only_the_owner_can_see_run_or_cancel_a_plan(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    path = f"/plans/{plan['plan_id']}"
    assert _call(client, "GET", path, "other").status_code == 404
    assert _call(client, "DELETE", path, "other").status_code == 404
    assert _call(client, "POST", path + "/run", "other", json={"confirm": plan["confirm_phrase"]}).status_code == 404
    assert server.databases == {}


def test_the_permission_is_required(world):
    client, designs, *_ = world
    designs.answers.append(json.dumps(GOOD))
    assert _call(client, "POST", "/plans", "nocreate", json={"question": "q"}).status_code == 403


def test_a_cancelled_plan_is_gone_and_cannot_run(world):
    client, designs, server, _ = world
    plan = _plan(client, designs).json()
    assert _call(client, "DELETE", f"/plans/{plan['plan_id']}").json()["status"] == "cancelled"
    assert _call(client, "GET", f"/plans/{plan['plan_id']}").status_code == 404
    assert _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]}).status_code == 404
    assert server.databases == {}


def test_a_plan_made_before_a_restart_is_still_there_after_it(world):
    client, designs, server, tmp = world
    plan = _plan(client, designs).json()
    app = client.app
    from api.platform_runtime import get_pipeline
    store_path = app.state.database_runtime_registry.work_dir / "platform.sqlite3"
    assert store_path.exists()
    task = get_pipeline(app).store.get(plan["plan_id"])
    assert task is not None and task.operation_id == "database.create"


def test_the_history_lists_it_with_a_readable_line(world):
    client, designs, *_ = world
    plan = _plan(client, designs).json()
    _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]})
    listed = client.get("/api/v1/tasks", headers={"X-Test-User": "creator"}).json()["tasks"]
    assert listed[0]["label"] == "Buat database" and listed[0]["title"] == "Buat database hai_toko"
    assert listed[0]["status"] == "completed" and listed[0]["verification"] == "verified"
    detail = client.get(f"/api/v1/tasks/{plan['plan_id']}", headers={"X-Test-User": "creator"}).json()
    assert detail["result"]["database"] == "hai_toko" and detail["result"]["mode"] == "create_database"
    assert [e["event"] for e in detail["timeline"]][-1] == "task_completed"


def test_tables_added_to_an_existing_database_have_their_own_phrase(world):
    client, designs, server, _ = world
    server.databases["hai_toko"] = {"tb_lama"}
    extra = {"database": "hai_toko", "tables": [{"name": "tb_supplier", "columns": [{"name": "nama", "type": "text"}]}]}
    plan = _plan(client, designs, extra).json()
    assert plan["mode"] == "add_tables" and plan["confirm_phrase"] == "TAMBAH TABEL hai_toko"
    done = _call(client, "POST", f"/plans/{plan['plan_id']}/run", json={"confirm": plan["confirm_phrase"]}).json()
    assert done["mode"] == "add_tables" and done["verification"] == "verified"
    assert server.databases["hai_toko"] == {"tb_lama", "tb_supplier"}
