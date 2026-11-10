"""Recording new data over HTTP: a clerk adds rows with one press of "Simpan", and can do nothing else."""
from __future__ import annotations

import dataclasses
import json
import time

from api.security.authentication import AuthenticatedPrincipal
from tests import test_data_changes_api as base
from tests.test_data_changes_api import _count, _call, world  # noqa: F401  (world is a fixture)

ENTRY = "data:entry:record"
ENTRIES = "/api/v1/data-entries"
base.USERS["clerk"] = AuthenticatedPrincipal(actor_id="clerk", permissions=(ENTRY,), auth_time=int(time.time()))
# A clerk whose last sign-in is old: recording must not demand a fresh login.
base.USERS["clerk-stale"] = dataclasses.replace(base.USERS["clerk"], auth_time=int(time.time()) - 3600)
base.USERS["clerk2"] = AuthenticatedPrincipal(actor_id="clerk2", permissions=(ENTRY,), auth_time=int(time.time()))

INSERT = {"supported": True, "operation": "insert", "table": "sales",
          "rows": [{"product": "kopi", "amount": 25}, {"product": "teh", "amount": 15}]}
UPDATE = {"supported": True, "operation": "update", "table": "sales", "set": {"amount": 1},
          "conditions": [{"column": "region", "operator": "eq", "value": "west"}]}


def _entry(client, method, path, user="clerk", **kw):
    return client.request(method, f"{ENTRIES}{path}",
                          headers={"X-Test-User": user, **kw.pop("headers", {})}, **kw)


def _plan(client, provider, answer, user="clerk"):
    provider.answers.append(json.dumps(answer))
    return _entry(client, "POST", "/plans", user, json={"question": "catat penjualan kopi dan teh"})


def test_a_clerk_records_rows_with_one_press_and_no_phrase_or_fresh_login(world):
    client, provider, db, _ = world
    before = _count(db)
    planned = _plan(client, provider, INSERT, user="clerk-stale")
    assert planned.status_code == 201, planned.text
    body = planned.json()
    assert body["status"] == "planned" and body["expected_rows"] == 2
    assert "confirm_phrase" not in body and _count(db) == before              # nothing recorded yet
    done = _entry(client, "POST", f"/plans/{body['plan_id']}/run", "clerk-stale")
    assert done.status_code == 200, done.text
    assert done.json()["rows_affected"] == 2 and done.json()["verification"] == "verified"
    assert _count(db) == before + 2


def test_pressing_simpan_twice_with_the_same_key_records_once(world):
    client, provider, db, _ = world
    before = _count(db)
    plan_id = _plan(client, provider, INSERT).json()["plan_id"]
    first = _entry(client, "POST", f"/plans/{plan_id}/run", headers={"Idempotency-Key": "k1"})
    again = _entry(client, "POST", f"/plans/{plan_id}/run", headers={"Idempotency-Key": "k1"})
    assert first.status_code == again.status_code == 200 and again.json()["replayed"] is True
    assert _count(db) == before + 2
    third = _entry(client, "POST", f"/plans/{plan_id}/run")                   # no key: a second run is refused
    assert third.status_code == 409 and _count(db) == before + 2


def test_a_clerk_cannot_update_or_delete_even_by_asking_in_words(world):
    client, provider, db, _ = world
    before = _count(db, "amount = 1")
    planned = _plan(client, provider, UPDATE)
    assert planned.status_code == 201 and planned.json()["status"] == "unsupported"
    assert planned.json()["plan_id"] is None and "Ubah data" in planned.json()["reason"]
    assert _count(db, "amount = 1") == before


def test_the_two_permissions_do_not_cross(world):
    client, provider, _, _ = world
    assert _call(client, "POST", "/plans", "clerk", json={"question": "q", "max_rows": 10}).status_code == 403
    provider.answers.append(json.dumps(INSERT))
    assert _entry(client, "POST", "/plans", "alice", json={"question": "q"}).status_code == 403
    assert _entry(client, "POST", "/plans", "reader", json={"question": "q"}).status_code == 403


def test_an_entry_belongs_to_the_person_who_made_it(world):
    client, provider, db, _ = world
    plan_id = _plan(client, provider, INSERT).json()["plan_id"]
    assert _entry(client, "GET", f"/plans/{plan_id}", "clerk2").status_code == 404
    assert _entry(client, "POST", f"/plans/{plan_id}/run", "clerk2").status_code == 404
    assert _entry(client, "DELETE", f"/plans/{plan_id}").json()["status"] == "cancelled"
    assert _entry(client, "POST", f"/plans/{plan_id}/run").status_code == 404


def test_a_clerks_entry_shows_in_their_task_history_labelled_catat_data(world):
    client, provider, _, _ = world
    plan_id = _plan(client, provider, INSERT).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{plan_id}/run").status_code == 200
    # the history permission is a separate grant; the clerk role carries it (see the role test)
    base.USERS["clerk-history"] = AuthenticatedPrincipal(actor_id="clerk", permissions=(ENTRY, "task:history:read"),
                                                         auth_time=int(time.time()))
    tasks = client.get("/api/v1/tasks", headers={"X-Test-User": "clerk-history"}).json()["tasks"]
    assert any(t.get("label") == "Catat data" for t in tasks), tasks


def test_a_clerk_sees_where_they_may_record_without_being_able_to_connect_databases(world):
    client, _, _, _ = world
    listed = _entry(client, "GET", "/databases").json()["databases"]
    assert listed == [{"name": "sales", "tables": ["sales"]}]
    assert client.get("/api/v1/database-connections", headers={"X-Test-User": "clerk"}).status_code == 403
    assert _entry(client, "GET", "/databases", "reader").status_code == 403


def test_recording_into_an_unknown_database_is_not_found(world):
    client, _, _, _ = world
    r = _entry(client, "POST", "/plans", json={"question": "x", "database": "nope"})
    assert r.status_code == 404
