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


# ------------------------------------------------------------------ taking an entry back
import sqlite3  # noqa: E402


def _record(client, provider, user="clerk"):
    plan_id = _plan(client, provider, INSERT, user=user).json()["plan_id"]
    done = _entry(client, "POST", f"/plans/{plan_id}/run", user)
    assert done.status_code == 200, done.text
    return plan_id


def _products(db):
    with sqlite3.connect(db) as c:
        return [r[0] for r in c.execute("SELECT product FROM sales WHERE product IN ('kopi', 'teh') ORDER BY id")]


def test_an_entry_can_be_taken_back_once_and_only_its_own_rows_go(world):
    client, provider, db, _ = world
    before = _count(db)
    entry_id = _record(client, provider)
    assert _count(db) == before + 2
    recent = _entry(client, "GET", "/recent").json()
    assert recent["entries"][0]["entry_id"] == entry_id and recent["entries"][0]["undoable"] is True
    assert recent["entries"][0]["rows"] == 2 and recent["entries"][0]["table"] == "sales"
    undone = _entry(client, "POST", f"/{entry_id}/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json() == {"status": "undone", "rows_removed": 2, "table": "sales", "verification": "verified"}
    assert _count(db) == before and _products(db) == []
    after = _entry(client, "GET", "/recent").json()["entries"][0]
    assert after["undone"] is True and after["undoable"] is False
    again = _entry(client, "POST", f"/{entry_id}/undo")
    assert again.status_code == 409 and "sudah dibatalkan" in again.json()["detail"]["message"]


def test_an_entry_is_not_taken_back_when_one_of_its_rows_was_edited_since(world):
    client, provider, db, _ = world
    before = _count(db)
    entry_id = _record(client, provider)
    with sqlite3.connect(db) as c:
        c.execute("UPDATE sales SET amount = 999 WHERE product = 'kopi'")
    refused = _entry(client, "POST", f"/{entry_id}/undo")
    assert refused.status_code == 409 and refused.json()["detail"]["error"] == "undo_refused"
    assert "changed after they were recorded" in refused.json()["detail"]["message"]
    assert _count(db) == before + 2                     # nothing was removed, not even the untouched row


def test_an_entry_is_not_taken_back_when_a_row_is_already_gone(world):
    client, provider, db, _ = world
    entry_id = _record(client, provider)
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM sales WHERE product = 'teh'")
    refused = _entry(client, "POST", f"/{entry_id}/undo")
    assert refused.status_code == 409 and "no longer there" in refused.json()["detail"]["message"]
    assert _products(db) == ["kopi"]


def test_only_the_author_can_take_an_entry_back(world):
    client, provider, db, _ = world
    entry_id = _record(client, provider)
    assert _entry(client, "POST", f"/{entry_id}/undo", "clerk2").status_code == 404
    assert _entry(client, "POST", f"/{entry_id}/undo", "alice").status_code == 403    # no entry permission
    assert _products(db) == ["kopi", "teh"]


def test_the_time_window_closes(world):
    client, provider, db, tmp = world
    entry_id = _record(client, provider)
    with sqlite3.connect(tmp / "work" / "platform.sqlite3") as c:
        c.execute("UPDATE platform_tasks SET updated_at = updated_at - 200000 WHERE task_id = ?", (entry_id,))
    listed = _entry(client, "GET", "/recent").json()["entries"][0]
    assert listed["undoable"] is False
    late = _entry(client, "POST", f"/{entry_id}/undo")
    assert late.status_code == 409 and "Batas waktu" in late.json()["detail"]["message"]
    assert _products(db) == ["kopi", "teh"]


def test_undo_needs_no_delete_approval_on_the_table_but_removes_nothing_else(world):
    client, provider, db, _ = world
    insert_only = {"tables": {"sales": {
        "columns": ["id", "region", "product", "amount"], "filterable_columns": ["id", "region", "product"],
        "key_column": "id", "writable": {"operations": ["insert"], "columns": ["product", "amount"]}}}}
    assert client.post("/api/v1/database-connections/sales/approve", json=insert_only,
                       headers={"X-Test-User": "alice"}).status_code == 200
    before = _count(db)
    entry_id = _record(client, provider)
    assert _entry(client, "POST", f"/{entry_id}/undo").status_code == 200
    assert _count(db) == before


def test_the_audit_names_the_keys_taken_back_and_holds_no_values(world, monkeypatch):
    client, provider, db, tmp = world
    monkeypatch.setenv("HAI_AUDIT_DETAIL", "minimal")
    entry_id = _record(client, provider)
    assert _entry(client, "POST", f"/{entry_id}/undo").status_code == 200
    lines = (tmp / "work" / "audit" / "changes-sales.jsonl").read_text("utf-8").splitlines()
    records = [json.loads(x) for x in lines]
    made = next(r for r in records if r.get("status") == "committed" and r.get("operation") == "insert")
    undo = next(r for r in records if r.get("event") == "undo")
    assert made["inserted_keys"] == undo["keys"] and undo["status"] == "committed"
    text = "\n".join(lines)
    # the person's own words are kept (that is what "the original instruction" means); the values are not
    for value in ('"product": "kopi"', '"product": "teh"', '"amount": 25', '"amount": 15', '"params": ["25"'):
        assert value not in text
    planned = next(r for r in records if r.get("event") == "entry_planned")
    assert planned["plan"] == {"operation": "insert", "table": "sales", "max_rows": 100, "rows_count": 2}


def test_an_insert_that_cannot_tell_its_own_rows_is_refused_and_leaves_nothing(world, monkeypatch):
    client, provider, db, _ = world
    from agent.workflow.data_change import DataChangeService
    before = _count(db)
    real = DataChangeService._max_key
    monkeypatch.setattr(DataChangeService, "_max_key", lambda self, cur, table, key: (real(self, cur, table, key) or 0) - 1)
    plan_id = _plan(client, provider, INSERT).json()["plan_id"]
    refused = _entry(client, "POST", f"/plans/{plan_id}/run")
    assert refused.status_code == 409 and "at the same moment" in refused.json()["detail"]["message"]
    assert _count(db) == before
