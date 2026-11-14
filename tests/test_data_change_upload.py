"""Updating many rows from an uploaded file: each row of the file names one existing row and its new values."""
from __future__ import annotations

import json
import sqlite3

from tests import test_data_entries_api  # noqa: F401  (registers the 'clerk' test user)
from tests.test_data_changes_api import CHANGES, _audit, _count, world  # noqa: F401  (world is a fixture)


def _upload(client, text, user="alice", query="", name="ubah.csv", headers=None):
    data = text.encode() if isinstance(text, str) else text
    return client.request("POST", f"{CHANGES}/file-plans?filename={name}{query}", content=data,
                          headers={"X-Test-User": user, "Content-Type": "application/octet-stream", **(headers or {})})


def _run(client, plan, user="alice", phrase=None, **kw):
    return client.request("POST", f"{CHANGES}/plans/{plan['plan_id']}/run",
                          json={"confirm": phrase or plan["confirm_phrase"]}, headers={"X-Test-User": user}, **kw)


def _row(db, ident):
    with sqlite3.connect(db) as c:
        return c.execute("SELECT product, amount FROM sales WHERE id = ?", (ident,)).fetchone()


def test_a_file_updates_the_rows_it_names_after_the_phrase_and_nothing_else(world):
    client, provider, db, _ = world
    plan = _upload(client, "id,amount\n1,500\n2,600\n3,700\n")
    assert plan.status_code == 201, plan.text
    body = plan.json()
    assert body["status"] == "planned" and body["expected_rows"] == 3 and provider.answers == []
    assert body["confirm_phrase"] == "UBAH 3 BARIS sales" and body["file"]["match"] == "id"
    assert [r["amount"] for r in body["preview"]["before"]] == [11.0, 12.0, 13.0]      # what is there now
    assert [r["amount"] for r in body["preview"]["after"]] == [500, 600, 700]          # what it would become
    assert _row(db, 1) == ("widget", 11.0)                                              # planning changed nothing
    wrong = _run(client, body, phrase="ya")
    assert wrong.status_code == 422 and _row(db, 1) == ("widget", 11.0)
    done = _run(client, body)
    assert done.status_code == 200, done.text
    assert done.json()["rows_affected"] == 3 and done.json()["verification"] == "verified"
    assert [_row(db, i)[1] for i in (1, 2, 3)] == [500, 600, 700] and _row(db, 4) == ("gadget", 14.0)


def test_an_empty_cell_leaves_that_value_as_it_is(world):
    client, _provider, db, _ = world
    body = _upload(client, "id,product,amount\n1,,500\n2,kopi,\n").json()
    assert body["status"] == "planned", body
    assert _run(client, body).status_code == 200
    assert _row(db, 1) == ("widget", 500) and _row(db, 2) == ("kopi", 12.0)


def test_more_than_twenty_rows_work(world):
    client, _provider, db, _ = world
    lines = "\n".join(f"{i},{1000 + i}" for i in range(1, 31))
    body = _upload(client, "id,amount\n" + lines + "\n").json()
    assert body["status"] == "planned" and body["expected_rows"] == 30
    done = _run(client, body)
    assert done.status_code == 200 and done.json()["verification"] == "verified"
    assert _row(db, 30) == ("gadget", 1030)


def test_a_value_that_matches_no_row_refuses_the_whole_file(world):
    client, _provider, db, _ = world
    body = _upload(client, "id,amount\n1,500\n9999,600\n").json()
    assert body["status"] == "refused" and "match no row" in body["reason"] and "9999" in body["reason"]
    assert _row(db, 1) == ("widget", 11.0)


def test_a_column_that_is_not_unique_cannot_find_rows(world):
    client, _provider, db, _ = world
    body = _upload(client, "region,amount\nwest,1\n", query="&match=region").json()
    assert body["status"] == "refused" and "more than one row" in body["reason"]
    assert _count(db, "amount = 1") == 0


def test_a_key_twice_in_the_file_or_a_row_with_nothing_to_change_is_refused(world):
    client, _provider, _db, _ = world
    assert "appears twice" in _upload(client, "id,amount\n1,5\n1,6\n").json()["reason"]
    assert "nothing to change" in _upload(client, "id,amount\n1,\n").json()["reason"]
    assert "Tidak ada kolom file yang cocok dengan id" in _upload(client, "amount\n5\n").json()["reason"]


def test_a_bad_value_names_its_row_and_column(world):
    client, _provider, _db, _ = world
    body = _upload(client, "id,amount\n1,5\n2,banyak\n").json()
    assert body["status"] == "refused" and "Row 3, column amount" in body["reason"]


def test_the_key_column_can_not_be_set_and_only_filterable_columns_find_rows(world):
    client, _provider, _db, _ = world
    bad = _upload(client, "amount,id\n5,1\n", query="&match=amount")
    assert bad.status_code == 422 and bad.json()["detail"]["error"] == "invalid_match"
    body = _upload(client, "id,amount,region\n1,5,west\n").json()                # region is not writable
    assert body["status"] == "needs_mapping"
    assert {c["name"] for c in body["columns"]} == {"id", "product", "amount"}


def test_unmatched_headers_get_a_proposal_from_names_only_and_the_corrected_mapping_plans(world):
    client, provider, db, _ = world
    provider.answers.append(json.dumps({"mapping": {"Kode": "id", "Harga": "amount"}}))
    text = "Kode,Harga\n1,500\n2,600\n"
    body = _upload(client, text).json()
    assert body["status"] == "needs_mapping" and body["mapping"] == {"Kode": "id", "Harga": "amount"}
    assert body["proposed_by_model"] is True and body["match"] == "id"
    from urllib.parse import quote
    plan = _upload(client, text, query="&mapping=" + quote(json.dumps(body["mapping"]))).json()
    assert plan["status"] == "planned" and _run(client, plan).status_code == 200
    assert _row(db, 2)[1] == 600


def test_it_needs_the_plan_permission_and_the_run_still_needs_a_fresh_login_and_the_phrase(world):
    client, _provider, db, _ = world
    assert _upload(client, "id,amount\n1,5\n", user="clerk").status_code == 403            # not a data writer
    assert _upload(client, "id,amount\n1,5\n", user="bob").status_code == 403              # an approver is not a planner
    body = _upload(client, "id,amount\n1,5\n", user="alice-stale").json()
    stale = _run(client, body, user="alice-stale")
    assert stale.status_code == 403 and stale.json()["detail"]["error"] == "step_up_required"
    assert _row(db, 1) == ("widget", 11.0)


def test_when_the_server_requires_a_second_person_a_file_update_waits_for_one(world, monkeypatch):
    client, _provider, db, _ = world
    monkeypatch.setenv("HAI_DATA_CHANGE_APPROVAL", "all")
    body = _upload(client, "id,amount\n1,5\n").json()
    assert body["requires_approval"] is True
    blocked = _run(client, body)
    assert blocked.status_code == 409 and _row(db, 1) == ("widget", 11.0)
    approved = client.request("POST", f"{CHANGES}/approvals/{body['plan_id']}/approve",
                              headers={"X-Test-User": "bob"})
    assert approved.status_code == 200, approved.text
    assert _run(client, body).status_code == 200 and _row(db, 1)[1] == 5


def test_a_row_that_disappears_before_the_run_changes_nothing(world):
    client, _provider, db, _ = world
    body = _upload(client, "id,amount\n1,500\n2,600\n").json()
    with sqlite3.connect(db) as c:
        c.execute("DELETE FROM sales WHERE id = 2")
    done = _run(client, body)
    assert done.status_code == 409 and done.json()["detail"]["error"] == "change_refused"
    assert _row(db, 1) == ("widget", 11.0)                                                 # all or nothing


def test_the_row_limit_applies(world):
    client, _provider, db, _ = world
    body = _upload(client, "id,amount\n1,5\n2,6\n3,7\n", query="&max_rows=2").json()
    assert body["status"] == "unsupported" and "1 to 2" in body["reason"]
    assert _row(db, 1) == ("widget", 11.0)


def test_the_audit_names_the_file_but_keeps_no_values(world, monkeypatch):
    client, _provider, _db, tmp = world
    monkeypatch.setenv("HAI_AUDIT_DETAIL", "minimal")           # the fixture asks for full detail
    body = _upload(client, "id,product,amount\n1,rahasia,12345.5\n").json()
    assert _run(client, body).status_code == 200
    text = (tmp / "work" / "audit" / "changes-sales.jsonl").read_text("utf-8")
    assert "rahasia" not in text and "12345.5" not in text
    planned = [e for e in _audit(tmp) if e.get("event") == "planned"][0]
    assert planned["plan"]["match_column"] == "id" and planned["plan"]["rows_count"] == 1
    assert planned["file"]["name"] == "ubah.csv"


def test_typed_text_sets_a_column_on_every_row_the_file_names(world):
    client, provider, db, _ = world
    provider.answers.append(json.dumps({"supported": True, "fill": {"product": {"value": "baru"}}}))
    plan = _upload(client, "id,amount\n1,500\n2,600\n", query="&question=semua%20product%20jadi%20baru")
    assert plan.status_code == 201, plan.text
    body = plan.json()
    assert body["status"] == "planned" and body["expected_rows"] == 2
    assert [r["product"] for r in body["preview"]["after"]] == ["baru", "baru"]
    assert _row(db, 1) == ("widget", 11.0)                       # still only a preview
    assert _run(client, body).status_code == 200
    assert _row(db, 1) == ("baru", 500) and _row(db, 2) == ("baru", 600)


def test_typed_text_that_adds_new_rows_is_refused_on_the_change_page(world):
    client, provider, db, _ = world
    provider.answers.append(json.dumps({"supported": True, "rows": [{"product": "gula", "amount": 3}]}))
    body = _upload(client, "id,amount\n1,500\n", query="&question=tambah%20gula").json()
    assert body["status"] == "refused" and body["plan_id"] is None
    assert _row(db, 1) == ("widget", 11.0)


def test_the_change_preview_names_the_table_column_order(world):
    client, _provider, _db, _ = world
    body = _upload(client, "id,amount\n1,500\n").json()
    assert body["column_order"] and "amount" in body["column_order"]


def test_the_connection_list_names_the_columns_that_may_find_rows(world):
    client, _provider, _db, _ = world
    listing = client.request("GET", "/api/v1/database-connections", headers={"X-Test-User": "alice"})
    assert listing.status_code == 200, listing.text
    item = next(c for c in listing.json()["connections"] if c.get("writable_tables"))
    choices = item["match_columns"]["sales"]
    assert choices and choices[0] == "id"


def test_text_that_only_restates_the_file_uses_the_file_as_it_is(world):
    client, provider, db, _ = world
    provider.answers.append(json.dumps({"supported": True, "fill": {}, "rows": []}))
    body = _upload(client, "id,amount\n1,500\n", query="&question=update%20dengan%20data%20di%20file").json()
    assert body["status"] == "planned" and body["expected_rows"] == 1
    assert [r["amount"] for r in body["preview"]["after"]] == [500]
