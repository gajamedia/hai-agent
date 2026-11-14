"""A database designed from an uploaded file, then filled from the same file: no language model anywhere."""
from __future__ import annotations

import sqlite3

from tests.test_sqlite_write import _call, _env, _tables, world  # noqa: F401  (fixtures)

CSV = (b"Produk,Qty,Harga,Tanggal,Telepon\n"
       b"kopi,10,25000.50,10/11/2026,0811000001\n"
       b"teh,5,15000,11/11/2026,0811000002\n")


def _upload(client, path, user, data=CSV, name="Penjualan Harian.csv", query=""):
    return client.request("POST", f"/api/v1{path}?filename={name}{query}", content=data,
                          headers={"X-Test-User": user, "Content-Type": "application/octet-stream"})


def test_a_database_is_designed_from_the_columns_of_a_file_and_made_after_the_phrase(world):
    client, provider, data = world
    plan = _upload(client, "/database-creation/file-plans", "maker", query="&database=hai_toko")
    assert plan.status_code == 201, plan.text
    body = plan.json()
    assert body["status"] == "planned" and provider.answers == []                  # no model was asked
    assert body["confirm_phrase"] == "BUAT DATABASE hai_toko"
    assert body["file"]["rows"] == 2
    assert [(c["name"], c["type"]) for c in body["file"]["columns"]] == [
        ("produk", "text"), ("qty", "integer"), ("harga", "decimal"), ("tanggal", "date"),
        ("telepon", "text")]                                                    # a phone number is not a number
    assert body["plan"]["tables"][0]["name"] == "tb_penjualan_harian"
    assert not (data / "hai_toko.sqlite3").exists()                              # nothing made before the phrase
    done = _call(client, "POST", f"/database-creation/plans/{body['plan_id']}/run", "maker",
                 json={"confirm": body["confirm_phrase"]})
    assert done.status_code == 200, done.text
    assert _tables(data / "hai_toko.sqlite3") == ["tb_penjualan_harian"]


def test_the_table_can_be_named_and_the_database_name_is_still_checked(world):
    client, _provider, data = world
    named = _upload(client, "/database-creation/file-plans", "maker", query="&database=hai_toko&table=tb_jual")
    assert named.json()["plan"]["tables"][0]["name"] == "tb_jual"
    wrong_prefix = _upload(client, "/database-creation/file-plans", "maker", query="&database=toko")
    assert wrong_prefix.json()["status"] == "unsupported" and "must start with" in wrong_prefix.json()["reason"]
    assert not list(data.glob("*.sqlite3"))


def test_a_file_that_cannot_make_valid_columns_is_refused_with_a_reason(world):
    client, _provider, _data = world
    for text, message in [(b"id,nama\n1,a\n", "id"), (b"1abc,nama\n1,a\n", "Invalid column name")]:
        body = _upload(client, "/database-creation/file-plans", "maker", data=text, query="&database=hai_toko").json()
        assert body["status"] == "unsupported" and message in body["reason"], body
    bad = _upload(client, "/database-creation/file-plans", "maker", data=b"x", name="a.pdf", query="&database=hai_toko")
    assert bad.json()["status"] == "refused"


def test_creating_from_a_file_needs_the_creation_permission_and_a_database_name(world):
    client, _provider, _data = world
    assert _upload(client, "/database-creation/file-plans", "sclerk", query="&database=hai_toko").status_code == 403
    assert _upload(client, "/database-creation/file-plans", "maker").status_code == 422


def test_the_same_file_then_fills_the_table_it_designed(world):
    client, _provider, data = world
    plan = _upload(client, "/database-creation/file-plans", "maker", query="&database=hai_toko&table=tb_jual").json()
    assert _call(client, "POST", f"/database-creation/plans/{plan['plan_id']}/run", "maker",
                 json={"confirm": plan["confirm_phrase"]}).status_code == 200
    assert _call(client, "POST", "/database-connections", "alice",
                 json={"name": "toko", "profile": "sqlite", "database": "hai_toko"}).status_code == 200
    selection = {"tables": {"tb_jual": {
        "columns": ["id", "produk", "qty", "harga", "tanggal", "telepon"], "filterable_columns": ["id", "produk"],
        "key_column": "id", "writable": {"operations": ["insert"],
                                         "columns": ["produk", "qty", "harga", "tanggal", "telepon"]}}}}
    assert _call(client, "POST", "/database-connections/toko/approve", "alice", json=selection).status_code == 200
    entry = _upload(client, "/data-entries/file-plans", "sclerk")
    assert entry.status_code == 201, entry.text
    assert entry.json()["expected_rows"] == 2
    saved = _call(client, "POST", f"/data-entries/plans/{entry.json()['plan_id']}/run", "sclerk")
    assert saved.status_code == 200 and saved.json()["verification"] == "verified"
    with sqlite3.connect(data / "hai_toko.sqlite3") as c:
        assert c.execute("SELECT produk, qty, harga, tanggal, telepon FROM tb_jual ORDER BY id").fetchall() == [
            ("kopi", 10, 25000.5, "2026-11-10", "0811000001"), ("teh", 5, 15000, "2026-11-11", "0811000002")]


def test_with_typed_text_the_file_and_the_text_are_read_together_by_the_model(world):
    import json
    client, provider, data = world
    provider.answers.append(json.dumps({"supported": True, "database": "hai_lain", "title": "Toko", "tables": [
        {"name": "tb_asal", "columns": [{"name": "produk", "type": "text", "nullable": False},
                                         {"name": "qty", "type": "integer", "nullable": True},
                                         {"name": "status", "type": "text", "nullable": True}]}]}))
    plan = _upload(client, "/database-creation/file-plans", "maker",
                   query="&database=hai_toko&table=tb_jual&question=tambah%20kolom%20status")
    assert plan.status_code == 201, plan.text
    body = plan.json()
    assert body["status"] == "planned" and provider.answers == []                      # the model was asked
    assert body["plan"]["database"] == "hai_toko"                                      # the form's names win
    assert body["plan"]["tables"][0]["name"] == "tb_jual"
    assert [c["name"] for c in body["plan"]["tables"][0]["columns"] if c["name"] != "id"] == ["produk", "qty", "status"]
    assert body["file"]["sent_to_model_rows"] == 2
    assert not (data / "hai_toko.sqlite3").exists()                                    # nothing made before the phrase


def test_the_model_reads_the_file_content_when_text_is_typed(world, monkeypatch):
    import json
    from agent.workflow import file_context
    client, provider, _data = world
    seen = {}
    real = file_context.describe_request

    def spy(**kw):
        text = real(**kw)
        seen["request"] = text
        return text

    monkeypatch.setattr("api.routes.database_creation.describe_request", spy)
    provider.answers.append(json.dumps({"supported": True, "database": "hai_toko", "tables": [
        {"name": "tb_x", "columns": [{"name": "produk", "type": "text", "nullable": True}]}]}))
    _upload(client, "/database-creation/file-plans", "maker", query="&database=hai_toko&question=rancang")
    assert "Produk,Qty,Harga,Tanggal,Telepon" in seen["request"] and "kopi,10,25000.50" in seen["request"]


def test_without_typed_text_no_model_sees_the_file(world):
    client, provider, _data = world
    body = _upload(client, "/database-creation/file-plans", "maker", query="&database=hai_toko").json()
    assert body["status"] == "planned" and body["file"]["sent_to_model_rows"] is None
