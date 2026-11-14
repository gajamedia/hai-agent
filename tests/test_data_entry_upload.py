"""Recording rows from an uploaded file: read by our parser, previewed, one press of Simpan, taken back as a unit."""
from __future__ import annotations

import sqlite3

from tests.test_data_entries_api import ENTRIES, _count, _entry, world  # noqa: F401  (world is a fixture)

CSV = b"Product;Amount\nkopi;25\nteh;15.5\nroti;7\n"


def _upload(client, data=CSV, name="penjualan.csv", user="clerk", query="", **kw):
    return _entry(client, "POST", f"/file-plans?filename={name}{query}", user, content=data,
                  headers={"Content-Type": "application/octet-stream", **kw.pop("headers", {})}, **kw)


def _products(db):
    with sqlite3.connect(db) as c:
        return [r[0] for r in c.execute("SELECT product FROM sales WHERE product IN ('kopi','teh','roti') ORDER BY id")]


def test_a_file_is_previewed_first_and_recorded_with_one_press(world):
    client, provider, db, _ = world
    before = _count(db)
    planned = _upload(client)
    assert planned.status_code == 201, planned.text
    body = planned.json()
    assert body["status"] == "planned" and body["expected_rows"] == 3 and body["file"]["rows"] == 3
    assert body["file"]["columns"] == ["Product", "Amount"] and body["file"]["already_recorded_at"] is None
    assert [r["product"] for r in body["preview"]["after"]] == ["kopi", "teh", "roti"]
    assert _count(db) == before and provider.answers == []                      # nothing recorded, no model used
    done = _entry(client, "POST", f"/plans/{body['plan_id']}/run")
    assert done.status_code == 200, done.text
    assert done.json()["rows_affected"] == 3 and done.json()["verification"] == "verified"
    assert _products(db) == ["kopi", "teh", "roti"] and _count(db) == before + 3


def test_an_uploaded_entry_can_be_taken_back_as_a_whole(world):
    client, _provider, db, _ = world
    before = _count(db)
    plan_id = _upload(client).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{plan_id}/run").status_code == 200
    undone = _entry(client, "POST", f"/{plan_id}/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json()["rows_removed"] == 3 and _count(db) == before and _products(db) == []


def test_the_same_file_twice_needs_an_explicit_second_yes(world):
    client, _provider, db, _ = world
    before = _count(db)
    first = _upload(client).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{first}/run").status_code == 200
    again = _upload(client)
    assert again.json()["file"]["already_recorded_at"] is not None
    blocked = _entry(client, "POST", f"/plans/{again.json()['plan_id']}/run")
    assert blocked.status_code == 409 and blocked.json()["detail"]["error"] == "file_already_recorded"
    assert _count(db) == before + 3
    again_id = _upload(client).json()["plan_id"]
    confirmed = _entry(client, "POST", f"/plans/{again_id}/run", headers={"X-Confirm-Duplicate": "yes"})
    assert confirmed.status_code == 200 and _count(db) == before + 6


def test_a_changed_file_is_not_taken_for_the_same_one(world):
    client, _provider, db, _ = world
    first = _upload(client).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{first}/run").status_code == 200
    other = _upload(client, data=CSV + b"gula;3\n")
    assert other.json()["file"]["already_recorded_at"] is None


def test_a_bad_value_refuses_the_whole_file_with_its_row_and_column(world):
    client, _provider, db, _ = world
    before = _count(db)
    body = _upload(client, data=b"product,amount\nkopi,25\nteh,banyak\n").json()
    assert body["status"] == "refused" and body["plan_id"] is None
    assert "Row 3, column amount" in body["reason"] and _count(db) == before


def test_a_column_that_cannot_be_written_never_reaches_the_table(world):
    client, _provider, db, _ = world
    before = _count(db)
    for data in (b"product,region\nkopi,west\n", b"id,product\n99,kopi\n"):    # not writable / the key column
        body = _upload(client, data=data).json()
        assert body["status"] == "needs_mapping" and body["plan_id"] is None
        assert {c["name"] for c in body["columns"]} == {"product", "amount"}       # only what may be filled is offered
    assert _count(db) == before


def test_unsupported_file_types_and_oversized_files_are_refused(world):
    client, _provider, _db, _ = world
    assert _upload(client, data=b"x", name="a.pdf").json()["status"] == "refused"
    big = _upload(client, data=b"a" * (5 * 1024 * 1024 + 1))
    assert big.status_code == 413 and big.json()["detail"]["error"] == "file_too_large"


def test_the_table_must_be_one_that_accepts_new_rows(world):
    client, _provider, _db, _ = world
    wrong = _upload(client, query="&table=users")
    assert wrong.status_code == 422 and wrong.json()["detail"]["error"] == "table_required"
    named = _upload(client, query="&table=sales")
    assert named.status_code == 201


def test_a_file_entry_belongs_to_its_author_and_needs_the_entry_permission(world):
    client, _provider, _db, _ = world
    plan_id = _upload(client).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{plan_id}/run", "clerk2").status_code == 404
    assert _upload(client, user="alice").status_code == 403                    # data-writer is not a clerk


def test_the_audit_names_the_file_but_holds_none_of_its_values(world, monkeypatch):
    import json
    monkeypatch.setenv("HAI_AUDIT_DETAIL", "minimal")      # the fixture asks for full detail
    client, _provider, _db, tmp = world
    plan_id = _upload(client).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{plan_id}/run").status_code == 200
    text = (tmp / "work" / "audit" / "changes-sales.jsonl").read_text(encoding="utf-8")
    lines = [json.loads(x) for x in text.splitlines()]
    assert any(x.get("file", {}).get("name") == "penjualan.csv" for x in lines)
    assert "kopi" not in text and "15.5" not in text


def test_an_excel_file_works_the_same_way(world):
    import io
    from openpyxl import Workbook
    client, _provider, db, _ = world
    book = Workbook()
    book.active.append(["product", "amount"])
    book.active.append(["kopi", 25])
    out = io.BytesIO()
    book.save(out)
    planned = _upload(client, data=out.getvalue(), name="penjualan.xlsx")
    assert planned.status_code == 201, planned.text
    assert _entry(client, "POST", f"/plans/{planned.json()['plan_id']}/run").status_code == 200
    assert _products(db) == ["kopi"]


MESSY = ("Nama Barang;Harga;Catatan Kasir\nkopi;25;pagi\nteh;15;siang\n").encode()


def _mapping_query(mapping):
    import json
    from urllib.parse import quote
    return "&mapping=" + quote(json.dumps(mapping))


def test_unmatched_headers_get_a_proposal_from_names_only_and_nothing_is_recorded(world):
    import json
    client, provider, db, _ = world
    seen = []
    original = provider.complete
    provider.complete = lambda request: (seen.append(str(request)), original(request))[1]
    provider.answers.append(json.dumps({"mapping": {"Nama Barang": "product", "Harga": "amount",
                                                    "Catatan Kasir": "region"}}))
    before = _count(db)
    body = _upload(client, data=MESSY).json()
    assert body["status"] == "needs_mapping" and body["proposed_by_model"] is True and _count(db) == before
    # "region" is not a column that may be filled, so that part of the proposal is dropped, not trusted
    assert body["mapping"] == {"Nama Barang": "product", "Harga": "amount", "Catatan Kasir": None}
    assert body["headers"] == ["Nama Barang", "Harga", "Catatan Kasir"]
    prompt = " ".join(seen)
    assert "Nama Barang" in prompt and "kopi" not in prompt and "teh" not in prompt and "25" not in prompt


def test_the_corrected_mapping_is_used_and_the_ignored_column_is_left_out(world):
    client, _provider, db, _ = world
    before = _count(db)
    mapping = {"Nama Barang": "product", "Harga": "amount", "Catatan Kasir": None}
    planned = _upload(client, data=MESSY, query=_mapping_query(mapping))
    assert planned.status_code == 201, planned.text
    assert [r["product"] for r in planned.json()["preview"]["after"]] == ["kopi", "teh"]
    assert "pagi" not in str(planned.json()["preview"])
    assert _entry(client, "POST", f"/plans/{planned.json()['plan_id']}/run").status_code == 200
    assert _products(db) == ["kopi", "teh"] and _count(db) == before + 2


def test_without_a_model_the_person_maps_the_columns_by_hand(world):
    client, provider, _db, _ = world                    # no scripted answer: the model "fails"
    body = _upload(client, data=MESSY).json()
    assert body["status"] == "needs_mapping" and body["proposed_by_model"] is False
    assert body["mapping"] == {"Nama Barang": None, "Harga": None, "Catatan Kasir": None}


def test_columns_that_match_by_name_are_kept_and_only_the_rest_are_proposed(world):
    import json
    client, provider, _db, _ = world
    provider.answers.append(json.dumps({"mapping": {"Harga": "amount", "Product": "amount"}}))
    body = _upload(client, data=b"Product,Harga\nkopi,25\n").json()
    assert body["mapping"] == {"Product": "product", "Harga": "amount"}


def test_a_bad_mapping_is_refused_not_trusted(world):
    client, _provider, db, _ = world
    before = _count(db)
    for mapping in ({"Nama Barang": "product", "Harga": "product", "Catatan Kasir": None},     # twice the same column
                    {"Nama Barang": "id", "Harga": "amount", "Catatan Kasir": None},             # the key column
                    {"Nama Barang": "product", "Harga": "amount"},                               # a file column left out
                    {"Nama Barang": None, "Harga": None, "Catatan Kasir": None}):                # nothing mapped
        body = _upload(client, data=MESSY, query=_mapping_query(mapping)).json()
        assert body["status"] == "refused" and body["plan_id"] is None, mapping
    bad = _upload(client, data=MESSY, query="&mapping=not-json")
    assert bad.status_code == 422 and _count(db) == before


def test_a_large_file_entry_can_be_taken_back_as_a_whole(world):
    client, _provider, db, _ = world
    before = _count(db)
    lines = ["product,amount"] + [f"barang{i},{i}" for i in range(60)]
    plan_id = _upload(client, data=("\n".join(lines) + "\n").encode()).json()["plan_id"]
    assert _entry(client, "POST", f"/plans/{plan_id}/run").status_code == 200
    assert _count(db) == before + 60
    undone = _entry(client, "POST", f"/{plan_id}/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json()["rows_removed"] == 60 and _count(db) == before


def _typed(provider, answer):
    import json
    provider.answers.append(json.dumps(answer))


def test_typed_text_fills_the_columns_the_file_leaves_out(world):
    client, provider, db, _ = world
    before = _count(db)
    _typed(provider, {"supported": True, "fill": {"amount": {"random_int": [10, 20]}}})
    planned = _upload(client, data=b"product\nkopi\nteh\n", query="&question=amount%20acak%2010%20sampai%2020")
    assert planned.status_code == 201, planned.text
    body = planned.json()
    assert body["status"] == "planned" and body["expected_rows"] == 2
    assert [r["product"] for r in body["preview"]["after"]] == ["kopi", "teh"]
    assert all(10 <= r["amount"] <= 20 for r in body["preview"]["after"])
    assert _count(db) == before
    assert _entry(client, "POST", f"/plans/{body['plan_id']}/run").status_code == 200
    assert _count(db) == before + 2


def test_typed_text_can_add_literal_rows_next_to_the_file(world):
    client, provider, db, _ = world
    _typed(provider, {"supported": True, "rows": [{"product": "gula", "amount": 3}]})
    body = _upload(client, query="&question=catat%20juga%20gula%203").json()
    assert body["status"] == "planned" and body["expected_rows"] == 4
    assert [r["product"] for r in body["preview"]["after"]] == ["kopi", "teh", "roti", "gula"]


def test_a_refusal_from_the_model_is_shown_and_nothing_is_recorded(world):
    client, provider, db, _ = world
    before = _count(db)
    _typed(provider, {"supported": False, "reason": "Nilai acak untuk NIK tidak dibuat."})
    body = _upload(client, query="&question=isi%20nik%20acak").json()
    assert body["status"] == "refused" and "NIK" in body["reason"] and body["plan_id"] is None
    assert _count(db) == before


def test_a_rule_for_a_column_the_file_already_fills_is_refused(world):
    client, provider, db, _ = world
    _typed(provider, {"supported": True, "fill": {"product": {"value": "x"}}})
    body = _upload(client, query="&question=semua%20x").json()
    assert body["status"] == "refused" and body["plan_id"] is None


def test_the_columns_that_came_from_the_file_are_named_for_the_preview(world):
    client, _provider, _db, _ = world
    body = _upload(client).json()
    assert body["file"]["filled"] == ["amount", "product"]


def test_the_preview_names_the_destination_tables_column_order(world):
    client, _provider, _db, _ = world
    body = _upload(client).json()
    assert body["column_order"] and set(body["column_order"]) >= {"product", "amount"}
