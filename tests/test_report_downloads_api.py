from __future__ import annotations

import io
import sqlite3
from zipfile import ZipFile

import pytest

from tests.test_table_report_plans_api import BASE, _create, _plan, world  # noqa: F401

DOWNLOAD = f"{BASE}/reports"
FORMATS = ["xlsx", "pdf", "csv", "html", "json"]
MEDIA = {"xlsx": "spreadsheetml.sheet", "pdf": "application/pdf", "csv": "text/csv",
         "html": "text/html", "json": "application/json"}


def _run(c, fmt, question="widget per region"):
    plan = c.post(f"{BASE}/plans", json={"question": question, "format": fmt}).json()
    return plan, c.post(f"{BASE}/plans/{plan['plan_id']}/run").json()


def _expected(tmp):
    with sqlite3.connect(tmp / "data" / "sales.sqlite3") as db:
        return dict(db.execute(
            "SELECT region, SUM(amount) FROM sales WHERE product='widget' GROUP BY region"))


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_is_planned_run_and_downloadable_with_safe_headers(world, fmt):  # noqa: F811
    build, tmp = world
    client, *_ = build(_plan(filters={"product": "widget"}))
    c = client()
    plan, body = _run(c, fmt)
    assert plan["plan"]["format"] == fmt                       # the user's choice is shown in the plan
    assert body["status"] == "succeeded", body
    info = body["download"]
    assert info["url"].endswith(f"/reports/{body['workflow_id']}/download")
    assert MEDIA[fmt] in info["media_type"] and info["size_bytes"] > 0

    r = c.get(info["url"].replace("/api/v1/table-reports", BASE))
    assert r.status_code == 200
    assert MEDIA[fmt] in r.headers["content-type"]
    assert r.headers["content-disposition"] == (
        f'attachment; filename="report-{body["workflow_id"][:8]}.{fmt}"')
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "private, no-store"
    assert "sandbox" in r.headers["content-security-policy"]
    assert len(r.content) == info["size_bytes"]

    if fmt == "pdf":
        assert r.content.startswith(b"%PDF-1.4")
    if fmt == "xlsx":
        with ZipFile(io.BytesIO(r.content)) as z:
            assert "xl/worksheets/sheet1.xml" in z.namelist()
    if fmt == "csv":
        assert b"west" in r.content or b"east" in r.content


def test_the_excel_file_holds_real_numbers_that_match_sql(world):  # noqa: F811
    openpyxl = pytest.importorskip("openpyxl")
    build, tmp = world
    client, *_ = build(_plan(filters={"product": "widget"}))
    c = client()
    _, body = _run(c, "xlsx")
    data = c.get(f"{DOWNLOAD}/{body['workflow_id']}/download").content
    ws = openpyxl.load_workbook(io.BytesIO(data)).active
    header = [cell.value for cell in ws[1]]
    got = {row[0].value: row[header.index("revenue")].value for row in ws.iter_rows(min_row=2)}
    assert got == pytest.approx(_expected(tmp))
    assert all(ws.cell(row=r, column=header.index("revenue") + 1).data_type == "n"
               for r in range(2, ws.max_row + 1))              # numeric, not text


def test_the_pdf_shows_the_same_numbers(world):  # noqa: F811
    pypdf = pytest.importorskip("pypdf")
    build, tmp = world
    client, *_ = build(_plan(filters={"product": "widget"}))
    c = client()
    _, body = _run(c, "pdf")
    text = pypdf.PdfReader(io.BytesIO(c.get(f"{DOWNLOAD}/{body['workflow_id']}/download").content)
                           ).pages[0].extract_text()
    for region, revenue in _expected(tmp).items():
        assert region in text and f"{revenue:g}" in text


def test_the_users_format_overrides_the_llms_choice(world):  # noqa: F811
    client, *_ = world[0](_plan(format="html"))
    plan = client().post(f"{BASE}/plans", json={"question": "q", "format": "pdf"}).json()
    assert plan["plan"]["format"] == "pdf"


def test_without_a_user_format_the_llms_choice_is_kept(world):  # noqa: F811
    client, *_ = world[0](_plan(format="csv"))
    assert _create(client()).json()["plan"]["format"] == "csv"


@pytest.mark.parametrize("fmt", ["docx", "exe", "", 5, ["pdf"]])
def test_unknown_formats_are_rejected_for_both_endpoints(world, fmt):  # noqa: F811
    client, *_ = world[0](_plan())
    c = client()
    assert c.post(f"{BASE}/plans", json={"question": "q", "format": fmt}).status_code == 422
    assert c.post(BASE, json={"question": "q", "format": fmt}).status_code == 422


def test_the_one_call_endpoint_supports_format_and_download(world):  # noqa: F811
    client, *_ = world[0](_plan())
    c = client()
    body = c.post(BASE, json={"question": "q", "format": "xlsx"}).json()
    assert body["plan"]["format"] == "xlsx" and body["download"]["filename"] == "report.xlsx"
    assert c.get(f"{DOWNLOAD}/{body['workflow_id']}/download").status_code == 200


def test_replaying_a_run_returns_the_same_download(world):  # noqa: F811
    client, *_ = world[0](_plan())
    c = client()
    plan = c.post(f"{BASE}/plans", json={"question": "q", "format": "pdf"}).json()
    key = {"Idempotency-Key": "abc"}
    first = c.post(f"{BASE}/plans/{plan['plan_id']}/run", headers=key).json()
    again = c.post(f"{BASE}/plans/{plan['plan_id']}/run", headers=key).json()
    assert again["download"] == first["download"] and again["replayed"] is True


def test_a_failed_report_offers_no_download(world):  # noqa: F811
    client, *_ = world[0](_plan(filters={"region": "atlantis"}))
    c = client()
    body = c.post(BASE, json={"question": "q", "format": "pdf"}).json()
    assert body["status"] == "failed" and body["download"] is None


def test_only_the_owner_can_download(world):  # noqa: F811
    client, *_ = world[0](_plan())
    alice, bob = client("alice"), client("bob")
    body = alice.post(BASE, json={"question": "q", "format": "pdf"}).json()
    url = f"{DOWNLOAD}/{body['workflow_id']}/download"
    assert alice.get(url).status_code == 200
    assert bob.get(url).status_code == 404                     # same answer as "no such report"


def test_download_needs_the_report_permission_and_unknown_ids_are_404(world):  # noqa: F811
    client, *_ = world[0](_plan())
    assert client("nobody", "other:thing:x").get(f"{DOWNLOAD}/abc/download").status_code == 403
    c = client()
    for bad in ("nope", "..", "%2e%2e%2f%2e%2e%2fsecret", "a" * 300):
        assert c.get(f"{DOWNLOAD}/{bad}/download").status_code == 404


def test_a_tampered_index_row_cannot_read_files_outside_the_artifacts_folder(world):  # noqa: F811
    build, tmp = world
    client, registry, _ = build(_plan())
    c = client()
    body = c.post(BASE, json={"question": "q", "format": "pdf"}).json()
    url = f"{DOWNLOAD}/{body['workflow_id']}/download"
    assert c.get(url).status_code == 200
    with sqlite3.connect(registry._db_path) as db:
        db.execute("UPDATE reports SET relative_path = '../../connections.sqlite3'")
    leak = c.get(url)
    assert leak.status_code == 404 and b"SQLite" not in leak.content


def test_a_deleted_file_is_a_404_not_a_crash(world):  # noqa: F811
    build, tmp = world
    client, *_ = build(_plan())
    c = client()
    body = c.post(BASE, json={"question": "q", "format": "csv"}).json()
    for path in (tmp / "work" / "artifacts").rglob("*report.csv"):
        path.unlink()
    assert c.get(f"{DOWNLOAD}/{body['workflow_id']}/download").status_code == 404


def test_the_download_index_survives_a_restart(world):  # noqa: F811
    from api.database_runtime_registry import DatabaseRuntimeRegistry
    from api.security.authorization import AuthorizationPolicy
    build, tmp = world
    client, registry, _ = build(_plan())
    body = client().post(BASE, json={"question": "q", "format": "xlsx"}).json()
    registry.shutdown()
    again = DatabaseRuntimeRegistry(
        data_root=tmp / "data", work_dir=tmp / "work", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None)
    try:
        found = again.find_report("alice", body["workflow_id"])
        assert found is not None and found["path"].is_file() and found["filename"] == "report.xlsx"
        assert again.find_report("bob", body["workflow_id"]) is None
    finally:
        again.shutdown()
