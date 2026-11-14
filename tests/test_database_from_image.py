"""A database designed from a picture of one (an ERD): checked by content, read by a model, shown before it is made."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent.llm.contracts import LLMImage, LLMMessage, LLMRequest
from agent.llm.claude_code_provider import ClaudeCodeProvider
from agent.llm.open_code_provider import OpenCodeProvider
from agent.workflow.image_design import read_image
from tests.test_sqlite_write import _call, _env, _tables, world  # noqa: F401  (fixtures)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
DESIGN = {"supported": True, "database": "hai_erd", "title": "ERD", "tables": [
    {"name": "tb_pelanggan", "columns": [{"name": "nama", "type": "text", "nullable": False}]},
    {"name": "tb_order", "columns": [{"name": "id_pelanggan", "type": "integer"},
                                     {"name": "total", "type": "decimal"}]}]}


def _upload(client, user, data=PNG, name="erd.png", query=""):
    return client.request("POST", f"/api/v1/database-creation/image-plans?filename={name}{query}", content=data,
                          headers={"X-Test-User": user, "Content-Type": "application/octet-stream"})


def test_an_erd_image_becomes_a_design_that_is_shown_and_then_made(world, monkeypatch):
    client, provider, data = world
    seen = []
    original = provider.complete

    def spy(request):
        seen.append(request)
        return original(request)

    monkeypatch.setattr(provider, "complete", spy)
    provider.answers.append(json.dumps(DESIGN))
    plan = _upload(client, "maker")
    assert plan.status_code == 201, plan.text
    body = plan.json()
    assert body["status"] == "planned" and body["confirm_phrase"] == "BUAT DATABASE hai_erd"
    assert [t["name"] for t in body["plan"]["tables"]] == ["tb_pelanggan", "tb_order"]
    assert body["file"] == {"name": "erd.png", "kind": "image", "sent_to_model": True}
    assert len(seen) == 1 and seen[0].images == (LLMImage("image/png", PNG),)       # the model got the picture
    assert "ERD" in seen[0].messages[0].content or "diagram" in seen[0].messages[0].content
    assert not (data / "hai_erd.sqlite3").exists()                                   # nothing made before the phrase
    done = _call(client, "POST", f"/database-creation/plans/{body['plan_id']}/run", "maker",
                 json={"confirm": body["confirm_phrase"]})
    assert done.status_code == 200, done.text
    assert _tables(data / "hai_erd.sqlite3") == ["tb_order", "tb_pelanggan"]


def test_the_names_typed_in_the_form_win_and_are_still_checked(world):
    client, provider, _ = world
    provider.answers.append(json.dumps(DESIGN))
    body = _upload(client, "maker", query="&database=hai_lain").json()
    assert body["plan"]["database"] == "hai_lain" and body["confirm_phrase"] == "BUAT DATABASE hai_lain"
    provider.answers.append(json.dumps(DESIGN))
    bad = _upload(client, "maker", query="&database=salah").json()
    assert bad["status"] == "unsupported" and bad["plan_id"] is None


def test_a_picture_that_is_not_a_design_is_reported_not_made(world):
    client, provider, _ = world
    provider.answers.append(json.dumps({"supported": False, "reason": "Gambar ini foto kucing."}))
    body = _upload(client, "maker").json()
    assert body["status"] == "unsupported" and "kucing" in body["reason"] and body["plan_id"] is None


def test_the_file_is_judged_by_its_content_and_size_not_its_name(world):
    client, provider, _ = world
    for name, data in (("erd.png", b"just text"), ("erd.exe", PNG), ("erd.png", b""), ("erd.png", PNG[:4])):
        out = _upload(client, "maker", data=data, name=name).json()
        assert out["status"] == "refused" and out["plan_id"] is None, (name, out)
    assert provider.answers == []                                                    # the model was never asked


def test_the_image_route_needs_the_creation_permission(world):
    client, provider, _ = world
    assert _upload(client, "sclerk").status_code == 403


def test_read_image_recognises_the_three_formats():
    assert read_image("a.jpg", b"\xff\xd8\xff\xe0" + b"0" * 8).media_type == "image/jpeg"
    assert read_image("a.webp", b"RIFF\x00\x00\x00\x00WEBPVP8 ").media_type == "image/webp"
    assert read_image("A.PNG", PNG).media_type == "image/png"


def _request():
    return LLMRequest(messages=(LLMMessage("system", "s"), LLMMessage("user", "lihat gambar")),
                      images=(LLMImage("image/png", PNG),))


def test_opencode_gets_the_image_as_a_file_that_is_removed_afterwards():
    calls = []

    def runner(command, *a, **k):
        file = next(c for c in command if c.startswith("--file=")).split("=", 1)[1]
        calls.append((list(command), Path(file), Path(file).read_bytes()))
        out = json.dumps({"type": "text", "part": {"type": "text", "text": "{}"}})
        return subprocess.CompletedProcess(command, 0, out + "\n", "")

    provider = OpenCodeProvider(executable="opencode", runner=runner)
    try:
        provider.complete(_request())
    except Exception:  # noqa: BLE001 - only the command and the cleanup matter here
        pass
    command, path, content = calls[0]
    assert content == PNG and path.name == "image-1.png" and command[-1].startswith("[SYSTEM]")
    assert not path.exists() and not path.parent.exists()


def test_claude_reads_the_image_only_through_the_read_tool_in_its_own_folder():
    calls = []

    def runner(command, cwd, timeout, env, stdin_text):
        folder = Path(command[command.index("--add-dir") + 1])
        calls.append((list(command), folder, stdin_text, sorted(p.name for p in folder.iterdir())))
        return subprocess.CompletedProcess(command, 0, json.dumps({"result": "{}"}), "")

    provider = ClaudeCodeProvider(executable="claude", runner=runner)
    provider.complete(_request())
    command, folder, prompt, files = calls[0]
    assert command[command.index("--tools") + 1] == "Read" and "--allowedTools" in command
    assert files == ["image-1.png"] and str(folder / "image-1.png") in prompt
    assert not folder.exists()
    plain = provider.build_command(model=None, system_prompt=None)
    assert plain[plain.index("--tools") + 1] == "" and "--add-dir" not in plain      # no tools without an image


# ------------------------------------------------------------------ the table limit (HAI_MAX_TABLES)
def _many(n):
    return {"supported": True, "database": "hai_besar", "title": "Besar", "tables": [
        {"name": f"tb_{i}", "columns": [{"name": "nama", "type": "text"}]} for i in range(n)]}


def test_the_table_limit_defaults_to_twenty_and_follows_the_setting(world, monkeypatch):
    from agent.workflow.database_creation import max_tables
    client, provider, _ = world
    monkeypatch.delenv("HAI_MAX_TABLES", raising=False)
    assert max_tables() == 20
    provider.answers += [json.dumps(_many(24)), json.dumps(_many(24))]               # the second is the retry
    refused = _upload(client, "maker").json()
    assert refused["status"] == "unsupported" and "at most 20" in refused["reason"]
    monkeypatch.setenv("HAI_MAX_TABLES", "40")
    assert max_tables() == 40
    provider.answers.append(json.dumps(_many(24)))
    planned = _upload(client, "maker").json()
    assert planned["status"] == "planned" and len(planned["plan"]["tables"]) == 24


def test_a_bad_setting_falls_back_to_twenty(monkeypatch):
    from agent.workflow.database_creation import max_tables
    for bad in ("abc", "0", "-3", "201", "2.5"):
        monkeypatch.setenv("HAI_MAX_TABLES", bad)
        assert max_tables() == 20, bad
    monkeypatch.setenv("HAI_MAX_TABLES", "200")
    assert max_tables() == 200


def test_the_same_limit_applies_to_approving_tables(tmp_path, monkeypatch):
    import sqlite3

    from tests import test_database_connections_api as conn

    gen = conn.world.__wrapped__(tmp_path) if hasattr(conn.world, "__wrapped__") else None
    assert gen is not None
    build, root = next(gen)
    try:
        with sqlite3.connect(root / "data" / "sales.sqlite3") as db:
            db.execute("CREATE TABLE extra (id INTEGER PRIMARY KEY, label TEXT)")
        client, _ = build()
        assert conn._connect(client).status_code == 200
        both = {"tables": {**conn.APPROVE["tables"], "extra": {"columns": ["id", "label"]}}}
        monkeypatch.setenv("HAI_MAX_TABLES", "1")
        refused = client.post(f"{conn.DB}/sales/approve", json=both)
        assert refused.status_code == 422 and "between 1 and 1 tables" in refused.text
        monkeypatch.setenv("HAI_MAX_TABLES", "2")
        assert client.post(f"{conn.DB}/sales/approve", json=both).status_code == 200
    finally:
        try:
            next(gen)
        except StopIteration:
            pass


# ------------------------------------------------------------------ every table name starts with tb_
def test_every_table_name_starts_with_tb_on_every_path(world):
    from agent.workflow.database_creation import DatabaseDesigner, table_prefix
    import pytest
    from agent.llm.service import LLMService
    from agent.workflow.table_report_planner import PlanValidationError
    client, provider, _ = world
    provider.answers.append(json.dumps({"supported": True, "database": "hai_x", "title": "x", "tables": [
        {"name": "customer", "columns": [{"name": "nama", "type": "text"}]},
        {"name": "tb_order", "columns": [{"name": "total", "type": "decimal"}]}]}))
    body = _upload(client, "maker").json()
    assert [t["name"] for t in body["plan"]["tables"]] == ["tb_customer", "tb_order"]
    assert table_prefix() == "tb_"
    designer = DatabaseDesigner(LLMService(provider), prefix="hai_")
    doc = lambda *names: {"database": "hai_x", "title": "x", "tables": [  # noqa: E731
        {"name": n, "columns": [{"name": "a", "type": "text"}]} for n in names]}
    assert [t.name for t in designer.validate(doc("Z".lower())).tables] == ["tb_z"]
    with pytest.raises(PlanValidationError, match="repeated"):                       # customer and tb_customer clash
        designer.validate(doc("customer", "tb_customer"))
    with pytest.raises(PlanValidationError, match="Invalid"):                        # a prefix must not push it too long
        designer.validate(doc("a" * 40))
    designer.validate(doc("a" * 38))                                                 # 38 + 3 = 41, the longest allowed


def _designer(world):
    from agent.llm.service import LLMService
    from agent.workflow.database_creation import DatabaseDesigner
    return DatabaseDesigner(LLMService(world[1]), prefix="hai_")


def _doc(*names):
    return {"database": "hai_x", "title": "x", "tables": [
        {"name": n, "columns": [{"name": "a", "type": "text"}]} for n in names]}


def test_the_table_prefix_comes_from_the_setting(world, monkeypatch):
    from agent.workflow.database_creation import table_prefix
    designer = _designer(world)
    monkeypatch.delenv("HAI_TABLE_PREFIX", raising=False)
    assert table_prefix() == "tb_"                                                    # unset: the default
    monkeypatch.setenv("HAI_TABLE_PREFIX", "t_")
    assert [t.name for t in designer.validate(_doc("customer", "t_order")).tables] == ["t_customer", "t_order"]
    assert "Table names start with t_" in designer._system_prompt() and '"t_x"' in designer._system_prompt()


def test_an_empty_prefix_leaves_names_as_they_are(world, monkeypatch):
    from agent.workflow.database_creation import table_prefix
    designer = _designer(world)
    for empty in ("", "   "):
        monkeypatch.setenv("HAI_TABLE_PREFIX", empty)
        assert table_prefix() == ""
        assert [t.name for t in designer.validate(_doc("customer", "tb_order")).tables] == ["customer", "tb_order"]
    assert "Table names start with" not in designer._system_prompt()


def test_a_bad_prefix_falls_back_to_the_default(monkeypatch):
    from agent.workflow.database_creation import table_prefix
    for bad in ("TB_", "1t_", "a b", "x;drop", "abcdefghijk", "../"):
        monkeypatch.setenv("HAI_TABLE_PREFIX", bad)
        assert table_prefix() == "tb_", bad
    monkeypatch.setenv("HAI_TABLE_PREFIX", "abcdefghij")
    assert table_prefix() == "abcdefghij"


def test_a_file_without_a_table_name_follows_the_prefix_too(world, monkeypatch):
    client, _, _ = world
    csv = b"Produk,Qty\nkopi,1\n"
    up = lambda: client.request("POST", "/api/v1/database-creation/file-plans?filename=Penjualan.csv&database=hai_toko",  # noqa: E731
                                content=csv, headers={"X-Test-User": "maker", "Content-Type": "application/octet-stream"})
    monkeypatch.setenv("HAI_TABLE_PREFIX", "")
    assert up().json()["plan"]["tables"][0]["name"] == "penjualan"
    monkeypatch.setenv("HAI_TABLE_PREFIX", "m_")
    assert up().json()["plan"]["tables"][0]["name"] == "m_penjualan"
