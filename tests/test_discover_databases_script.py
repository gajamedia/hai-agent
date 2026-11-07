from __future__ import annotations

import importlib.util
import json
import socket
import sqlite3
import threading
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "discover_databases.py"
spec = importlib.util.spec_from_file_location("discover", SCRIPT)
tool = importlib.util.module_from_spec(spec)
import sys
sys.modules["discover"] = tool
spec.loader.exec_module(tool)


def _server(reply: bytes, *, speak_first: bool):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        try:
            conn, _ = listener.accept()
            with conn:
                if speak_first:
                    conn.sendall(reply)
                else:
                    conn.settimeout(2)
                    conn.recv(64)
                    conn.sendall(reply)
        except OSError:
            pass
        finally:
            listener.close()

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def test_a_mysql_greeting_is_recognised_and_mariadb_is_told_apart():
    greeting = b"\x4a\x00\x00\x00\x0a" + b"8.0.36\x00" + b"\x00" * 40
    port = _server(greeting, speak_first=True)
    hit = tool.probe_mysql("127.0.0.1", port)
    assert (hit.kind, hit.detail) == ("mysql", "MySQL 8.0.36")
    maria = b"\x4a\x00\x00\x00\x0a" + b"5.5.5-10.11.6-MariaDB\x00" + b"\x00" * 40
    assert tool.probe_mysql("127.0.0.1", _server(maria, speak_first=True)).detail.startswith("MariaDB")


def test_a_postgresql_answer_to_the_ssl_request_is_recognised():
    assert tool.probe_postgresql("127.0.0.1", _server(b"S", speak_first=False)).kind == "postgresql"
    assert tool.probe_postgresql("127.0.0.1", _server(b"N", speak_first=False)).kind == "postgresql"


def test_a_tds_response_is_recognised_as_sql_server():
    assert tool.probe_sqlserver("127.0.0.1", _server(b"\x04\x01\x00\x20\x00\x00\x01\x00", speak_first=False)).kind == "sqlserver"


def test_something_else_on_the_port_is_not_mistaken_for_a_database():
    assert tool.probe_mysql("127.0.0.1", _server(b"HTTP/1.1 200 OK\r\n", speak_first=True)) is None
    assert tool.probe_postgresql("127.0.0.1", _server(b"HTTP/1.1 400\r\n", speak_first=False)) is None
    free = socket.socket(); free.bind(("127.0.0.1", 0)); port = free.getsockname()[1]; free.close()
    assert tool.probe_mysql("127.0.0.1", port, timeout=0.3) is None   # nothing listening


def test_windows_services_are_matched_by_name():
    text = ("SERVICE_NAME: MySQL80\nSERVICE_NAME: postgresql-x64-16\nSERVICE_NAME: MSSQL$SQLEXPRESS\n"
            "SERVICE_NAME: SQLWriter\nSERVICE_NAME: SQLBrowser\nSERVICE_NAME: Spooler\nSERVICE_NAME: MSSQLSERVER\n")
    got = tool.windows_services(lambda *cmd: text)
    assert got == {"mysql": ["MySQL80"], "postgresql": ["postgresql-x64-16"], "sqlserver": ["MSSQL$SQLEXPRESS", "MSSQLSERVER"]}


def test_sqlite_files_are_found_by_header_not_by_name(tmp_path):
    good = tmp_path / "a.db"
    sqlite3.connect(good).execute("CREATE TABLE t(x)")
    (tmp_path / "fake.db").write_bytes(b"not a database at all")
    (tmp_path / "notes.txt").write_text("x")
    sub = tmp_path / "sub"; sub.mkdir()
    sqlite3.connect(sub / "b.sqlite3").execute("CREATE TABLE t(x)")
    hidden = tmp_path / ".hai-work"; hidden.mkdir()
    sqlite3.connect(hidden / "workflow.sqlite3").execute("CREATE TABLE t(x)")     # the agent's own file
    assert [p.name for p in tool.find_sqlite_files(tmp_path)] == ["a.db", "b.sqlite3"]
    assert tool.inside_root(good, tmp_path) and not tool.inside_root(good, tmp_path / "sub") and not tool.inside_root(good, None)


def _run(tmp_path, found, answers, extra=()):
    profiles = tmp_path / "profiles.json"
    lines, replies = [], iter(answers)
    code = tool.main(["--profiles", str(profiles), "--env", str(tmp_path / "none.env"), *extra],
                     hosts_probe=lambda hosts: found, services_probe=lambda: {},
                     ask=lambda prompt: next(replies), out=lines.append)
    return code, "\n".join(lines), profiles


MYSQL = tool.Found("mysql", "127.0.0.1", 3306, "MySQL 8.0.36")
SQLSERVER = tool.Found("sqlserver", "127.0.0.1", 1433, "Microsoft SQL Server")


def test_looking_only_reports_and_writes_nothing(tmp_path):
    code, text, profiles = _run(tmp_path, [MYSQL, SQLSERVER], [])
    assert code == 0 and "MySQL 8.0.36" in text and "NOT registered" in text and "no SQL Server engine" in text
    assert "--add" in text and not profiles.exists()


def test_add_writes_a_profile_without_any_password_and_names_the_next_step(tmp_path):
    code, text, profiles = _run(tmp_path, [MYSQL], ["y", "", "dbsales"], ["--add"])
    data = json.loads(profiles.read_text("utf-8"))
    assert code == 0 and data == {"local_mysql": {
        "type": "mysql", "host": "127.0.0.1", "port": 3306, "database": "dbsales", "user": "hai_reader",
        "password_env": "HAI_LOCAL_MYSQL_PASSWORD", "ssl": "require"}}
    assert "password" not in {k for k in data["local_mysql"] if k == "password"}
    assert "setup_mysql_readonly_user.py local_mysql" in text


def test_an_already_registered_server_is_left_alone_and_sql_server_is_never_added(tmp_path):
    (tmp_path / "profiles.json").write_text(json.dumps({"local_mysql": {
        "type": "mysql", "host": "localhost", "port": 3306, "database": "d", "user": "u", "password_env": "P"}}))
    code, text, profiles = _run(tmp_path, [MYSQL, SQLSERVER], [], ["--add"])     # no prompts are asked
    assert code == 0 and "registered as profile 'local_mysql'" in text
    assert list(json.loads(profiles.read_text("utf-8"))) == ["local_mysql"]


def test_bad_answers_are_skipped_and_declining_adds_nothing(tmp_path):
    _, text, profiles = _run(tmp_path, [MYSQL], ["y", "Bad Name!", "x; drop"], ["--add"])
    assert "skipped" in text and not profiles.exists()
    _, _, profiles = _run(tmp_path, [MYSQL], ["n"], ["--add"])
    assert not profiles.exists()


def test_a_postgresql_profile_uses_sslmode_and_an_odd_host_is_refused(tmp_path):
    pg = tool.Found("postgresql", "127.0.0.1", 5432, "PostgreSQL")
    assert tool.draft_profile(pg, name="local_pg", database="shop")["sslmode"] == "require"
    with pytest.raises(ValueError):
        tool.draft_profile(SQLSERVER, name="x1", database="d")
    code, text, _ = _run(tmp_path, [], [], ["--host", "10.0.0.1; rm -rf"])
    assert code == 2 and "plain host name" in text


def test_a_profile_written_by_the_script_is_accepted_by_the_app(tmp_path):
    from api.database_engines import parse_profiles
    _, _, profiles = _run(tmp_path, [MYSQL], ["y", "", "dbsales"], ["--add"])
    assert "local_mysql" in parse_profiles(json.loads(profiles.read_text("utf-8")))
