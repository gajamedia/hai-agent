"""scripts/setup_mysql_writer_user.py: the statements it would run, and (live) what it really creates."""
from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_mysql_writer_user.py"
spec = importlib.util.spec_from_file_location("setup_writer", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

PORT = os.getenv("HAI_MYSQL_TEST_PORT")
HOST = os.getenv("HAI_MYSQL_TEST_HOST", "127.0.0.1")
ROOT_PW = os.getenv("HAI_MYSQL_TEST_ROOT_PASSWORD", "")
live = pytest.mark.skipif(not PORT, reason="set HAI_MYSQL_TEST_PORT for live MySQL tests")


def test_the_grants_are_data_changes_on_one_database_and_nothing_else():
    statements = tool.plan_statements("dbsales", "hai_writer", "pw", ("localhost", "127.0.0.1"))
    sql = " ".join(s for s, _ in statements)
    assert sql.count("GRANT SELECT, INSERT, UPDATE, DELETE ON `dbsales`.* TO") == 2
    for forbidden in ("DROP", "ALTER TABLE", "CREATE TABLE", "ALL PRIVILEGES ON", "WITH GRANT OPTION", "*.*"):
        assert forbidden not in sql.replace("REVOKE ALL PRIVILEGES, GRANT OPTION", "")
    assert all("pw" not in s for s, _ in statements)           # the password is a bound parameter


def test_tables_can_narrow_the_grant():
    sql = " ".join(s for s, _ in tool.plan_statements("db", "w", "p", ("localhost",), ("tb_bahan", "tb_x")))
    assert "ON `db`.`tb_bahan`" in sql and "ON `db`.`tb_x`" in sql and "`db`.*" not in sql


@pytest.mark.parametrize("db,user,hosts,tables", [
    ("db;drop", "w", ("localhost",), ()), ("db", "w'x", ("localhost",), ()),
    ("db", "w", ("a b",), ()), ("db", "w", ("localhost",), ("t;x",)), ("db", "w", (), ())])
def test_unsafe_names_are_refused(db, user, hosts, tables):
    with pytest.raises(tool.SetupError):
        tool.plan_statements(db, user, "p", hosts, tables)


@live
def test_it_creates_a_writer_the_server_accepts_and_wires_up_the_files(tmp_path, monkeypatch):
    import pymysql
    tag = uuid.uuid4().hex[:8]
    db, reader, writer = f"wsetup_{tag}", f"rd_{tag}", f"wr_{tag}"
    admin = pymysql.connect(host=HOST, port=int(PORT), user="root", password=ROOT_PW, autocommit=True)
    cur = admin.cursor()
    cur.execute(f"CREATE DATABASE {db}")
    cur.execute(f"CREATE TABLE {db}.t (id INT PRIMARY KEY AUTO_INCREMENT, v VARCHAR(5))")
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"loc": {
        "type": "mysql", "host": HOST, "port": int(PORT), "database": db, "user": reader,
        "password_env": "HAI_TEST_W_READER_PW", "ssl": "require"}}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    from api.database_engines import load_profiles
    lines = []
    try:
        code = tool.run(profile_name="loc", profiles=load_profiles(profiles), profiles_path=profiles,
                        env_path=env, admin_user="root", admin_password=ROOT_PW, writer_user=writer,
                        tables=(), confirm=lambda _q: True, out=lines.append,
                        hosts=("%",))
        assert code == 0, lines
        data = json.loads(profiles.read_text("utf-8"))
        assert data["loc"]["writer"] == {"user": writer, "password_env": "HAI_LOC_WRITER_PASSWORD"}
        password = [l for l in env.read_text("utf-8").splitlines() if l.startswith("HAI_LOC_WRITER_PASSWORD=")]
        assert password and len(password[0].split("=", 1)[1]) >= 20
        assert password[0].split("=", 1)[1] not in "\n".join(lines)      # never printed
        cur.execute(f"SHOW GRANTS FOR '{writer}'@'%'")
        grants = " ".join(r[0] for r in cur.fetchall())
        assert "SELECT, INSERT, UPDATE, DELETE" in grants and "DROP" not in grants
    finally:
        cur.execute(f"DROP DATABASE {db}")
        cur.execute(f"DROP USER IF EXISTS '{writer}'@'%'")
        admin.close()
        monkeypatch.delenv("HAI_LOC_WRITER_PASSWORD", raising=False)
        os.environ.pop("HAI_LOC_WRITER_PASSWORD", None)
