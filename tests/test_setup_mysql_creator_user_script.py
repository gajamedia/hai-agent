"""scripts/setup_mysql_creator_user.py: the statements it would run, and (live) what it really creates."""
from __future__ import annotations

import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_mysql_creator_user.py"
spec = importlib.util.spec_from_file_location("setup_creator", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

PORT = os.getenv("HAI_MYSQL_TEST_PORT")
HOST = os.getenv("HAI_MYSQL_TEST_HOST", "127.0.0.1")
ROOT_PW = os.getenv("HAI_MYSQL_TEST_ROOT_PASSWORD", "")
live = pytest.mark.skipif(not PORT, reason="set HAI_MYSQL_TEST_PORT for live MySQL tests")
PATTERN = "`hai\\_%%`.*"   # %% because the driver formats statements that have parameters


def test_the_creator_gets_only_create_on_the_prefix_pattern():
    statements = tool.plan_statements("hai_", "hai_creator", "pw", "rd", "wr", ("localhost",))
    sql = [s for s, _ in statements]
    assert f"GRANT CREATE ON {PATTERN} TO %s@%s" in sql
    creator_grants = [s for s, a in statements if s.startswith("GRANT") and a[0] == "hai_creator"]
    assert creator_grants == [f"GRANT CREATE ON {PATTERN} TO %s@%s"]
    assert f"GRANT SELECT ON {PATTERN} TO %s@%s" in sql
    assert f"GRANT SELECT, INSERT, UPDATE, DELETE ON {PATTERN} TO %s@%s" in sql
    assert not any("DROP" in s.replace("REVOKE ALL PRIVILEGES, GRANT OPTION", "") for s in sql)
    assert all("pw" not in s for s in sql)


@pytest.mark.parametrize("prefix,creator,hosts", [
    ("HAI_", "c", ("localhost",)), ("hai", "c", ("localhost",)), ("hai_%", "c", ("localhost",)),
    ("hai_", "c;x", ("localhost",)), ("hai_", "c", ("a b",)), ("hai_", "c", ())])
def test_unsafe_input_is_refused(prefix, creator, hosts):
    with pytest.raises(tool.SetupError):
        tool.plan_statements(prefix, creator, "p", "rd", "", hosts)


@live
def test_it_creates_a_creator_the_server_accepts_and_wires_up_the_files(tmp_path):
    import pymysql
    tag = uuid.uuid4().hex[:6]
    db, reader, creator = f"csetup_{tag}", f"rd_{tag}", f"cr_{tag}"
    admin = pymysql.connect(host=HOST, port=int(PORT), user="root", password=ROOT_PW, autocommit=True)
    cur = admin.cursor()
    cur.execute(f"CREATE DATABASE {db}")
    cur.execute(f"CREATE USER '{reader}'@'%' IDENTIFIED BY 'x'")
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"loc": {
        "type": "mysql", "host": HOST, "port": int(PORT), "database": db, "user": reader,
        "password_env": "HAI_TEST_C_READER_PW", "ssl": "require"}}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    from api.database_engines import load_profiles
    lines = []
    try:
        code = tool.run(profile_name="loc", profiles=load_profiles(profiles), profiles_path=profiles,
                        env_path=env, admin_user="root", admin_password=ROOT_PW, creator_user=creator,
                        prefix="hai_", confirm=lambda _q: True, out=lines.append, hosts=("%",))
        assert code == 0, lines
        data = json.loads(profiles.read_text("utf-8"))
        assert data["loc"]["creator"] == {"user": creator, "password_env": "HAI_LOC_CREATOR_PASSWORD",
                                          "prefix": "hai_"}
        password = [l for l in env.read_text("utf-8").splitlines() if l.startswith("HAI_LOC_CREATOR_PASSWORD=")]
        assert password and password[0].split("=", 1)[1] not in "\n".join(lines)
        cur.execute(f"SHOW GRANTS FOR '{creator}'@'%'")
        grants = " ".join(r[0] for r in cur.fetchall())
        assert "CREATE ON `hai\\_%`.*" in grants and "DROP" not in grants
    finally:
        cur.execute(f"DROP DATABASE {db}")
        for user in (reader, creator):
            cur.execute(f"DROP USER IF EXISTS '{user}'@'%'")
        admin.close()
        os.environ.pop("HAI_LOC_CREATOR_PASSWORD", None)
