"""MySQL engine tests.

Offline tests always run. Live tests need a disposable MySQL server:

    docker run -d --name hai_test_mysql -e MYSQL_ROOT_PASSWORD=rootpw -e MYSQL_DATABASE=salesdb \
        -p 127.0.0.1:53306:3306 mysql:8.4
    HAI_MYSQL_TEST_PORT=53306 HAI_MYSQL_TEST_ROOT_PASSWORD=rootpw pytest tests/test_mysql_engine.py
"""
from __future__ import annotations

import json
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from agent.llm.contracts import LLMResponse
from api.database_engines import (
    EngineError, MySQLProfile, _usable, check_mysql_grants, map_mysql_type, parse_profiles,
)
from api.database_runtime_registry import DatabaseRuntimeRegistry, RegistryError
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from tests.engine_conformance import ConformanceSuite, Harness
from tests.test_table_report_job_e2e import _context  # noqa: F401
from tests.test_table_reports_api import PERMISSION, READ, _Attach, _principal

CONNECT = "database:connection:manage"
PORT = os.getenv("HAI_MYSQL_TEST_PORT")
HOST = os.getenv("HAI_MYSQL_TEST_HOST", "127.0.0.1")  # CI service containers use their own host name
ROOT_PW = os.getenv("HAI_MYSQL_TEST_ROOT_PASSWORD", "")
live = pytest.mark.skipif(not PORT, reason="set HAI_MYSQL_TEST_PORT for live MySQL tests")
GOOD = {"type": "mysql", "host": "h", "database": "d", "user": "u", "password_env": "P"}


# ------------------------------------------------------------------ offline
def test_mysql_profile_defaults_are_safe():
    p = parse_profiles({"crm": GOOD})["crm"]
    assert isinstance(p, MySQLProfile) and (p.port, p.ssl) == (3306, "require")


@pytest.mark.parametrize("change", [
    {"ssl": "maybe"}, {"ssl": "disable"}, {"sslmode": "require"}, {"schema": "x"}, {"password": "p"},
    {"port": 70000}, {"statement_timeout_ms": 5}])
def test_invalid_mysql_profiles_are_rejected(change):
    with pytest.raises(EngineError):
        parse_profiles({"crm": {**GOOD, **change}})


def test_profiles_of_both_engines_can_share_a_file():
    pg = {"type": "postgresql", "host": "h", "database": "d", "user": "u", "password_env": "P"}
    got = parse_profiles({"a_pg": pg, "b_my": GOOD})
    assert (got["a_pg"].type, got["b_my"].type) == ("postgresql", "mysql")


def test_mysql_type_mapping():
    assert [map_mysql_type(t) for t in (
        "int", "bigint", "tinyint", "decimal", "double", "varchar", "text", "date", "datetime",
        "timestamp", "json")] == [
        "integer", "integer", "integer", "number", "number", "string", "string", "date",
        "datetime", "datetime", "string"]


@pytest.mark.parametrize("grants", [
    ["GRANT USAGE ON *.* TO `r`@`%`", "GRANT SELECT ON `d`.* TO `r`@`%`"],
    ["GRANT SELECT, USAGE ON *.* TO `r`@`%`"],
])
def test_select_only_grants_are_accepted(grants):
    check_mysql_grants(grants)


@pytest.mark.parametrize("grants,needle", [
    (["GRANT ALL PRIVILEGES ON *.* TO `root`@`%` WITH GRANT OPTION"], "ALL PRIVILEGES"),
    (["GRANT SELECT, INSERT ON `d`.* TO `r`@`%`"], "INSERT"),
    (["GRANT SELECT ON `d`.* TO `r`@`%`", "GRANT DELETE ON `d`.`t` TO `r`@`%`"], "DELETE"),
    (["GRANT `reporting_role`@`%` TO `r`@`%`"], "cannot be verified"),
    (["something unexpected"], "cannot be verified"),
])
def test_accounts_with_more_than_select_are_refused(grants, needle):
    with pytest.raises(EngineError, match=needle):
        check_mysql_grants(grants)


def test_only_names_the_adapter_cannot_represent_are_hidden():
    # Reserved words and mixed case are fine: every identifier is quoted.
    assert all(_usable(n) for n in ("sales", "order_total", "order", "user", "group", "Sales", "key"))
    assert not any(_usable(n) for n in ("a-b", "1abc", "with space", "naive\u00ef", "x;drop", ""))


def test_registry_lists_profiles_of_every_engine_without_hosts(tmp_path):
    pg = {"type": "postgresql", "host": "secret-host", "database": "d", "user": "u", "password_env": "P"}
    (tmp_path / "d").mkdir()
    reg = DatabaseRuntimeRegistry(
        data_root=tmp_path / "d", work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None, profiles=parse_profiles({"a_pg": pg, "b_my": GOOD}))
    assert reg.profiles() == [{"name": "a_pg", "type": "postgresql"},
                              {"name": "b_my", "type": "mysql"}]
    assert reg.profile_engine("b_my") == "mysql" and reg.profile_engine("zzz") is None
    assert reg.available_engines() == ["mysql", "postgresql", "sqlite"]


# --------------------------------------------------------------------- live
class _Provider:
    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)

    def complete(self, request):
        return LLMResponse(content=self.answers.pop(0), provider=self.name)


def _plan(**extra):
    return json.dumps({"supported": True, "table": "sales", "group_by": ["region"],
                       "metrics": [{"output": "revenue", "operation": "sum", "column": "amount"},
                                   {"output": "orders", "operation": "count"}], **extra})


@pytest.fixture
def my(monkeypatch, tmp_path):
    import pymysql
    tag = uuid.uuid4().hex[:8]
    db, reader, writer = f"rep_{tag}", f"rd_{tag}", f"wr_{tag}"
    admin = pymysql.connect(host=HOST, port=int(PORT), user="root", password=ROOT_PW,
                            autocommit=True)
    cur = admin.cursor()
    cur.execute(f"CREATE DATABASE {db}")
    cur.execute(f"CREATE TABLE {db}.sales (id INT PRIMARY KEY, region VARCHAR(20), "
                f"product VARCHAR(20), amount DECIMAL(10,2), sold_on DATE)")
    cur.execute(f"ALTER TABLE {db}.sales ADD COLUMN `weird name` VARCHAR(5)")
    cur.execute(f"CREATE TABLE {db}.`order` (id INT PRIMARY KEY, `group` VARCHAR(10), "
                f"`user` VARCHAR(10), amount DECIMAL(8,2))")
    cur.executemany(f"INSERT INTO {db}.`order` VALUES (%s,%s,%s,%s)",
                    [(i, ["a", "b"][i % 2], "u1", f"{i}.50") for i in range(1, 9)])
    rows = [(i, ["west", "east", "north"][i % 3], "widget" if i % 2 else "gadget",
             f"{10 + i}.25", f"2026-01-{(i % 28) + 1:02d}") for i in range(1, 31)]
    cur.executemany(f"INSERT INTO {db}.sales (id, region, product, amount, sold_on) "
                    f"VALUES (%s,%s,%s,%s,%s)", rows)
    for user in (reader, writer):
        cur.execute(f"CREATE USER '{user}'@'%' IDENTIFIED BY 'rolepw'")
        cur.execute(f"GRANT SELECT ON {db}.* TO '{user}'@'%'")
    cur.execute(f"GRANT INSERT ON {db}.sales TO '{writer}'@'%'")
    monkeypatch.setenv("HAI_TEST_MY_ROLE_PW", "rolepw")
    monkeypatch.setenv("HAI_TEST_MY_ROOT_PW", ROOT_PW)

    def profile(user, pw_env="HAI_TEST_MY_ROLE_PW", **extra):
        return {"type": "mysql", "host": HOST, "port": int(PORT), "database": db,
                "user": user, "password_env": pw_env, "ssl": "require", **extra}

    profiles = parse_profiles({
        "my_reader": profile(reader), "my_writer": profile(writer),
        "my_root": profile("root", "HAI_TEST_MY_ROOT_PW"),
        "my_wrongpw": profile(reader, "HAI_TEST_MY_ROOT_PW")})
    made = []

    def registry(*answers):
        r = DatabaseRuntimeRegistry(
            data_root=tmp_path, work_dir=tmp_path / "work", authorization_policy=AuthorizationPolicy(),
            llm_provider_factory=lambda: _Provider(*answers), profiles=profiles)
        made.append(r)
        return r
    try:
        yield SimpleNamespace(registry=registry, db=db, admin=cur, reader=reader)
    finally:
        for r in made:
            r.shutdown()
        cur.execute(f"DROP DATABASE {db}")
        for user in (reader, writer):
            cur.execute(f"DROP USER '{user}'@'%'")
        admin.close()


@live
def test_select_only_account_connects_and_hidden_names_are_reported(my):
    out = my.registry().connect("mysales", engine="mysql", source="my_reader", actor_id="alice")
    tables = {t["name"]: t for t in out["schema"]["tables"]}
    assert set(tables) == {"sales", "order"}  # reserved word is fine; it is quoted
    assert "weird name" not in {c["name"] for c in tables["sales"]["columns"]}
    cols = {c["name"]: c for c in tables["sales"]["columns"]}
    assert cols["id"]["primary_key"] is True and cols["amount"]["declared_type"] == "decimal"
    assert any("hidden" in w for w in out["warnings"])


@live
def test_session_is_read_only_and_uses_tls(my):
    reg = my.registry()
    reg.connect("mysales", engine="mysql", source="my_reader", actor_id="alice")
    with reg._connections["mysales"].manager.connection("mysales") as c:
        cur = c.cursor()
        cur.execute("SELECT @@session.transaction_read_only")
        assert cur.fetchone()[0] == 1
        cur.execute("SHOW SESSION STATUS LIKE 'Ssl_cipher'")
        assert cur.fetchone()[1]  # a negotiated cipher means the link is encrypted
        with pytest.raises(Exception):
            cur.execute(f"CREATE TABLE {my.db}.x (a INT)")


@live
def test_root_and_write_capable_accounts_are_refused(my):
    reg = my.registry()
    with pytest.raises(RegistryError, match="more than SELECT"):
        reg.connect("a1", engine="mysql", source="my_root", actor_id="alice")
    with pytest.raises(RegistryError, match="INSERT"):
        reg.connect("a2", engine="mysql", source="my_writer", actor_id="alice")
    assert reg.list() == []


@live
def test_mysql_connection_survives_restart_and_api_connects_by_profile(my):
    reg = my.registry()
    reg.connect("mysales", engine="mysql", source="my_reader", actor_id="alice")
    reg.activate("mysales", {"sales": {"columns": ["id", "region", "amount"], "key_column": "id"}},
                 actor_id="bob")
    reg.shutdown()
    again = my.registry()
    (item,) = again.list()
    assert item["status"] == "active" and item["database"] == "mysql"
    again.disconnect("mysales")

    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda _a: again)
    app.add_middleware(_Attach, principal=_principal((CONNECT, PERMISSION, READ)))
    with TestClient(app) as client:
        listing = client.get("/api/v1/database-connections/profiles")
        assert {"name": "my_reader", "type": "mysql"} in listing.json()["profiles"]
        assert HOST not in listing.text
        ok = client.post("/api/v1/database-connections", json={"name": "viaapi", "profile": "my_reader"})
        assert ok.status_code == 200 and ok.json()["database"] == "mysql"
        assert client.post("/api/v1/database-connections",
                           json={"name": "x2", "profile": "my_reader", "type": "postgresql"}
                           ).status_code == 422


@live
class TestMySQLConformance(ConformanceSuite):
    """The shared engine contract (see tests/engine_conformance.py)."""

    @pytest.fixture
    def harness(self, my):
        def probe(connection):
            cur = connection.cursor()
            cur.execute("SELECT @@session.transaction_read_only")
            assert cur.fetchone()[0] == 1
            with pytest.raises(Exception):
                cur.execute(f"CREATE TABLE {my.db}.x (a INT)")

        def sql(query):
            my.admin.execute(query)
            return my.admin.fetchall()
        return Harness(
            engine="mysql", registry=my.registry, reader="my_reader", writer="my_writer",
            privileged="my_root", wrong_password="my_wrongpw", sql=sql,
            q_sales=f"{my.db}.sales", q_order=f"{my.db}.`order`", order_name="order",
            quote=lambda column: f"`{column}`", assert_read_only=probe,
            leak_tokens=[HOST, my.reader, my.db, ROOT_PW])


def test_a_profiles_file_saved_with_a_utf8_bom_is_accepted(tmp_path):
    """Windows PowerShell 5.1 (Set-Content -Encoding utf8) and Notepad add a BOM."""
    from api.database_engines import load_profiles
    path = tmp_path / "profiles.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"crm": GOOD}).encode("utf-8"))
    assert load_profiles(path)["crm"].type == "mysql"


def test_a_profiles_file_without_a_bom_still_loads(tmp_path):
    from api.database_engines import load_profiles
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"crm": GOOD}), encoding="utf-8")
    assert load_profiles(path)["crm"].host == "h"
