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


def test_a_missing_profiles_file_says_what_to_do(tmp_path):
    from api.database_engines import load_profiles
    with pytest.raises(EngineError) as info:
        load_profiles(tmp_path / "profiles.json")
    text = str(info.value)
    assert "not found" in text and "HAI_DATABASE_PROFILES" in text and "TRY_DATA_REPORTS_LOCALLY" in text


def test_a_missing_data_folder_says_how_to_create_it(tmp_path):
    with pytest.raises(RegistryError) as info:
        DatabaseRuntimeRegistry(data_root=tmp_path / "nope", work_dir=tmp_path / "w",
                                authorization_policy=AuthorizationPolicy(),
                                llm_provider_factory=lambda: None)
    assert "make_sample_sales_db.py" in str(info.value) and "HAI_DATABASE_ROOT" in str(info.value)


def test_the_real_connection_error_goes_to_the_server_log_but_not_to_the_browser(tmp_path, caplog):
    """The operator can read why a connection failed; the HTTP caller only sees a generic message."""
    class Failing:
        paramstyle, quote, map_type = "format", "`", staticmethod(lambda t: "string")

        def open(self, manager, name, source, auth_ref):
            raise Exception(1045, "Access denied for user 'hai_reader'@'172.17.0.1' (using password: YES)")

    (tmp_path / "d").mkdir()
    registry = DatabaseRuntimeRegistry(
        data_root=tmp_path / "d", work_dir=tmp_path / "w", authorization_policy=AuthorizationPolicy(),
        llm_provider_factory=lambda: None, engines={"mysql": Failing()})
    with caplog.at_level("WARNING", logger="hai_agent.database_registry"):
        with pytest.raises(RegistryError) as info:
            registry.connect("shop", engine="mysql", source="local_mysql", actor_id="alice")
    assert str(info.value) == "cannot open database: connection failed"
    assert "1045" not in str(info.value) and "172.17" not in str(info.value)
    log = caplog.text
    assert "Access denied for user 'hai_reader'@'172.17.0.1'" in log
    assert "local_mysql" in log and "check_database_profile.py" in log


# ------------------------------------------------------------ changing data (live)
@pytest.fixture
def rw(my, monkeypatch, tmp_path):
    """A reader (SELECT) plus a separate writer account with data-changing rights on one database."""
    writer = f"dw_{uuid.uuid4().hex[:8]}"
    my.admin.execute(f"CREATE USER '{writer}'@'%' IDENTIFIED BY 'rolepw'")
    my.admin.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {my.db}.* TO '{writer}'@'%'")
    my.admin.execute(f"ALTER TABLE {my.db}.sales MODIFY id INT NOT NULL AUTO_INCREMENT")  # inserts get their key
    monkeypatch.setenv("HAI_TEST_MY_ROLE_PW", "rolepw")
    base = {"type": "mysql", "host": HOST, "port": int(PORT), "database": my.db, "user": my.reader,
            "password_env": "HAI_TEST_MY_ROLE_PW", "ssl": "require"}
    profiles = parse_profiles({
        "my_rw": {**base, "writer": {"user": writer, "password_env": "HAI_TEST_MY_ROLE_PW"}},
        "my_ro": base})

    def build(profile="my_rw"):
        reg = DatabaseRuntimeRegistry(
            data_root=tmp_path, work_dir=tmp_path / "w2", authorization_policy=AuthorizationPolicy(),
            llm_provider_factory=lambda: _Provider(), profiles=profiles)
        reg.connect("sales_rw", engine="mysql", source=profile, actor_id="alice")
        reg.activate("sales_rw", {"sales": {
            "columns": ["id", "region", "product", "amount", "sold_on"],
            "filterable_columns": ["id", "region", "product", "amount", "sold_on"], "key_column": "id",
            "writable": {"operations": ["insert", "update", "delete"],
                         "columns": ["region", "product", "amount", "sold_on"]}}}, actor_id="bob")
        return reg

    made = []
    try:
        yield SimpleNamespace(build=lambda *a, **k: made.append(build(*a, **k)) or made[-1],
                              admin=my.admin, db=my.db, tmp=tmp_path)
    finally:
        for r in made:
            r.shutdown()
        my.admin.execute(f"DROP USER '{writer}'@'%'")


def _mysql_row_count(rw, where):
    rw.admin.execute(f"SELECT COUNT(*) FROM {rw.db}.sales WHERE {where}")
    return rw.admin.fetchone()[0]


def _run_change(service, payload, max_rows=100, approver=None):
    request = service.planner.validate(payload, max_rows=max_rows)
    preview = service.preview(request)
    return service.execute(request, expected_rows=preview.expected_rows, actor="alice",
                           approver=approver, plan_id="p", question="q"), preview


@live
def test_update_with_a_date_range_and_bound_values_matches_independent_sql(rw):
    service = rw.build().data_changes("sales_rw")
    expected = _mysql_row_count(rw, "sold_on BETWEEN '2026-01-05' AND '2026-01-10' AND region = 'west'")
    assert expected
    done, preview = _run_change(service, {
        "operation": "update", "table": "sales", "set": {"product": "x'; DROP TABLE s;--"},
        "conditions": [{"column": "sold_on", "operator": "between", "value": ["2026-01-05", "2026-01-10"]},
                       {"column": "region", "operator": "eq", "value": "west"}]})
    assert done.rows_affected == preview.expected_rows == expected
    assert _mysql_row_count(rw, "product LIKE 'x%DROP TABLE s%'") == expected    # stored as plain text
    assert _mysql_row_count(rw, "1=1") == 30                                    # the table is still there


@live
def test_setting_a_value_the_row_already_has_still_counts_the_row(rw):
    """MySQL normally counts only rows that changed; the safety check needs rows that matched."""
    service = rw.build().data_changes("sales_rw")
    done, _ = _run_change(service, {
        "operation": "update", "table": "sales", "set": {"amount": 11.25},
        "conditions": [{"column": "id", "operator": "eq", "value": 1}]})     # id 1 already has 11.25
    assert done.rows_affected == 1


@live
def test_delete_and_insert_work_and_the_audit_keeps_the_deleted_rows(rw):
    service = rw.build().data_changes("sales_rw")
    done, _ = _run_change(service, {
        "operation": "delete", "table": "sales",
        "conditions": [{"column": "id", "operator": "in", "value": [1, 2, 3]}]}, approver="bob")
    assert done.rows_affected == 3 and _mysql_row_count(rw, "id IN (1,2,3)") == 0
    assert sorted(r["id"] for r in done.before_image) == [1, 2, 3]
    ins, _ = _run_change(service, {
        "operation": "insert", "table": "sales",
        "rows": [{"region": "south", "product": "gadget", "amount": 5, "sold_on": "2026-02-01"}]})
    assert ins.rows_affected == 1 and _mysql_row_count(rw, "region = 'south'") == 1


@live
def test_the_users_row_limit_stops_a_change_that_is_too_big(rw):
    from agent.workflow.data_change import ChangeRefused
    service = rw.build().data_changes("sales_rw")
    request = service.planner.validate({
        "operation": "update", "table": "sales", "set": {"product": "z"},
        "conditions": [{"column": "region", "operator": "eq", "value": "west"}]}, max_rows=3)
    with pytest.raises(ChangeRefused, match="more than 3 rows"):
        service.preview(request)
    assert _mysql_row_count(rw, "product = 'z'") == 0


@live
def test_the_reader_connection_still_cannot_write_even_on_a_connection_that_can_change_data(rw):
    reg = rw.build()
    with reg._connections["sales_rw"].manager.connection("sales_rw") as c:
        cur = c.cursor()
        with pytest.raises(Exception):
            cur.execute(f"UPDATE {rw.db}.sales SET amount = 0")
    assert _mysql_row_count(rw, "amount = 0") == 0


@live
def test_a_profile_without_a_writer_cannot_enable_writing(rw):
    with pytest.raises(RegistryError, match="no writer account"):
        rw.build("my_ro")


@live
def test_a_writer_with_schema_changing_rights_is_refused(rw, monkeypatch):
    writer = f"dx_{uuid.uuid4().hex[:8]}"
    rw.admin.execute(f"CREATE USER '{writer}'@'%' IDENTIFIED BY 'rolepw'")
    rw.admin.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE, DROP ON {rw.db}.* TO '{writer}'@'%'")
    try:
        base = {"type": "mysql", "host": HOST, "port": int(PORT), "database": rw.db, "user": "unused_reader",
                "password_env": "HAI_TEST_MY_ROLE_PW", "ssl": "require"}
        profile = parse_profiles({"my_bad": {**base, "writer": {"user": writer,
                                                                  "password_env": "HAI_TEST_MY_ROLE_PW"}}})
        from api.database_engines import MySQLEngine
        with pytest.raises(EngineError, match="more than"):
            MySQLEngine(profile).open_writer("my_bad")
    finally:
        rw.admin.execute(f"DROP USER '{writer}'@'%'")
