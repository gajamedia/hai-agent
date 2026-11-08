from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

from api import database_admin_setup as admin
from api import database_server_setup as setup
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy
from tests.test_table_reports_api import _Attach

MANAGE, CREATE = "database:connection:manage", "database:create:manage"
DB = "/api/v1/database-connections"
ADMIN_PASSWORD = "Adm1n-Secret-Pass"


class FakeConnection:
    def __init__(self, log, rows=None):
        self.log, self.rows, self.closed = log, rows or {}, False
        self.autocommit = True

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, conn):
        self.conn, self._last = conn, ""

    def execute(self, sql, args=()):
        self.conn.log.append((sql, tuple(args)))
        self._last = sql

    def fetchall(self):
        for key, value in self.conn.rows.items():
            if key in self._last:
                return value
        return []

    def fetchone(self):
        return (0, 1)

    def close(self):
        pass


def fake_admin(log, rows=None, fail=None):
    def opener(kind, host, port, user, password, *, database="master", connect=None):
        if fail:
            raise fail
        return FakeConnection(log, rows)
    return opener


def test_account_names_are_derived_from_the_profile_and_fit_mysql():
    names = admin.account_names("a_very_long_profile_name_of_32_chrs", "mysql")
    assert names["reader"].startswith("hai_rd_") and all(len(n) <= 32 for n in names.values())
    assert admin.account_names("srv", "sqlserver")["creator"] == "hai_cr_srv"


def test_mysql_accounts_get_exactly_the_documented_rights(monkeypatch):
    log = []
    monkeypatch.setattr(admin, "open_admin", fake_admin(log))
    made = admin.provision("mysql", "127.0.0.1", "", "root", ADMIN_PASSWORD, "srv", database="dbsales")
    sql = [s for s, _ in log]
    assert made["names"] == {"reader": "hai_rd_srv", "creator": "hai_cr_srv", "writer": "hai_wr_srv"}
    assert any("GRANT SELECT ON `dbsales`.* TO" in s for s in sql)
    assert any("GRANT CREATE ON `hai\\_%%`.* TO" in s for s in sql)
    assert any("GRANT SELECT, INSERT, UPDATE, DELETE ON `dbsales`.* TO" in s for s in sql)
    assert not any(s.startswith("GRANT ALL") or "GRANT OPTION TO" in s for s in sql)
    assert {a[-1] for s, a in log if s.startswith("CREATE USER")} == {made["passwords"][r] for r in made["names"]}
    assert ADMIN_PASSWORD not in json.dumps(made) and all(len(p) == 24 for p in made["passwords"].values())
    assert {a[1] for s, a in log if s.startswith("CREATE USER")} == {"localhost", "127.0.0.1"}


def test_a_remote_mysql_server_gets_accounts_for_any_host(monkeypatch):
    log = []
    monkeypatch.setattr(admin, "open_admin", fake_admin(log))
    admin.provision("mysql", "10.62.10.34", 3306, "root", ADMIN_PASSWORD, "srv", database="dbsales",
                    roles=("reader",))
    assert {a[1] for s, a in log if s.startswith("CREATE USER")} == {"%"}
    assert not any("CREATE ON" in s for s, _ in log)


def test_sqlserver_creator_holds_only_create_any_database(monkeypatch):
    log = []
    monkeypatch.setattr(admin, "open_admin", fake_admin(log))
    admin.provision("sqlserver", "10.62.10.34", 1433, "sa", ADMIN_PASSWORD, "srv", roles=("reader", "creator"))
    sql = " ".join(s for s, _ in log)
    assert "GRANT CREATE ANY DATABASE TO [hai_cr_srv]" in sql
    assert "sysadmin" not in sql and "db_owner" not in sql and "dbcreator" not in sql


def test_a_database_or_a_creator_is_needed_and_the_read_account_is_always_made(monkeypatch):
    monkeypatch.setattr(admin, "open_admin", fake_admin([]))
    with pytest.raises(admin.AdminSetupError, match="give a database"):
        admin.provision("mysql", "127.0.0.1", "", "root", ADMIN_PASSWORD, "srv", roles=("reader",))
    with pytest.raises(admin.AdminSetupError, match="always created"):
        admin.provision("mysql", "127.0.0.1", "", "root", ADMIN_PASSWORD, "srv", database="d", roles=("creator",))
    with pytest.raises(admin.AdminSetupError, match="letters, digits"):
        admin.provision("mysql", "127.0.0.1", "", "root", ADMIN_PASSWORD, "srv", database="d`; DROP")


def test_a_wrong_admin_password_is_explained_without_raw_driver_text(monkeypatch):
    monkeypatch.setattr(admin, "open_admin", fake_admin([], fail=Exception(
        1045, f"Access denied for user 'root'@'10.1.1.1' (using password: YES) {ADMIN_PASSWORD}")))
    with pytest.raises(admin.AdminSetupError) as caught:
        admin.provision("mysql", "10.1.1.1", "", "root", ADMIN_PASSWORD, "srv", database="d")
    assert "administrator user or password is wrong" in str(caught.value)
    assert ADMIN_PASSWORD not in str(caught.value) and "10.1.1.1" not in str(caught.value)


def test_inspect_lists_existing_accounts_and_databases_and_changes_nothing(monkeypatch):
    log = []
    rows = {"mysql.user": [("hai_rd_srv",), ("hai_reader",)],
            "SHOW DATABASES": [("information_schema",), ("mysql",), ("dbsales",), ("hai_toko",)]}
    monkeypatch.setattr(admin, "open_admin", fake_admin(log, rows))
    found = admin.inspect("mysql", "127.0.0.1", "", "root", ADMIN_PASSWORD, "srv")
    assert found["databases"] == ["dbsales", "hai_toko"]
    assert found["plan"]["reader"] == {"user": "hai_rd_srv", "exists": True}
    assert found["plan"]["creator"] == {"user": "hai_cr_srv", "exists": False}
    assert all(s.startswith(("SELECT", "SHOW")) for s, _ in log)


# ----------------------------------------------------------------------------------- the API
def _user(perms, age=0):
    return AuthenticatedPrincipal(actor_id="admin-1", permissions=tuple(perms), auth_time=int(time.time()) - age,
                                  issuer="test", subject="admin-1", is_authenticated=True)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "check_profile", lambda n, p: (True, ["[ok]"]))
    monkeypatch.setattr(admin, "open_admin", fake_admin([], {"mysql.user": [], "SHOW DATABASES": [("dbsales",)]}))
    data = tmp_path / "data"
    data.mkdir()

    def build(principal, enabled=True):
        def factory(app):
            registry = DatabaseRuntimeRegistry(data_root=data, work_dir=tmp_path / "work",
                                               authorization_policy=app.state.authorization_policy,
                                               llm_provider_factory=lambda: None)
            registry.profiles_path, registry.env_path = tmp_path / "profiles.json", tmp_path / ".env"
            registry.admin_setup_enabled = enabled
            return registry
        app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                         database_runtime_registry_factory=factory)
        app.add_middleware(_Attach, principal=principal)
        return TestClient(app)
    yield build, tmp_path
    for key in [k for k in os.environ if k.startswith("HAI_DB_")]:
        del os.environ[key]


BODY = {"name": "srv", "type": "mysql", "host": "127.0.0.1", "admin_user": "root",
        "admin_password": ADMIN_PASSWORD, "database": "dbsales"}


def test_one_request_creates_the_accounts_and_registers_the_server(world):
    build, tmp = world
    with build(_user([MANAGE, CREATE])) as client:
        assert client.get(f"{DB}/profiles").json()["account_setup"] is True
        found = client.post(f"{DB}/servers/inspect", json=BODY)
        assert found.status_code == 200 and found.json()["databases"] == ["dbsales"]
        r = client.post(f"{DB}/servers/auto", json=BODY)
        assert r.status_code == 200, r.text
        assert r.json()["accounts"]["reader"] == "hai_rd_srv" and r.json()["type"] == "mysql"
        assert any(p["name"] == "srv" for p in client.get(f"{DB}/profiles").json()["profiles"])
        assert client.post(f"{DB}/servers/auto", json=BODY).status_code == 422          # name now taken
    stored = (tmp / "profiles.json").read_text() + r.text
    env = (tmp / ".env").read_text()
    assert ADMIN_PASSWORD not in stored and ADMIN_PASSWORD not in env
    assert "HAI_DB_SRV_PASSWORD=" in env and "HAI_DB_SRV_CREATOR_PASSWORD=" in env
    assert ADMIN_PASSWORD not in json.dumps(dict(os.environ))


def test_it_needs_the_create_permission_and_a_fresh_login(world):
    build, _ = world
    with build(_user([MANAGE])) as client:
        assert client.post(f"{DB}/servers/auto", json=BODY).status_code == 403
    with build(_user([MANAGE, CREATE], age=3600)) as client:
        r = client.post(f"{DB}/servers/auto", json=BODY)
        assert r.status_code == 403 and r.json()["detail"]["error"] == "step_up_required"
        assert client.post(f"{DB}/servers/inspect", json=BODY).status_code == 403


def test_it_can_be_switched_off(world):
    build, _ = world
    with build(_user([MANAGE, CREATE]), enabled=False) as client:
        assert client.get(f"{DB}/profiles").json()["account_setup"] is False
        r = client.post(f"{DB}/servers/auto", json=BODY)
        assert r.status_code == 422 and "switched off" in r.json()["detail"]["message"]
