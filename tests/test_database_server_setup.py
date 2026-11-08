from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from api import database_server_setup as setup
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from tests.test_table_reports_api import PERMISSION, READ, _Attach, _principal

CONNECT = "database:connection:manage"
DB = "/api/v1/database-connections"
SQLSERVER = {"name": "srv_two", "type": "sqlserver", "host": "10.62.10.34", "database": "master_data",
             "user": "hai_reader", "password": "Reader-pass-1",
             "creator": {"user": "hai_creator", "password": "Creator-pass-1"},
             "writer": {"user": "hai_writer", "password": "Writer-pass-1"}}


@pytest.fixture
def passing(monkeypatch):
    seen = {}

    def fake(name, profiles):
        seen["env"] = {k: os.environ.get(k) for k in ("HAI_DB_SRV_TWO_PASSWORD",)}
        seen["profile"] = profiles[name]
        return True, ["[ok]   connected"]
    monkeypatch.setattr(setup, "check_profile", fake)
    yield seen
    for key in [k for k in os.environ if k.startswith("HAI_DB_")]:
        del os.environ[key]


def test_entry_names_variables_after_the_profile_and_keeps_passwords_out_of_the_json():
    name, entry, secrets = setup.build_entry(SQLSERVER)
    assert name == "srv_two" and entry["server"] == "10.62.10.34" and "host" not in entry
    assert entry["password_env"] == "HAI_DB_SRV_TWO_PASSWORD"
    assert entry["creator"] == {"user": "hai_creator", "password_env": "HAI_DB_SRV_TWO_CREATOR_PASSWORD"}
    assert entry["writer"] == {"user": "hai_writer", "password_env": "HAI_DB_SRV_TWO_WRITER_PASSWORD"}
    assert "Reader-pass-1" not in json.dumps(entry)
    assert set(secrets.values()) == {"Reader-pass-1", "Creator-pass-1", "Writer-pass-1"}


@pytest.mark.parametrize("change,fragment", [
    ({"name": "Bad Name"}, "name"), ({"type": "oracle"}, "type must be"), ({"host": ""}, "host is required"),
    ({"password": "has space"}, ".env file cannot hold"), ({"password": ""}, "password is required"),
    ({"port": "abc"}, "port must be a number"), ({"extra": 1}, "unsupported fields"),
])
def test_bad_input_is_refused(change, fragment):
    with pytest.raises(setup.ServerSetupError, match=fragment):
        setup.build_entry({**SQLSERVER, **change})


def test_database_is_optional_only_when_a_creator_account_is_given():
    spec = {k: v for k, v in SQLSERVER.items() if k != "database"}
    assert setup.build_entry(spec)[1]["database"] == setup.NO_DATABASE
    with pytest.raises(setup.ServerSetupError, match="unless a creator account"):
        setup.build_entry({k: v for k, v in spec.items() if k not in {"creator", "writer"}})


def test_creator_only_server_is_tested_through_the_creator_account(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(setup, "check_profile", lambda *a: pytest.fail("no database to read"))
    from api import database_engines as engines
    monkeypatch.setattr(engines.SQLServerEngine, "open_creator",
                        lambda self, name: called.append(name) or (lambda: None))
    spec = {k: v for k, v in SQLSERVER.items() if k != "database"}
    profile = setup.register(spec, profiles_path=tmp_path / "p.json", env_path=tmp_path / ".env", taken=set())
    assert called == ["srv_two"] and profile.database == setup.NO_DATABASE
    for key in [k for k in os.environ if k.startswith("HAI_DB_")]:
        del os.environ[key]


def test_creator_only_server_with_a_refused_creator_saves_nothing(tmp_path, monkeypatch):
    from api import database_engines as engines

    def refuse(self, name):
        raise engines.EngineError("the creator login holds more than CREATE DATABASE")
    monkeypatch.setattr(engines.SQLServerEngine, "open_creator", refuse)
    spec = {k: v for k, v in SQLSERVER.items() if k != "database"}
    with pytest.raises(setup.ServerSetupError, match="creator account was refused"):
        setup.register(spec, profiles_path=tmp_path / "p.json", env_path=tmp_path / ".env", taken=set())
    assert not (tmp_path / "p.json").exists() and "HAI_DB_SRV_TWO_PASSWORD" not in os.environ


def test_postgresql_cannot_have_a_creator_or_writer():
    with pytest.raises(setup.ServerSetupError, match="read-only"):
        setup.build_entry({**SQLSERVER, "type": "postgresql"})


def test_a_working_server_is_saved_and_the_password_goes_only_to_env(tmp_path, passing):
    profiles, env = tmp_path / "profiles.json", tmp_path / ".env"
    profile = setup.register(SQLSERVER, profiles_path=profiles, env_path=env, taken=set())
    assert profile.host == "10.62.10.34" and profile.creator_user == "hai_creator"
    assert passing["env"]["HAI_DB_SRV_TWO_PASSWORD"] == "Reader-pass-1"        # set while testing
    assert "Reader-pass-1" not in profiles.read_text() and "Creator-pass-1" not in profiles.read_text()
    assert "HAI_DB_SRV_TWO_PASSWORD=Reader-pass-1" in env.read_text()
    assert json.loads(profiles.read_text())["srv_two"]["type"] == "sqlserver"


def test_a_failing_test_saves_nothing_and_leaves_no_variable_behind(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "check_profile", lambda n, p: (False, [
        "[FAIL] could not connect: OperationalError: host 10.62.10.34 user hai_reader",
        "       hint: the server is not reachable (is it running? check host and port)"]))
    profiles, env = tmp_path / "profiles.json", tmp_path / ".env"
    with pytest.raises(setup.ServerSetupError) as caught:
        setup.register(SQLSERVER, profiles_path=profiles, env_path=env, taken=set())
    assert "not reachable" in str(caught.value)
    assert "OperationalError" not in str(caught.value) and "10.62" not in str(caught.value)
    assert not profiles.exists() and not env.exists()
    assert "HAI_DB_SRV_TWO_PASSWORD" not in os.environ


def test_an_existing_name_is_refused_and_existing_profiles_are_kept(tmp_path, passing):
    profiles, env = tmp_path / "profiles.json", tmp_path / ".env"
    profiles.write_text(json.dumps({"local_mysql": {"type": "mysql"}}))
    with pytest.raises(setup.ServerSetupError, match="already exists"):
        setup.register(SQLSERVER, profiles_path=profiles, env_path=env, taken={"srv_two"})
    setup.register({**SQLSERVER, "name": "other"}, profiles_path=profiles, env_path=env, taken=set())
    assert set(json.loads(profiles.read_text())) == {"local_mysql", "other"}


# ----------------------------------------------------------------------------- through the API
@pytest.fixture
def client(tmp_path, passing):
    data = tmp_path / "data"
    data.mkdir()

    def factory(app):
        registry = DatabaseRuntimeRegistry(data_root=data, work_dir=tmp_path / "work",
                                           authorization_policy=app.state.authorization_policy,
                                           llm_provider_factory=lambda: None)
        registry.profiles_path, registry.env_path = tmp_path / "profiles.json", tmp_path / ".env"
        return registry
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=factory)
    app.add_middleware(_Attach, principal=_principal((CONNECT, PERMISSION, READ)))
    with TestClient(app) as test_client:
        yield test_client


def test_added_server_appears_in_the_lists_without_a_restart(client):
    assert client.get(f"{DB}/profiles").json()["server_registration"] is True
    r = client.post(f"{DB}/servers", json=SQLSERVER)
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "srv_two", "type": "sqlserver", "prefix": "hai_"}
    assert {"name": "srv_two", "type": "sqlserver", "prefix": "hai_"} in client.get(f"{DB}/profiles").json()["profiles"]
    assert "Reader-pass-1" not in r.text
    assert client.post(f"{DB}/servers", json=SQLSERVER).status_code == 422          # duplicate


def test_bad_request_is_a_clear_422(client):
    r = client.post(f"{DB}/servers", json={**SQLSERVER, "password": "has space"})
    assert r.status_code == 422 and ".env file cannot hold" in r.json()["detail"]["message"]


def test_server_can_be_removed_again(client):
    client.post(f"{DB}/servers", json=SQLSERVER)
    assert client.delete(f"{DB}/servers/srv_two").status_code == 200
    assert all(p["name"] != "srv_two" for p in client.get(f"{DB}/profiles").json()["profiles"])
    assert client.delete(f"{DB}/servers/srv_two").status_code == 404


def test_feature_is_off_when_no_files_are_configured(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    app = create_app(authorization_policy_factory=lambda _a: AuthorizationPolicy(),
                     database_runtime_registry_factory=lambda a: DatabaseRuntimeRegistry(
                         data_root=data, work_dir=tmp_path / "work",
                         authorization_policy=a.state.authorization_policy, llm_provider_factory=lambda: None))
    app.add_middleware(_Attach, principal=_principal((CONNECT, PERMISSION, READ)))
    with TestClient(app) as c:
        assert c.get(f"{DB}/profiles").json()["server_registration"] is False
        assert c.post(f"{DB}/servers", json=SQLSERVER).status_code == 422
