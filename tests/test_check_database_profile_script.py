from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from api.database_engines import MySQLEngine, parse_profiles
from tests.test_mysql_engine import (  # noqa: F401  (fixture and live gate shared with the engine tests)
    GOOD, HOST, PORT, ROOT_PW, live, my,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_database_profile.py"
spec = importlib.util.spec_from_file_location("check_database_profile", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _profiles(**change):
    return parse_profiles({"local_mysql": {**GOOD, "password_env": "HAI_TEST_CHECK_PW", **change}})


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    """The tool loads .env into os.environ (it is a command line program). Whatever a test
    makes it load must never leak into the rest of the suite, so restore os.environ after."""
    import os
    saved = dict(os.environ)
    monkeypatch.setenv("HAI_TEST_CHECK_PW", "very-secret-password")
    yield
    os.environ.clear()
    os.environ.update(saved)


def _text(lines):
    return "\n".join(lines)


def test_unknown_profile_lists_the_available_ones():
    ok, lines = tool.check_profile("nope", _profiles())
    assert not ok and "local_mysql" in _text(lines)


def test_a_missing_password_variable_is_named_but_never_the_value(monkeypatch):
    monkeypatch.delenv("HAI_TEST_CHECK_PW")
    ok, lines = tool.check_profile("local_mysql", _profiles())
    text = _text(lines)
    assert not ok and "HAI_TEST_CHECK_PW is not set" in text


@pytest.mark.parametrize("code,needle", [
    (1045, "wrong user or password"), (1044, "no access to this database"),
    (1049, "does not exist"), (2003, "not reachable"), (2026, "TLS handshake"),
])
def test_known_mysql_errors_come_with_a_plain_hint(monkeypatch, code, needle):
    def boom(self, *a, **k):
        raise Exception(code, "driver said something")
    monkeypatch.setattr(MySQLEngine, "open", boom)
    ok, lines = tool.check_profile("local_mysql", _profiles())
    text = _text(lines)
    assert not ok and "driver said something" in text and needle in text
    assert "very-secret-password" not in text


def test_an_unknown_error_is_still_shown_without_a_made_up_hint(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("something odd")
    monkeypatch.setattr(MySQLEngine, "open", boom)
    ok, lines = tool.check_profile("local_mysql", _profiles())
    assert not ok and "RuntimeError: something odd" in _text(lines) and "hint:" not in _text(lines)


def test_a_missing_driver_says_how_to_install_it(monkeypatch):
    def boom(self, *a, **k):
        raise ImportError("No module named pymysql")
    monkeypatch.setattr(MySQLEngine, "open", boom)
    ok, lines = tool.check_profile("local_mysql", _profiles())
    assert not ok and "PyMySQL" in _text(lines)


def test_safety_rule_refusals_are_reported_as_such(monkeypatch):
    from api.database_engines import EngineError

    def refuse(self, *a, **k):
        raise EngineError("the database account holds more than SELECT (ALL PRIVILEGES)")
    monkeypatch.setattr(MySQLEngine, "open", refuse)
    ok, lines = tool.check_profile("local_mysql", _profiles())
    assert not ok and "safety rules" in _text(lines) and "ALL PRIVILEGES" in _text(lines)


def test_main_reports_a_bad_or_missing_profiles_file(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("HAI_DATABASE_PROFILES", raising=False)
    assert tool.main(["x", "--env-file", str(tmp_path / "absent.env")]) == 1
    assert "no profiles file" in capsys.readouterr().err
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"xx": {"type": "mysql", "password": "inline"}}), encoding="utf-8")
    assert tool.main(["xx", "--profiles", str(bad), "--env-file", str(tmp_path / "absent.env")]) == 1
    assert "password_env" in capsys.readouterr().err


@live
def test_a_working_select_only_account_passes_against_a_real_server(my, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setenv("HAI_TEST_MY_ROLE_PW", "rolepw")
    profiles = parse_profiles({"reader": {
        "type": "mysql", "host": HOST, "port": int(PORT), "database": my.db, "user": my.reader,
        "password_env": "HAI_TEST_MY_ROLE_PW", "ssl": "require"}})
    ok, lines = tool.check_profile("reader", profiles)
    text = _text(lines)
    assert ok, text
    assert "SELECT only" in text and "sales (5 columns)" in text and "order" in text
    assert "rolepw" not in text


@live
def test_real_failures_explain_themselves(my, monkeypatch):  # noqa: F811
    base = {"type": "mysql", "host": HOST, "port": int(PORT), "database": my.db, "ssl": "require"}
    monkeypatch.setenv("HAI_TEST_MY_ROLE_PW", "rolepw")
    monkeypatch.setenv("HAI_TEST_MY_ROOT_PW", ROOT_PW)
    monkeypatch.setenv("HAI_TEST_BAD_PW", "definitely-wrong")
    profiles = parse_profiles({
        "badpw": {**base, "user": my.reader, "password_env": "HAI_TEST_BAD_PW"},
        "root": {**base, "user": "root", "password_env": "HAI_TEST_MY_ROOT_PW"},
        "nodb": {**base, "user": my.reader, "password_env": "HAI_TEST_MY_ROLE_PW", "database": "no_such_db"},
        "noport": {**base, "user": my.reader, "password_env": "HAI_TEST_MY_ROLE_PW", "port": 1}})
    ok, lines = tool.check_profile("badpw", profiles)
    assert not ok and "wrong user or password" in _text(lines) and "definitely-wrong" not in _text(lines)
    ok, lines = tool.check_profile("root", profiles)
    assert not ok and "safety rules" in _text(lines) and ROOT_PW not in _text(lines)
    ok, lines = tool.check_profile("nodb", profiles)
    assert not ok and ("no access" in _text(lines) or "does not exist" in _text(lines))
    ok, lines = tool.check_profile("noport", profiles)
    assert not ok and "not reachable" in _text(lines)


def test_the_env_file_is_loaded_like_the_server_does_and_never_overrides_the_terminal(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    import os
    windows_path = r"C:\Projects\x\data"
    env = tmp_path / ".env"
    env.write_text("\n".join([
        "# comment",
        "HAI_TEST_ENVFILE_A=from-file",
        "HAI_TEST_ENVFILE_B='quoted value'",
        f"HAI_TEST_ENVFILE_C={windows_path}",
    ]) + "\n", encoding="utf-8")
    monkeypatch.setenv("HAI_TEST_ENVFILE_A", "from-terminal")
    for name in ("HAI_TEST_ENVFILE_B", "HAI_TEST_ENVFILE_C"):
        monkeypatch.delenv(name, raising=False)
    loaded = tool.load_env_file(env)
    assert sorted(loaded) == ["HAI_TEST_ENVFILE_B", "HAI_TEST_ENVFILE_C"]  # names only
    assert os.environ["HAI_TEST_ENVFILE_A"] == "from-terminal"
    assert os.environ["HAI_TEST_ENVFILE_B"] == "quoted value"
    assert os.environ["HAI_TEST_ENVFILE_C"] == windows_path
    for name in ("HAI_TEST_ENVFILE_B", "HAI_TEST_ENVFILE_C"):
        monkeypatch.delenv(name, raising=False)


def test_a_missing_env_file_is_simply_skipped(tmp_path):
    assert tool.load_env_file(tmp_path / "nope.env") == []


def test_main_uses_the_profiles_file_named_in_the_env_file(tmp_path, monkeypatch, capsys):
    pytest.importorskip("dotenv")
    monkeypatch.delenv("HAI_DATABASE_PROFILES", raising=False)
    profiles = tmp_path / "p.json"
    profiles.write_text(json.dumps({"local_mysql": {**GOOD, "password_env": "HAI_TEST_CHECK_PW"}}), encoding="utf-8")
    env = tmp_path / ".env"
    env.write_text(f"HAI_DATABASE_PROFILES={profiles}\n", encoding="utf-8")
    monkeypatch.setattr(MySQLEngine, "open", lambda self, *a, **k: (_ for _ in ()).throw(Exception(2003, "refused")))
    code = tool.main(["local_mysql", "--env-file", str(env)])
    out = capsys.readouterr().out
    monkeypatch.delenv("HAI_DATABASE_PROFILES", raising=False)
    assert code == 1 and "loaded 1 setting(s)" in out and "not reachable" in out


def test_loading_an_env_file_inside_a_test_does_not_leak_into_the_next_test(tmp_path):
    """Guards the isolation fixture itself (this once polluted the whole suite with the real .env)."""
    import os
    assert "HAI_TEST_LEAK_PROBE" not in os.environ
    env = tmp_path / ".env"
    env.write_text("HAI_TEST_LEAK_PROBE=1\n", encoding="utf-8")
    pytest.importorskip("dotenv")
    tool.load_env_file(env)
    assert os.environ["HAI_TEST_LEAK_PROBE"] == "1"


def test_the_probe_from_the_previous_test_is_gone():
    import os
    assert "HAI_TEST_LEAK_PROBE" not in os.environ


def test_the_password_length_is_shown_but_never_the_password(monkeypatch):
    monkeypatch.setattr(MySQLEngine, "open", lambda self, *a, **k: (_ for _ in ()).throw(Exception(1045, "denied")))
    ok, lines = tool.check_profile("local_mysql", _profiles())
    text = _text(lines)
    assert "(20 characters)" in text and "very-secret-password" not in text


@pytest.mark.parametrize("value,warning", [
    ('"quoted"', "quote character"), ("'single'", "quote character"), ("trailing ", "ends with a space"),
])
def test_values_that_look_mangled_get_a_warning(monkeypatch, value, warning):
    monkeypatch.setenv("HAI_TEST_CHECK_PW", value)
    monkeypatch.setattr(MySQLEngine, "open", lambda self, *a, **k: (_ for _ in ()).throw(Exception(1045, "denied")))
    ok, lines = tool.check_profile("local_mysql", _profiles())
    assert warning in _text(lines)
