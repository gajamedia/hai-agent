"""The optional MySQL writer account: profile parsing and the grants it may (not) hold."""
from __future__ import annotations

import pytest

from api.database_engines import (
    EngineError, MySQLEngine, check_mysql_writer_grants, parse_profiles,
)

BASE = {"type": "mysql", "host": "127.0.0.1", "database": "dbsales", "user": "hai_reader",
        "password_env": "R_PW"}


def test_a_writer_is_optional_and_described_by_names_only():
    plain = parse_profiles({"my": BASE})["my"]
    assert plain.writer_user == "" and plain.writer_password_env == ""
    with_writer = parse_profiles({"my": {**BASE, "writer": {"user": "hai_writer", "password_env": "W_PW"}}})["my"]
    assert (with_writer.writer_user, with_writer.writer_password_env) == ("hai_writer", "W_PW")


@pytest.mark.parametrize("writer", [
    {"user": "hai_reader", "password_env": "W_PW"},          # the same account as the reader
    {"user": "w"}, {"password_env": "W"}, {"user": "w", "password_env": "W", "password": "x"}, "w"])
def test_a_bad_writer_block_is_refused(writer):
    with pytest.raises(EngineError):
        parse_profiles({"my": {**BASE, "writer": writer}})


def test_postgresql_profiles_have_no_writer_yet():
    pg = {"type": "postgresql", "host": "h", "database": "d", "user": "u", "password_env": "P",
          "writer": {"user": "w", "password_env": "W"}}
    with pytest.raises(EngineError, match="unsupported"):
        parse_profiles({"pg": pg})


OK = ["GRANT USAGE ON *.* TO `w`@`%`",
      "GRANT SELECT, INSERT, UPDATE, DELETE ON `db_x`.* TO `w`@`%`"]


def test_a_writer_limited_to_data_changes_in_its_database_is_accepted():
    check_mysql_writer_grants(OK, "db_x")
    check_mysql_writer_grants(["GRANT SELECT, INSERT ON `db\\_x`.`tb_bahan` TO `w`@`%`"], "db_x")


@pytest.mark.parametrize("grant,needle", [
    ("GRANT DROP ON `db_x`.* TO `w`@`%`", "more than"),
    ("GRANT ALL PRIVILEGES ON `db_x`.* TO `w`@`%`", "more than"),
    ("GRANT CREATE, ALTER ON `db_x`.* TO `w`@`%`", "more than"),
    ("GRANT SELECT, UPDATE ON *.* TO `w`@`%`", "outside"),
    ("GRANT SELECT ON `other`.* TO `w`@`%`", "outside"),
    ("GRANT SELECT ON `db_x`.* TO `w`@`%` WITH GRANT OPTION", "grant privileges"),
    ("GRANT `some_role`@`%` TO `w`@`%`", "role or proxy"),
])
def test_a_writer_with_more_power_than_data_changes_is_refused(grant, needle):
    with pytest.raises(EngineError, match=needle):
        check_mysql_writer_grants([*OK[:1], grant], "db_x")


def test_open_writer_checks_the_grants_and_never_reuses_the_reader(monkeypatch):
    profile = parse_profiles({"my": {**BASE, "database": "db_x",
                                    "writer": {"user": "hai_writer", "password_env": "W_PW"}}})["my"]
    monkeypatch.setenv("W_PW", "secret")
    seen = []

    class Cursor:
        def __init__(self, grants): self.grants = grants
        def execute(self, sql, params=()): pass
        def fetchall(self): return [(g,) for g in self.grants]
        def close(self): pass

    class Connection:
        def __init__(self, **kw): seen.append(kw); self.closed = False
        def cursor(self): return Cursor(GRANTS)
        def close(self): self.closed = True

    GRANTS = OK
    engine = MySQLEngine({"my": profile}, connect=Connection)
    assert engine.has_writer("my") and not engine.has_writer("unknown")
    factory = engine.open_writer("my")
    assert seen[0]["user"] == "hai_writer" and seen[0]["autocommit"] is False

    GRANTS = ["GRANT SELECT, DROP ON `db_x`.* TO `w`@`%`"]
    with pytest.raises(EngineError, match="more than"):
        engine.open_writer("my")
    monkeypatch.delenv("W_PW")
    with pytest.raises(EngineError, match="not set"):
        engine.open_writer("my")
    assert callable(factory)
