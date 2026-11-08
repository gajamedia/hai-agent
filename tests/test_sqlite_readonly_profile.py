import sqlite3

import pytest

from agent.database import (
    ConnectionManager,
    SQLiteReadOnlyConfigurationError,
    register_sqlite_read_only_profile,
)


def create_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE report_data (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO report_data (value) VALUES ('safe')")


def test_registers_existing_sqlite_file_as_read_only(tmp_path):
    path = tmp_path / "reporting.sqlite3"
    create_database(path)
    manager = ConnectionManager()
    profile = register_sqlite_read_only_profile(
        manager,
        profile_name="reporting_ro",
        database_path=path,
        authorization_ref="user-approval:reporting-001",
    )

    assert profile.database == "sqlite"
    assert manager.health_check("reporting_ro").healthy is True
    with manager.connection("reporting_ro") as connection:
        assert connection.execute("SELECT value FROM report_data").fetchone()[0] == "safe"
        connection.execute("PRAGMA query_only = OFF")
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO report_data (value) VALUES ('blocked')")

    with sqlite3.connect(path) as verification:
        assert verification.execute("SELECT COUNT(*) FROM report_data").fetchone()[0] == 1
    manager.shutdown()


def test_never_creates_missing_sqlite_database_file(tmp_path):
    missing = tmp_path / "must-not-be-created.sqlite3"
    with pytest.raises(SQLiteReadOnlyConfigurationError):
        register_sqlite_read_only_profile(
            ConnectionManager(),
            profile_name="missing",
            database_path=missing,
            authorization_ref="user-approval:missing",
        )
    assert not missing.exists()
