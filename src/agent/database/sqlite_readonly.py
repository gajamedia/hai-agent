"""Explicit SQLite read-only profile configuration.

The caller supplies a user-approved existing database file and an approval
reference.  This helper never creates a database file, discovers tables, or
accepts SQL from the caller.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .connection_manager import ConnectionManager, ConnectionProfile


class SQLiteReadOnlyConfigurationError(ValueError):
    """The requested SQLite file/profile cannot be safely configured."""


def register_sqlite_read_only_profile(
    manager: ConnectionManager,
    *,
    profile_name: str,
    database_path: str | Path,
    authorization_ref: str,
) -> ConnectionProfile:
    """Register an existing SQLite file using SQLite's immutable read-only mode.

    `mode=ro` is enforced during file opening.  `PRAGMA query_only=ON` is a
    defense-in-depth session policy and is verified on every new connection.
    """
    if not isinstance(manager, ConnectionManager):
        raise TypeError("manager must be ConnectionManager")
    path = Path(database_path).expanduser().resolve(strict=False)
    if not path.is_file():
        raise SQLiteReadOnlyConfigurationError(
            "SQLite database path must reference an existing regular file."
        )
    if not isinstance(authorization_ref, str) or not authorization_ref.strip():
        raise SQLiteReadOnlyConfigurationError(
            "authorization_ref must identify the user's approval."
        )
    uri = f"{path.as_uri()}?mode=ro"

    def connection_factory() -> sqlite3.Connection:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def read_only_guard(connection: sqlite3.Connection) -> None:
        connection.execute("PRAGMA query_only = ON")
        query_only = connection.execute("PRAGMA query_only").fetchone()
        database_list = connection.execute("PRAGMA database_list").fetchall()
        if not query_only or query_only[0] != 1:
            raise PermissionError("SQLite read-only session policy could not be verified.")
        # A fresh profile must point to exactly its main database; this detects
        # unexpected connection setup before the adapter gets access.
        if len(database_list) != 1 or database_list[0][1] != "main":
            raise PermissionError("SQLite database attachment state is invalid.")

    return manager.register(
        name=profile_name,
        database="sqlite",
        authorization_ref=authorization_ref.strip(),
        connection_factory=connection_factory,
        read_only_guard=read_only_guard,
        read_only=True,
        enabled=True,
    )


__all__ = ["SQLiteReadOnlyConfigurationError", "register_sqlite_read_only_profile"]
