from __future__ import annotations

"""Database engines the runtime registry can connect to.

Each engine knows how to open a read-only profile in a ``ConnectionManager``,
discover its schema, and map native column types to the report type system.

Credentials never travel through requests. Networked engines are described by a
server-side *profile* (host, database, user and the **name** of an environment
variable holding the password). A user can only pick a profile by name.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.database import ConnectionManager
from agent.database.sqlite_readonly import register_sqlite_read_only_profile
from agent.database.sqlite_schema import (
    SQLiteColumn, SQLiteSchema, SQLiteTable, discover_sqlite_schema,
)

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
MAX_TABLES = 200


class EngineError(ValueError):
    """A connection cannot be opened or violates the read-only policy."""


def map_sqlite_type(declared: str) -> str:
    t = (declared or "").upper()
    if "INT" in t:
        return "integer"
    if any(k in t for k in ("REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL")):
        return "number"
    if "BOOL" in t:
        return "boolean"
    if "DATETIME" in t or "TIMESTAMP" in t:
        return "datetime"
    if "DATE" in t:
        return "date"
    return "string"


_PG_INTEGER = {"smallint", "integer", "bigint"}
_PG_NUMBER = {"numeric", "real", "double precision", "money"}


def map_postgres_type(declared: str) -> str:
    t = (declared or "").lower()
    if t in _PG_INTEGER:
        return "integer"
    if t in _PG_NUMBER:
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "date":
        return "date"
    if t.startswith("timestamp"):
        return "datetime"
    return "string"


# ------------------------------------------------------------------ SQLite
class SQLiteEngine:
    kind = "sqlite"
    paramstyle = "qmark"
    map_type = staticmethod(map_sqlite_type)

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def describe_source(self, source: str) -> str:
        return source

    def open(self, manager: ConnectionManager, name: str, source: str, auth_ref: str):
        from api.database_runtime_registry import resolve_in_root  # avoids import cycle at load
        register_sqlite_read_only_profile(
            manager, profile_name=name, database_path=resolve_in_root(self.data_root, source),
            authorization_ref=auth_ref)
        return discover_sqlite_schema(manager, profile_name=name), []


# ---------------------------------------------------------------- PostgreSQL
@dataclass(frozen=True)
class PostgresProfile:
    name: str
    host: str
    port: int
    database: str
    user: str
    password_env: str
    schema: str = "public"
    sslmode: str = "require"
    statement_timeout_ms: int = 15000


def parse_profiles(data: Any) -> dict[str, PostgresProfile]:
    if not isinstance(data, Mapping):
        raise EngineError("profiles file must be a JSON object")
    profiles: dict[str, PostgresProfile] = {}
    for name, raw in data.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", name):
            raise EngineError(f"invalid profile name: {name!r}")
        if not isinstance(raw, Mapping) or raw.get("type") != "postgresql":
            raise EngineError(f"profile {name!r}: only type 'postgresql' is supported")
        if "password" in raw:
            raise EngineError(f"profile {name!r}: put the password in an environment "
                              "variable and reference it with password_env")
        unknown = set(raw) - {"type", "host", "port", "database", "user", "password_env",
                              "schema", "sslmode", "statement_timeout_ms"}
        if unknown:
            raise EngineError(f"profile {name!r}: unsupported fields {sorted(unknown)}")
        try:
            profile = PostgresProfile(
                name=name, host=str(raw["host"]).strip(), port=int(raw.get("port", 5432)),
                database=str(raw["database"]).strip(), user=str(raw["user"]).strip(),
                password_env=str(raw["password_env"]).strip(),
                schema=str(raw.get("schema", "public")),
                sslmode=str(raw.get("sslmode", "require")),
                statement_timeout_ms=int(raw.get("statement_timeout_ms", 15000)))
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"profile {name!r} is incomplete or invalid: {exc}") from exc
        if not all((profile.host, profile.database, profile.user, profile.password_env)):
            raise EngineError(f"profile {name!r} has empty fields")
        if not 1 <= profile.port <= 65535 or not _IDENT.match(profile.schema):
            raise EngineError(f"profile {name!r}: invalid port or schema")
        if profile.sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            raise EngineError(f"profile {name!r}: invalid sslmode")
        if not 100 <= profile.statement_timeout_ms <= 120000:
            raise EngineError(f"profile {name!r}: statement_timeout_ms out of range")
        profiles[name] = profile
    return profiles


def load_profiles(path: str | Path) -> dict[str, PostgresProfile]:
    try:
        return parse_profiles(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        if isinstance(exc, EngineError):
            raise
        raise EngineError(f"cannot read profiles file: {exc}") from exc


class PostgresEngine:
    kind = "postgresql"
    paramstyle = "format"
    map_type = staticmethod(map_postgres_type)

    def __init__(self, profiles: Mapping[str, PostgresProfile], *, connect=None) -> None:
        self.profiles = dict(profiles)
        self._connect = connect  # injectable for tests

    def describe_source(self, source: str) -> str:
        return source

    def open(self, manager: ConnectionManager, name: str, source: str, auth_ref: str):
        profile = self.profiles.get(source)
        if profile is None:
            raise EngineError("unknown database profile")
        password = os.environ.get(profile.password_env)
        if not password:
            raise EngineError(f"environment variable {profile.password_env} is not set")
        connect = self._connect
        if connect is None:
            import psycopg  # optional dependency, imported only when this engine is used
            connect = psycopg.connect

        def factory():
            conn = connect(
                host=profile.host, port=profile.port, dbname=profile.database,
                user=profile.user, password=os.environ.get(profile.password_env, ""),
                sslmode=profile.sslmode, connect_timeout=5, application_name="hai-agent-reports",
                options=(f"-c default_transaction_read_only=on "
                         f"-c statement_timeout={profile.statement_timeout_ms} "
                         f"-c idle_in_transaction_session_timeout=30000 "
                         f"-c search_path={profile.schema}"))
            conn.read_only = True  # BEGIN READ ONLY for every transaction
            return conn

        def guard(connection) -> None:
            cur = connection.cursor()
            try:
                cur.execute("SHOW transaction_read_only")
                row = cur.fetchone()
            finally:
                cur.close()
            if not row or str(row[0]).lower() != "on":
                raise PermissionError("PostgreSQL read-only session could not be verified.")

        manager.register(name=name, database="postgresql", authorization_ref=auth_ref,
                         connection_factory=factory, read_only_guard=guard,
                         read_only=True, enabled=True)
        with manager.connection(name) as connection:
            warnings = self._check_role(connection, profile)
            schema = self._discover(connection, name, profile)
        return schema, warnings

    @staticmethod
    def _rows(connection, sql: str, params: tuple = ()) -> list[tuple]:
        cur = connection.cursor()
        try:
            cur.execute(sql, params)
            return list(cur.fetchall())
        finally:
            cur.close()

    def _check_role(self, connection, profile: PostgresProfile) -> list[str]:
        """Refuse roles that could write; read-only session flags are only a second layer."""
        (superuser,) = self._rows(
            connection, "SELECT rolsuper FROM pg_roles WHERE rolname = current_user")[0]
        if superuser:
            raise EngineError("the database role is a superuser; use a SELECT-only role")
        writable = self._rows(
            connection,
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s "
            "AND table_type = 'BASE TABLE' AND has_table_privilege("
            "current_user, format('%%I.%%I', table_schema, table_name), "
            "'INSERT,UPDATE,DELETE,TRUNCATE') ORDER BY 1", (profile.schema,))
        if writable:
            names = ", ".join(r[0] for r in writable[:5])
            raise EngineError("the database role can modify data (" + names +
                              "); grant it SELECT only")
        return []

    def _discover(self, connection, name: str, profile: PostgresProfile) -> SQLiteSchema:
        cols = self._rows(
            connection,
            "SELECT c.table_name, c.column_name, c.data_type, c.is_nullable "
            "FROM information_schema.columns c JOIN information_schema.tables t "
            "ON t.table_schema = c.table_schema AND t.table_name = c.table_name "
            "AND t.table_type = 'BASE TABLE' WHERE c.table_schema = %s "
            "ORDER BY c.table_name, c.ordinal_position", (profile.schema,))
        # pg_catalog (not information_schema): constraint views hide primary keys
        # from roles that only hold SELECT, which is exactly the role we require.
        pks = {(r[0], r[1]) for r in self._rows(
            connection,
            "SELECT c.relname, a.attname FROM pg_index i JOIN pg_class c ON c.oid = i.indrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indisprimary AND n.nspname = %s", (profile.schema,))}
        tables: dict[str, list[SQLiteColumn]] = {}
        for table, column, data_type, nullable in cols:
            # Only lowercase identifiers: the adapter does not quote, and Postgres
            # would fold or reject anything else.
            if not (_IDENT.match(table) and _IDENT.match(column)):
                continue
            tables.setdefault(table, []).append(SQLiteColumn(
                name=column, declared_type=data_type, nullable=(nullable == "YES"),
                primary_key=(table, column) in pks))
        items = tuple(SQLiteTable(name=t, columns=tuple(c)) for t, c in sorted(tables.items()))
        return SQLiteSchema(profile_name=name, tables=items[:MAX_TABLES])


__all__ = ["EngineError", "PostgresEngine", "PostgresProfile", "SQLiteEngine",
           "load_profiles", "map_postgres_type", "map_sqlite_type", "parse_profiles"]
