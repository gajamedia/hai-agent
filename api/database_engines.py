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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from agent.database import ConnectionManager
from agent.database.sqlite_readonly import register_sqlite_read_only_profile
from agent.database.sqlite_schema import (
    SQLiteColumn, SQLiteSchema, SQLiteTable, discover_sqlite_schema,
)

_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")  # profile schema names only
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")  # what the adapter accepts
MAX_TABLES = 200

# The adapter quotes every identifier, so reserved words and mixed case are fine.
# Only names it cannot represent (spaces, hyphens, non-ASCII) are hidden.
def _usable(name: str) -> bool:
    return bool(_NAME.match(name))


def _skipped_warning(count: int) -> list[str]:
    if not count:
        return []
    return [f"{count} table(s)/column(s) are hidden because their names contain "
            "characters other than letters, digits and underscores."]


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


_MSSQL_INTEGER = {"tinyint", "smallint", "int", "bigint"}
_MSSQL_NUMBER = {"decimal", "numeric", "float", "real", "money", "smallmoney"}
_MSSQL_DATETIME = {"datetime", "datetime2", "smalldatetime", "datetimeoffset"}


def map_sqlserver_type(declared: str) -> str:
    t = (declared or "").lower()
    if t in _MSSQL_INTEGER:
        return "integer"
    if t in _MSSQL_NUMBER:
        return "number"
    if t == "bit":
        return "boolean"
    if t == "date":
        return "date"
    if t in _MSSQL_DATETIME:
        return "datetime"
    return "string"


# ------------------------------------------------------------------ SQLite
class SQLiteEngine:
    kind = "sqlite"
    paramstyle = "qmark"
    quote = "`"  # backticks: SQLite never reinterprets them as string literals
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
    type: str = "postgresql"


@dataclass(frozen=True)
class MySQLProfile:
    name: str
    host: str
    port: int
    database: str
    user: str
    password_env: str
    ssl: str = "require"
    statement_timeout_ms: int = 15000
    type: str = "mysql"
    # Optional second account that may INSERT/UPDATE/DELETE. The reader above never can.
    writer_user: str = ""
    writer_password_env: str = ""
    # Optional third account that may only CREATE databases whose name starts with creator_prefix.
    creator_user: str = ""
    creator_password_env: str = ""
    creator_prefix: str = "hai_"


@dataclass(frozen=True)
class SQLServerProfile:
    r"""``host`` is the server as ODBC writes it: ``(local)\SQLEXPRESS``, ``name\instance`` or a host name.
    ``port`` 0 means "let the driver find it" (shared memory on this computer, or the SQL Browser)."""
    name: str
    host: str
    port: int
    database: str
    user: str
    password_env: str
    schema: str = "dbo"
    ssl: str = "require"
    driver: str = ""
    statement_timeout_ms: int = 15000
    type: str = "sqlserver"
    # Optional second login that may only CREATE DATABASE (and so owns what it creates).
    creator_user: str = ""
    creator_password_env: str = ""
    creator_prefix: str = "hai_"


_COMMON = {"type", "host", "port", "database", "user", "password_env", "statement_timeout_ms"}
_EXTRA = {"postgresql": {"schema", "sslmode"}, "mysql": {"ssl", "writer", "creator"},
          "sqlserver": {"server", "schema", "ssl", "driver", "creator"}}
_DEFAULT_PORT = {"postgresql": 5432, "mysql": 3306, "sqlserver": 0}
_MSSQL_SERVER = re.compile(r"^[A-Za-z0-9_.\-\\()]{1,100}$")
_MSSQL_NAME = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_MSSQL_DRIVER = re.compile(r"^[A-Za-z0-9 ._\-]{1,60}$")


def parse_profiles(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise EngineError("profiles file must be a JSON object")
    profiles: dict[str, Any] = {}
    for name, raw in data.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", name):
            raise EngineError(f"invalid profile name: {name!r}")
        kind = raw.get("type") if isinstance(raw, Mapping) else None
        if kind not in _EXTRA:
            raise EngineError(f"profile {name!r}: type must be one of {sorted(_EXTRA)}")
        if "password" in raw:
            raise EngineError(f"profile {name!r}: put the password in an environment "
                              "variable and reference it with password_env")
        unknown = set(raw) - _COMMON - _EXTRA[kind]
        if unknown:
            raise EngineError(f"profile {name!r}: unsupported fields {sorted(unknown)}")
        try:
            if kind == "sqlserver" and "server" in raw and "host" in raw:
                raise EngineError(f"profile {name!r}: give either server or host, not both")
            host = raw["server"] if kind == "sqlserver" and "server" in raw else raw["host"]
            common = dict(
                name=name, host=str(host).strip(),
                port=int(raw.get("port", _DEFAULT_PORT[kind])),
                database=str(raw["database"]).strip(), user=str(raw["user"]).strip(),
                password_env=str(raw["password_env"]).strip(),
                statement_timeout_ms=int(raw.get("statement_timeout_ms", 15000)))
            if kind == "postgresql":
                profile = PostgresProfile(
                    **common, schema=str(raw.get("schema", "public")),
                    sslmode=str(raw.get("sslmode", "require")))
            elif kind == "sqlserver":
                creator = raw.get("creator") or {}
                if creator and (not isinstance(creator, Mapping)
                                or set(creator) - {"user", "password_env", "prefix"}
                                or not creator.get("user") or not creator.get("password_env")
                                or not _MSSQL_NAME.match(str(creator.get("user")))
                                or not re.fullmatch(r"[a-z][a-z0-9]{0,15}_", str(creator.get("prefix", "hai_")))):
                    raise EngineError(f"profile {name!r}: creator must be "
                                      '{"user": "...", "password_env": "...", "prefix": "hai_"}')
                if creator and str(creator["user"]).strip() == common["user"]:
                    raise EngineError(f"profile {name!r}: the creator must be its own login")
                profile = SQLServerProfile(
                    **common, schema=str(raw.get("schema", "dbo")), ssl=str(raw.get("ssl", "require")),
                    driver=str(raw.get("driver", "")).strip(),
                    creator_user=str(creator.get("user", "")).strip(),
                    creator_password_env=str(creator.get("password_env", "")).strip(),
                    creator_prefix=str(creator.get("prefix", "hai_")))
            else:
                writer = raw.get("writer") or {}
                if writer and (not isinstance(writer, Mapping) or set(writer) - {"user", "password_env"}
                               or not writer.get("user") or not writer.get("password_env")):
                    raise EngineError(f"profile {name!r}: writer must be "
                                      '{"user": "...", "password_env": "..."}')
                if writer and str(writer["user"]).strip() == common["user"]:
                    raise EngineError(f"profile {name!r}: the writer must be a different account "
                                      "than the read-only user")
                creator = raw.get("creator") or {}
                if creator and (not isinstance(creator, Mapping)
                                or set(creator) - {"user", "password_env", "prefix"}
                                or not creator.get("user") or not creator.get("password_env")
                                or not re.fullmatch(r"[a-z][a-z0-9]{0,15}_", str(creator.get("prefix", "hai_")))):
                    raise EngineError(f"profile {name!r}: creator must be "
                                      '{"user": "...", "password_env": "...", "prefix": "hai_"}')
                if creator and str(creator["user"]).strip() in {common["user"], str(writer.get("user", ""))}:
                    raise EngineError(f"profile {name!r}: the creator must be its own account")
                profile = MySQLProfile(
                    **common, ssl=str(raw.get("ssl", "require")),
                    creator_user=str(creator.get("user", "")).strip(),
                    creator_password_env=str(creator.get("password_env", "")).strip(),
                    creator_prefix=str(creator.get("prefix", "hai_")),
                    writer_user=str(writer.get("user", "")).strip(),
                    writer_password_env=str(writer.get("password_env", "")).strip())
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(f"profile {name!r} is incomplete or invalid: {exc}") from exc
        if not all((profile.host, profile.database, profile.user, profile.password_env)):
            raise EngineError(f"profile {name!r} has empty fields")
        if not (0 if kind == "sqlserver" else 1) <= profile.port <= 65535:
            raise EngineError(f"profile {name!r}: invalid port")
        if not 100 <= profile.statement_timeout_ms <= 120000:
            raise EngineError(f"profile {name!r}: statement_timeout_ms out of range")
        if kind == "postgresql":
            if not _IDENT.match(profile.schema):
                raise EngineError(f"profile {name!r}: invalid schema")
            if profile.sslmode not in {"disable", "allow", "prefer", "require",
                                       "verify-ca", "verify-full"}:
                raise EngineError(f"profile {name!r}: invalid sslmode")
        elif kind == "sqlserver":
            # These values go into an ODBC connection string, so only plain characters are allowed.
            if not _MSSQL_SERVER.match(profile.host):
                raise EngineError(f"profile {name!r}: invalid server name")
            if not (_MSSQL_NAME.match(profile.database) and _MSSQL_NAME.match(profile.user)
                    and _MSSQL_NAME.match(profile.schema)):
                raise EngineError(f"profile {name!r}: database, user and schema use letters, digits and _ only")
            if profile.driver and not _MSSQL_DRIVER.match(profile.driver):
                raise EngineError(f"profile {name!r}: invalid driver name")
            if profile.ssl not in {"require", "verify"}:
                raise EngineError(f"profile {name!r}: ssl must be 'require' or 'verify' "
                                  "(SQL Server connections are always encrypted)")
        elif profile.ssl not in {"require", "verify"}:
            raise EngineError(f"profile {name!r}: ssl must be 'require' or 'verify' "
                              "(MySQL connections are always encrypted)")
        profiles[name] = profile
    return profiles


def load_profiles(path: str | Path) -> dict[str, Any]:
    try:
        return parse_profiles(json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig")))
    except FileNotFoundError as exc:
        raise EngineError(
            f"profiles file not found: {path}. Create it (docs/TRY_DATA_REPORTS_LOCALLY.md, section 5) "
            "or remove HAI_DATABASE_PROFILES from .env.") from exc
    except (OSError, ValueError) as exc:
        if isinstance(exc, EngineError):
            raise
        raise EngineError(f"cannot read profiles file: {exc}") from exc


class PostgresEngine:
    kind = "postgresql"
    paramstyle = "format"
    quote = '"'
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
            self._check_role(connection, profile)
            schema, skipped = self._discover(connection, name, profile)
        return schema, _skipped_warning(skipped)

    @staticmethod
    def _rows(connection, sql: str, params: tuple = ()) -> list[tuple]:
        cur = connection.cursor()
        try:
            cur.execute(sql, params)
            return list(cur.fetchall())
        finally:
            cur.close()

    def _check_role(self, connection, profile: PostgresProfile) -> None:
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

    def _discover(self, connection, name: str, profile: PostgresProfile):
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
        skipped = 0
        for table, column, data_type, nullable in cols:
            if not (_usable(table) and _usable(column)):
                skipped += 1
                continue
            tables.setdefault(table, []).append(SQLiteColumn(
                name=column, declared_type=data_type, nullable=(nullable == "YES"),
                primary_key=(table, column) in pks))
        items = tuple(SQLiteTable(name=t, columns=tuple(c)) for t, c in sorted(tables.items()))
        return SQLiteSchema(profile_name=name, tables=items[:MAX_TABLES]), skipped


# --------------------------------------------------------------------- MySQL
_MYSQL_INTEGER = {"tinyint", "smallint", "mediumint", "int", "integer", "bigint"}
_MYSQL_NUMBER = {"decimal", "numeric", "float", "double", "real"}
_MYSQL_ALLOWED_PRIVILEGES = {"USAGE", "SELECT"}
_GRANT = re.compile(r"^GRANT\s+(.+?)\s+ON\s+(.+?)\s+TO\s+", re.IGNORECASE | re.DOTALL)


def map_mysql_type(declared: str) -> str:
    t = (declared or "").lower()
    if t in _MYSQL_INTEGER:
        return "integer"
    if t in _MYSQL_NUMBER:
        return "number"
    if t == "date":
        return "date"
    if t in {"datetime", "timestamp"}:
        return "datetime"
    return "string"


def check_mysql_grants(grants: list[str]) -> None:
    """Refuse any account that holds more than SELECT (or an unverifiable role grant)."""
    for line in grants:
        match = _GRANT.match(line.strip())
        if not match:
            raise EngineError("the account has role or proxy grants that cannot be verified; "
                              "use an account with direct SELECT grants only")
        privileges = {p.strip().upper() for p in match.group(1).split(",")}
        excess = privileges - _MYSQL_ALLOWED_PRIVILEGES
        if excess:
            raise EngineError("the database account holds more than SELECT ("
                              + ", ".join(sorted(excess)[:5]) + "); grant it SELECT only")


_MYSQL_WRITER_PRIVILEGES = {"USAGE", "SELECT", "INSERT", "UPDATE", "DELETE"}


def check_mysql_writer_grants(grants: list[str], database: str, prefix: str = "",
                              also: tuple[str, ...] = ()) -> None:
    """The writer may only SELECT/INSERT/UPDATE/DELETE inside the profile's one database.

    No DDL (DROP, ALTER, CREATE, TRUNCATE needs DROP), no GRANT OPTION, no global grants, no roles.
    """
    for line in grants:
        text = line.strip()
        match = _GRANT.match(text)
        if not match:
            raise EngineError("the writer account has role or proxy grants that cannot be verified; "
                              "use an account with direct grants only")
        privileges = {p.strip().upper() for p in match.group(1).split(",")}
        excess = privileges - _MYSQL_WRITER_PRIVILEGES
        if excess:
            raise EngineError("the writer account holds more than SELECT, INSERT, UPDATE and DELETE ("
                              + ", ".join(sorted(excess)[:5]) + ")")
        if re.search(r"WITH\s+GRANT\s+OPTION\s*$", text, re.IGNORECASE):
            raise EngineError("the writer account can grant privileges to others (GRANT OPTION)")
        scope = match.group(2).replace("`", "").replace("\\_", "_").strip()
        in_scope = (any(scope.startswith(f"{d}.") for d in (database, *also))
                    or (prefix and scope.startswith(f"{prefix}%.")))
        if privileges != {"USAGE"} and not in_scope:
            raise EngineError("the writer account has privileges outside the database "
                              f"'{database}'; limit it to that database")


def check_mysql_creator_grants(grants: list[str], prefix: str) -> None:
    """The creator may only CREATE, and only databases whose name starts with the prefix."""
    for line in grants:
        text = line.strip()
        match = _GRANT.match(text)
        if not match:
            raise EngineError("the creator account has role or proxy grants that cannot be verified")
        privileges = {p.strip().upper() for p in match.group(1).split(",")}
        if privileges - {"USAGE", "CREATE"}:
            raise EngineError("the creator account may hold only CREATE (it can do more: "
                              + ", ".join(sorted(privileges - {"USAGE", "CREATE"})[:5]) + ")")
        if re.search(r"WITH\s+GRANT\s+OPTION\s*$", text, re.IGNORECASE):
            raise EngineError("the creator account can grant privileges to others (GRANT OPTION)")
        scope = match.group(2).replace("`", "").replace("\\_", "_").strip()
        if privileges != {"USAGE"} and scope != f"{prefix}%.*":
            raise EngineError(f"the creator account must be limited to databases named '{prefix}%'")


_DB_OVERRIDE = re.compile(r"^[a-z][a-z0-9_]{1,56}$")


class MySQLEngine:
    kind = "mysql"
    paramstyle = "format"
    quote = "`"
    map_type = staticmethod(map_mysql_type)

    def __init__(self, profiles: Mapping[str, MySQLProfile], *, connect=None) -> None:
        self.profiles = dict(profiles)
        self._connect = connect  # injectable for tests

    def open(self, manager: ConnectionManager, name: str, source: str, auth_ref: str):
        profile = self._resolve(source)
        if profile is None:
            raise EngineError("unknown database profile")
        if not os.environ.get(profile.password_env):
            raise EngineError(f"environment variable {profile.password_env} is not set")
        connect = self._connect
        if connect is None:
            import pymysql  # optional dependency, imported only when this engine is used
            connect = pymysql.connect
        tls: dict[str, Any] = {}
        if profile.ssl == "require":      # encrypted, certificate not verified
            tls = {"ssl": {"check_hostname": False}, "ssl_verify_cert": False}
        elif profile.ssl == "verify":     # encrypted and verified against system CAs
            tls = {"ssl": {}, "ssl_verify_cert": True, "ssl_verify_identity": True}
        timeout_s = max(1, profile.statement_timeout_ms // 1000)

        def factory():
            conn = connect(
                host=profile.host, port=profile.port, user=profile.user,
                password=os.environ.get(profile.password_env, ""), database=profile.database,
                connect_timeout=5, read_timeout=timeout_s, write_timeout=timeout_s,
                charset="utf8mb4", autocommit=False, **tls)
            cur = conn.cursor()
            try:
                cur.execute("SET SESSION TRANSACTION READ ONLY")
                cur.execute("SET SESSION max_execution_time = %s", (profile.statement_timeout_ms,))
            finally:
                cur.close()
            return conn

        def guard(connection) -> None:
            cur = connection.cursor()
            try:
                cur.execute("SELECT @@session.transaction_read_only")
                row = cur.fetchone()
            finally:
                cur.close()
            if not row or int(row[0]) != 1:
                raise PermissionError("MySQL read-only session could not be verified.")

        manager.register(name=name, database="mysql", authorization_ref=auth_ref,
                         connection_factory=factory, read_only_guard=guard,
                         read_only=True, enabled=True)
        with manager.connection(name) as connection:
            check_mysql_grants([r[0] for r in PostgresEngine._rows(connection, "SHOW GRANTS")])
            schema, skipped = self._discover(connection, name, profile)
        return schema, _skipped_warning(skipped)

    def _resolve(self, source: str):
        """A profile name, or "profile/database" for a database made by the creator (prefix only)."""
        name, _, database = source.partition("/")
        profile = self.profiles.get(name)
        if profile is None or not database:
            return profile
        if (not profile.creator_user or not _DB_OVERRIDE.match(database)
                or not database.startswith(profile.creator_prefix)
                or database == profile.creator_prefix):
            raise EngineError(f"only databases named '{profile.creator_prefix}...' can be chosen on this profile")
        return replace(profile, database=database)

    def has_writer(self, source: str) -> bool:
        profile = self._resolve(source)
        return bool(profile and profile.writer_user and profile.writer_password_env)

    def has_creator(self, name: str) -> bool:
        profile = self.profiles.get(name)
        return bool(profile and profile.creator_user and profile.creator_password_env)

    def open_creator(self, name: str):
        """A function opening connections for the account that may only CREATE prefixed databases."""
        profile = self.profiles.get(name)
        if profile is None or not self.has_creator(name):
            raise EngineError("this database profile has no creator account configured")
        if not os.environ.get(profile.creator_password_env):
            raise EngineError(f"environment variable {profile.creator_password_env} is not set")
        connect = self._connect
        if connect is None:
            import pymysql
            connect = pymysql.connect
        tls: dict[str, Any] = {}
        if profile.ssl == "require":
            tls = {"ssl": {"check_hostname": False}, "ssl_verify_cert": False}
        elif profile.ssl == "verify":
            tls = {"ssl": {}, "ssl_verify_cert": True, "ssl_verify_identity": True}
        timeout_s = max(1, profile.statement_timeout_ms // 1000)

        def factory():
            return connect(
                host=profile.host, port=profile.port, user=profile.creator_user,
                password=os.environ.get(profile.creator_password_env, ""),
                connect_timeout=5, read_timeout=timeout_s, write_timeout=timeout_s,
                charset="utf8mb4", autocommit=True, **tls)

        connection = factory()
        try:
            check_mysql_creator_grants(
                [r[0] for r in PostgresEngine._rows(connection, "SHOW GRANTS")], profile.creator_prefix)
        finally:
            connection.close()
        return factory

    def open_writer(self, source: str):
        """A function that opens write-capable connections for the profile's writer account.

        The account is verified once, here: it must hold only data-changing privileges on the
        profile's one database. The reader connections are never reused for writing.
        """
        profile = self._resolve(source)
        if profile is None or not self.has_writer(source):
            raise EngineError("this database profile has no writer account configured")
        if not os.environ.get(profile.writer_password_env):
            raise EngineError(f"environment variable {profile.writer_password_env} is not set")
        connect = self._connect
        # By default MySQL reports only rows whose value actually changed. The safety check compares the
        # statement's row count with the rows that were previewed, so it must count rows that MATCHED.
        flags: dict[str, Any] = {}
        if connect is None:
            import pymysql
            from pymysql.constants import CLIENT
            connect = pymysql.connect
            flags = {"client_flag": CLIENT.FOUND_ROWS}
        tls: dict[str, Any] = {}
        if profile.ssl == "require":
            tls = {"ssl": {"check_hostname": False}, "ssl_verify_cert": False}
        elif profile.ssl == "verify":
            tls = {"ssl": {}, "ssl_verify_cert": True, "ssl_verify_identity": True}
        timeout_s = max(1, profile.statement_timeout_ms // 1000)

        def factory():
            return connect(
                host=profile.host, port=profile.port, user=profile.writer_user,
                password=os.environ.get(profile.writer_password_env, ""), database=profile.database,
                connect_timeout=5, read_timeout=timeout_s, write_timeout=timeout_s,
                charset="utf8mb4", autocommit=False, **flags, **tls)

        connection = factory()
        try:
            check_mysql_writer_grants(
                [r[0] for r in PostgresEngine._rows(connection, "SHOW GRANTS")], profile.database,
                profile.creator_prefix if profile.creator_user else "",
                (self.profiles[source.partition("/")[0]].database,))
        finally:
            connection.close()
        return factory

    @staticmethod
    def _discover(connection, name: str, profile: MySQLProfile):
        rows = PostgresEngine._rows(
            connection,
            "SELECT c.table_name, c.column_name, c.data_type, c.is_nullable, c.column_key "
            "FROM information_schema.columns c JOIN information_schema.tables t "
            "ON t.table_schema = c.table_schema AND t.table_name = c.table_name "
            "AND t.table_type = 'BASE TABLE' WHERE c.table_schema = %s "
            "ORDER BY c.table_name, c.ordinal_position", (profile.database,))
        tables: dict[str, list[SQLiteColumn]] = {}
        skipped = 0
        for table, column, data_type, nullable, key in rows:
            if not (_usable(table) and _usable(column)):
                skipped += 1
                continue
            tables.setdefault(table, []).append(SQLiteColumn(
                name=column, declared_type=str(data_type), nullable=(nullable == "YES"),
                primary_key=(key == "PRI")))
        items = tuple(SQLiteTable(name=t, columns=tuple(c)) for t, c in sorted(tables.items()))
        return SQLiteSchema(profile_name=name, tables=items[:MAX_TABLES]), skipped


# ----------------------------------------------------------------- SQL Server
# Effective permissions the reading account may hold (database level, then server level).
_MSSQL_DB_ALLOWED = {"CONNECT", "SELECT", "REFERENCES", "VIEW DEFINITION", "VIEW DATABASE STATE",
                     "VIEW ANY COLUMN ENCRYPTION KEY DEFINITION", "VIEW ANY COLUMN MASTER KEY DEFINITION"}
_MSSQL_SERVER_ALLOWED = {"CONNECT SQL", "VIEW ANY DATABASE"}


_MSSQL_FIXED_ROLES = ("sysadmin", "securityadmin", "serveradmin", "setupadmin", "processadmin", "diskadmin",
                      "dbcreator", "bulkadmin")


def _server_roles(connection) -> list[str]:
    """Fixed server roles the connected login belongs to. fn_my_permissions does not show everything a role
    implies (dbcreator, for one), so the roles are asked directly."""
    sql = "SELECT " + ", ".join(f"CAST(IS_SRVROLEMEMBER('{r}') AS int)" for r in _MSSQL_FIXED_ROLES)
    row = PostgresEngine._rows(connection, sql)[0]
    return [role for role, member in zip(_MSSQL_FIXED_ROLES, row) if member == 1]


def check_sqlserver_permissions(database_permissions: list[str], server_permissions: list[str],
                                writable_tables: list[str], server_roles: list[str] = ()) -> None:
    """Refuse any account that can do more than read. Session flags do not exist in SQL Server,
    so this check (and a SELECT-only account) is the read-only guarantee, next to the SELECT-only
    statements the adapter builds."""
    if server_roles:
        raise EngineError("the login belongs to the server role(s) " + ", ".join(server_roles)
                          + "; use a login with read access to one database only")
    excess = {p.upper() for p in database_permissions} - _MSSQL_DB_ALLOWED
    if excess:
        raise EngineError("the database account holds more than SELECT ("
                          + ", ".join(sorted(excess)[:5]) + "); use a db_datareader-only login")
    excess = {p.upper() for p in server_permissions} - _MSSQL_SERVER_ALLOWED
    if excess:
        raise EngineError("the login holds server-wide rights (" + ", ".join(sorted(excess)[:5])
                          + "); use a login with read access to one database only")
    if writable_tables:
        raise EngineError("the database account can modify data (" + ", ".join(writable_tables[:5])
                          + "); grant it SELECT only")


def check_sqlserver_creator_permissions(server_permissions: list[str], server_roles: list[str] = ()) -> None:
    """The creator may hold CREATE ANY DATABASE and nothing else. (The dbcreator role is refused: it
    also carries ALTER ANY DATABASE, which reaches databases this application did not create.)"""
    if server_roles:
        raise EngineError("the creator login belongs to the server role(s) " + ", ".join(server_roles)
                          + "; use a login that holds only CREATE ANY DATABASE")
    perms = {p.upper() for p in server_permissions}
    excess = perms - {"CONNECT SQL", "VIEW ANY DATABASE", "CREATE ANY DATABASE"}
    if excess:
        raise EngineError("the creator login may hold only CREATE ANY DATABASE (it can do more: "
                          + ", ".join(sorted(excess)[:5]) + ")")
    if "CREATE ANY DATABASE" not in perms:
        raise EngineError("the creator login cannot create databases (GRANT CREATE ANY DATABASE)")


def _odbc_value(value: str) -> str:
    """Braces make an ODBC value literal, so a password with ; or = cannot add keywords."""
    return "{" + value.replace("}", "}}") + "}"


class SQLServerEngine:
    kind = "sqlserver"
    paramstyle = "qmark"
    quote = "["
    dialect = "top"          # SELECT TOP (n) ... instead of LIMIT n
    map_type = staticmethod(map_sqlserver_type)

    def __init__(self, profiles: Mapping[str, SQLServerProfile], *, connect=None) -> None:
        self.profiles = dict(profiles)
        self._connect = connect  # injectable for tests: connect(connection_string, timeout=...)

    def _connection_string(self, profile: SQLServerProfile, driver: str, *, user: str = "",
                           password_env: str = "", database: str = "") -> str:
        server = f"{profile.host},{profile.port}" if profile.port else profile.host
        trust = "no" if profile.ssl == "verify" else "yes"      # require: encrypted, certificate not verified
        password = os.environ.get(password_env or profile.password_env, "")
        return (f"DRIVER={{{driver}}};SERVER={server};DATABASE={database or profile.database};"
                f"UID={user or profile.user};PWD={_odbc_value(password)};Encrypt=yes;"
                f"TrustServerCertificate={trust};APP=hai-agent-reports")

    def _resolve(self, source: str):
        """A profile name, or "profile/database" for a database made by the creator (prefix only)."""
        name, _, database = source.partition("/")
        profile = self.profiles.get(name)
        if profile is None or not database:
            return profile
        if (not profile.creator_user or not _DB_OVERRIDE.match(database)
                or not database.startswith(profile.creator_prefix)
                or database == profile.creator_prefix):
            raise EngineError(f"only databases named '{profile.creator_prefix}...' can be chosen on this profile")
        return replace(profile, database=database)

    def has_creator(self, name: str) -> bool:
        profile = self.profiles.get(name)
        return bool(profile and profile.creator_user and profile.creator_password_env)

    def _driver(self, profile: SQLServerProfile):
        connect, driver = self._connect, profile.driver or "ODBC Driver 17 for SQL Server"
        if connect is None:
            import pyodbc  # optional dependency, imported only when this engine is used
            connect = pyodbc.connect
            if not profile.driver:
                installed = list(pyodbc.drivers())
                driver = next((d for d in ("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server",
                                           "ODBC Driver 13 for SQL Server") if d in installed), "")
                if not driver:
                    raise EngineError("no 'ODBC Driver 17/18 for SQL Server' is installed on this computer")
        return connect, driver

    def open_creator(self, name: str):
        """A function opening autocommit connections for the login that may only CREATE DATABASE."""
        profile = self.profiles.get(name)
        if profile is None or not self.has_creator(name):
            raise EngineError("this database profile has no creator login configured")
        if not os.environ.get(profile.creator_password_env):
            raise EngineError(f"environment variable {profile.creator_password_env} is not set")
        connect, driver = self._driver(profile)
        timeout_s = max(1, profile.statement_timeout_ms // 1000)
        text = self._connection_string(profile, driver, user=profile.creator_user,
                                       password_env=profile.creator_password_env, database="master")

        def factory():
            conn = connect(text, timeout=5, autocommit=True)   # CREATE DATABASE cannot run in a transaction
            conn.timeout = timeout_s
            return conn

        connection = factory()
        try:
            perms = [r[0] for r in PostgresEngine._rows(
                connection, "SELECT permission_name FROM sys.fn_my_permissions(NULL, 'SERVER')")]
            check_sqlserver_creator_permissions(perms, _server_roles(connection))
        finally:
            connection.close()
        return factory

    def open(self, manager: ConnectionManager, name: str, source: str, auth_ref: str):
        profile = self._resolve(source)
        if profile is None:
            raise EngineError("unknown database profile")
        if not os.environ.get(profile.password_env):
            raise EngineError(f"environment variable {profile.password_env} is not set")
        connect, driver = self._driver(profile)
        timeout_s = max(1, profile.statement_timeout_ms // 1000)
        text = self._connection_string(profile, driver)

        def factory():
            conn = connect(text, timeout=5)
            conn.timeout = timeout_s          # query timeout, seconds
            return conn

        def guard(connection) -> None:
            cur = connection.cursor()
            try:
                cur.execute("SELECT CAST(IS_SRVROLEMEMBER('sysadmin') AS int), CAST(IS_MEMBER('db_owner') AS int)")
                row = cur.fetchone()
            finally:
                cur.close()
            if not row or row[0] == 1 or row[1] == 1:
                raise PermissionError("the SQL Server login is an administrator or database owner; "
                                      "use a db_datareader-only login.")

        manager.register(name=name, database="sqlserver", authorization_ref=auth_ref,
                         connection_factory=factory, read_only_guard=guard,
                         read_only=True, enabled=True)
        with manager.connection(name) as connection:
            self._check_account(connection, profile)
            schema, skipped = self._discover(connection, name, profile)
        return schema, _skipped_warning(skipped)

    @staticmethod
    def _check_account(connection, profile: SQLServerProfile) -> None:
        rows = PostgresEngine._rows
        db_perms = [r[0] for r in rows(connection, "SELECT permission_name FROM sys.fn_my_permissions(NULL, 'DATABASE')")]
        server_perms = [r[0] for r in rows(connection, "SELECT permission_name FROM sys.fn_my_permissions(NULL, 'SERVER')")]
        object_name = "QUOTENAME(s.name) + '.' + QUOTENAME(t.name)"
        tests = " OR ".join(f"HAS_PERMS_BY_NAME({object_name}, 'OBJECT', '{p}') = 1"
                            for p in ("INSERT", "UPDATE", "DELETE", "ALTER"))
        writable = [r[0] for r in rows(
            connection,
            "SELECT t.name FROM sys.tables t JOIN sys.schemas s ON s.schema_id = t.schema_id "
            f"WHERE s.name = ? AND ({tests}) ORDER BY 1", (profile.schema,))]
        check_sqlserver_permissions(db_perms, server_perms, writable, _server_roles(connection))

    @staticmethod
    def _discover(connection, name: str, profile: SQLServerProfile):
        rows = PostgresEngine._rows(
            connection,
            "SELECT c.TABLE_NAME, c.COLUMN_NAME, c.DATA_TYPE, c.IS_NULLABLE "
            "FROM INFORMATION_SCHEMA.COLUMNS c JOIN INFORMATION_SCHEMA.TABLES t "
            "ON t.TABLE_SCHEMA = c.TABLE_SCHEMA AND t.TABLE_NAME = c.TABLE_NAME "
            "AND t.TABLE_TYPE = 'BASE TABLE' WHERE c.TABLE_SCHEMA = ? "
            "ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION", (profile.schema,))
        pks = {(r[0], r[1]) for r in PostgresEngine._rows(
            connection,
            "SELECT t.name, c.name FROM sys.indexes i "
            "JOIN sys.index_columns ic ON ic.object_id = i.object_id AND ic.index_id = i.index_id "
            "JOIN sys.columns c ON c.object_id = ic.object_id AND c.column_id = ic.column_id "
            "JOIN sys.tables t ON t.object_id = i.object_id JOIN sys.schemas s ON s.schema_id = t.schema_id "
            "WHERE i.is_primary_key = 1 AND s.name = ?", (profile.schema,))}
        tables: dict[str, list[SQLiteColumn]] = {}
        skipped = 0
        for table, column, data_type, nullable in rows:
            if not (_usable(table) and _usable(column)):
                skipped += 1
                continue
            tables.setdefault(table, []).append(SQLiteColumn(
                name=column, declared_type=str(data_type), nullable=(nullable == "YES"),
                primary_key=(table, column) in pks))
        items = tuple(SQLiteTable(name=t, columns=tuple(c)) for t, c in sorted(tables.items()))
        return SQLiteSchema(profile_name=name, tables=items[:MAX_TABLES]), skipped


__all__ = ["EngineError", "MySQLEngine", "check_mysql_writer_grants", "check_mysql_creator_grants", "MySQLProfile", "PostgresEngine", "PostgresProfile",
           "SQLiteEngine", "SQLServerEngine", "SQLServerProfile", "check_mysql_grants",
           "check_sqlserver_permissions", "check_sqlserver_creator_permissions", "map_sqlserver_type", "load_profiles", "map_mysql_type",
           "map_postgres_type", "map_sqlite_type", "parse_profiles"]
