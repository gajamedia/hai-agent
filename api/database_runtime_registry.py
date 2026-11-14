from __future__ import annotations

"""Runtime registry of user-approved, read-only database connections.

A user (with the ``database:connect`` permission) asks to connect a database and
then approves which tables/columns may be used. Each connection becomes its own
read-only gateway + report service. Nothing here lets a request or the LLM name a
filesystem path, credentials or SQL:

* SQLite files are chosen by *name inside a server-owned data root*; anything
  that resolves outside the root (``..``, absolute paths, symlinks) is refused.
* The LLM only ever sees the catalog the user approved for one connection.
* Every connection has an audit ``authorization_ref`` naming who approved it.

Engines: SQLite (a file inside the data root) and PostgreSQL (a server-side
*profile* chosen by name; its password lives in an environment variable and the
role must be SELECT-only). See ``api.database_engines``.
"""

import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from agent.database import ConnectionManager
from agent.database.capability_gateway import build_business_database_gateway
from agent.llm.business_database_adapter import ReadOnlyTableSpec, RelationBook
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.service import LLMService
from agent.llm.trusted_capability_invocation_port import (
    CapabilityInvocationSpec,
    TrustedCapabilityInvocationPort,
)
from agent.workflow import SQLiteWorkflowStore
from agent.workflow.analytics import WorkflowAnalyticalTaskExecutor
from agent.workflow.invocation import WorkflowCapabilityTaskExecutor
from agent.workflow.reports import LocalArtifactStore, WorkflowReportTaskExecutor
from agent.workflow.service import WorkflowApplicationService
from agent.workflow.job_experience import JobExperienceStore
from agent.workflow.table_report_job import READ_ACTION, TableReportJobRunner
from agent.workflow.table_report_planner import TableCatalogEntry, TableRelation, TableReportPlanner
from agent.workflow.table_report_service import TableReportService
from agent.workflow.database_creation import (
    DIALECTS, DatabaseCreationService, DatabaseDesigner, SQLiteCreationService, max_tables,
)
from agent.workflow.data_change import (
    DataChangePlanner, DataChangeService, OPERATIONS as CHANGE_OPERATIONS, WritableTable,
)
from api.data_change_settings import settings as data_change_settings
from api.relation_suggestions import suggest_relations
from api.database_engines import (
    EngineError, MySQLEngine, PostgresEngine, SQLiteEngine, SQLServerEngine, any_database_allowed,
    map_mysql_type, map_postgres_type, map_sqlite_type,
)

logger = logging.getLogger("hai_agent.database_registry")
_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
LIST_READ_LIMIT = 1001
MAX_COLUMNS_PER_TABLE = 100
MAX_RELATIONS = 30


class RegistryError(ValueError):
    """Invalid request to the registry (maps to HTTP 4xx)."""


class RegistryNotFound(RegistryError):
    pass


def _relation_to_dict(rel: TableRelation) -> dict[str, str]:
    return {"table": rel.table, "column": rel.column, "ref_table": rel.ref_table,
            "ref_column": rel.ref_column, "source": rel.source}


def resolve_in_root(root: Path, filename: Any) -> Path:
    """Resolve ``filename`` strictly inside ``root`` or raise RegistryError."""
    if not isinstance(filename, str) or not filename.strip() or "\x00" in filename:
        raise RegistryError("filename must be a non-empty text")
    candidate = Path(filename.strip())
    if candidate.is_absolute() or candidate.drive or ".." in candidate.parts:
        raise RegistryError("filename must be a relative name inside the data folder")
    resolved = (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RegistryError("filename resolves outside the data folder") from exc
    if resolved.suffix.lower() not in _SQLITE_SUFFIXES:
        raise RegistryError("only .db, .sqlite and .sqlite3 files are supported")
    if not resolved.is_file():
        raise RegistryNotFound("database file not found in the data folder")
    return resolved


@dataclass
class _Connection:
    name: str
    filename: str  # source: file name (sqlite) or profile name (postgresql)
    engine: str
    approved_by: str
    approved_at: str
    manager: ConnectionManager
    schema: Any
    service: TableReportService | None = None
    tables: tuple[str, ...] = ()
    tables_approved_by: str | None = None
    tables_approved_at: str | None = None
    selection: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    changes: DataChangeService | None = None  # only when tables were approved for writing
    relations: tuple[TableRelation, ...] = ()  # approved joins (database foreign keys or declared by a user)
    relation_book: Any = None  # what the adapter of this connection may join along (grows on confirmation)


class DatabaseRuntimeRegistry:
    def __init__(self, *, data_root: str | Path, work_dir: str | Path,
                 authorization_policy: Any, llm_provider_factory: Callable[[], Any],
                 max_connections: int = 10, restore: bool = True,
                 profiles: Mapping[str, Any] | None = None,
                 engines: Mapping[str, Any] | None = None) -> None:
        self.data_root = Path(data_root).expanduser()
        if not self.data_root.is_dir():
            raise RegistryError(
                f"data_root is not a directory: {self.data_root}. Create the folder first "
                "(for example: python scripts/make_sample_sales_db.py <folder>), "
                "or fix HAI_DATABASE_ROOT in .env.")
        self.work_dir = Path(work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.authorization_policy = authorization_policy
        self.llm_provider_factory = llm_provider_factory
        self.max_connections = max_connections
        # Where servers added from the browser are saved; None switches that feature off.
        self.profiles_path: Path | None = None
        self.env_path: Path | None = None
        # Creating the limited accounts with a one-time administrator password (see database_admin_setup).
        self.admin_setup_enabled = False
        self._engines: dict[str, Any] = {"sqlite": SQLiteEngine(self.data_root)}
        by_type: dict[str, dict[str, Any]] = {}
        for pname, profile in (profiles or {}).items():
            by_type.setdefault(profile.type, {})[pname] = profile
        if "postgresql" in by_type:
            self._engines["postgresql"] = PostgresEngine(by_type["postgresql"])
        if "mysql" in by_type:
            self._engines["mysql"] = MySQLEngine(by_type["mysql"])
        if "sqlserver" in by_type:
            self._engines["sqlserver"] = SQLServerEngine(by_type["sqlserver"])
        if engines:
            self._engines.update(engines)
        self._connections: dict[str, _Connection] = {}
        self._creators: dict[str, DatabaseCreationService] = {}
        self._lock = threading.RLock()
        self._store = SQLiteWorkflowStore(self.work_dir / "workflow.sqlite3")
        self._llm: Any = None
        self._experience = JobExperienceStore(self.work_dir / "experience.sqlite3")
        # Connections that could not be safely restored (file gone, schema changed).
        self._degraded: dict[str, dict[str, Any]] = {}
        self._db_path = self.work_dir / "connections.sqlite3"
        self._init_persistence()
        if restore:
            self.restore()

    # -------------------------------------------------------- persistence
    # Only names, file names, approvers and the approved table selection are
    # stored; never credentials, SQL or data.
    def _persist_conn(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_persistence(self) -> None:
        db = self._persist_conn()
        try:
            with db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS connections ("
                    "name TEXT PRIMARY KEY, filename TEXT NOT NULL, connected_by TEXT NOT NULL,"
                    "connected_at TEXT NOT NULL, approved_by TEXT NOT NULL, approved_at TEXT NOT NULL,"
                    "selection_json TEXT NOT NULL, fingerprint TEXT NOT NULL)")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS reports (workflow_id TEXT PRIMARY KEY, "
                    "owner TEXT NOT NULL, database_name TEXT NOT NULL, relative_path TEXT NOT NULL, "
                    "filename TEXT NOT NULL, media_type TEXT NOT NULL, size_bytes INTEGER NOT NULL, "
                    "created_at TEXT NOT NULL)")
                have = {r[1] for r in db.execute("PRAGMA table_info(connections)")}
                if "engine" not in have:
                    db.execute("ALTER TABLE connections ADD COLUMN engine TEXT NOT NULL "
                               "DEFAULT 'sqlite'")
                if "relations_json" not in have:
                    db.execute("ALTER TABLE connections ADD COLUMN relations_json TEXT NOT NULL "
                               "DEFAULT '[]'")
        finally:
            db.close()

    def _fingerprint(self, engine: str, schema: Any, selection: Mapping[str, Any]) -> str:
        map_type = self._engines[engine].map_type
        by_table = {t.name: t for t in schema.tables}
        shape = []
        for table in sorted(selection):
            found = by_table.get(table)
            wanted = selection[table].get("columns", [])
            cols = (sorted((c.name, map_type(c.declared_type))
                           for c in found.columns if c.name in wanted) if found else None)
            if cols is not None and len(cols) != len(set(wanted)):
                cols = None  # an approved column disappeared
            shape.append([table, cols])
        return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()

    def _save(self, conn: _Connection) -> None:
        db = self._persist_conn()
        try:
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO connections (name, filename, connected_by, "
                    "connected_at, approved_by, approved_at, selection_json, fingerprint, engine, "
                    "relations_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (conn.name, conn.filename, conn.approved_by, conn.approved_at,
                     conn.tables_approved_by or conn.approved_by,
                     conn.tables_approved_at or conn.approved_at,
                     json.dumps(conn.selection, sort_keys=True),
                     self._fingerprint(conn.engine, conn.schema, conn.selection), conn.engine,
                     json.dumps([_relation_to_dict(r) for r in conn.relations], sort_keys=True)))
        finally:
            db.close()

    def _forget(self, name: str) -> None:
        db = self._persist_conn()
        try:
            with db:
                db.execute("DELETE FROM connections WHERE name = ?", (name,))
        finally:
            db.close()

    def restore(self) -> None:
        """Reload saved connections; re-activate only if the approved schema is unchanged."""
        db = self._persist_conn()
        try:
            rows = [dict(r) for r in db.execute("SELECT * FROM connections ORDER BY name")]
        finally:
            db.close()
        for row in rows:
            name, engine = row["name"], row["engine"]
            degraded = {"name": name, "file": row["filename"], "database": engine}
            try:
                selection = json.loads(row["selection_json"])
                saved = json.loads(row["relations_json"] or "[]")  # validated again below
                conn = self._open(name, engine, row["filename"], row["connected_by"],
                                  row["connected_at"])
            except Exception as exc:
                self._degraded[name] = {**degraded, "status": "unavailable", "reason": str(exc)}
                continue
            self._connections[name] = conn
            if self._fingerprint(engine, conn.schema, selection) != row["fingerprint"]:
                self._degraded[name] = {
                    **degraded, "status": "needs_reapproval",
                    "reason": "approved tables or columns changed since approval"}
                continue
            try:
                self._activate_locked(
                    conn, selection, row["approved_by"], row["approved_at"],
                    relations=self._check_relations(conn, selection, saved, trusted=True))
            except Exception as exc:
                self._degraded[name] = {**degraded, "status": "needs_reapproval",
                                        "reason": str(exc)}

    # ------------------------------------------------------------ connect
    def _open(self, name: str, engine: str, source: Any, actor_id: str,
              approved_at: str) -> _Connection:
        impl = self._engines.get(engine)
        if impl is None:
            raise RegistryError(f"database type '{engine}' is not available")
        if not isinstance(source, str) or not source.strip():
            raise RegistryError("a source (file name or profile) is required")
        manager = ConnectionManager()
        try:
            schema, warnings = impl.open(
                manager, name, source.strip(), f"user-approval:{actor_id}:{approved_at}:{name}")
        except Exception as exc:
            manager.shutdown()
            if isinstance(exc, RegistryError):
                raise
            # Driver errors can echo host names or users, so the browser only gets our own
            # messages. The operator still needs the real cause: it goes to the server log
            # (which never holds the password) and to scripts/check_database_profile.py.
            if not isinstance(exc, EngineError):
                logger.warning(
                    "Could not open %s source %r for connection %r: %s: %s "
                    "(run scripts/check_database_profile.py for a step by step check)",
                    engine, source, name, type(exc).__name__, exc)
            detail = str(exc) if isinstance(exc, EngineError) else "connection failed"
            raise RegistryError(f"cannot open database: {detail}") from exc
        return _Connection(name=name, filename=source.strip(), engine=engine,
                           approved_by=actor_id, approved_at=approved_at, manager=manager,
                           schema=schema, warnings=tuple(warnings))

    # ------------------------------------------------------------- reports
    def record_report(self, *, owner: str, database: str, workflow_id: str,
                      artifact: Mapping[str, Any]) -> None:
        """Remember who produced a report file so only they can download it."""
        db = self._persist_conn()
        try:
            with db:
                db.execute(
                    "INSERT OR REPLACE INTO reports VALUES (?,?,?,?,?,?,?,?)",
                    (workflow_id, owner, database, str(artifact["relative_path"]),
                     str(artifact["filename"]), str(artifact["media_type"]),
                     int(artifact["size_bytes"]), datetime.now(timezone.utc).isoformat()))
        finally:
            db.close()

    def find_report(self, owner: str, workflow_id: str) -> dict[str, Any] | None:
        """The report's metadata and a verified path, or None (unknown or not the owner's)."""
        db = self._persist_conn()
        try:
            row = db.execute("SELECT * FROM reports WHERE workflow_id = ? AND owner = ?",
                             (workflow_id, owner)).fetchone()
        finally:
            db.close()
        if row is None:
            return None
        root = (self.work_dir / "artifacts" / row["database_name"]).resolve()
        path = (root / row["relative_path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return None  # a tampered index row must never reach outside the artifacts folder
        if not path.is_file():
            return None
        return {**dict(row), "path": path}

    def supported_engine_names(self) -> list[str]:
        """Product names users might type when asking for an engine (already supported ones)."""
        aliases = {"sqlite": ["sqlite"], "postgresql": ["postgresql", "postgres"],
                   "mysql": ["mysql", "mariadb"], "sqlserver": ["sqlserver", "sql server", "mssql"]}
        return sorted({alias for kind in {"sqlite", "postgresql", "mysql", "sqlserver"}
                       for alias in aliases[kind]})

    def available_engines(self) -> list[str]:
        return sorted(self._engines)

    def profiles(self) -> list[dict[str, str]]:
        """Names and types only: never hosts, users or secrets."""
        out = []
        for kind, engine in self._engines.items():
            for n, profile in getattr(engine, "profiles", {}).items():
                item = {"name": n, "type": kind}
                if getattr(engine, "has_creator", lambda _n: False)(n):
                    item["prefix"] = profile.creator_prefix     # databases made by the agent can be chosen
                if kind in {"mysql", "sqlserver"} and any_database_allowed():
                    item["any_database"] = True                 # any database the reader can see (read only)
                out.append(item)
        return sorted(out, key=lambda item: item["name"])

    # ----------------------------------------------- servers added from the form
    @property
    def can_register_servers(self) -> bool:
        return self.profiles_path is not None and self.env_path is not None

    def register_server(self, spec: Any) -> dict[str, Any]:
        """Test and save a server definition, then make it usable at once (no restart)."""
        from api import database_server_setup as setup
        if not self.can_register_servers:
            raise RegistryError("adding servers from the browser is not configured on this server")
        with self._lock:
            taken = {p["name"] for p in self.profiles()}
            try:
                profile = setup.register(spec, profiles_path=self.profiles_path, env_path=self.env_path,
                                         taken=taken)
            except setup.ServerSetupError as exc:
                raise RegistryError(str(exc)) from exc
            engine_class = {"postgresql": PostgresEngine, "mysql": MySQLEngine, "sqlserver": SQLServerEngine}
            impl = self._engines.get(profile.type)
            if impl is None:
                self._engines[profile.type] = engine_class[profile.type]({profile.name: profile})
            else:
                impl.profiles[profile.name] = profile
            self._creators.pop(profile.name, None)
        return next(p for p in self.profiles() if p["name"] == profile.name)

    @property
    def can_setup_accounts(self) -> bool:
        return self.can_register_servers and self.admin_setup_enabled

    def _admin_fields(self, payload: Any) -> dict[str, Any]:
        from api import database_admin_setup as admin
        if not self.can_setup_accounts:
            raise RegistryError("automatic account setup is switched off on this server")
        allowed = {"name", "type", "host", "port", "admin_user", "admin_password", "database", "roles"}
        if not isinstance(payload, Mapping) or set(payload) - allowed:
            raise RegistryError("unsupported fields in the request")
        name = payload.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", name):
            raise RegistryError("name: lowercase letters, digits or _ (2-32 characters, starts with a letter)")
        return {"name": name, "kind": payload.get("type"), "host": payload.get("host"), "port": payload.get("port"),
                "admin_user": payload.get("admin_user"), "admin_password": payload.get("admin_password")}

    def inspect_server(self, payload: Any) -> dict[str, Any]:
        """Look at a server with a one-time administrator login; nothing is changed or kept."""
        from api import database_admin_setup as admin
        fields = self._admin_fields(payload)
        taken = fields["name"] in {p["name"] for p in self.profiles()}
        try:
            found = admin.inspect(**fields)
        except admin.AdminSetupError as exc:
            raise RegistryError(str(exc)) from exc
        return {**found, "name_taken": taken}

    def provision_server(self, payload: Any) -> dict[str, Any]:
        """Create the limited accounts with a one-time administrator login, then register the server."""
        from api import database_admin_setup as admin
        fields = self._admin_fields(payload)
        if fields["name"] in {p["name"] for p in self.profiles()}:
            raise RegistryError(f"a server named '{fields['name']}' already exists")
        roles = payload.get("roles", list(admin.ROLES))
        if not isinstance(roles, list) or set(roles) - set(admin.ROLES):
            raise RegistryError("roles must be a list of reader, creator, writer")
        database = payload.get("database") or ""
        if not isinstance(database, str):
            raise RegistryError("database is not valid")
        try:
            made = admin.provision(**fields, database=database.strip(), roles=tuple(roles))
        except admin.AdminSetupError as exc:
            raise RegistryError(str(exc)) from exc
        finally:
            fields["admin_password"] = None
        names, passwords = made["names"], made["passwords"]
        spec: dict[str, Any] = {"name": fields["name"], "type": fields["kind"], "host": fields["host"],
                                "user": names["reader"], "password": passwords["reader"]}
        if database.strip():
            spec["database"] = database.strip()
        if fields["port"] not in (None, ""):
            spec["port"] = fields["port"]
        for role in ("creator", "writer"):
            if role in names:
                spec[role] = {"user": names[role], "password": passwords[role]}
        try:
            added = self.register_server(spec)
        except RegistryError as exc:
            raise RegistryError(f"the accounts were created but the server could not be registered: {exc}") from exc
        return {**added, "accounts": names, "reset": made["reset"]}

    def remove_server(self, name: str) -> None:
        """Forget a server added from the form; refused while a connection still uses it."""
        from api import database_server_setup as setup
        if not self.can_register_servers:
            raise RegistryError("adding servers from the browser is not configured on this server")
        with self._lock:
            impl = next((e for e in self._engines.values()
                         if name in getattr(e, "profiles", {}) and e.kind != "sqlite"), None)
            if impl is None:
                raise RegistryNotFound(f"no server named '{name}'")
            if any(c.filename == name or str(c.filename).startswith(f"{name}/")
                   for c in self._connections.values()):
                raise RegistryError("a connection still uses this server; disconnect it first")
            try:
                setup.forget(name, profiles_path=self.profiles_path)
            except setup.ServerSetupError as exc:
                raise RegistryError(str(exc)) from exc
            del impl.profiles[name]
            self._creators.pop(name, None)

    def list_source_databases(self, source: Any) -> list[dict[str, Any]]:
        """The databases on a server profile (or the ones this application made, for the SQLite profile), each
        marked with whether it may be connected and whether it is connected already. Names only."""
        with self._lock:
            impl = next((e for e in self._engines.values()
                         if isinstance(source, str) and source in getattr(e, "profiles", {})), None)
            connected = {c.filename for c in self._connections.values()}
        if impl is None or not hasattr(impl, "list_databases"):
            raise RegistryNotFound(f"no server named '{source}'")
        try:
            items = impl.list_databases(source)
        except Exception as exc:
            # Driver errors can echo hosts and users: the browser gets our own message, the log the real cause.
            if not isinstance(exc, EngineError):
                logger.warning("Could not list the databases of %r: %s: %s", source, type(exc).__name__, exc)
            detail = str(exc) if isinstance(exc, EngineError) else "connection failed"
            raise RegistryError(f"cannot list databases: {detail}") from exc
        return [{**item, "connected": f"{source}/{item['name']}" in connected
                 or (bool(item.get("default")) and source in connected)} for item in items]

    def list_source_files(self) -> list[dict[str, Any]]:
        """SQLite files in the data folder, each marked with whether a connection uses it already."""
        impl = self._engines.get("sqlite")
        if impl is None or not hasattr(impl, "list_files"):
            return []
        with self._lock:
            connected = {c.filename for c in self._connections.values()}
        return [{"name": f, "connected": f in connected or f"sqlite/{Path(f).stem}" in connected}
                for f in impl.list_files()]

    def profile_engine(self, name: Any) -> str | None:
        """Engine type of a named server profile, or None if there is no such profile."""
        for item in self.profiles():
            if item["name"] == name:
                return item["type"]
        return None

    def connect(self, name: Any, *, engine: str, source: Any, actor_id: str) -> dict[str, Any]:
        if not isinstance(name, str) or not _NAME.match(name):
            raise RegistryError("name must be lowercase letters, digits or _ (2-32 chars)")
        if engine == "sqlite":
            # Fail fast before taking the lock. "sqlite/<database>" names a database this application made.
            finder = getattr(self._engines.get("sqlite"), "_file", None)
            try:
                finder(source) if finder else resolve_in_root(self.data_root, source)
            except EngineError as exc:
                raise RegistryError(str(exc)) from exc
        with self._lock:
            if name in self._connections or name in self._degraded:
                raise RegistryError(f"connection '{name}' already exists")
            if len(self._connections) >= self.max_connections:
                raise RegistryError("too many active connections")
            conn = self._open(name, engine, source, actor_id,
                              datetime.now(timezone.utc).isoformat())
            self._connections[name] = conn
        return {"name": name, "database": engine, "status": "connected_pending_approval",
                "warnings": list(conn.warnings), "schema": conn.schema.to_dict()}

    def connect_sqlite(self, name: Any, filename: Any, *, actor_id: str) -> dict[str, Any]:
        return self.connect(name, engine="sqlite", source=filename, actor_id=actor_id)

    # ----------------------------------------------------------- activate
    def activate(self, name: str, tables: Any, *, actor_id: str | None = None,
                 relations: Any = None) -> dict[str, Any]:
        with self._lock:
            conn = self._get(name)
            self._activate_locked(conn, tables, actor_id or conn.approved_by,
                                  datetime.now(timezone.utc).isoformat(), persist=True,
                                  relations=relations)
            return self._describe(conn)

    @staticmethod
    def _check_relations(conn: _Connection, selection: Mapping[str, Any],
                         raw: Any, *, trusted: bool = False) -> tuple[TableRelation, ...]:
        """Relations the user approves for joining. Both columns must be approved columns of approved
        tables. Whether a relation is the database's own foreign key is decided here from the schema,
        never taken from the request."""
        if raw in (None, []):
            return ()
        if not isinstance(raw, list) or len(raw) > MAX_RELATIONS:
            raise RegistryError(f"relations must be a list of at most {MAX_RELATIONS} items")
        declared = {frozenset({(k.table, k.column), (k.ref_table, k.ref_column)})
                    for k in getattr(conn.schema, "foreign_keys", ())}
        result: dict[frozenset, TableRelation] = {}
        for item in raw:
            if not isinstance(item, Mapping) or set(item) - {"table", "column", "ref_table", "ref_column",
                                                              "source"}:
                raise RegistryError("each relation must be {table, column, ref_table, ref_column}")
            parts = [item.get(k) for k in ("table", "column", "ref_table", "ref_column")]
            if not all(isinstance(x, str) for x in parts):
                raise RegistryError("a relation needs table, column, ref_table and ref_column as text")
            table, column, ref_table, ref_column = parts
            for t, c in ((table, column), (ref_table, ref_column)):
                if t not in selection or c not in selection[t].get("columns", []):
                    raise RegistryError(f"relation uses {t}.{c}, which is not an approved column")
            if table == ref_table:
                raise RegistryError("a relation must connect two different tables")
            key = frozenset({(table, column), (ref_table, ref_column)})
            # "ai" (found by the AI, then approved by running a plan) is only believed when it comes from our
            # own server code or our own store, never from a request.
            source = ("foreign_key" if key in declared
                      else "ai" if trusted and item.get("source") == "ai" else "manual")
            result.setdefault(key, TableRelation(table, column, ref_table, ref_column, source))
        return tuple(result.values())

    def _activate_locked(self, conn: _Connection, tables: Any, actor: str, at: str,
                         *, persist: bool = False, relations: Any = None) -> None:
        name = conn.name
        map_type = self._engines[conn.engine].map_type
        if True:
            limit = max_tables()
            if not isinstance(tables, Mapping) or not tables or len(tables) > limit:
                raise RegistryError(f"select between 1 and {limit} tables")
            by_table = {t.name: t for t in conn.schema.tables}
            specs: dict[str, ReadOnlyTableSpec] = {}
            entries: list[TableCatalogEntry] = []
            writables: list[WritableTable] = []
            for table, sel in tables.items():
                found = by_table.get(table)
                if found is None:
                    raise RegistryError(f"table '{table}' is not in the discovered schema")
                if not isinstance(sel, Mapping):
                    raise RegistryError(f"selection for '{table}' must be an object")
                unknown = set(sel) - {"columns", "filterable_columns", "key_column",
                                      "required_non_null_columns", "description", "writable"}
                if unknown:
                    raise RegistryError(f"unsupported fields for '{table}': {sorted(unknown)}")
                columns = sel.get("columns")
                if (not isinstance(columns, list) or not columns
                        or len(columns) > MAX_COLUMNS_PER_TABLE
                        or not all(isinstance(c, str) for c in columns)):
                    raise RegistryError(f"'{table}': columns must be a non-empty list of names")
                types = {c.name: map_type(c.declared_type) for c in found.columns}
                filterable = sel.get("filterable_columns", [])
                required = sel.get("required_non_null_columns", [])
                if not isinstance(filterable, list) or not isinstance(required, list):
                    raise RegistryError(f"'{table}': filterable/required must be lists")
                try:
                    # Every approved column may be used to sort. max_limit 1001 lets a listing
                    # hold 1000 rows plus the one extra row that shows "there is more".
                    spec = ReadOnlyTableSpec(
                        table=table, columns=tuple(columns), filterable_columns=tuple(filterable),
                        sortable_columns=tuple(columns), max_limit=LIST_READ_LIMIT)
                    unknown_cols = [c for c in columns if c not in types]
                    if unknown_cols:
                        raise RegistryError(f"'{table}': unknown columns {unknown_cols}")
                    entry = TableCatalogEntry.from_table_spec(
                        spec, types, description=str(sel.get("description", ""))[:200],
                        key_column=sel.get("key_column"),
                        required_non_null_columns=tuple(required))
                except RegistryError:
                    raise
                except (TypeError, ValueError) as exc:
                    raise RegistryError(f"'{table}': {exc}") from exc
                specs[table] = spec
                entries.append(entry)
                if sel.get("writable"):
                    writables.append(self._writable_table(conn, table, sel, entry))
            approved_relations = (relations if isinstance(relations, tuple)
                                  else self._check_relations(conn, tables, relations))
            changes = self._build_changes(conn, writables) if writables else None
            conn.changes = changes
            conn.service = self._build_service(conn, specs, entries, approved_relations)
            conn.relations = approved_relations
            conn.tables = tuple(specs)
            conn.selection = {t: dict(v) for t, v in tables.items()}
            conn.tables_approved_by, conn.tables_approved_at = actor, at
            self._degraded.pop(name, None)
            if persist:
                self._save(conn)

    # ---------------------------------------------------------- data changes
    def _writer_engine(self, conn: _Connection):
        impl = self._engines[conn.engine]
        if not (hasattr(impl, "open_writer") and impl.has_writer(conn.filename)):
            raise RegistryError(
                "changing data is not possible on this connection: its server profile has no "
                "writer account (see docs/TRY_DATA_REPORTS_LOCALLY.md, 'Changing data')")
        return impl

    def _writable_table(self, conn: _Connection, table: str, sel: Mapping[str, Any],
                        entry: TableCatalogEntry) -> WritableTable:
        self._writer_engine(conn)
        cfg = sel["writable"]
        if not isinstance(cfg, Mapping) or set(cfg) - {"operations", "columns"}:
            raise RegistryError(f"'{table}': writable must be {{operations, columns}}")
        operations, columns = cfg.get("operations"), cfg.get("columns", [])
        if (not isinstance(operations, list) or not operations
                or not set(operations) <= set(CHANGE_OPERATIONS)):
            raise RegistryError(f"'{table}': operations must be a non-empty list of {list(CHANGE_OPERATIONS)}")
        if not isinstance(columns, list) or not set(columns) <= set(entry.columns):
            raise RegistryError(f"'{table}': writable columns must be approved columns")
        if entry.key_column and entry.key_column in columns:
            raise RegistryError(f"'{table}': the key column cannot be made writable")
        if {"insert", "update"} & set(operations) and not columns:
            raise RegistryError(f"'{table}': tick the columns that may be written")
        if {"update", "delete"} & set(operations) and not entry.filterable_columns:
            raise RegistryError(f"'{table}': update and delete need at least one 'boleh difilter' "
                                "column to select the rows")
        return WritableTable(entry=entry, operations=tuple(operations), writable_columns=tuple(columns))

    def _build_changes(self, conn: _Connection, tables: list[WritableTable]) -> DataChangeService:
        impl = self._writer_engine(conn)
        try:
            factory = impl.open_writer(conn.filename)
        except EngineError as exc:
            raise RegistryError(f"cannot use the writer account: {exc}") from exc
        except Exception as exc:
            # Driver errors can echo hosts and users: the browser gets our own message, the log the real cause.
            logger.warning("Could not open the writer for connection %r (source %r): %s: %s",
                           conn.name, conn.filename, type(exc).__name__, exc)
            code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else None
            if getattr(impl, "kind", "mysql") == "sqlserver":
                denied = ("4060" in str(exc) or "18456" in str(exc))
                fix = ("python scripts/connect_sqlserver.py --create-writer --database <name>, then restart the backend")
            else:
                denied = code == 1044
                fix = ("scripts/setup_mysql_creator_user.py <profile> (for databases the agent created) or "
                       "scripts/setup_mysql_writer_user.py <profile> (for the profile's own database), then "
                       "restart the backend")
            detail = ("the writer account has no access to this database. An administrator must run " + fix
                      if denied else "connection failed")
            raise RegistryError(f"cannot use the writer account: {detail}") from exc
        if self._llm is None:
            self._llm = LLMService(self.llm_provider_factory())
        return DataChangeService(
            planner=DataChangePlanner(self._llm, tables), connect=factory, quote=impl.quote,
            paramstyle=impl.paramstyle, lock_clause=getattr(impl, "lock_clause", " FOR UPDATE"),
            dialect=getattr(impl, "dialect", "limit"),
            audit_path=self.work_dir / "audit" / f"changes-{conn.name}.jsonl",
            audit_detail=lambda: data_change_settings()["audit_detail"])

    # --------------------------------------------------- creating databases
    def creatable_profiles(self) -> list[dict[str, str]]:
        """Profiles whose server definition has a creator account (names and prefix only)."""
        out = []
        for engine in self._engines.values():
            for pname, profile in getattr(engine, "profiles", {}).items():
                if getattr(engine, "has_creator", lambda _n: False)(pname):
                    out.append({"name": pname, "prefix": profile.creator_prefix, "type": engine.kind})
        return sorted(out, key=lambda item: item["name"])

    def language_model(self) -> Any:
        """The shared language-model service (built on first use), for callers that only classify text."""
        with self._lock:
            if self._llm is None:
                self._llm = LLMService(self.llm_provider_factory())
            return self._llm

    def database_creator(self, profile: Any = None) -> DatabaseCreationService:
        with self._lock:
            able = self.creatable_profiles()
            names = [p["name"] for p in able]
            if profile is None:
                if len(names) != 1:
                    raise RegistryError("name the server profile" if names
                                        else "no server profile can create databases")
                profile = names[0]
            if profile not in names:
                raise RegistryNotFound(f"profile '{profile}' cannot create databases")
            cached = self._creators.get(profile)
            if cached is not None:
                return cached
            impl = next(e for e in self._engines.values() if profile in getattr(e, "profiles", {}))
            if impl.kind == "sqlite":
                # A SQLite database is a file in the data folder: there is no server, account or grant.
                if self._llm is None:
                    self._llm = LLMService(self.llm_provider_factory())
                service = SQLiteCreationService(
                    designer=DatabaseDesigner(self._llm, prefix=impl.profiles[profile].creator_prefix,
                                              dialect=DIALECTS["sqlite"]),
                    data_root=self.data_root, profile=profile,
                    audit_path=self.work_dir / "audit" / f"creations-{profile}.jsonl")
                self._creators[profile] = service
                return service
            try:
                factory = impl.open_creator(profile)
            except EngineError as exc:
                raise RegistryError(f"cannot use the creator account: {exc}") from exc
            if self._llm is None:
                self._llm = LLMService(self.llm_provider_factory())
            prefix = impl.profiles[profile].creator_prefix
            service = DatabaseCreationService(
                designer=DatabaseDesigner(self._llm, prefix=prefix, dialect=DIALECTS[impl.kind]),
                connect=factory, profile=profile,
                grant_reader=impl.profiles[profile].user if impl.kind == "sqlserver" else "",
                grant_writer=impl.profiles[profile].writer_user if impl.kind == "sqlserver" else "",
                audit_path=self.work_dir / "audit" / f"creations-{profile}.jsonl")
            self._creators[profile] = service
            return service

    def data_changes(self, name: Any) -> DataChangeService:
        with self._lock:
            conn = self._get(name)
            if conn.changes is None:
                raise RegistryError("no table of this connection is approved for changing data")
            return conn.changes

    def change_token(self, name: Any) -> str | None:
        """Fingerprint of what data changes on ``name`` may touch (tables, columns, operations and the
        engine). A change plan remembers it; if the writable approvals change, the token changes and the
        plan is refused. Unlike object identity it survives a restart."""
        with self._lock:
            conn = self._get(name)
            if conn.changes is None:
                return None
            tables = []
            for table, writable in sorted(conn.changes.planner.tables.items()):
                entry = writable.entry
                tables.append([table, sorted((c, str(t)) for c, t in entry.columns.items()),
                               sorted(entry.filterable_columns), entry.key_column,
                               sorted(entry.required_non_null_columns), sorted(writable.operations),
                               sorted(writable.writable_columns)])
            body = {"name": conn.name, "engine": conn.engine, "tables": tables}
            return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()

    def entry_databases(self) -> list[dict[str, Any]]:
        """Connections where new rows may be recorded (a table approved for insert): name and tables only."""
        with self._lock:
            out = []
            for conn_name, conn in sorted(self._connections.items()):
                if conn.changes is None:
                    continue
                tables = sorted(t for t, w in conn.changes.planner.tables.items() if "insert" in w.operations)
                if tables:
                    out.append({"name": conn_name, "tables": tables})
            return out

    def resolve_entry_name(self, name: Any = None) -> str:
        """The connection to record into: the named one, or the only one that accepts new rows."""
        able = [d["name"] for d in self.entry_databases()]
        if name is not None:
            if name not in able:
                raise RegistryNotFound(f"'{name}' does not accept new data")
            return name
        if len(able) != 1:
            raise RegistryError("name the database to record into" if able
                                else "no connection has tables approved for adding data")
        return able[0]

    def resolve_change_name(self, name: Any = None) -> str:
        """The connection to use for a change: the named one, or the only one that can change data."""
        with self._lock:
            if name is not None:
                return self._get(name) and name
            able = [n for n, c in self._connections.items() if c.changes is not None]
            if len(able) != 1:
                raise RegistryError("name the database to change" if able
                                    else "no connection has tables approved for changing data")
            return able[0]

    def _build_service(self, conn: _Connection, specs: Mapping[str, ReadOnlyTableSpec],
                       entries: list[TableCatalogEntry],
                       relations: tuple[TableRelation, ...] = ()) -> TableReportService:
        book = RelationBook(specs, [(r.table, r.column, r.ref_table, r.ref_column) for r in relations])
        conn.relation_book = book
        assembly = build_business_database_gateway(
            connection_manager=conn.manager, profile_name=conn.name,
            authorization_policy=self.authorization_policy, table_specs=specs,
            trace_path=self.work_dir / f"trace-{conn.name}.jsonl",
            paramstyle=self._engines[conn.engine].paramstyle,
            identifier_quote=getattr(self._engines[conn.engine], "quote", None),
            dialect=getattr(self._engines[conn.engine], "dialect", "limit"),
            relations=book)
        port = TrustedCapabilityInvocationPort(
            gateway=assembly.gateway,
            specs={READ_ACTION: CapabilityInvocationSpec(
                action_name=READ_ACTION, target="business_database",
                capability_id="business_database.select")})
        reader = SimpleNamespace(
            actions={}, unified_executor=object(),
            execution_security_policy=ExecutionSecurityPolicy(action_risks={READ_ACTION: 0.1}))
        store = self._store
        runner = TableReportJobRunner(
            service=WorkflowApplicationService(store=store), store=store,
            capability_executor=WorkflowCapabilityTaskExecutor(
                store=store, agent=reader, trusted_capability_invocation_port=port),
            analytical_executor=WorkflowAnalyticalTaskExecutor(store=store),
            report_executor=WorkflowReportTaskExecutor(
                store=store, artifact_store=LocalArtifactStore(self.work_dir / "artifacts" / conn.name)))
        if self._llm is None:
            self._llm = LLMService(self.llm_provider_factory())
        experience = self._experience.bind(conn.name)
        # Guesses from column names for the AI to build on when a question needs a relation nobody approved.
        columns = {t: set(s.columns) for t, s in specs.items()}
        hints = [TableRelation(g["table"], g["column"], g["ref_table"], g["ref_column"], "manual")
                 for g in suggest_relations(conn.schema, self._engines[conn.engine].map_type)
                 if g["column"] in columns.get(g["table"], ()) and g["ref_column"] in columns.get(g["ref_table"], ())]
        planner = TableReportPlanner(self._llm, entries, experience=experience, relations=relations,
                                     relation_hints=hints)
        name = conn.name
        return TableReportService(
            planner=planner, runner=runner, experience=experience,
            relation_approver=lambda rels, context: self._confirm_relations(name, rels, context))

    def _confirm_relations(self, name: str, relations: Any, context: Any) -> None:
        """A person ran a plan that uses relations the AI found: from now on they are approved relations.

        Only tables and columns that are already approved can be related, so this never exposes anything
        new; it only lets approved tables be combined. The relation is remembered (source "ai") and can be
        removed in the approval form."""
        actor = getattr(getattr(context, "principal", None), "actor_id", "unknown")
        with self._lock:
            conn = self._get(name)
            if conn.service is None or conn.relation_book is None or conn.selection is None:
                raise RegistryError("this connection has no approved tables")
            items = [{"table": r.table, "column": r.column, "ref_table": r.ref_table,
                      "ref_column": r.ref_column, "source": "ai"} for r in relations]
            checked = self._check_relations(conn, conn.selection, items, trusted=True)
            have = {frozenset({r.left, r.right}) for r in conn.relations}
            added = tuple(r for r in checked if frozenset({r.left, r.right}) not in have)
            if not added:
                return
            conn.relation_book.add([(r.table, r.column, r.ref_table, r.ref_column) for r in added])
            conn.relations = (*conn.relations, *added)
            conn.service.planner.relations = conn.relations
            self._save(conn)
            logger.info("Relations approved by running a plan (connection %r, by %s): %s", name, actor,
                        ", ".join(f"{r.left} = {r.right}" for r in added))

    # ------------------------------------------------------------- access
    def _get(self, name: Any) -> _Connection:
        conn = self._connections.get(name) if isinstance(name, str) else None
        if conn is None:
            raise RegistryNotFound(f"connection '{name}' does not exist")
        return conn

    @staticmethod
    def _describe(conn: _Connection) -> dict[str, Any]:
        return {"name": conn.name, "database": conn.engine, "file": conn.filename,
                "approved_by": conn.approved_by, "approved_at": conn.approved_at,
                "tables_approved_by": conn.tables_approved_by,
                "status": "active" if conn.service else "connected_pending_approval",
                "tables": list(conn.tables),
                "writable_tables": sorted(conn.changes.planner.tables) if conn.changes else [],
                # Per table that allows update: the columns a file may use to find the rows (key first).
                "match_columns": ({
                    t: sorted(w.entry.filterable_columns, key=lambda c: c != w.entry.key_column)
                    for t, w in conn.changes.planner.tables.items() if "update" in w.operations
                } if conn.changes else {}),
                "relations": [_relation_to_dict(r) for r in conn.relations]}

    def schema(self, name: str) -> dict[str, Any]:
        with self._lock:
            conn = self._get(name)
            data = conn.schema.to_dict()
            # What is approved right now, so the page can show it and let the user change it.
            data["approved_relations"] = [_relation_to_dict(r) for r in conn.relations]
            # Guesses from names for relations the database does not declare; used only if a person ticks them.
            data["suggested_relations"] = suggest_relations(conn.schema, self._engines[conn.engine].map_type)
            data["approved"] = {t: {"columns": list(v.get("columns", [])),
                                    "filterable_columns": list(v.get("filterable_columns", [])),
                                    "writable": v.get("writable")}
                                for t, v in (conn.selection or {}).items()}
            impl = self._engines[conn.engine]
            data["writer_available"] = bool(
                hasattr(impl, "has_writer") and impl.has_writer(conn.filename))
            return data

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            items = []
            for c in self._connections.values():
                d = self._describe(c)
                if c.name in self._degraded:
                    d.update(status=self._degraded[c.name]["status"],
                             reason=self._degraded[c.name]["reason"])
                items.append(d)
            items += [dict(d, tables=[]) for n, d in self._degraded.items()
                      if n not in self._connections]
            return items

    def service(self, name: str | None) -> TableReportService:
        """Service for ``name``; with no name, the only active connection."""
        with self._lock:
            if name is None:
                active = [c for c in self._connections.values() if c.service]
                if len(active) != 1:
                    raise RegistryError(
                        "specify 'database'; " + ("none is active" if not active
                                                  else "several are active"))
                return active[0].service
            conn = self._get(name)
            if conn.service is None:
                raise RegistryError(f"connection '{name}' has no approved tables yet")
            return conn.service

    def service_token(self, name: str | None) -> str | None:
        """Stable fingerprint of what a report on ``name`` may touch (engine, approved tables/columns,
        approved relations). A plan remembers it; if the approvals change, the token changes and the plan
        is refused. Unlike object identity it survives a restart."""
        with self._lock:
            conn = self._get(self.resolve_name(name))
            if conn.service is None:
                return None
            relations = sorted((r.table, r.column, r.ref_table, r.ref_column) for r in conn.relations)
            body = {"name": conn.name, "engine": conn.engine, "selection": conn.selection,
                    "relations": relations}
            return hashlib.sha256(
                json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def resolve_name(self, name: str | None) -> str:
        """Name of the connection ``service(name)`` would use, so callers can pin it."""
        with self._lock:
            if name is None:
                active = [c.name for c in self._connections.values() if c.service]
                if len(active) != 1:
                    raise RegistryError("specify 'database'; " + (
                        "none is active" if not active else "several are active"))
                return active[0]
            return self._get(name).name

    def refresh(self, name: str, *, actor_id: str) -> dict[str, Any]:
        """Read the schema again (for example after tables were added) and keep what is approved.

        A fresh connection is opened first; the old one is replaced only if the approved selection still
        fits the new schema, so a failure leaves the connection exactly as it was."""
        with self._lock:
            old = self._get(name)
            fresh = self._open(name, old.engine, old.filename, old.approved_by, old.approved_at)
            try:
                if old.selection:
                    self._activate_locked(fresh, old.selection, old.tables_approved_by or old.approved_by,
                                          old.tables_approved_at or old.approved_at,
                                          relations=old.relations)
            except Exception:
                fresh.manager.shutdown()
                raise
            self._connections[name] = fresh
        old.manager.shutdown()
        return self._describe(fresh)

    def disconnect(self, name: str) -> None:
        with self._lock:
            self._experience.bind(name).forget()  # learned values may be business data
            if name in self._degraded and name not in self._connections:
                del self._degraded[name]
                self._forget(name)
                return
            conn = self._get(name)
            del self._connections[name]
            self._degraded.pop(name, None)
            self._forget(name)
        conn.manager.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            conns, self._connections = list(self._connections.values()), {}
        for conn in conns:
            conn.manager.shutdown()


def registry_factory_from_environment(app: Any) -> DatabaseRuntimeRegistry:
    """HAI_DATABASE_ROOT (data folder) [+ HAI_DATABASE_WORK_DIR]; auth must be on."""
    import os
    from api.dependencies import create_llm_provider
    root = os.environ["HAI_DATABASE_ROOT"].strip()
    work = os.getenv("HAI_DATABASE_WORK_DIR", "").strip() or str(Path(root) / ".hai-work")
    policy = getattr(app.state, "authorization_policy", None)
    if policy is None:
        raise RuntimeError("HAI_DATABASE_ROOT requires HAI_AUTH_ENABLED=true "
                           "(runtime database connections need an authorization policy).")
    profiles = None
    repo = Path(__file__).resolve().parents[1]
    profiles_path = os.getenv("HAI_DATABASE_PROFILES", "").strip()
    default_profiles = repo / "data" / "reports" / "profiles.json"
    if not profiles_path and default_profiles.is_file():
        profiles_path = str(default_profiles)          # servers added from the form land here
    if profiles_path:
        from api.database_engines import load_profiles
        profiles = load_profiles(profiles_path)
    registry = DatabaseRuntimeRegistry(data_root=root, work_dir=work, authorization_policy=policy,
                                       llm_provider_factory=create_llm_provider, profiles=profiles)
    if os.getenv("HAI_SERVER_REGISTRATION", "true").strip().lower() not in {"0", "false", "no", "off"}:
        registry.profiles_path = Path(profiles_path or default_profiles)
        registry.env_path = Path(os.getenv("HAI_ENV_FILE", "").strip() or repo / ".env")
        registry.admin_setup_enabled = os.getenv("HAI_ADMIN_SETUP", "true").strip().lower() not in {
            "0", "false", "no", "off"}
    return registry


__all__ = ["registry_factory_from_environment", "DatabaseRuntimeRegistry", "RegistryError", "RegistryNotFound",
           "map_sqlite_type", "map_postgres_type", "map_mysql_type", "resolve_in_root"]
