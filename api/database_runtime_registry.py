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
from agent.llm.business_database_adapter import ReadOnlyTableSpec
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
from agent.workflow.table_report_planner import TableCatalogEntry, TableReportPlanner
from agent.workflow.table_report_service import TableReportService
from api.database_engines import (
    EngineError, MySQLEngine, PostgresEngine, SQLiteEngine, map_mysql_type, map_postgres_type,
    map_sqlite_type,
)

_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
_SQLITE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}
MAX_TABLES = 20
MAX_COLUMNS_PER_TABLE = 100


class RegistryError(ValueError):
    """Invalid request to the registry (maps to HTTP 4xx)."""


class RegistryNotFound(RegistryError):
    pass


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


class DatabaseRuntimeRegistry:
    def __init__(self, *, data_root: str | Path, work_dir: str | Path,
                 authorization_policy: Any, llm_provider_factory: Callable[[], Any],
                 max_connections: int = 10, restore: bool = True,
                 profiles: Mapping[str, Any] | None = None,
                 engines: Mapping[str, Any] | None = None) -> None:
        self.data_root = Path(data_root).expanduser()
        if not self.data_root.is_dir():
            raise RegistryError(f"data_root is not a directory: {self.data_root}")
        self.work_dir = Path(work_dir).expanduser()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.authorization_policy = authorization_policy
        self.llm_provider_factory = llm_provider_factory
        self.max_connections = max_connections
        self._engines: dict[str, Any] = {"sqlite": SQLiteEngine(self.data_root)}
        by_type: dict[str, dict[str, Any]] = {}
        for pname, profile in (profiles or {}).items():
            by_type.setdefault(profile.type, {})[pname] = profile
        if "postgresql" in by_type:
            self._engines["postgresql"] = PostgresEngine(by_type["postgresql"])
        if "mysql" in by_type:
            self._engines["mysql"] = MySQLEngine(by_type["mysql"])
        if engines:
            self._engines.update(engines)
        self._connections: dict[str, _Connection] = {}
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
                have = {r[1] for r in db.execute("PRAGMA table_info(connections)")}
                if "engine" not in have:
                    db.execute("ALTER TABLE connections ADD COLUMN engine TEXT NOT NULL "
                               "DEFAULT 'sqlite'")
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
                    "connected_at, approved_by, approved_at, selection_json, fingerprint, engine) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (conn.name, conn.filename, conn.approved_by, conn.approved_at,
                     conn.tables_approved_by or conn.approved_by,
                     conn.tables_approved_at or conn.approved_at,
                     json.dumps(conn.selection, sort_keys=True),
                     self._fingerprint(conn.engine, conn.schema, conn.selection), conn.engine))
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
                self._activate_locked(conn, selection, row["approved_by"], row["approved_at"])
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
            # Driver errors can echo host names or users; keep only our own messages.
            detail = str(exc) if isinstance(exc, EngineError) else "connection failed"
            raise RegistryError(f"cannot open database: {detail}") from exc
        return _Connection(name=name, filename=source.strip(), engine=engine,
                           approved_by=actor_id, approved_at=approved_at, manager=manager,
                           schema=schema, warnings=tuple(warnings))

    def supported_engine_names(self) -> list[str]:
        """Product names users might type when asking for an engine (already supported ones)."""
        aliases = {"sqlite": ["sqlite"], "postgresql": ["postgresql", "postgres"],
                   "mysql": ["mysql", "mariadb"]}
        return sorted({alias for kind in {"sqlite", "postgresql", "mysql"}
                       for alias in aliases[kind]})

    def available_engines(self) -> list[str]:
        return sorted(self._engines)

    def profiles(self) -> list[dict[str, str]]:
        """Names and types only: never hosts, users or secrets."""
        out = [{"name": n, "type": kind}
               for kind, engine in self._engines.items()
               for n in getattr(engine, "profiles", {})]
        return sorted(out, key=lambda item: item["name"])

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
            resolve_in_root(self.data_root, source)  # fail fast before taking the lock
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
    def activate(self, name: str, tables: Any, *, actor_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            conn = self._get(name)
            self._activate_locked(conn, tables, actor_id or conn.approved_by,
                                  datetime.now(timezone.utc).isoformat(), persist=True)
            return self._describe(conn)

    def _activate_locked(self, conn: _Connection, tables: Any, actor: str, at: str,
                         *, persist: bool = False) -> None:
        name = conn.name
        map_type = self._engines[conn.engine].map_type
        if True:
            if not isinstance(tables, Mapping) or not tables or len(tables) > MAX_TABLES:
                raise RegistryError(f"select between 1 and {MAX_TABLES} tables")
            by_table = {t.name: t for t in conn.schema.tables}
            specs: dict[str, ReadOnlyTableSpec] = {}
            entries: list[TableCatalogEntry] = []
            for table, sel in tables.items():
                found = by_table.get(table)
                if found is None:
                    raise RegistryError(f"table '{table}' is not in the discovered schema")
                if not isinstance(sel, Mapping):
                    raise RegistryError(f"selection for '{table}' must be an object")
                unknown = set(sel) - {"columns", "filterable_columns", "key_column",
                                      "required_non_null_columns", "description"}
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
                    spec = ReadOnlyTableSpec(
                        table=table, columns=tuple(columns), filterable_columns=tuple(filterable))
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
            conn.service = self._build_service(conn, specs, entries)
            conn.tables = tuple(specs)
            conn.selection = {t: dict(v) for t, v in tables.items()}
            conn.tables_approved_by, conn.tables_approved_at = actor, at
            self._degraded.pop(name, None)
            if persist:
                self._save(conn)

    def _build_service(self, conn: _Connection, specs: Mapping[str, ReadOnlyTableSpec],
                       entries: list[TableCatalogEntry]) -> TableReportService:
        assembly = build_business_database_gateway(
            connection_manager=conn.manager, profile_name=conn.name,
            authorization_policy=self.authorization_policy, table_specs=specs,
            trace_path=self.work_dir / f"trace-{conn.name}.jsonl",
            paramstyle=self._engines[conn.engine].paramstyle,
            identifier_quote=getattr(self._engines[conn.engine], "quote", None),
            dialect=getattr(self._engines[conn.engine], "dialect", "limit"))
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
        planner = TableReportPlanner(self._llm, entries, experience=experience)
        return TableReportService(planner=planner, runner=runner, experience=experience)

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
                "tables": list(conn.tables)}

    def schema(self, name: str) -> dict[str, Any]:
        with self._lock:
            return self._get(name).schema.to_dict()

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
    profiles_path = os.getenv("HAI_DATABASE_PROFILES", "").strip()
    if profiles_path:
        from api.database_engines import load_profiles
        profiles = load_profiles(profiles_path)
    return DatabaseRuntimeRegistry(data_root=root, work_dir=work, authorization_policy=policy,
                                   llm_provider_factory=create_llm_provider, profiles=profiles)


__all__ = ["registry_factory_from_environment", "DatabaseRuntimeRegistry", "RegistryError", "RegistryNotFound",
           "map_sqlite_type", "map_postgres_type", "map_mysql_type", "resolve_in_root"]
