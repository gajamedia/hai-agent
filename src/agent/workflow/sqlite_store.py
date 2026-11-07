from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from .contracts import (
    TaskStatus,
    Workflow,
    WorkflowEvent,
    WorkflowStatus,
    WorkflowTask,
)


class WorkflowStoreError(RuntimeError):
    pass


class WorkflowNotFoundError(WorkflowStoreError):
    pass


class WorkflowAlreadyExistsError(WorkflowStoreError):
    pass


class TaskNotFoundError(WorkflowStoreError):
    pass


class TaskAlreadyExistsError(WorkflowStoreError):
    pass


class WorkflowConcurrencyError(WorkflowStoreError):
    pass


class WorkflowStateError(WorkflowStoreError):
    pass


_WORKFLOW_TRANSITIONS = {
    WorkflowStatus.DRAFT: {WorkflowStatus.PLANNED, WorkflowStatus.CANCELLED},
    WorkflowStatus.PLANNED: {WorkflowStatus.READY, WorkflowStatus.CANCELLED},
    WorkflowStatus.READY: {WorkflowStatus.RUNNING, WorkflowStatus.CANCELLED},
    WorkflowStatus.RUNNING: {
        WorkflowStatus.BLOCKED, WorkflowStatus.SUCCEEDED,
        WorkflowStatus.FAILED, WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.BLOCKED: {
        WorkflowStatus.READY, WorkflowStatus.RUNNING,
        WorkflowStatus.FAILED, WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.SUCCEEDED: set(),
    WorkflowStatus.FAILED: set(),
    WorkflowStatus.CANCELLED: set(),
}

_TASK_TRANSITIONS = {
    TaskStatus.PENDING: {TaskStatus.READY, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.READY: {TaskStatus.RUNNING, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.SUCCEEDED, TaskStatus.FAILED,
        TaskStatus.BLOCKED, TaskStatus.CANCELLED,
    },
    TaskStatus.BLOCKED: {TaskStatus.READY, TaskStatus.CANCELLED},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: {TaskStatus.READY, TaskStatus.CANCELLED},
    TaskStatus.CANCELLED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(value: Mapping[str, Any] | None, field: str) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must contain finite JSON-compatible values") from exc


def _decode_object(value: str, field: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise WorkflowStateError(f"Persisted {field} is invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise WorkflowStateError(f"Persisted {field} must be a JSON object")
    return decoded


class SQLiteWorkflowStore:
    """Durable workflow metadata store. It never executes actions or connects to business DBs."""

    SCHEMA_VERSION = 2

    def __init__(self, database_path: str | Path, *, timeout: float = 10.0):
        raw = str(database_path).strip() if database_path is not None else ""
        if not raw or raw == ":memory:":
            raise ValueError("database_path must be a non-empty filesystem path")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        path = Path(raw).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database_path = str(path.resolve())
        self.timeout = float(timeout)
        self._lock = RLock()
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.database_path, timeout=self.timeout, isolation_level=None
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = FULL")
        return conn

    def _initialize_schema(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS workflow_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO workflow_schema(singleton, schema_version)
                VALUES (1, 1);

                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                      ('draft','planned','ready','running','blocked','succeeded','failed','cancelled')),
                    version INTEGER NOT NULL CHECK(version >= 1),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS workflow_tasks (
                    task_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
                    task_key TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                      ('pending','ready','running','blocked','succeeded','failed','cancelled')),
                    version INTEGER NOT NULL CHECK(version >= 1),
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 0),
                    max_attempts INTEGER NOT NULL CHECK(max_attempts >= 1),
                    input_json TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(workflow_id, task_key)
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_tasks_workflow
                ON workflow_tasks(workflow_id, created_at, task_key);

                CREATE TABLE IF NOT EXISTS workflow_events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
                    task_id TEXT REFERENCES workflow_tasks(task_id) ON DELETE RESTRICT,
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    workflow_version INTEGER,
                    task_version INTEGER,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workflow_events_workflow_time
                ON workflow_events(workflow_id, occurred_at, event_id);

                CREATE TRIGGER IF NOT EXISTS trg_workflow_events_no_update
                BEFORE UPDATE ON workflow_events
                BEGIN SELECT RAISE(ABORT, 'workflow events are append-only'); END;

                CREATE TRIGGER IF NOT EXISTS trg_workflow_events_no_delete
                BEFORE DELETE ON workflow_events
                BEGIN SELECT RAISE(ABORT, 'workflow events are append-only'); END;

                CREATE TRIGGER IF NOT EXISTS trg_workflow_terminal_state
                BEFORE UPDATE OF status ON workflows
                FOR EACH ROW WHEN OLD.status IN ('succeeded','failed','cancelled')
                  AND NEW.status != OLD.status
                BEGIN SELECT RAISE(ABORT, 'workflow terminal state is immutable'); END;

                CREATE TRIGGER IF NOT EXISTS trg_task_terminal_state
                BEFORE UPDATE OF status ON workflow_tasks
                FOR EACH ROW WHEN OLD.status IN ('succeeded','cancelled')
                  AND NEW.status != OLD.status
                BEGIN SELECT RAISE(ABORT, 'task terminal state is immutable'); END;
            """)
            row = conn.execute(
                "SELECT schema_version FROM workflow_schema WHERE singleton=1"
            ).fetchone()
            if row is None:
                raise WorkflowStateError("Workflow schema metadata is missing")
            current_version = int(row["schema_version"])
            if current_version == 1:
                # Additive migration only: preserve every existing workflow/task/event row.
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute("""CREATE TABLE IF NOT EXISTS workflow_task_dependencies (
                        workflow_id TEXT NOT NULL REFERENCES workflows(workflow_id) ON DELETE RESTRICT,
                        task_key TEXT NOT NULL,
                        depends_on_task_key TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(workflow_id, task_key, depends_on_task_key),
                        FOREIGN KEY(workflow_id, task_key) REFERENCES workflow_tasks(workflow_id, task_key) ON DELETE RESTRICT,
                        FOREIGN KEY(workflow_id, depends_on_task_key) REFERENCES workflow_tasks(workflow_id, task_key) ON DELETE RESTRICT,
                        CHECK(task_key != depends_on_task_key)
                    )""")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_workflow_task_dependencies_dep ON workflow_task_dependencies(workflow_id, depends_on_task_key)")
                    conn.execute("UPDATE workflow_schema SET schema_version=2 WHERE singleton=1 AND schema_version=1")
                    conn.commit()
                except BaseException:
                    conn.rollback()
                    raise
                current_version = 2
            if current_version != self.SCHEMA_VERSION:
                raise WorkflowStateError("Unsupported workflow store schema version")

    @staticmethod
    def _workflow(row: sqlite3.Row) -> Workflow:
        return Workflow(
            workflow_id=row["workflow_id"],
            idempotency_key=row["idempotency_key"],
            goal=row["goal"],
            status=WorkflowStatus(row["status"]),
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_decode_object(row["metadata_json"], "workflow metadata"),
        )

    @staticmethod
    def _task(row: sqlite3.Row) -> WorkflowTask:
        return WorkflowTask(
            task_id=row["task_id"], workflow_id=row["workflow_id"],
            task_key=row["task_key"], task_type=row["task_type"],
            description=row["description"], status=TaskStatus(row["status"]),
            version=row["version"], attempt_count=row["attempt_count"],
            max_attempts=row["max_attempts"],
            input_data=_decode_object(row["input_json"], "task input"),
            result_data=None if row["result_json"] is None else
                _decode_object(row["result_json"], "task result"),
            error_code=row["error_code"], created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> WorkflowEvent:
        return WorkflowEvent(
            event_id=row["event_id"], workflow_id=row["workflow_id"],
            task_id=row["task_id"], event_type=row["event_type"],
            from_status=row["from_status"], to_status=row["to_status"],
            occurred_at=row["occurred_at"],
            workflow_version=row["workflow_version"],
            task_version=row["task_version"],
            metadata=_decode_object(row["metadata_json"], "event metadata"),
        )

    @staticmethod
    def _insert_event(conn, workflow_id, event_type, to_status, *,
                      task_id=None, from_status=None, workflow_version=None,
                      task_version=None, metadata=None, occurred_at=None):
        conn.execute(
            """INSERT INTO workflow_events
            (event_id,workflow_id,task_id,event_type,from_status,to_status,
             occurred_at,workflow_version,task_version,metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), workflow_id, task_id, event_type, from_status,
             to_status, occurred_at or _now(), workflow_version, task_version,
             _json_object(metadata, "event metadata")),
        )

    def create_workflow(self, *, goal: str, idempotency_key: str,
                        metadata: Mapping[str, Any] | None = None,
                        workflow_id: str | None = None,
                        now: str | None = None) -> Workflow:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")
        workflow_id = workflow_id or str(uuid.uuid4())
        try:
            workflow_id = str(uuid.UUID(workflow_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("workflow_id must be a UUID") from exc
        timestamp = now or _now()
        metadata_json = _json_object(metadata, "metadata")
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    "SELECT * FROM workflows WHERE idempotency_key=?",
                    (idempotency_key.strip(),),
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return self._workflow(existing)
                conn.execute(
                    """INSERT INTO workflows
                    (workflow_id,idempotency_key,goal,status,version,created_at,updated_at,metadata_json)
                    VALUES (?,?,?,'draft',1,?,?,?)""",
                    (workflow_id, idempotency_key.strip(), goal.strip(),
                     timestamp, timestamp, metadata_json),
                )
                self._insert_event(conn, workflow_id, "workflow.created", "draft",
                                   workflow_version=1, metadata={"idempotency_key": idempotency_key.strip()},
                                   occurred_at=timestamp)
                row = conn.execute("SELECT * FROM workflows WHERE workflow_id=?",
                                   (workflow_id,)).fetchone()
                conn.commit()
                return self._workflow(row)
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise WorkflowAlreadyExistsError("Workflow identifier or idempotency key already exists") from exc
            except BaseException:
                conn.rollback()
                raise

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM workflows WHERE workflow_id=?",
                               (workflow_id,)).fetchone()
            return None if row is None else self._workflow(row)

    def get_by_idempotency_key(self, key: str) -> Workflow | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM workflows WHERE idempotency_key=?",
                               (key,)).fetchone()
            return None if row is None else self._workflow(row)

    def list_workflows(self, *, limit: int = 100) -> tuple[Workflow, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM workflows ORDER BY created_at DESC, workflow_id LIMIT ?",
                (limit,),
            ).fetchall()
            return tuple(self._workflow(row) for row in rows)

    def transition_workflow(self, workflow_id: str, target: WorkflowStatus | str, *,
                            expected_version: int, metadata: Mapping[str, Any] | None = None,
                            now: str | None = None) -> Workflow:
        target = WorkflowStatus(target)
        timestamp = now or _now()
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT * FROM workflows WHERE workflow_id=?",
                                   (workflow_id,)).fetchone()
                if row is None:
                    raise WorkflowNotFoundError(workflow_id)
                current = WorkflowStatus(row["status"])
                if row["version"] != expected_version:
                    raise WorkflowConcurrencyError("Workflow version changed; reload before retrying")
                if target not in _WORKFLOW_TRANSITIONS[current]:
                    raise WorkflowStateError(f"Illegal workflow transition: {current.value} -> {target.value}")
                version = expected_version + 1
                cursor = conn.execute(
                    """UPDATE workflows SET status=?,version=?,updated_at=?
                    WHERE workflow_id=? AND version=?""",
                    (target.value, version, timestamp, workflow_id, expected_version),
                )
                if cursor.rowcount != 1:
                    raise WorkflowConcurrencyError("Workflow changed concurrently")
                self._insert_event(conn, workflow_id, "workflow.status_changed",
                                   target.value, from_status=current.value,
                                   workflow_version=version, metadata=metadata,
                                   occurred_at=timestamp)
                updated = conn.execute("SELECT * FROM workflows WHERE workflow_id=?",
                                       (workflow_id,)).fetchone()
                conn.commit()
                return self._workflow(updated)
            except BaseException:
                conn.rollback()
                raise

    def create_task(self, workflow_id: str, *, task_key: str, task_type: str,
                    description: str, input_data: Mapping[str, Any] | None = None,
                    max_attempts: int = 3, task_id: str | None = None,
                    now: str | None = None) -> WorkflowTask:
        for name, value in (("task_key", task_key), ("task_type", task_type),
                            ("description", description)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        task_id = task_id or str(uuid.uuid4())
        try:
            task_id = str(uuid.UUID(task_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("task_id must be a UUID") from exc
        timestamp = now or _now()
        input_json = _json_object(input_data, "input_data")
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                workflow = conn.execute("SELECT status FROM workflows WHERE workflow_id=?",
                                        (workflow_id,)).fetchone()
                if workflow is None:
                    raise WorkflowNotFoundError(workflow_id)
                if WorkflowStatus(workflow["status"]) not in {WorkflowStatus.DRAFT, WorkflowStatus.PLANNED}:
                    raise WorkflowStateError("Tasks can only be added while workflow is draft or planned")
                conn.execute(
                    """INSERT INTO workflow_tasks
                    (task_id,workflow_id,task_key,task_type,description,status,version,
                     attempt_count,max_attempts,input_json,result_json,error_code,created_at,updated_at)
                    VALUES (?,?,?,?,?,'pending',1,0,?,?,NULL,NULL,?,?)""",
                    (task_id, workflow_id, task_key.strip(), task_type.strip(),
                     description.strip(), max_attempts, input_json, timestamp, timestamp),
                )
                self._insert_event(conn, workflow_id, "task.created", "pending",
                                   task_id=task_id, task_version=1, occurred_at=timestamp)
                row = conn.execute("SELECT * FROM workflow_tasks WHERE task_id=?",
                                   (task_id,)).fetchone()
                conn.commit()
                return self._task(row)
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise TaskAlreadyExistsError("Task identifier or task_key already exists in workflow") from exc
            except BaseException:
                conn.rollback()
                raise

    def create_tasks_from_plan(self, workflow_id: str, plan, *, now: str | None = None) -> tuple[WorkflowTask, ...]:
        """Atomically persist a previously validated WorkflowPlan and all dependency edges."""
        specs = tuple(getattr(plan, "tasks", ()))
        order = tuple(getattr(plan, "execution_order", ()))
        if not specs or len(order) != len(specs):
            raise ValueError("plan must be a validated non-empty WorkflowPlan")
        by_key = {spec.task_key: spec for spec in specs}
        if len(by_key) != len(specs) or set(order) != set(by_key):
            raise ValueError("plan task keys/order are inconsistent")
        # Revalidate graph independently at persistence boundary.
        visiting, visited = set(), set()
        def visit(key):
            if key in visiting:
                raise WorkflowStateError("Dependency cycle detected")
            if key in visited:
                return
            visiting.add(key)
            for dep in by_key[key].depends_on:
                if dep not in by_key:
                    raise WorkflowStateError(f"Unknown dependency: {dep}")
                visit(dep)
            visiting.remove(key)
            visited.add(key)
        for key in by_key:
            visit(key)
        timestamp = now or _now()
        created_ids = {key: str(uuid.uuid4()) for key in order}
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                wf = conn.execute("SELECT status FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
                if wf is None:
                    raise WorkflowNotFoundError(workflow_id)
                if WorkflowStatus(wf["status"]) not in {WorkflowStatus.DRAFT, WorkflowStatus.PLANNED}:
                    raise WorkflowStateError("Tasks can only be added while workflow is draft or planned")
                for key in order:
                    spec = by_key[key]
                    conn.execute("""INSERT INTO workflow_tasks
                        (task_id,workflow_id,task_key,task_type,description,status,version,attempt_count,max_attempts,input_json,result_json,error_code,created_at,updated_at)
                        VALUES (?,?,?,?,?,'pending',1,0,?,?,NULL,NULL,?,?)""",
                        (created_ids[key], workflow_id, spec.task_key, spec.task_type, spec.description,
                         spec.max_attempts, _json_object(spec.input_data, "input_data"), timestamp, timestamp))
                    self._insert_event(conn, workflow_id, "task.created", "pending", task_id=created_ids[key], task_version=1, occurred_at=timestamp)
                    for dep in spec.depends_on:
                        conn.execute("INSERT INTO workflow_task_dependencies(workflow_id,task_key,depends_on_task_key,created_at) VALUES(?,?,?,?)", (workflow_id, key, dep, timestamp))
                    if spec.depends_on:
                        self._insert_event(conn, workflow_id, "task.dependencies_set", "pending", task_id=created_ids[key], task_version=1,
                                           metadata={"task_key": key, "depends_on": list(spec.depends_on)}, occurred_at=timestamp)
                rows = conn.execute("SELECT * FROM workflow_tasks WHERE workflow_id=? AND task_key IN (%s)" % ",".join("?" for _ in order), (workflow_id, *order)).fetchall()
                by_row = {row["task_key"]: self._task(row) for row in rows}
                conn.commit()
                return tuple(by_row[key] for key in order)
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise TaskAlreadyExistsError("Task identifier or task_key already exists in workflow") from exc
            except BaseException:
                conn.rollback()
                raise

    def get_task(self, task_id: str) -> WorkflowTask | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM workflow_tasks WHERE task_id=?",
                               (task_id,)).fetchone()
            return None if row is None else self._task(row)

    def list_tasks(self, workflow_id: str) -> tuple[WorkflowTask, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM workflow_tasks WHERE workflow_id=? ORDER BY created_at,task_key",
                (workflow_id,),
            ).fetchall()
            return tuple(self._task(row) for row in rows)

    def transition_task(self, task_id: str, target: TaskStatus | str, *,
                        expected_version: int, result_data: Mapping[str, Any] | None = None,
                        error_code: str | None = None,
                        metadata: Mapping[str, Any] | None = None,
                        now: str | None = None) -> WorkflowTask:
        target = TaskStatus(target)
        timestamp = now or _now()
        result_json = None if result_data is None else _json_object(result_data, "result_data")
        if error_code is not None and (not isinstance(error_code, str) or not error_code.strip()):
            raise ValueError("error_code must be non-empty text or None")
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT * FROM workflow_tasks WHERE task_id=?",
                                   (task_id,)).fetchone()
                if row is None:
                    raise TaskNotFoundError(task_id)
                current = TaskStatus(row["status"])
                if row["version"] != expected_version:
                    raise WorkflowConcurrencyError("Task version changed; reload before retrying")
                if target not in _TASK_TRANSITIONS[current]:
                    raise WorkflowStateError(f"Illegal task transition: {current.value} -> {target.value}")
                if target == TaskStatus.READY:
                    dependency_rows = conn.execute("""SELECT t.status FROM workflow_task_dependencies d
                        JOIN workflow_tasks t ON t.workflow_id=d.workflow_id AND t.task_key=d.depends_on_task_key
                        WHERE d.workflow_id=? AND d.task_key=?""", (row["workflow_id"], row["task_key"])).fetchall()
                    if any(TaskStatus(item["status"]) != TaskStatus.SUCCEEDED for item in dependency_rows):
                        raise WorkflowStateError("Task cannot become ready until all dependencies have succeeded")
                attempts = row["attempt_count"] + (1 if target == TaskStatus.RUNNING else 0)
                if attempts > row["max_attempts"]:
                    raise WorkflowStateError("Task maximum attempts exceeded")
                version = expected_version + 1
                cursor = conn.execute(
                    """UPDATE workflow_tasks SET status=?,version=?,attempt_count=?,
                    result_json=COALESCE(?,result_json),error_code=?,updated_at=?
                    WHERE task_id=? AND version=?""",
                    (target.value, version, attempts, result_json,
                     error_code, timestamp, task_id, expected_version),
                )
                if cursor.rowcount != 1:
                    raise WorkflowConcurrencyError("Task changed concurrently")
                self._insert_event(conn, row["workflow_id"], "task.status_changed",
                                   target.value, task_id=task_id,
                                   from_status=current.value, task_version=version,
                                   metadata=metadata, occurred_at=timestamp)
                updated = conn.execute("SELECT * FROM workflow_tasks WHERE task_id=?",
                                       (task_id,)).fetchone()
                conn.commit()
                return self._task(updated)
            except BaseException:
                conn.rollback()
                raise

    def set_task_dependencies(self, workflow_id: str, task_key: str, depends_on: tuple[str, ...] | list[str], *, now: str | None = None) -> tuple[str, ...]:
        """Persist dependency edges for a pending task before workflow execution starts."""
        if not isinstance(task_key, str) or not task_key.strip():
            raise ValueError("task_key must be non-empty text")
        if isinstance(depends_on, (str, bytes)):
            raise ValueError("depends_on must be a sequence of task keys")
        dependencies = tuple(depends_on)
        if any(not isinstance(item, str) or not item.strip() for item in dependencies):
            raise ValueError("dependency keys must be non-empty text")
        dependencies = tuple(item.strip() for item in dependencies)
        if len(set(dependencies)) != len(dependencies):
            raise WorkflowStateError("Duplicate task dependency")
        if task_key.strip() in dependencies:
            raise WorkflowStateError("A task cannot depend on itself")
        timestamp = now or _now()
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                wf = conn.execute("SELECT status FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
                if wf is None:
                    raise WorkflowNotFoundError(workflow_id)
                if WorkflowStatus(wf["status"]) not in {WorkflowStatus.DRAFT, WorkflowStatus.PLANNED}:
                    raise WorkflowStateError("Dependencies can only be changed before workflow execution")
                task = conn.execute("SELECT status FROM workflow_tasks WHERE workflow_id=? AND task_key=?", (workflow_id, task_key.strip())).fetchone()
                if task is None:
                    raise TaskNotFoundError(task_key)
                if TaskStatus(task["status"]) != TaskStatus.PENDING:
                    raise WorkflowStateError("Dependencies can only be assigned to pending tasks")
                for dep in dependencies:
                    if conn.execute("SELECT 1 FROM workflow_tasks WHERE workflow_id=? AND task_key=?", (workflow_id, dep)).fetchone() is None:
                        raise TaskNotFoundError(f"Unknown dependency task_key: {dep}")
                conn.execute("DELETE FROM workflow_task_dependencies WHERE workflow_id=? AND task_key=?", (workflow_id, task_key.strip()))
                conn.executemany("INSERT INTO workflow_task_dependencies(workflow_id,task_key,depends_on_task_key,created_at) VALUES(?,?,?,?)", [(workflow_id, task_key.strip(), dep, timestamp) for dep in dependencies])
                self._insert_event(conn, workflow_id, "task.dependencies_set", "pending", task_id=None,
                                   metadata={"task_key": task_key.strip(), "depends_on": list(dependencies)}, occurred_at=timestamp)
                # Validate the whole persisted graph before commit.
                graph_rows = conn.execute("SELECT task_key,depends_on_task_key FROM workflow_task_dependencies WHERE workflow_id=?", (workflow_id,)).fetchall()
                keys = [r["task_key"] for r in conn.execute("SELECT task_key FROM workflow_tasks WHERE workflow_id=?", (workflow_id,)).fetchall()]
                graph = {key: [] for key in keys}
                for edge in graph_rows:
                    graph[edge["task_key"]].append(edge["depends_on_task_key"])
                visiting, visited = set(), set()
                def visit(node):
                    if node in visiting:
                        raise WorkflowStateError("Dependency cycle detected")
                    if node in visited:
                        return
                    visiting.add(node)
                    for parent in graph[node]:
                        visit(parent)
                    visiting.remove(node)
                    visited.add(node)
                for key in keys:
                    visit(key)
                conn.commit()
                return dependencies
            except BaseException:
                conn.rollback()
                raise

    def get_task_dependencies(self, workflow_id: str, task_key: str) -> tuple[str, ...]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT depends_on_task_key FROM workflow_task_dependencies WHERE workflow_id=? AND task_key=? ORDER BY depends_on_task_key", (workflow_id, task_key)).fetchall()
            return tuple(row["depends_on_task_key"] for row in rows)

    def refresh_task_readiness(self, workflow_id: str, *, now: str | None = None) -> tuple[WorkflowTask, ...]:
        """Mark eligible pending tasks ready; block tasks whose prerequisite failed/cancelled."""
        timestamp = now or _now()
        changed = []
        with self._lock, closing(self._connect()) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                wf = conn.execute("SELECT status FROM workflows WHERE workflow_id=?", (workflow_id,)).fetchone()
                if wf is None:
                    raise WorkflowNotFoundError(workflow_id)
                if WorkflowStatus(wf["status"]) not in {WorkflowStatus.DRAFT, WorkflowStatus.PLANNED, WorkflowStatus.READY, WorkflowStatus.RUNNING, WorkflowStatus.BLOCKED}:
                    raise WorkflowStateError("Cannot refresh task readiness for a terminal workflow")
                rows = conn.execute("SELECT * FROM workflow_tasks WHERE workflow_id=? AND status='pending' ORDER BY created_at,task_key", (workflow_id,)).fetchall()
                for row in rows:
                    deps = conn.execute("""SELECT t.status FROM workflow_task_dependencies d
                        JOIN workflow_tasks t ON t.workflow_id=d.workflow_id AND t.task_key=d.depends_on_task_key
                        WHERE d.workflow_id=? AND d.task_key=?""", (workflow_id, row["task_key"])).fetchall()
                    statuses = [TaskStatus(item["status"]) for item in deps]
                    target = None
                    if any(status in {TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED} for status in statuses):
                        target = TaskStatus.BLOCKED
                    elif all(status == TaskStatus.SUCCEEDED for status in statuses):
                        target = TaskStatus.READY
                    if target is None:
                        continue
                    version = row["version"] + 1
                    conn.execute("UPDATE workflow_tasks SET status=?,version=?,updated_at=? WHERE task_id=? AND version=?", (target.value, version, timestamp, row["task_id"], row["version"]))
                    self._insert_event(conn, workflow_id, "task.status_changed", target.value,
                                       task_id=row["task_id"], from_status="pending", task_version=version,
                                       metadata={"reason": "dependency_evaluation"}, occurred_at=timestamp)
                    updated = conn.execute("SELECT * FROM workflow_tasks WHERE task_id=?", (row["task_id"],)).fetchone()
                    changed.append(self._task(updated))
                conn.commit()
                return tuple(changed)
            except BaseException:
                conn.rollback()
                raise

    def list_events(self, workflow_id: str, *, limit: int = 1000) -> tuple[WorkflowEvent, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT * FROM workflow_events WHERE workflow_id=?
                ORDER BY event_seq LIMIT ?""",
                (workflow_id, limit),
            ).fetchall()
            return tuple(self._event(row) for row in rows)


__all__ = [
    "SQLiteWorkflowStore", "WorkflowStoreError", "WorkflowNotFoundError",
    "WorkflowAlreadyExistsError", "TaskNotFoundError", "TaskAlreadyExistsError",
    "WorkflowConcurrencyError", "WorkflowStateError",
]
