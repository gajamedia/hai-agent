"""Durable task store and audit log for the platform pipeline (SQLite).

Every status change and its audit record are written in **one transaction**, so
an audit that cannot be written also stops the change (fail closed), and a
recorded change always has its audit record. Status changes are compare-and-set
on a version number, so two callers can never both move a task out of the same
state (for example, run it twice).

The store keeps the plan and result of each task as JSON so a task survives a
restart. Audit records hold a hash of the plan, never its contents.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .contracts import TaskStatus, can_transition

_SCHEMA = """
CREATE TABLE IF NOT EXISTS platform_tasks (
    task_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    preview_json TEXT,
    confirm_phrase TEXT,
    result_json TEXT,
    verification_json TEXT,
    error TEXT,
    approved_by TEXT,
    run_key TEXT,
    exec_key TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS platform_tasks_run_key
    ON platform_tasks(owner, operation_id, run_key) WHERE run_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS platform_audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    event TEXT NOT NULL,
    task_id TEXT,
    operation_id TEXT,
    actor TEXT,
    detail_json TEXT NOT NULL
);
"""


class StoreError(RuntimeError):
    """Base class for store errors."""


class TaskNotFound(StoreError):
    """Unknown task (or not visible to the caller; the two are indistinguishable)."""


class TaskConflict(StoreError):
    """The task changed under the caller, or the step is not allowed from its state."""


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    operation_id: str
    owner: str
    status: TaskStatus
    version: int
    plan: Mapping[str, Any]
    preview: Optional[Mapping[str, Any]]
    confirm_phrase: Optional[str]
    result: Optional[Mapping[str, Any]]
    verification: Optional[Mapping[str, Any]]
    error: Optional[str]
    approved_by: Optional[str]
    run_key: Optional[str]
    exec_key: Optional[str]
    created_at: float
    updated_at: float
    expires_at: float


def _dump(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_reject)


def _reject(value: Any):
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _load(text: Optional[str]) -> Any:
    return None if text is None else json.loads(text)


class SQLiteTaskStore:
    def __init__(self, path: str | Path, *, clock: Callable[[], float]) -> None:
        self._path = str(path)
        self._clock = clock
        self._lock = threading.RLock()
        self._memory = self._path == ":memory:"
        self._shared: Optional[sqlite3.Connection] = None
        if not self._memory:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        if self._memory:
            if self._shared is None:
                self._shared = sqlite3.connect(":memory:", check_same_thread=False)
                self._shared.row_factory = sqlite3.Row
            return self._shared
        conn = sqlite3.connect(self._path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def _record(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            task_id=row["task_id"], operation_id=row["operation_id"], owner=row["owner"],
            status=TaskStatus(row["status"]), version=row["version"], plan=_load(row["plan_json"]),
            preview=_load(row["preview_json"]), confirm_phrase=row["confirm_phrase"],
            result=_load(row["result_json"]), verification=_load(row["verification_json"]),
            error=row["error"], approved_by=row["approved_by"], run_key=row["run_key"],
            exec_key=row["exec_key"],
            created_at=row["created_at"], updated_at=row["updated_at"], expires_at=row["expires_at"])

    def _write_audit(self, conn: sqlite3.Connection, event: str, task_id: Optional[str],
                     operation_id: Optional[str], actor: Optional[str],
                     detail: Mapping[str, Any]) -> None:
        conn.execute(
            "INSERT INTO platform_audit(at, event, task_id, operation_id, actor, detail_json) "
            "VALUES (?,?,?,?,?,?)",
            (self._clock(), event, task_id, operation_id, actor, _dump(dict(detail))))

    # -- writes ------------------------------------------------------------

    def create(self, *, operation_id: str, owner: str, status: TaskStatus, plan: Mapping[str, Any],
               preview: Optional[Mapping[str, Any]], confirm_phrase: Optional[str],
               run_key: Optional[str], ttl_seconds: float, audit_event: str,
               audit_detail: Mapping[str, Any]) -> TaskRecord:
        now = self._clock()
        task_id = uuid.uuid4().hex
        with self._lock, self._conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO platform_tasks(task_id, operation_id, owner, status, version, plan_json, "
                    "preview_json, confirm_phrase, run_key, created_at, updated_at, expires_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (task_id, operation_id, owner, status.value, 1, _dump(plan), _dump(preview),
                     confirm_phrase, run_key, now, now, now + ttl_seconds))
            except sqlite3.IntegrityError as exc:
                raise TaskConflict("a task with this idempotency key already exists") from exc
            self._write_audit(conn, audit_event, task_id, operation_id, owner, audit_detail)
            return self._record(conn.execute(
                "SELECT * FROM platform_tasks WHERE task_id=?", (task_id,)).fetchone())

    def transition(self, task_id: str, *, expected_version: int, new_status: TaskStatus,
                   actor: Optional[str], audit_event: str, audit_detail: Mapping[str, Any],
                   result: Any = None, verification: Any = None, error: Optional[str] = None,
                   approved_by: Optional[str] = None, exec_key: Optional[str] = None,
                   extend_ttl: Optional[float] = None) -> TaskRecord:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM platform_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise TaskNotFound("task not found")
            current = TaskStatus(row["status"])
            if row["version"] != expected_version:
                raise TaskConflict("the task changed; reload it and try again")
            if not can_transition(current, new_status):
                raise TaskConflict(f"cannot go from {current.value} to {new_status.value}")
            now = self._clock()
            conn.execute(
                "UPDATE platform_tasks SET status=?, version=version+1, updated_at=?, "
                "result_json=COALESCE(?, result_json), verification_json=COALESCE(?, verification_json), "
                "error=COALESCE(?, error), approved_by=COALESCE(?, approved_by), "
                "expires_at=COALESCE(?, expires_at), exec_key=COALESCE(?, exec_key) WHERE task_id=?",
                (new_status.value, now, _dump(result), _dump(verification), error, approved_by,
                 None if extend_ttl is None else now + extend_ttl, exec_key, task_id))
            self._write_audit(conn, audit_event, task_id, row["operation_id"], actor,
                              {"from": current.value, "to": new_status.value, **dict(audit_detail)})
            return self._record(conn.execute(
                "SELECT * FROM platform_tasks WHERE task_id=?", (task_id,)).fetchone())

    def count_active(self, operation_id: str) -> int:
        """Tasks of this operation that are still waiting (planned, awaiting approval or approved)
        and have not expired."""
        with self._lock, self._conn() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM platform_tasks WHERE operation_id=? AND expires_at>? "
                "AND status IN ('planned','awaiting_approval','approved')",
                (operation_id, self._clock())).fetchone()[0]

    def audit(self, event: str, *, task_id: Optional[str] = None, operation_id: Optional[str] = None,
              actor: Optional[str] = None, detail: Optional[Mapping[str, Any]] = None) -> None:
        """Record a decision that is not a status change (refusals, denials)."""
        with self._lock, self._conn() as conn:
            self._write_audit(conn, event, task_id, operation_id, actor, detail or {})

    # -- reads -------------------------------------------------------------

    def get(self, task_id: str) -> TaskRecord:
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM platform_tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            raise TaskNotFound("task not found")
        return self._record(row)

    def find_by_run_key(self, owner: str, operation_id: str, run_key: str) -> Optional[TaskRecord]:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM platform_tasks WHERE owner=? AND operation_id=? AND run_key=?",
                (owner, operation_id, run_key)).fetchone()
        return None if row is None else self._record(row)

    def list_by_status(self, statuses: Sequence[TaskStatus],
                       operation_id: Optional[str] = None) -> list[TaskRecord]:
        marks = ",".join("?" for _ in statuses)
        sql = f"SELECT * FROM platform_tasks WHERE status IN ({marks})"
        args: list[Any] = [s.value for s in statuses]
        if operation_id is not None:
            sql += " AND operation_id=?"
            args.append(operation_id)
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql + " ORDER BY created_at", args).fetchall()
        return [self._record(r) for r in rows]

    def audit_records(self, task_id: Optional[str] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM platform_audit"
        args: tuple = ()
        if task_id is not None:
            sql += " WHERE task_id=?"
            args = (task_id,)
        with self._lock, self._conn() as conn:
            rows = conn.execute(sql + " ORDER BY seq", args).fetchall()
        return [{"seq": r["seq"], "at": r["at"], "event": r["event"], "task_id": r["task_id"],
                 "operation_id": r["operation_id"], "actor": r["actor"],
                 "detail": _load(r["detail_json"])} for r in rows]


__all__ = ["SQLiteTaskStore", "StoreError", "TaskConflict", "TaskNotFound", "TaskRecord"]
