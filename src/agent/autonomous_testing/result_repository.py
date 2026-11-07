"""SQLite persistence for autonomous-testing batch definitions and results.

This repository is data-only. It never creates runtimes, dispatches actions,
registers capabilities, or writes to agent memory.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any, Optional

from .contracts import (
    AutonomousTestBatch,
    BatchStatus,
    TestBatchResult,
    TestCaseResult,
    canonical_json,
)


class ResultRepositoryError(RuntimeError):
    """Raised when the autonomous-testing result store cannot be used."""


class DuplicateBatchError(ResultRepositoryError):
    """Raised when a batch ID already exists."""


class ResultNotFoundError(ResultRepositoryError):
    """Raised when a requested batch or case result does not exist."""


class SQLiteTestResultRepository:
    """Small, transaction-safe SQLite repository dedicated to test results."""

    SCHEMA_VERSION = 3

    def __init__(self, database_path: str | Path) -> None:
        raw_path = str(database_path)
        if not raw_path.strip():
            raise ValueError("database_path must be non-empty")
        self._uri = raw_path == ":memory:"
        self.database_path = (
            f"file:autonomous_testing_{uuid4().hex}?mode=memory&cache=shared"
            if self._uri
            else str(Path(raw_path).expanduser())
        )
        self._anchor = (
            sqlite3.connect(self.database_path, uri=True)
            if self._uri
            else None
        )
        if not self._uri:
            Path(self.database_path).resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
            isolation_level=None,
            uri=self._uri,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS schema_info (
                        version INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS test_batches (
                        batch_id TEXT PRIMARY KEY,
                        seed INTEGER NOT NULL,
                        difficulty TEXT NOT NULL,
                        requested_cases INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        started_at TEXT,
                        finished_at TEXT,
                        worker_id TEXT,
                        heartbeat_at TEXT,
                        batch_json TEXT NOT NULL,
                        result_json TEXT
                    );

                    CREATE TABLE IF NOT EXISTS test_case_results (
                        batch_id TEXT NOT NULL,
                        case_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        result_json TEXT NOT NULL,
                        PRIMARY KEY (batch_id, case_id),
                        FOREIGN KEY (batch_id)
                            REFERENCES test_batches(batch_id)
                            ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_test_batches_created
                        ON test_batches(created_at);
                    CREATE INDEX IF NOT EXISTS idx_case_results_status
                        ON test_case_results(status);

                    CREATE TABLE IF NOT EXISTS batch_reconciliation_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        batch_id TEXT NOT NULL,
                        previous_status TEXT NOT NULL,
                        previous_worker_id TEXT,
                        previous_heartbeat_at TEXT,
                        reconciled_at TEXT NOT NULL,
                        reconciled_by TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        FOREIGN KEY (batch_id)
                            REFERENCES test_batches(batch_id)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_batch_reconciliation_events
                        ON batch_reconciliation_events(batch_id, event_id);
                    """
                )
                row = connection.execute(
                    "SELECT version FROM schema_info LIMIT 1"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO schema_info(version) VALUES (?)",
                        (self.SCHEMA_VERSION,),
                    )
                else:
                    version = int(row["version"])
                    if version > self.SCHEMA_VERSION or version < 1:
                        raise ResultRepositoryError(
                            f"Unsupported result-store schema version: {version}"
                        )
                    if version in (1, 2):
                        columns = {
                            item["name"]
                            for item in connection.execute(
                                "PRAGMA table_info(test_batches)"
                            ).fetchall()
                        }
                        if "worker_id" not in columns:
                            connection.execute(
                                "ALTER TABLE test_batches ADD COLUMN worker_id TEXT"
                            )
                        if "heartbeat_at" not in columns:
                            connection.execute(
                                "ALTER TABLE test_batches ADD COLUMN heartbeat_at TEXT"
                            )
                        connection.execute(
                            "UPDATE schema_info SET version = ?",
                            (self.SCHEMA_VERSION,),
                        )
        except sqlite3.Error as exc:
            raise ResultRepositoryError(f"Could not initialize SQLite store: {exc}") from exc

    def create_batch(self, batch: AutonomousTestBatch) -> None:
        if not isinstance(batch, AutonomousTestBatch):
            raise TypeError("batch must be AutonomousTestBatch")
        payload = canonical_json(batch)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO test_batches
                    (batch_id, seed, difficulty, requested_cases, status, batch_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch.batch_id,
                        batch.seed,
                        batch.difficulty.value,
                        batch.requested_cases,
                        batch.status.value,
                        payload,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "UNIQUE" in str(exc).upper() or "PRIMARY KEY" in str(exc).upper():
                raise DuplicateBatchError(
                    f"Batch already exists: {batch.batch_id}"
                ) from exc
            raise ResultRepositoryError(str(exc)) from exc
        except sqlite3.Error as exc:
            raise ResultRepositoryError(f"Could not create batch: {exc}") from exc

    def claim_batch_running(
        self, batch_id: str, started_at: str, worker_id: Optional[str] = None
    ) -> bool:
        """Atomically claim a pending batch and persist its execution owner.

        The owner token is an opaque per-execution identifier, not a machine
        identity. Heartbeats from any other owner are rejected.
        """
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        if not isinstance(started_at, str) or not started_at.strip():
            raise ValueError("started_at must be a non-empty string")
        resolved_worker_id = worker_id or uuid4().hex
        if not isinstance(resolved_worker_id, str) or not resolved_worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE test_batches
                    SET status = ?, started_at = ?, worker_id = ?, heartbeat_at = ?
                    WHERE batch_id = ? AND status = ?
                    """,
                    (
                        BatchStatus.RUNNING.value,
                        started_at,
                        resolved_worker_id,
                        started_at,
                        batch_id,
                        BatchStatus.PENDING.value,
                    ),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise ResultRepositoryError(
                f"Could not claim batch for execution: {exc}"
            ) from exc

    def heartbeat_batch(
        self, batch_id: str, worker_id: str, heartbeat_at: str
    ) -> bool:
        """Refresh a running batch lease only for its current owner."""
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise ValueError("batch_id must be a non-empty string")
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be a non-empty string")
        if not isinstance(heartbeat_at, str) or not heartbeat_at.strip():
            raise ValueError("heartbeat_at must be a non-empty string")
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE test_batches
                    SET heartbeat_at = ?
                    WHERE batch_id = ? AND status = ? AND worker_id = ?
                    """,
                    (
                        heartbeat_at,
                        batch_id,
                        BatchStatus.RUNNING.value,
                        worker_id,
                    ),
                )
                return cursor.rowcount == 1
        except sqlite3.Error as exc:
            raise ResultRepositoryError(
                f"Could not update batch heartbeat: {exc}"
            ) from exc

    def mark_batch_running(self, batch_id: str, started_at: str) -> None:
        """Backward-compatible strict transition from pending to running."""
        if not self.claim_batch_running(batch_id, started_at):
            try:
                current = self.get_batch(batch_id)
            except ResultNotFoundError:
                raise
            raise ResultRepositoryError(
                f"Batch cannot transition to running from status: {current['status']}"
            )

    def reconcile_stale_batch(
        self,
        batch_id: str,
        *,
        stale_before: str,
        reconciled_at: str,
        reconciled_by: str,
        reason: str,
    ) -> dict[str, Any]:
        """Close a stale RUNNING batch without replaying it.

        The transition is serialized with BEGIN IMMEDIATE and re-checks the
        heartbeat under the write lock. Existing case results are preserved.
        A durable audit event is inserted in the same transaction.
        """
        for value, name in (
            (batch_id, "batch_id"),
            (stale_before, "stale_before"),
            (reconciled_at, "reconciled_at"),
            (reconciled_by, "reconciled_by"),
            (reason, "reason"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len(reason.strip()) < 10 or len(reason) > 1000:
            raise ValueError("reason must contain between 10 and 1000 characters")
        try:
            # Validate timestamps and normalize comparison to aware UTC.
            cutoff = datetime.fromisoformat(stale_before)
            finished = datetime.fromisoformat(reconciled_at)
            if cutoff.tzinfo is None or cutoff.utcoffset() is None:
                raise ValueError("stale_before must be timezone-aware")
            if finished.tzinfo is None or finished.utcoffset() is None:
                raise ValueError("reconciled_at must be timezone-aware")
            cutoff = cutoff.astimezone(timezone.utc)
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """SELECT status, worker_id, heartbeat_at
                       FROM test_batches WHERE batch_id = ?""",
                    (batch_id,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    raise ResultNotFoundError(f"Batch not found: {batch_id}")
                if row["status"] != BatchStatus.RUNNING.value:
                    connection.rollback()
                    raise ResultRepositoryError(
                        f"Batch is not running; current status is {row['status']}"
                    )
                heartbeat_at = row["heartbeat_at"]
                if heartbeat_at:
                    try:
                        heartbeat = datetime.fromisoformat(heartbeat_at)
                        if heartbeat.tzinfo is None or heartbeat.utcoffset() is None:
                            heartbeat = heartbeat.replace(tzinfo=timezone.utc)
                        heartbeat = heartbeat.astimezone(timezone.utc)
                    except ValueError as exc:
                        connection.rollback()
                        raise ResultRepositoryError(
                            "Batch heartbeat timestamp is invalid; manual investigation is required"
                        ) from exc
                    if heartbeat > cutoff:
                        connection.rollback()
                        raise ResultRepositoryError(
                            "Batch heartbeat is still fresh; reconciliation is not permitted"
                        )
                cursor = connection.execute(
                    """UPDATE test_batches
                       SET status = ?, finished_at = ?
                       WHERE batch_id = ? AND status = ? AND worker_id IS ?
                         AND heartbeat_at IS ?""",
                    (
                        BatchStatus.FAILED.value,
                        finished.astimezone(timezone.utc).isoformat(),
                        batch_id,
                        BatchStatus.RUNNING.value,
                        row["worker_id"],
                        heartbeat_at,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise ResultRepositoryError(
                        "Batch state changed during reconciliation; refresh and inspect again"
                    )
                connection.execute(
                    """INSERT INTO batch_reconciliation_events
                       (batch_id, previous_status, previous_worker_id,
                        previous_heartbeat_at, reconciled_at, reconciled_by, reason)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        batch_id,
                        BatchStatus.RUNNING.value,
                        row["worker_id"],
                        heartbeat_at,
                        finished.astimezone(timezone.utc).isoformat(),
                        reconciled_by.strip(),
                        reason.strip(),
                    ),
                )
                connection.commit()
                return {
                    "batch_id": batch_id,
                    "status": BatchStatus.FAILED.value,
                    "reconciled_at": finished.astimezone(timezone.utc).isoformat(),
                    "reconciled_by": reconciled_by.strip(),
                    "reason": reason.strip(),
                    "replay_performed": False,
                }
        except (ResultNotFoundError, ResultRepositoryError, ValueError):
            raise
        except sqlite3.Error as exc:
            raise ResultRepositoryError(
                f"Could not reconcile interrupted batch: {exc}"
            ) from exc

    def list_reconciliation_events(self, batch_id: str) -> list[dict[str, Any]]:
        try:
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM test_batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if exists is None:
                    raise ResultNotFoundError(f"Batch not found: {batch_id}")
                rows = connection.execute(
                    """SELECT event_id, batch_id, previous_status,
                              previous_worker_id, previous_heartbeat_at,
                              reconciled_at, reconciled_by, reason
                       FROM batch_reconciliation_events
                       WHERE batch_id = ? ORDER BY event_id""",
                    (batch_id,),
                ).fetchall()
                return [dict(row) for row in rows]
        except ResultNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ResultRepositoryError(
                f"Could not read batch reconciliation history: {exc}"
            ) from exc

    def save_case_result(self, batch_id: str, result: TestCaseResult) -> None:
        if not isinstance(result, TestCaseResult):
            raise TypeError("result must be TestCaseResult")
        try:
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM test_batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if exists is None:
                    raise ResultNotFoundError(f"Batch not found: {batch_id}")
                connection.execute(
                    """
                    INSERT INTO test_case_results(batch_id, case_id, status, result_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(batch_id, case_id) DO UPDATE SET
                        status = excluded.status,
                        saved_at = CURRENT_TIMESTAMP,
                        result_json = excluded.result_json
                    """,
                    (
                        batch_id,
                        result.case_id,
                        result.status.value,
                        canonical_json(result),
                    ),
                )
        except ResultNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ResultRepositoryError(f"Could not save case result: {exc}") from exc

    def complete_batch(
        self, result: TestBatchResult, *, worker_id: Optional[str] = None
    ) -> None:
        if not isinstance(result, TestBatchResult):
            raise TypeError("result must be TestBatchResult")
        payload = canonical_json(result)
        if worker_id is None:
            self._update_batch(
                result.batch_id,
                """
                UPDATE test_batches
                SET status = ?, finished_at = ?, result_json = ?
                WHERE batch_id = ?
                """,
                (
                    result.status.value,
                    result.finished_at.isoformat() if result.finished_at else None,
                    payload,
                    result.batch_id,
                ),
            )
            return
        self._update_owned_batch(
            result.batch_id,
            """
            UPDATE test_batches
            SET status = ?, finished_at = ?, result_json = ?, heartbeat_at = ?
            WHERE batch_id = ? AND status = ? AND worker_id = ?
            """,
            (
                result.status.value,
                result.finished_at.isoformat() if result.finished_at else None,
                payload,
                result.finished_at.isoformat() if result.finished_at else None,
                result.batch_id,
                BatchStatus.RUNNING.value,
                worker_id,
            ),
        )

    def mark_batch_failed(
        self,
        batch_id: str,
        finished_at: str,
        error: str,
        *,
        worker_id: Optional[str] = None,
    ) -> None:
        sql = """
            UPDATE test_batches
            SET status = ?, finished_at = ?, result_json = ?
            WHERE batch_id = ?
        """
        params = (
            BatchStatus.FAILED.value,
            finished_at,
            json.dumps({"error": str(error)}, sort_keys=True),
            batch_id,
        )
        if worker_id is None:
            self._update_batch(batch_id, sql, params)
            return
        self._update_owned_batch(
            batch_id,
            sql.rstrip().removesuffix("WHERE batch_id = ?")
            + "WHERE batch_id = ? AND status = ? AND worker_id = ?",
            params + (BatchStatus.RUNNING.value, worker_id),
        )

    def _update_batch(self, batch_id: str, sql: str, params: tuple) -> None:
        try:
            with self._connect() as connection:
                cursor = connection.execute(sql, params)
                if cursor.rowcount == 0:
                    raise ResultNotFoundError(f"Batch not found: {batch_id}")
        except ResultNotFoundError:
            raise
        except sqlite3.Error as exc:
            raise ResultRepositoryError(f"Could not update batch: {exc}") from exc


    def _update_owned_batch(self, batch_id: str, sql: str, params: tuple) -> None:
        try:
            with self._connect() as connection:
                cursor = connection.execute(sql, params)
                if cursor.rowcount == 0:
                    raise ResultRepositoryError(
                        f"Batch ownership or running status changed: {batch_id}"
                    )
        except ResultRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise ResultRepositoryError(f"Could not update owned batch: {exc}") from exc

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT batch_id, seed, difficulty, requested_cases, status,
                           created_at, started_at, finished_at, worker_id, heartbeat_at,
                           batch_json, result_json
                    FROM test_batches WHERE batch_id = ?
                    """,
                    (batch_id,),
                ).fetchone()
                if row is None:
                    raise ResultNotFoundError(f"Batch not found: {batch_id}")
                data = dict(row)
                data["batch"] = json.loads(data.pop("batch_json"))
                result_json = data.pop("result_json")
                data["result"] = json.loads(result_json) if result_json else None
                return data
        except ResultNotFoundError:
            raise
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise ResultRepositoryError(f"Could not read batch: {exc}") from exc

    def get_case_result(self, batch_id: str, case_id: str) -> dict[str, Any]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT batch_id, case_id, status, saved_at, result_json
                    FROM test_case_results
                    WHERE batch_id = ? AND case_id = ?
                    """,
                    (batch_id, case_id),
                ).fetchone()
                if row is None:
                    raise ResultNotFoundError(
                        f"Case result not found: {batch_id}/{case_id}"
                    )
                data = dict(row)
                data["result"] = json.loads(data.pop("result_json"))
                return data
        except ResultNotFoundError:
            raise
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise ResultRepositoryError(f"Could not read case result: {exc}") from exc

    def close(self) -> None:
        """Release the anchor connection used by the shared in-memory database."""
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None

    def list_case_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Return saved case results for a batch in insertion order."""
        try:
            with self._connect() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM test_batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if exists is None:
                    raise ResultNotFoundError(f"Batch not found: {batch_id}")
                rows = connection.execute(
                    """
                    SELECT batch_id, case_id, status, saved_at, result_json
                    FROM test_case_results
                    WHERE batch_id = ?
                    ORDER BY rowid ASC
                    """,
                    (batch_id,),
                ).fetchall()
                items = []
                for row in rows:
                    item = dict(row)
                    item["result"] = json.loads(item.pop("result_json"))
                    items.append(item)
                return items
        except ResultNotFoundError:
            raise
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise ResultRepositoryError(f"Could not list case results: {exc}") from exc

    def list_batches(self, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT batch_id, seed, difficulty, requested_cases, status,
                           created_at, started_at, finished_at, heartbeat_at
                    FROM test_batches
                    ORDER BY created_at DESC, batch_id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise ResultRepositoryError(f"Could not list batches: {exc}") from exc


__all__ = [
    "SQLiteTestResultRepository",
    "ResultRepositoryError",
    "DuplicateBatchError",
    "ResultNotFoundError",
]
