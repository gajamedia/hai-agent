"""Durable idempotency ledger and global mutation gate for simulation execution.

Execution and SQLite writes cannot be one atomic transaction. A persistent gate
therefore remains held while a mutation is in progress and is retained as
``unknown`` if its outcome cannot be established. Operators must explicitly
reconcile the observed simulation state before clearing an unresolved gate.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class PlanRunIdempotencyConflict(RuntimeError):
    """An idempotency key was reused with a different request fingerprint."""


class PlanRunIdempotencyInProgress(RuntimeError):
    """A matching request already exists and is not safely replayable."""


class PlanRunExecutionBlocked(RuntimeError):
    """Another unresolved simulation mutation blocks new mutations."""


@dataclass(frozen=True)
class PlanRunIdempotencyClaim:
    status: str
    response: Mapping[str, Any] | None = None


class PlanRunIdempotencyStore:
    """SQLite ledger with a single durable gate shared by all mutating routes."""

    def __init__(self, database_path: str | Path, *, timeout: float = 10.0) -> None:
        self.database_path = str(database_path)
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.database_path != ":memory:":
            Path(self.database_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS plan_run_idempotency (
                    idempotency_key TEXT PRIMARY KEY,
                    request_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'rejected', 'unknown')),
                    response_json TEXT,
                    http_status INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS execution_mutation_gate (
                    gate_id INTEGER PRIMARY KEY CHECK(gate_id = 1),
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('running', 'unknown')),
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS plan_run_reconciliations (
                    idempotency_key TEXT PRIMARY KEY,
                    notes TEXT NOT NULL,
                    reconciled_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            # Upgrade databases created by the earlier request-only ledger. Any
            # unresolved historical row is promoted to the global fail-closed gate.
            gate = connection.execute(
                "SELECT owner, status FROM execution_mutation_gate WHERE gate_id = 1"
            ).fetchone()
            if gate is None:
                unresolved = connection.execute(
                    """SELECT p.idempotency_key, p.status
                       FROM plan_run_idempotency p
                       LEFT JOIN plan_run_reconciliations r
                         ON r.idempotency_key = p.idempotency_key
                       WHERE p.status IN ('running', 'unknown')
                         AND r.idempotency_key IS NULL
                       ORDER BY CASE p.status WHEN 'unknown' THEN 0 ELSE 1 END,
                                p.created_at
                       LIMIT 1"""
                ).fetchone()
                if unresolved is not None:
                    connection.execute(
                        "INSERT INTO execution_mutation_gate (gate_id, owner, status) VALUES (1, ?, ?)",
                        (f"plan-run:{unresolved['idempotency_key']}", str(unresolved["status"])),
                    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path, timeout=self.timeout, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return connection

    @staticmethod
    def _text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _decode_response(raw: str | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Stored plan-run idempotency response is invalid JSON") from exc
        if not isinstance(value, dict):
            raise RuntimeError("Stored plan-run idempotency response must be a JSON object")
        return value

    @staticmethod
    def _gate_exists(connection: sqlite3.Connection) -> bool:
        return connection.execute(
            "SELECT 1 FROM execution_mutation_gate WHERE gate_id = 1"
        ).fetchone() is not None

    def claim(self, *, idempotency_key: str, request_fingerprint: str) -> PlanRunIdempotencyClaim:
        """Atomically reserve a key and the global mutation gate for a new request."""
        key = self._text(idempotency_key, "idempotency_key")
        fingerprint = self._text(request_fingerprint, "request_fingerprint")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_fingerprint, status, response_json FROM plan_run_idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                if row["request_fingerprint"] != fingerprint:
                    raise PlanRunIdempotencyConflict(
                        f"Idempotency key {key!r} is already bound to a different request."
                    )
                status = str(row["status"])
                response = self._decode_response(row["response_json"])
                connection.commit()
                if status == "running":
                    raise PlanRunIdempotencyInProgress(
                        f"Request {key!r} is already running or was interrupted before a result was recorded."
                    )
                return PlanRunIdempotencyClaim(status, response)

            if self._gate_exists(connection):
                gate = connection.execute(
                    "SELECT owner, status FROM execution_mutation_gate WHERE gate_id = 1"
                ).fetchone()
                raise PlanRunExecutionBlocked(
                    f"Simulation mutation is blocked by unresolved owner {gate['owner']!r} ({gate['status']})."
                )
            unresolved = connection.execute(
                """SELECT p.idempotency_key, p.status
                   FROM plan_run_idempotency p
                   LEFT JOIN plan_run_reconciliations r ON r.idempotency_key = p.idempotency_key
                   WHERE p.status IN ('running', 'unknown') AND r.idempotency_key IS NULL
                   LIMIT 1"""
            ).fetchone()
            if unresolved is not None:
                raise PlanRunExecutionBlocked(
                    f"Previous plan-run {unresolved['idempotency_key']!r} requires reconciliation."
                )
            connection.execute(
                "INSERT INTO plan_run_idempotency (idempotency_key, request_fingerprint, status) VALUES (?, ?, 'running')",
                (key, fingerprint),
            )
            connection.execute(
                "INSERT INTO execution_mutation_gate (gate_id, owner, status) VALUES (1, ?, 'running')",
                (f"plan-run:{key}",),
            )
            connection.commit()
            return PlanRunIdempotencyClaim("running")
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _update(
        self, *, idempotency_key: str, request_fingerprint: str, status: str,
        response: Mapping[str, Any] | None = None, http_status: int | None = None,
    ) -> None:
        key = self._text(idempotency_key, "idempotency_key")
        fingerprint = self._text(request_fingerprint, "request_fingerprint")
        if status not in {"completed", "rejected", "unknown"}:
            raise ValueError("Unsupported terminal idempotency status")
        response_json = None if response is None else json.dumps(
            dict(response), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """UPDATE plan_run_idempotency
                   SET status = ?, response_json = ?, http_status = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE idempotency_key = ? AND request_fingerprint = ? AND status = 'running'""",
                (status, response_json, http_status, key, fingerprint),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Plan-run idempotency reservation is missing or no longer running")
            owner = f"plan-run:{key}"
            if status == "unknown":
                gate = connection.execute(
                    "SELECT owner FROM execution_mutation_gate WHERE gate_id=1"
                ).fetchone()
                if gate is not None and str(gate["owner"]) != owner:
                    raise RuntimeError("Cannot mark unknown: the global mutation gate is owned by another request")
                if gate is None:
                    connection.execute(
                        "INSERT INTO execution_mutation_gate (gate_id, owner, status) VALUES (1, ?, 'unknown')",
                        (owner,),
                    )
                else:
                    connection.execute(
                        "UPDATE execution_mutation_gate SET status='unknown', updated_at=CURRENT_TIMESTAMP WHERE gate_id=1 AND owner=?",
                        (owner,),
                    )
            else:
                connection.execute(
                    "DELETE FROM execution_mutation_gate WHERE gate_id = 1 AND owner = ?",
                    (owner,),
                )
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def complete(self, *, idempotency_key: str, request_fingerprint: str, response: Mapping[str, Any]) -> None:
        self._update(idempotency_key=idempotency_key, request_fingerprint=request_fingerprint,
                     status="completed", response=response, http_status=200)

    def reject(self, *, idempotency_key: str, request_fingerprint: str,
               status_code: int, detail: Mapping[str, Any]) -> None:
        self._update(idempotency_key=idempotency_key, request_fingerprint=request_fingerprint,
                     status="rejected", response={"status_code": int(status_code), "detail": dict(detail)},
                     http_status=int(status_code))

    def mark_unknown(self, *, idempotency_key: str, request_fingerprint: str) -> None:
        self._update(
            idempotency_key=idempotency_key, request_fingerprint=request_fingerprint,
            status="unknown", response={
                "error": "idempotency_outcome_unknown",
                "message": "Execution may have started; reconcile state before retrying.",
            }, http_status=409,
        )

    def acquire_mutation(self, owner: str) -> None:
        """Acquire the persistent global gate for a non-idempotent mutation."""
        owner = self._text(owner, "owner")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            gate = connection.execute(
                "SELECT owner, status FROM execution_mutation_gate WHERE gate_id = 1"
            ).fetchone()
            if gate is not None:
                raise PlanRunExecutionBlocked(
                    f"Simulation mutation is blocked by unresolved owner {gate['owner']!r} ({gate['status']})."
                )
            unresolved = connection.execute(
                """SELECT p.idempotency_key, p.status FROM plan_run_idempotency p
                   LEFT JOIN plan_run_reconciliations r ON r.idempotency_key = p.idempotency_key
                   WHERE p.status IN ('running', 'unknown') AND r.idempotency_key IS NULL LIMIT 1"""
            ).fetchone()
            if unresolved is not None:
                raise PlanRunExecutionBlocked(
                    f"Previous plan-run {unresolved['idempotency_key']!r} requires reconciliation."
                )
            connection.execute(
                "INSERT INTO execution_mutation_gate (gate_id, owner, status) VALUES (1, ?, 'running')",
                (owner,),
            )
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def release_mutation(self, owner: str) -> None:
        owner = self._text(owner, "owner")
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM execution_mutation_gate WHERE gate_id = 1 AND owner = ?",
                (owner,),
            )

    def mark_mutation_unknown(self, owner: str) -> None:
        owner = self._text(owner, "owner")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE execution_mutation_gate SET status='unknown', updated_at=CURRENT_TIMESTAMP WHERE gate_id=1 AND owner=?",
                (owner,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Execution mutation gate is missing or owned by another request")

    def reconcile(self, *, notes: str) -> dict[str, Any]:
        """Clear the global gate after an operator explicitly reviews the state.

        Unknown per-key rows remain unknown and can never be replayed. A reconciliation
        record allows new keys after the operator has reviewed the actual environment.
        """
        notes = self._text(notes, "notes")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            gate = connection.execute(
                "SELECT owner, status FROM execution_mutation_gate WHERE gate_id=1"
            ).fetchone()
            if gate is None:
                connection.commit()
                return {"cleared": False, "message": "No unresolved mutation gate exists."}
            unresolved_rows = connection.execute(
                """SELECT p.idempotency_key FROM plan_run_idempotency p
                   LEFT JOIN plan_run_reconciliations r ON r.idempotency_key=p.idempotency_key
                   WHERE p.status IN ('running','unknown') AND r.idempotency_key IS NULL"""
            ).fetchall()
            for row in unresolved_rows:
                connection.execute(
                    "INSERT OR REPLACE INTO plan_run_reconciliations (idempotency_key, notes) VALUES (?, ?)",
                    (row["idempotency_key"], notes),
                )
            connection.execute("DELETE FROM execution_mutation_gate WHERE gate_id=1")
            connection.commit()
            return {
                "cleared": True,
                "previous_owner": str(gate["owner"]),
                "previous_status": str(gate["status"]),
                "reconciled_keys": [str(row["idempotency_key"]) for row in unresolved_rows],
            }
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def get_record(self, idempotency_key: str) -> PlanRunIdempotencyClaim | None:
        key = self._text(idempotency_key, "idempotency_key")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, response_json FROM plan_run_idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return PlanRunIdempotencyClaim(str(row["status"]), self._decode_response(row["response_json"]))
