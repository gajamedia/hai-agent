"""Durable request-level idempotency for the simulation plan-run API.

This ledger does not make execution and ledger writes one atomic transaction:
plan execution changes a separate runtime/environment. To avoid duplicate actions,
interrupted requests are marked unknown and must be reconciled rather than retried.
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


@dataclass(frozen=True)
class PlanRunIdempotencyClaim:
    status: str
    response: Mapping[str, Any] | None = None


class PlanRunIdempotencyStore:
    """SQLite ledger for completed, rejected, running, and unknown plan requests."""

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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            isolation_level=None,
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

    def claim(self, *, idempotency_key: str, request_fingerprint: str) -> PlanRunIdempotencyClaim:
        key = self._text(idempotency_key, "idempotency_key")
        fingerprint = self._text(request_fingerprint, "request_fingerprint")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT request_fingerprint, status, response_json FROM plan_run_idempotency WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO plan_run_idempotency (idempotency_key, request_fingerprint, status) VALUES (?, ?, 'running')",
                    (key, fingerprint),
                )
                connection.commit()
                return PlanRunIdempotencyClaim("running")
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
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _update(
        self,
        *,
        idempotency_key: str,
        request_fingerprint: str,
        status: str,
        response: Mapping[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        key = self._text(idempotency_key, "idempotency_key")
        fingerprint = self._text(request_fingerprint, "request_fingerprint")
        if status not in {"completed", "rejected", "unknown"}:
            raise ValueError("Unsupported terminal idempotency status")
        response_json = None
        if response is not None:
            response_json = json.dumps(
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
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def complete(self, *, idempotency_key: str, request_fingerprint: str, response: Mapping[str, Any]) -> None:
        self._update(
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            status="completed",
            response=response,
            http_status=200,
        )

    def reject(
        self, *, idempotency_key: str, request_fingerprint: str,
        status_code: int, detail: Mapping[str, Any],
    ) -> None:
        self._update(
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            status="rejected",
            response={"status_code": int(status_code), "detail": dict(detail)},
            http_status=int(status_code),
        )

    def mark_unknown(self, *, idempotency_key: str, request_fingerprint: str) -> None:
        self._update(
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            status="unknown",
            response={
                "error": "idempotency_outcome_unknown",
                "message": "Execution may have started; reconcile state before retrying.",
            },
            http_status=409,
        )

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
