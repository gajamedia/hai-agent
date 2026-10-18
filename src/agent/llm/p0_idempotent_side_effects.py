"""Step 39.6C: durable idempotency for transactional side effects.

This helper guarantees one committed effect per idempotency key only when the
side effect and ledger write use the same SQLite database transaction. It does
not make arbitrary HTTP/API/file-system side effects exactly-once; those require
idempotency support at the downstream system or an outbox/inbox protocol.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for a different action."""


class IdempotencyRecordError(RuntimeError):
    """A stored idempotency record is malformed or unsupported."""


@dataclass(frozen=True)
class IdempotentEffectResult:
    idempotency_key: str
    action_name: str
    result: Mapping[str, Any]
    replayed: bool


TransactionalEffect = Callable[[sqlite3.Connection], Mapping[str, Any]]


class SQLiteIdempotentSideEffectStore:
    """Run and record effects once per key using a single SQLite transaction.

    The callback must perform all state changes through the provided connection.
    Do not use this class alone to wrap non-transactional external effects.
    """

    def __init__(self, database_path: str | Path, *, timeout: float = 10.0) -> None:
        self.database_path = str(database_path)
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS p0_idempotent_effects (
                    idempotency_key TEXT PRIMARY KEY,
                    action_name TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=self.timeout, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = %d" % int(self.timeout * 1000))
        return connection

    @staticmethod
    def _validate_text(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty string")
        return value.strip()

    def execute_once(
        self,
        *,
        idempotency_key: str,
        action_name: str,
        effect: TransactionalEffect,
    ) -> IdempotentEffectResult:
        key = self._validate_text(idempotency_key, "idempotency_key")
        action = self._validate_text(action_name, "action_name")
        if not callable(effect):
            raise TypeError("effect must be callable")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT action_name, result_json FROM p0_idempotent_effects WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if existing is not None:
                if existing["action_name"] != action:
                    raise IdempotencyConflictError(
                        f"Idempotency key {key!r} is already bound to action {existing['action_name']!r}."
                    )
                try:
                    stored_result = json.loads(existing["result_json"])
                except (TypeError, json.JSONDecodeError) as exc:
                    raise IdempotencyRecordError(f"Stored result for {key!r} is invalid JSON.") from exc
                if not isinstance(stored_result, dict):
                    raise IdempotencyRecordError(f"Stored result for {key!r} must be a JSON object.")
                connection.commit()
                return IdempotentEffectResult(key, action, stored_result, True)

            result = effect(connection)
            if not isinstance(result, Mapping):
                raise TypeError("effect must return a mapping that can be stored as JSON")
            # Reject NaN/Infinity and unsupported objects so replay is deterministic.
            result_json = json.dumps(dict(result), sort_keys=True, separators=(",", ":"), allow_nan=False)
            connection.execute(
                "INSERT INTO p0_idempotent_effects (idempotency_key, action_name, result_json) VALUES (?, ?, ?)",
                (key, action, result_json),
            )
            connection.commit()
            return IdempotentEffectResult(key, action, json.loads(result_json), False)
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def get_record(self, idempotency_key: str) -> IdempotentEffectResult | None:
        key = self._validate_text(idempotency_key, "idempotency_key")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT action_name, result_json FROM p0_idempotent_effects WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        try:
            result = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise IdempotencyRecordError(f"Stored result for {key!r} is invalid JSON.") from exc
        if not isinstance(result, dict):
            raise IdempotencyRecordError(f"Stored result for {key!r} must be a JSON object.")
        return IdempotentEffectResult(key, row["action_name"], result, True)


__all__ = [
    "IdempotencyConflictError",
    "IdempotencyRecordError",
    "IdempotentEffectResult",
    "SQLiteIdempotentSideEffectStore",
]
