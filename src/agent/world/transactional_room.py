"""SQLite-backed Room simulation with transactional, epoch-scoped idempotency.

This is an additive prototype and is not wired into ExecutionRuntime.
For a given Room epoch, state mutation and idempotency result commit atomically.
Reset advances the epoch, so keys from an earlier Room session do not replay into
the new session. This does not cover external I/O or external side effects.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping


class IdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for a different request in the same epoch."""


class SQLiteRoomEnvironment:
    def __init__(
        self,
        database_path: str | Path,
        *,
        spare_key_available: bool = True,
        timeout: float = 10.0,
    ) -> None:
        self.database_path = str(database_path)
        self.spare_key_available = bool(spare_key_available)
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.database_path == ":memory:":
            raise ValueError("Use a file-backed database for multi-connection transactions.")
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS room_state (
                state_key TEXT PRIMARY KEY, state_value_json TEXT NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS room_meta (
                meta_key TEXT PRIMARY KEY, meta_value TEXT NOT NULL)""")
            con.execute("""CREATE TABLE IF NOT EXISTS room_idempotency (
                epoch INTEGER NOT NULL,
                idempotency_key TEXT NOT NULL,
                action_name TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL,
                committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (epoch, idempotency_key))""")
            con.execute("BEGIN IMMEDIATE")
            try:
                # Refuse a legacy schema rather than silently claiming epoch semantics.
                columns = {row["name"] for row in con.execute(
                    "PRAGMA table_info(room_idempotency)"
                ).fetchall()}
                pk_columns = [row["name"] for row in sorted(
                    (r for r in con.execute("PRAGMA table_info(room_idempotency)").fetchall()
                     if r["pk"]),
                    key=lambda r: r["pk"],
                )]
                if "epoch" not in columns or pk_columns != ["epoch", "idempotency_key"]:
                    raise RuntimeError(
                        "Legacy room_idempotency schema detected. Use a new database "
                        "or migrate the existing ledger explicitly before enabling epoch scope."
                    )
                con.execute(
                    "INSERT OR IGNORE INTO room_meta(meta_key, meta_value) VALUES ('epoch', '1')"
                )
                count = con.execute("SELECT COUNT(*) FROM room_state").fetchone()[0]
                if count == 0:
                    self._write_state(con, self._initial_state())
                con.commit()
            except BaseException:
                if con.in_transaction:
                    con.rollback()
                raise

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(
            self.database_path, timeout=self.timeout, isolation_level=None
        )
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        return con

    def _initial_state(self) -> dict[str, Any]:
        return {
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "available" if self.spare_key_available else "unavailable",
            "agent_location": "room",
        }

    @staticmethod
    def _write_state(con: sqlite3.Connection, state: Mapping[str, Any]) -> None:
        con.execute("DELETE FROM room_state")
        con.executemany(
            "INSERT INTO room_state(state_key, state_value_json) VALUES (?, ?)",
            [(k, json.dumps(v, ensure_ascii=False, allow_nan=False))
             for k, v in state.items()],
        )

    @staticmethod
    def _read_state(con: sqlite3.Connection) -> dict[str, Any]:
        return {
            row["state_key"]: json.loads(row["state_value_json"])
            for row in con.execute(
                "SELECT state_key, state_value_json FROM room_state"
            ).fetchall()
        }

    @staticmethod
    def _canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )

    @staticmethod
    def _current_epoch(con: sqlite3.Connection) -> int:
        row = con.execute(
            "SELECT meta_value FROM room_meta WHERE meta_key = 'epoch'"
        ).fetchone()
        if row is None:
            raise RuntimeError("Room epoch metadata is missing.")
        return int(row["meta_value"])

    def observe(self) -> dict[str, Any]:
        with self._connect() as con:
            return self._read_state(con)

    @property
    def epoch(self) -> int:
        with self._connect() as con:
            return self._current_epoch(con)

    def reset(self) -> None:
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            epoch = self._current_epoch(con) + 1
            self._write_state(con, self._initial_state())
            con.execute(
                "UPDATE room_meta SET meta_value = ? WHERE meta_key = 'epoch'",
                (str(epoch),),
            )
            con.commit()
        except BaseException:
            if con.in_transaction:
                con.rollback()
            raise
        finally:
            con.close()

    @staticmethod
    def _apply_action(state: dict[str, Any], action: str) -> dict[str, Any]:
        if action == "open_box":
            if state["box"] == "closed":
                state["box"] = "open"
                return {"success": True, "message": "The box is now open."}
            return {"success": False, "message": "The box is already open."}
        if action == "take_key":
            if state["box"] != "open":
                return {"success": False, "message": "The box is closed."}
            if state["key"] != "inside_box":
                return {"success": False, "message": "The key is not inside the box."}
            state["key"] = "agent"
            return {"success": True, "message": "The agent picked up the key."}
        if action == "find_spare_key":
            if state["spare_key"] != "available":
                return {"success": False, "message": "The spare key is not available."}
            state["spare_key"] = "found"
            state["key"] = "agent"
            return {"success": True, "message": "The agent found the spare key."}
        if action == "unlock_door":
            if state["key"] != "agent":
                return {"success": False, "message": "The agent does not have the key."}
            if state["door"] != "locked":
                return {"success": False, "message": "The door is already unlocked."}
            state["door"] = "unlocked"
            return {"success": True, "message": "The door is now unlocked."}
        return {"success": False, "message": f"Unknown action: {action}"}

    def execute(self, action: str) -> dict[str, Any]:
        """Execute transactionally without deduplication."""
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            state = self._read_state(con)
            result = self._apply_action(state, action.strip())
            if result.get("success"):
                self._write_state(con, state)
            con.commit()
            return result
        except BaseException:
            if con.in_transaction:
                con.rollback()
            raise
        finally:
            con.close()

    def execute_idempotent(
        self,
        *,
        idempotency_key: str,
        action: str,
        request_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(idempotency_key, str) or not idempotency_key.strip():
            raise ValueError("idempotency_key must be a non-empty string")
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non-empty string")
        if request_fingerprint is not None and (
            not isinstance(request_fingerprint, str) or not request_fingerprint.strip()
        ):
            raise ValueError("request_fingerprint must be a non-empty string when provided")

        key, name = idempotency_key.strip(), action.strip()
        fingerprint = request_fingerprint
        if fingerprint is None:
            fingerprint = hashlib.sha256(
                self._canonical_json({"action": name}).encode("utf-8")
            ).hexdigest()

        con = self._connect()
        try:
            con.execute("BEGIN IMMEDIATE")
            epoch = self._current_epoch(con)
            row = con.execute(
                """SELECT action_name, request_fingerprint, result_json
                   FROM room_idempotency
                   WHERE epoch = ? AND idempotency_key = ?""",
                (epoch, key),
            ).fetchone()
            if row is not None:
                if row["action_name"] != name or row["request_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        "Idempotency key is already bound to a different request in this Room epoch."
                    )
                result = json.loads(row["result_json"])
                con.commit()
                return {**result, "replayed": True, "epoch": epoch}

            state = self._read_state(con)
            result = self._apply_action(state, name)
            if result.get("success"):
                self._write_state(con, state)
            con.execute(
                """INSERT INTO room_idempotency
                   (epoch, idempotency_key, action_name, request_fingerprint, result_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (epoch, key, name, fingerprint, self._canonical_json(result)),
            )
            con.commit()
            return {**result, "replayed": False, "epoch": epoch}
        except BaseException:
            if con.in_transaction:
                con.rollback()
            raise
        finally:
            con.close()


__all__ = ["IdempotencyConflictError", "SQLiteRoomEnvironment"]
