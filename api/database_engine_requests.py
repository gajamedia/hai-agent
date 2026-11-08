from __future__ import annotations

"""Requests from users for database engines the platform does not support yet.

A request is only data: an engine name and a reason. Nothing here generates,
loads or runs code. A reviewer reads it and, if accepted, the engine is built
through the normal reviewed process (docs/ADDING_A_DATABASE_ENGINE.md).
"""

import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ENGINE = re.compile(r"^[A-Za-z][A-Za-z0-9 ._+\-]{1,39}$")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MAX_REASON = 500
MAX_NOTE = 500
MAX_OPEN_PER_USER = 5
STATUSES = ("open", "accepted", "rejected", "delivered")
_TRANSITIONS = {"open": {"accepted", "rejected"}, "accepted": {"delivered", "rejected"},
                "rejected": set(), "delivered": set()}


class EngineRequestError(ValueError):
    """Invalid input (HTTP 422)."""


class EngineRequestConflict(EngineRequestError):
    """Valid input that conflicts with current state (HTTP 409)."""


class EngineRequestNotFound(EngineRequestError):
    """HTTP 404."""


def _clean(value: Any, label: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise EngineRequestError(f"{label} must be text")
    text = value.strip()
    if required and not text:
        raise EngineRequestError(f"{label} is required")
    if len(text) > limit or _CONTROL.search(text):
        raise EngineRequestError(f"{label} is too long or contains control characters")
    return text


class EngineRequestStore:
    def __init__(self, path: str | Path, *, supported: frozenset[str] = frozenset()) -> None:
        self.path = Path(path)
        self.supported = {s.casefold() for s in supported}
        self._lock = threading.Lock()
        db = self._db()
        try:
            with db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS engine_requests (id TEXT PRIMARY KEY, "
                    "engine TEXT NOT NULL, reason TEXT NOT NULL, requested_by TEXT NOT NULL, "
                    "requested_at TEXT NOT NULL, status TEXT NOT NULL, reviewed_by TEXT, "
                    "reviewed_at TEXT, note TEXT)")
        finally:
            db.close()

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def create(self, engine: Any, reason: Any, *, requested_by: str) -> dict[str, Any]:
        name = _clean(engine, "engine", 40)
        if not _ENGINE.match(name):
            raise EngineRequestError("engine must be a short product name, e.g. 'Oracle 19c'")
        why = _clean(reason, "reason", MAX_REASON)
        if name.casefold() in self.supported:
            raise EngineRequestConflict(f"{name} is already supported")
        with self._lock:
            db = self._db()
            try:
                (open_count,) = db.execute(
                    "SELECT COUNT(*) FROM engine_requests WHERE requested_by = ? AND status = 'open'",
                    (requested_by,)).fetchone()
                if open_count >= MAX_OPEN_PER_USER:
                    raise EngineRequestConflict(
                        f"you already have {MAX_OPEN_PER_USER} open requests")
                record = {"id": uuid.uuid4().hex, "engine": name, "reason": why,
                          "requested_by": requested_by,
                          "requested_at": datetime.now(timezone.utc).isoformat(), "status": "open",
                          "reviewed_by": None, "reviewed_at": None, "note": None}
                with db:
                    db.execute("INSERT INTO engine_requests VALUES (:id, :engine, :reason, "
                               ":requested_by, :requested_at, :status, :reviewed_by, "
                               ":reviewed_at, :note)", record)
                return record
            finally:
                db.close()

    def list(self, *, requested_by: str | None = None) -> list[dict[str, Any]]:
        db = self._db()
        try:
            sql, params = "SELECT * FROM engine_requests", ()
            if requested_by is not None:
                sql, params = sql + " WHERE requested_by = ?", (requested_by,)
            return [dict(r) for r in db.execute(sql + " ORDER BY requested_at DESC LIMIT 200", params)]
        finally:
            db.close()

    def review(self, request_id: Any, status: Any, *, reviewed_by: str,
               note: Any = "") -> dict[str, Any]:
        if status not in STATUSES or status == "open":
            raise EngineRequestError(f"status must be one of {list(STATUSES[1:])}")
        text = _clean(note or "", "note", MAX_NOTE, required=False)
        with self._lock:
            db = self._db()
            try:
                row = db.execute("SELECT * FROM engine_requests WHERE id = ?",
                                 (str(request_id),)).fetchone()
                if row is None:
                    raise EngineRequestNotFound("request not found")
                if status not in _TRANSITIONS[row["status"]]:
                    raise EngineRequestConflict(
                        f"cannot change a {row['status']} request to {status}")
                with db:
                    db.execute("UPDATE engine_requests SET status = ?, reviewed_by = ?, "
                               "reviewed_at = ?, note = ? WHERE id = ?",
                               (status, reviewed_by, datetime.now(timezone.utc).isoformat(),
                                text or None, row["id"]))
                return dict(db.execute("SELECT * FROM engine_requests WHERE id = ?",
                                       (row["id"],)).fetchone())
            finally:
                db.close()


__all__ = ["EngineRequestConflict", "EngineRequestError", "EngineRequestNotFound",
           "EngineRequestStore", "STATUSES"]
