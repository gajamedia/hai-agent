from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from uuid import uuid4

from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalStateError,
    ProposalCategory,
    ProposalStatus,
)
from agent.llm.capability_proposal_store import CapabilityProposalStore


# SQLite's journal_mode change is a database-level operation. Serialize schema
# initialization among store instances in this process, while retaining bounded
# retry for other processes or external SQLite clients.
_INITIALIZATION_LOCKS_GUARD = Lock()
_INITIALIZATION_LOCKS: dict[str, RLock] = {}


def _initialization_lock_for(database_path: str) -> RLock:
    key = database_path if database_path == ":memory:" else str(
        Path(database_path).expanduser().resolve()
    )
    with _INITIALIZATION_LOCKS_GUARD:
        lock = _INITIALIZATION_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _INITIALIZATION_LOCKS[key] = lock
        return lock


class SQLiteCapabilityProposalStore(CapabilityProposalStore):
    """Durable SQLite proposal store with concurrent initialization and writes.

    The database stores declarative proposal metadata only. It never stores
    handlers, callables, import paths, or executable source.
    """

    _ACTIVE_STATUSES = (
        ProposalStatus.PENDING.value,
        ProposalStatus.APPROVED.value,
        ProposalStatus.REGISTERED.value,
    )

    def __init__(
        self,
        database_path: str | Path,
        *,
        validator=None,
        timeout: float = 10.0,
    ) -> None:
        if database_path is None:
            raise ValueError(
                "database_path is required; use ':memory:' explicitly for tests."
            )
        raw_path = str(database_path).strip()
        if not raw_path:
            raise ValueError("database_path must be non-empty.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        self.database_path = raw_path
        self.timeout = float(timeout)
        self._lock = RLock()
        super().__init__(validator=validator)

        if raw_path != ":memory:":
            path = Path(raw_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.database_path = str(path.resolve())

        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        connection.execute("PRAGMA foreign_keys = ON")
        if self.database_path != ":memory:":
            # synchronous is connection-local. WAL mode itself is established
            # once during serialized schema initialization, not on every connect.
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _enable_wal_with_retry(self, connection: sqlite3.Connection) -> None:
        deadline = time.monotonic() + self.timeout
        delay = 0.01
        last_error: sqlite3.OperationalError | None = None

        while True:
            try:
                row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
                if row is None or str(row[0]).lower() != "wal":
                    raise sqlite3.OperationalError(
                        "SQLite did not enable WAL journal mode."
                    )
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise
                last_error = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise sqlite3.OperationalError(
                        "Timed out while enabling SQLite WAL journal mode."
                    ) from last_error
                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.25)

    def _initialize_schema(self) -> None:
        initialization_lock = _initialization_lock_for(self.database_path)
        with initialization_lock, closing(self._connect()) as connection:
            if self.database_path != ":memory:":
                self._enable_wal_with_retry(connection)

            # DDL and index creation are idempotent. A bounded retry around WAL
            # setup handles cross-process startup contention; SQLite's busy
            # timeout handles ordinary concurrent DDL/write locks.
            connection.execute(
                """CREATE TABLE IF NOT EXISTS capability_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    capability_id TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','registered')),
                    version INTEGER NOT NULL CHECK(version >= 1),
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_capability_proposals_status_created "
                "ON capability_proposals(status, created_at, proposal_id)"
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_capability_proposals_active_capability
                   ON capability_proposals(capability_id)
                   WHERE status IN ('pending','approved','registered')"""
            )
            connection.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS uq_capability_proposals_active_action
                   ON capability_proposals(action_name)
                   WHERE status IN ('pending','approved','registered')"""
            )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, dict) or hasattr(value, "items"):
            return {
                str(key): SQLiteCapabilityProposalStore._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                SQLiteCapabilityProposalStore._json_value(item)
                for item in value
            ]
        return value

    @classmethod
    def _serialize(cls, proposal: CapabilityProposal) -> str:
        payload = {
            "proposal_id": proposal.proposal_id,
            "capability_id": proposal.capability_id,
            "action_name": proposal.action_name,
            "description": proposal.description,
            "category": proposal.category.value,
            "risk_level": proposal.risk_level,
            "permissions": list(proposal.permissions),
            "preconditions": cls._json_value(proposal.preconditions),
            "effects": cls._json_value(proposal.effects),
            "parameters": cls._json_value(proposal.parameters),
            "status": proposal.status.value,
            "created_at": proposal.created_at,
            "approved_by": proposal.approved_by,
            "rejected_by": proposal.rejected_by,
            "rejection_reason": proposal.rejection_reason,
            "registered_by": proposal.registered_by,
            "version": proposal.version,
        }
        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise CapabilityProposalStateError(
                "Proposal contains data that cannot be safely persisted as JSON."
            ) from exc

    @staticmethod
    def _deserialize(payload_json: str) -> CapabilityProposal:
        try:
            payload = json.loads(payload_json)
            return CapabilityProposal(
                proposal_id=payload["proposal_id"],
                capability_id=payload["capability_id"],
                action_name=payload["action_name"],
                description=payload["description"],
                category=ProposalCategory(payload["category"]),
                risk_level=payload["risk_level"],
                permissions=tuple(payload.get("permissions", ())),
                preconditions=payload.get("preconditions", {}),
                effects=payload.get("effects", {}),
                parameters=payload.get("parameters", {}),
                status=ProposalStatus(payload["status"]),
                created_at=payload.get("created_at", ""),
                approved_by=payload.get("approved_by"),
                rejected_by=payload.get("rejected_by"),
                rejection_reason=payload.get("rejection_reason"),
                registered_by=payload.get("registered_by"),
                version=int(payload.get("version", 1)),
            )
        except Exception as exc:
            raise CapabilityProposalStateError(
                "A persisted capability proposal is malformed or unsupported."
            ) from exc

    def _decode_row(self, row: sqlite3.Row) -> CapabilityProposal:
        proposal = self._deserialize(row["payload_json"])
        if (
            proposal.proposal_id != row["proposal_id"]
            or proposal.capability_id != row["capability_id"]
            or proposal.action_name != row["action_name"]
            or proposal.status.value != row["status"]
            or proposal.version != int(row["version"])
        ):
            raise CapabilityProposalStateError(
                "Persisted proposal index fields do not match its canonical payload."
            )
        return proposal

    def submit(self, proposal: CapabilityProposal) -> CapabilityProposal:
        self.validator.validate(proposal)
        with self._lock, closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                stored = replace(
                    proposal,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    approved_by=None,
                    rejected_by=None,
                    rejection_reason=None,
                    registered_by=None,
                    status=ProposalStatus.PENDING,
                    version=1,
                )
                payload = self._serialize(stored)
                now = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """INSERT INTO capability_proposals
                       (proposal_id, capability_id, action_name, status, version,
                        created_at, payload_json, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        stored.proposal_id,
                        stored.capability_id,
                        stored.action_name,
                        stored.status.value,
                        stored.version,
                        stored.created_at,
                        payload,
                        now,
                    ),
                )
                connection.commit()
                return stored
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                message = str(exc).lower()
                if "proposal_id" in message:
                    detail = f"Proposal already exists: {proposal.proposal_id}"
                elif "capability" in message:
                    detail = (
                        "An unresolved proposal already exists for capability: "
                        f"{proposal.capability_id}"
                    )
                elif "action" in message:
                    detail = (
                        "An unresolved proposal already uses action name: "
                        f"{proposal.action_name}"
                    )
                else:
                    detail = "Proposal conflicts with an existing persistent record."
                raise CapabilityProposalStateError(detail) from exc
            except BaseException:
                connection.rollback()
                raise

    def create(
        self,
        *,
        capability_id: str,
        action_name: str,
        description: str,
        category,
        risk_level: str,
        permissions=(),
        preconditions=None,
        effects=None,
        parameters=None,
    ) -> CapabilityProposal:
        proposal = CapabilityProposal(
            proposal_id=uuid4().hex,
            capability_id=capability_id,
            action_name=action_name,
            description=description,
            category=category,
            risk_level=risk_level,
            permissions=tuple(permissions),
            preconditions=preconditions or {},
            effects=effects or {},
            parameters=parameters or {},
        )
        return self.submit(proposal)

    def get(self, proposal_id: str) -> CapabilityProposal:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM capability_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
            if row is None:
                raise CapabilityProposalStateError(
                    f"Proposal not found: {proposal_id}"
                )
            return self._decode_row(row)

    def list_all(self) -> tuple[CapabilityProposal, ...]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM capability_proposals ORDER BY created_at, proposal_id"
            ).fetchall()
            return tuple(self._decode_row(row) for row in rows)

    def transition(
        self,
        proposal_id: str,
        *,
        expected_status: ProposalStatus,
        new_status: ProposalStatus,
        actor: str,
        reason: str | None = None,
    ) -> CapabilityProposal:
        if not isinstance(actor, str) or not actor.strip():
            raise CapabilityProposalStateError("actor must be non-empty.")
        allowed = {
            (ProposalStatus.PENDING, ProposalStatus.APPROVED),
            (ProposalStatus.PENDING, ProposalStatus.REJECTED),
            (ProposalStatus.APPROVED, ProposalStatus.REGISTERED),
        }
        if (expected_status, new_status) not in allowed:
            raise CapabilityProposalStateError(
                f"Invalid proposal transition: {expected_status.value} -> {new_status.value}."
            )

        with self._lock, closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM capability_proposals WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone()
                if row is None:
                    raise CapabilityProposalStateError(
                        f"Proposal not found: {proposal_id}"
                    )
                current = self._decode_row(row)
                if current.status != expected_status:
                    raise CapabilityProposalStateError(
                        f"Expected {expected_status.value}, found {current.status.value}."
                    )

                updates = {
                    "status": new_status,
                    "version": current.version + 1,
                }
                if new_status == ProposalStatus.APPROVED:
                    updates["approved_by"] = actor.strip()
                elif new_status == ProposalStatus.REJECTED:
                    updates["rejected_by"] = actor.strip()
                    updates["rejection_reason"] = (reason or "").strip() or None
                elif new_status == ProposalStatus.REGISTERED:
                    updates["registered_by"] = actor.strip()

                updated = replace(current, **updates)
                payload = self._serialize(updated)
                cursor = connection.execute(
                    """UPDATE capability_proposals
                       SET status=?, version=?, payload_json=?, updated_at=?
                       WHERE proposal_id=? AND status=? AND version=?""",
                    (
                        updated.status.value,
                        updated.version,
                        payload,
                        datetime.now(timezone.utc).isoformat(),
                        proposal_id,
                        expected_status.value,
                        current.version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CapabilityProposalStateError(
                        "Proposal changed concurrently; reload before retrying."
                    )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise


__all__ = ["SQLiteCapabilityProposalStore"]
