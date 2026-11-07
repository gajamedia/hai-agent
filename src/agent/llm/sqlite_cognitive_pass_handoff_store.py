
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Mapping

from .cognitive_pass_handoff import (
    CognitivePassHandoff,
    CognitivePassHandoffValidationError,
    validate_cognitive_pass_handoff,
)


_INITIALIZATION_LOCKS_GUARD = Lock()
_INITIALIZATION_LOCKS: dict[str, RLock] = {}


def _initialization_lock_for(database_path: str) -> RLock:
    key = str(Path(database_path).expanduser().resolve())
    with _INITIALIZATION_LOCKS_GUARD:
        lock = _INITIALIZATION_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _INITIALIZATION_LOCKS[key] = lock
        return lock


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None) -> str:
    resolved = value or _utc_now()

    if not isinstance(resolved, datetime):
        raise TypeError("now must be a datetime")

    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    return resolved.astimezone(timezone.utc).isoformat()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Persisted timestamp must include a timezone")

    return parsed.astimezone(timezone.utc)


class CognitivePassHandoffStoreError(RuntimeError):
    """Base error for persistent handoff store operations."""


class CognitivePassHandoffNotFoundError(CognitivePassHandoffStoreError):
    pass


class CognitivePassHandoffAlreadyExistsError(CognitivePassHandoffStoreError):
    pass


class CognitivePassHandoffAlreadyConsumedError(CognitivePassHandoffStoreError):
    pass


class CognitivePassHandoffExpiredError(CognitivePassHandoffStoreError):
    pass


class CognitivePassHandoffStateError(CognitivePassHandoffStoreError):
    pass


@dataclass(frozen=True)
class StoredCognitivePassHandoff:
    handoff: CognitivePassHandoff
    status: str
    created_at: str
    consumed_at: str | None
    updated_at: str

    @property
    def handoff_id(self) -> str:
        return self.handoff.handoff_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff": self.handoff.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "consumed_at": self.consumed_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class StoredCognitivePassHandoffEvent:
    """Immutable audit event for a successful handoff lifecycle transition."""

    event_id: str
    handoff_id: str
    event_type: str
    from_status: str | None
    to_status: str
    occurred_at: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "handoff_id": self.handoff_id,
            "event_type": self.event_type,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "occurred_at": self.occurred_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CognitivePassHandoffAuditIssue:
    """One detected inconsistency in the persisted handoff audit history."""

    code: str
    handoff_id: str | None
    message: str
    event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "handoff_id": self.handoff_id,
            "message": self.message,
            "event_id": self.event_id,
        }


@dataclass(frozen=True)
class CognitivePassHandoffAuditIntegrityReport:
    """Read-only summary of lifecycle/audit consistency verification."""

    is_valid: bool
    checked_handoffs: int
    verified_handoffs: int
    untracked_handoffs: int
    issues: tuple[CognitivePassHandoffAuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "checked_handoffs": self.checked_handoffs,
            "verified_handoffs": self.verified_handoffs,
            "untracked_handoffs": self.untracked_handoffs,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class SQLiteCognitivePassHandoffStore:
    """Durable, atomic, single-use storage for declarative handoff plans.

    This store persists data only. It never invokes actions, registers
    capabilities, or authorizes execution.

    Lifecycle:
        prepared -> consumed
        prepared -> expired

    consumed and expired are terminal states.
    """

    STATUSES = ("prepared", "consumed", "expired")

    TABLE_NAME = "cognitive_pass_handoffs"

    LIFECYCLE_TRIGGER = "trg_cognitive_pass_handoff_lifecycle"

    EVENTS_TABLE_NAME = "cognitive_pass_handoff_events"
    EVENT_TYPES = ("created", "consumed", "expired")

    def __init__(
        self,
        database_path: str | Path,
        *,
        timeout: float = 10.0,
    ) -> None:
        if database_path is None:
            raise ValueError("database_path is required")

        raw_path = str(database_path).strip()

        if not raw_path or raw_path == ":memory:":
            raise ValueError(
                "database_path must be a non-empty filesystem path"
            )

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")

        path = Path(raw_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        self.database_path = str(path.resolve())
        self.timeout = float(timeout)
        self._lock = RLock()

        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=self.timeout,
            isolation_level=None,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            f"PRAGMA busy_timeout = {int(self.timeout * 1000)}"
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")

        return connection

    def _enable_wal_with_retry(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        deadline = time.monotonic() + self.timeout
        delay = 0.01

        while True:
            try:
                row = connection.execute(
                    "PRAGMA journal_mode = WAL"
                ).fetchone()

                if row is None or str(row[0]).lower() != "wal":
                    raise sqlite3.OperationalError(
                        "SQLite did not enable WAL mode"
                    )

                return

            except sqlite3.OperationalError as exc:
                message = str(exc).lower()

                if "locked" not in message and "busy" not in message:
                    raise

                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    raise sqlite3.OperationalError(
                        "Timed out while enabling SQLite WAL mode"
                    ) from exc

                time.sleep(min(delay, remaining))
                delay = min(delay * 2, 0.25)

    def _initialize_schema(self) -> None:
        lock = _initialization_lock_for(self.database_path)

        with lock, closing(self._connect()) as connection:
            self._enable_wal_with_retry(connection)

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cognitive_pass_handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL
                        CHECK(status IN ('prepared', 'consumed', 'expired')),
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT,
                    updated_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_cognitive_handoffs_status_expiry
                ON cognitive_pass_handoffs(
                    status,
                    expires_at,
                    created_at
                )
                """
            )

            # Database-level lifecycle enforcement.
            #
            # Same-status metadata updates remain allowed.
            # A status change is allowed only from prepared to a terminal
            # state. Terminal states can never transition again.
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS
                    trg_cognitive_pass_handoff_lifecycle
                BEFORE UPDATE OF status ON cognitive_pass_handoffs
                FOR EACH ROW
                WHEN
                    NEW.status != OLD.status
                    AND (
                        OLD.status != 'prepared'
                        OR NEW.status NOT IN ('consumed', 'expired')
                    )
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'illegal cognitive pass handoff lifecycle transition'
                    );
                END
                """
            )

            # Audit history is append-only. Existing handoff rows are deliberately
            # not backfilled: their historical transitions cannot be reconstructed.
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cognitive_pass_handoff_events (
                    event_id TEXT PRIMARY KEY,
                    handoff_id TEXT NOT NULL,
                    event_type TEXT NOT NULL
                        CHECK(event_type IN ('created', 'consumed', 'expired')),
                    from_status TEXT
                        CHECK(from_status IS NULL OR from_status IN ('prepared', 'consumed', 'expired')),
                    to_status TEXT NOT NULL
                        CHECK(to_status IN ('prepared', 'consumed', 'expired')),
                    occurred_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(handoff_id) REFERENCES cognitive_pass_handoffs(handoff_id)
                        ON DELETE RESTRICT
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cognitive_handoff_events_handoff_time
                ON cognitive_pass_handoff_events(handoff_id, occurred_at, event_id)
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_cognitive_pass_handoff_events_no_update
                BEFORE UPDATE ON cognitive_pass_handoff_events
                BEGIN
                    SELECT RAISE(ABORT, 'cognitive pass handoff events are append-only');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_cognitive_pass_handoff_events_no_delete
                BEFORE DELETE ON cognitive_pass_handoff_events
                BEGIN
                    SELECT RAISE(ABORT, 'cognitive pass handoff events are append-only');
                END
                """
            )

    @staticmethod
    def _serialize(handoff: CognitivePassHandoff) -> str:
        try:
            return json.dumps(
                handoff.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise CognitivePassHandoffStateError(
                "Handoff cannot be serialized safely"
            ) from exc

    @staticmethod
    def _decode_row(
        row: sqlite3.Row,
    ) -> StoredCognitivePassHandoff:
        try:
            payload = json.loads(row["payload_json"])

            if not isinstance(payload, dict):
                raise CognitivePassHandoffStateError(
                    "Persisted handoff payload must be a JSON object"
                )

            handoff = CognitivePassHandoff.from_dict(payload)

            status = row["status"]
            created_at = row["created_at"]
            expires_at = row["expires_at"]
            consumed_at = row["consumed_at"]
            updated_at = row["updated_at"]

            if (
                handoff.handoff_id != row["handoff_id"]
                or handoff.created_at != created_at
                or handoff.expires_at != expires_at
            ):
                raise CognitivePassHandoffStateError(
                    "Persisted handoff index fields do not match "
                    "canonical payload"
                )

            if status not in SQLiteCognitivePassHandoffStore.STATUSES:
                raise CognitivePassHandoffStateError(
                    "Persisted handoff status is invalid"
                )

            created = _parse_timestamp(created_at)
            expires = _parse_timestamp(expires_at)
            updated = _parse_timestamp(updated_at)

            if expires <= created:
                raise CognitivePassHandoffStateError(
                    "Persisted handoff expiration must be after creation"
                )

            if status == "prepared":
                if updated < created:
                    raise CognitivePassHandoffStateError(
                        "Prepared handoff updated_at precedes created_at"
                    )

                if consumed_at is not None:
                    raise CognitivePassHandoffStateError(
                        "Prepared handoff cannot have consumed_at"
                    )

                if updated >= expires:
                    raise CognitivePassHandoffStateError(
                        "Prepared handoff has an expired lifecycle timestamp"
                    )

            elif status == "consumed":
                if consumed_at is None:
                    raise CognitivePassHandoffStateError(
                        "Consumed handoff must have consumed_at"
                    )

                consumed = _parse_timestamp(consumed_at)

                if consumed < created or consumed >= expires:
                    raise CognitivePassHandoffStateError(
                        "Consumed handoff timestamp is outside "
                        "its validity window"
                    )

                if updated != consumed:
                    raise CognitivePassHandoffStateError(
                        "Consumed handoff updated_at must equal consumed_at"
                    )

            elif status == "expired":
                if consumed_at is not None:
                    raise CognitivePassHandoffStateError(
                        "Expired handoff cannot have consumed_at"
                    )

                if updated < expires:
                    raise CognitivePassHandoffStateError(
                        "Expired handoff timestamp precedes expires_at"
                    )

            return StoredCognitivePassHandoff(
                handoff=handoff,
                status=status,
                created_at=created_at,
                consumed_at=consumed_at,
                updated_at=updated_at,
            )

        except CognitivePassHandoffStateError:
            raise

        except Exception as exc:
            raise CognitivePassHandoffStateError(
                "Persisted handoff is malformed or fails integrity validation"
            ) from exc

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        *,
        handoff_id: str,
        event_type: str,
        from_status: str | None,
        to_status: str,
        occurred_at: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if event_type not in SQLiteCognitivePassHandoffStore.EVENT_TYPES:
            raise ValueError("Unsupported handoff audit event type")
        if to_status not in SQLiteCognitivePassHandoffStore.STATUSES:
            raise ValueError("Unsupported handoff audit target status")
        if from_status is not None and from_status not in SQLiteCognitivePassHandoffStore.STATUSES:
            raise ValueError("Unsupported handoff audit source status")
        safe_metadata = dict(metadata or {})
        if not isinstance(safe_metadata, dict):
            raise TypeError("Audit metadata must be a JSON object")
        try:
            metadata_json = json.dumps(
                safe_metadata, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise CognitivePassHandoffStateError(
                "Audit metadata cannot be serialized safely"
            ) from exc
        connection.execute(
            """
            INSERT INTO cognitive_pass_handoff_events (
                event_id, handoff_id, event_type, from_status, to_status,
                occurred_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), handoff_id, event_type, from_status,
             to_status, occurred_at, metadata_json),
        )

    @staticmethod
    def _decode_event_row(row: sqlite3.Row) -> StoredCognitivePassHandoffEvent:
        try:
            metadata = json.loads(row["metadata_json"])
            if not isinstance(metadata, dict):
                raise ValueError("Audit metadata must be a JSON object")
            if row["event_type"] not in SQLiteCognitivePassHandoffStore.EVENT_TYPES:
                raise ValueError("Invalid audit event type")
            if row["to_status"] not in SQLiteCognitivePassHandoffStore.STATUSES:
                raise ValueError("Invalid audit target status")
            if row["from_status"] is not None and row["from_status"] not in SQLiteCognitivePassHandoffStore.STATUSES:
                raise ValueError("Invalid audit source status")
            _parse_timestamp(row["occurred_at"])
            return StoredCognitivePassHandoffEvent(
                event_id=row["event_id"], handoff_id=row["handoff_id"],
                event_type=row["event_type"], from_status=row["from_status"],
                to_status=row["to_status"], occurred_at=row["occurred_at"],
                metadata=metadata,
            )
        except Exception as exc:
            raise CognitivePassHandoffStateError(
                "Persisted handoff audit event is malformed"
            ) from exc

    def create(
        self,
        handoff: CognitivePassHandoff | Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> StoredCognitivePassHandoff:
        validated = validate_cognitive_pass_handoff(
            handoff,
            now=now,
        )

        created_at = validated.created_at
        expires_at = validated.expires_at
        updated_at = _timestamp(now)
        payload_json = self._serialize(validated)

        with self._lock, closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")

                connection.execute(
                    """
                    INSERT INTO cognitive_pass_handoffs (
                        handoff_id,
                        status,
                        created_at,
                        expires_at,
                        consumed_at,
                        updated_at,
                        payload_json
                    )
                    VALUES (?, 'prepared', ?, ?, NULL, ?, ?)
                    """,
                    (
                        validated.handoff_id,
                        created_at,
                        expires_at,
                        updated_at,
                        payload_json,
                    ),
                )

                self._insert_event(
                    connection, handoff_id=validated.handoff_id,
                    event_type="created", from_status=None,
                    to_status="prepared", occurred_at=updated_at,
                )
                connection.commit()

            except sqlite3.IntegrityError as exc:
                connection.rollback()
                message = str(exc).lower()
                if "unique constraint" in message or "primary key" in message:
                    raise CognitivePassHandoffAlreadyExistsError(
                        f"Handoff already exists: {validated.handoff_id}"
                    ) from exc
                raise

            except BaseException:
                connection.rollback()
                raise

        return StoredCognitivePassHandoff(
            handoff=validated,
            status="prepared",
            created_at=created_at,
            consumed_at=None,
            updated_at=updated_at,
        )

    def get(
        self,
        handoff_id: str,
    ) -> StoredCognitivePassHandoff | None:
        if not isinstance(handoff_id, str) or not handoff_id.strip():
            raise ValueError("handoff_id must be a non-empty string")

        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM cognitive_pass_handoffs
                WHERE handoff_id = ?
                """,
                (handoff_id,),
            ).fetchone()

        return None if row is None else self._decode_row(row)

    def consume(
        self,
        handoff_id: str,
        *,
        expected_goal: str | None = None,
        expected_observation: Any = None,
        now: datetime | None = None,
    ) -> StoredCognitivePassHandoff:
        if not isinstance(handoff_id, str) or not handoff_id.strip():
            raise ValueError("handoff_id must be a non-empty string")

        now_text = _timestamp(now)
        current = _parse_timestamp(now_text)

        expired_after_commit = False
        result: StoredCognitivePassHandoff | None = None

        with self._lock, closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")

                row = connection.execute(
                    """
                    SELECT *
                    FROM cognitive_pass_handoffs
                    WHERE handoff_id = ?
                    """,
                    (handoff_id,),
                ).fetchone()

                if row is None:
                    raise CognitivePassHandoffNotFoundError(
                        f"Handoff not found: {handoff_id}"
                    )

                stored = self._decode_row(row)

                if stored.status == "consumed":
                    raise CognitivePassHandoffAlreadyConsumedError(
                        f"Handoff has already been consumed: {handoff_id}"
                    )

                if stored.status == "expired":
                    raise CognitivePassHandoffExpiredError(
                        f"Handoff has expired: {handoff_id}"
                    )

                expires = _parse_timestamp(stored.handoff.expires_at)
                created = _parse_timestamp(stored.handoff.created_at)

                if current < created:
                    raise CognitivePassHandoffValidationError(
                        "handoff is not yet valid."
                    )

                if current >= expires:
                    cursor = connection.execute(
                        """
                        UPDATE cognitive_pass_handoffs
                        SET status = 'expired',
                            updated_at = ?
                        WHERE handoff_id = ?
                          AND status = 'prepared'
                        """,
                        (now_text, handoff_id),
                    )

                    if cursor.rowcount != 1:
                        raise CognitivePassHandoffStateError(
                            "Handoff changed concurrently during expiration"
                        )

                    self._insert_event(
                        connection, handoff_id=handoff_id,
                        event_type="expired", from_status="prepared",
                        to_status="expired", occurred_at=now_text,
                    )
                    expired_after_commit = True

                else:
                    validate_cognitive_pass_handoff(
                        stored.handoff,
                        expected_goal=expected_goal,
                        expected_observation=expected_observation,
                        now=current,
                    )

                    cursor = connection.execute(
                        """
                        UPDATE cognitive_pass_handoffs
                        SET status = 'consumed',
                            consumed_at = ?,
                            updated_at = ?
                        WHERE handoff_id = ?
                          AND status = 'prepared'
                        """,
                        (now_text, now_text, handoff_id),
                    )

                    if cursor.rowcount != 1:
                        raise CognitivePassHandoffStateError(
                            "Handoff changed concurrently; reload before retrying"
                        )

                    self._insert_event(
                        connection, handoff_id=handoff_id,
                        event_type="consumed", from_status="prepared",
                        to_status="consumed", occurred_at=now_text,
                    )

                    result = StoredCognitivePassHandoff(
                        handoff=stored.handoff,
                        status="consumed",
                        created_at=stored.created_at,
                        consumed_at=now_text,
                        updated_at=now_text,
                    )

                connection.commit()

            except BaseException:
                connection.rollback()
                raise

        if expired_after_commit:
            raise CognitivePassHandoffExpiredError(
                f"Handoff has expired: {handoff_id}"
            )

        assert result is not None
        return result

    def expire_due(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        now_text = _timestamp(now)

        with self._lock, closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    """
                    SELECT * FROM cognitive_pass_handoffs
                    WHERE status = 'prepared' AND expires_at <= ?
                    ORDER BY expires_at, created_at, handoff_id
                    """,
                    (now_text,),
                ).fetchall()

                count = 0
                for row in rows:
                    stored = self._decode_row(row)
                    cursor = connection.execute(
                        """
                        UPDATE cognitive_pass_handoffs
                        SET status = 'expired', updated_at = ?
                        WHERE handoff_id = ? AND status = 'prepared'
                        """,
                        (now_text, stored.handoff_id),
                    )
                    if cursor.rowcount != 1:
                        raise CognitivePassHandoffStateError(
                            "Handoff changed concurrently during expiration"
                        )
                    self._insert_event(
                        connection, handoff_id=stored.handoff_id,
                        event_type="expired", from_status="prepared",
                        to_status="expired", occurred_at=now_text,
                    )
                    count += 1

                connection.commit()
                return count

            except BaseException:
                connection.rollback()
                raise

    def count(self, status: str | None = None) -> int:
        if status is not None and status not in self.STATUSES:
            raise ValueError(f"status must be one of {self.STATUSES}")

        with self._lock, closing(self._connect()) as connection:
            if status is None:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM cognitive_pass_handoffs
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM cognitive_pass_handoffs
                    WHERE status = ?
                    """,
                    (status,),
                ).fetchone()

        return int(row["total"])

    def list_all(
        self,
        status: str | None = None,
    ) -> tuple[StoredCognitivePassHandoff, ...]:
        if status is not None and status not in self.STATUSES:
            raise ValueError(f"status must be one of {self.STATUSES}")

        with self._lock, closing(self._connect()) as connection:
            if status is None:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM cognitive_pass_handoffs
                    ORDER BY created_at, handoff_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM cognitive_pass_handoffs
                    WHERE status = ?
                    ORDER BY created_at, handoff_id
                    """,
                    (status,),
                ).fetchall()

        return tuple(self._decode_row(row) for row in rows)


    def list_events(
        self,
        *,
        handoff_id: str | None = None,
        event_type: str | None = None,
    ) -> tuple[StoredCognitivePassHandoffEvent, ...]:
        if handoff_id is not None and (
            not isinstance(handoff_id, str) or not handoff_id.strip()
        ):
            raise ValueError("handoff_id must be a non-empty string")
        if event_type is not None and event_type not in self.EVENT_TYPES:
            raise ValueError(f"event_type must be one of {self.EVENT_TYPES}")

        clauses: list[str] = []
        parameters: list[Any] = []
        if handoff_id is not None:
            clauses.append("handoff_id = ?")
            parameters.append(handoff_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            parameters.append(event_type)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""

        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM cognitive_pass_handoff_events" + where
                + " ORDER BY occurred_at, event_id",
                tuple(parameters),
            ).fetchall()

        return tuple(self._decode_event_row(row) for row in rows)


    def verify_audit_integrity(
        self,
    ) -> CognitivePassHandoffAuditIntegrityReport:
        """Verify handoff lifecycle records against append-only audit events.

        The method is strictly read-only. Rows with no audit events are treated
        as legacy/untracked records, not as proof of corruption, because the
        audit table was introduced after handoff persistence already existed.
        """
        issues: list[CognitivePassHandoffAuditIssue] = []
        checked = 0
        verified = 0
        untracked = 0

        def add_issue(
            code: str,
            handoff_id: str | None,
            message: str,
            event_id: str | None = None,
        ) -> None:
            issues.append(CognitivePassHandoffAuditIssue(
                code=code,
                handoff_id=handoff_id,
                message=message,
                event_id=event_id,
            ))

        with self._lock, closing(self._connect()) as connection:
            # A single read transaction ensures the handoff and event snapshots
            # are mutually consistent while writers remain serialized by SQLite.
            connection.execute("BEGIN")
            try:
                handoff_rows = connection.execute(
                    "SELECT * FROM cognitive_pass_handoffs ORDER BY handoff_id"
                ).fetchall()
                event_rows = connection.execute(
                    "SELECT rowid AS audit_sequence, * "
                    "FROM cognitive_pass_handoff_events ORDER BY rowid"
                ).fetchall()

                events_by_handoff: dict[str, list[sqlite3.Row]] = {}
                for event_row in event_rows:
                    event_handoff_id = event_row["handoff_id"]
                    events_by_handoff.setdefault(event_handoff_id, []).append(event_row)

                known_ids = {row["handoff_id"] for row in handoff_rows}
                for event_row in event_rows:
                    if event_row["handoff_id"] not in known_ids:
                        add_issue(
                            "orphan_event",
                            event_row["handoff_id"],
                            "Audit event references a handoff record that does not exist.",
                            event_row["event_id"],
                        )

                for row in handoff_rows:
                    checked += 1
                    handoff_id = row["handoff_id"]
                    try:
                        stored = self._decode_row(row)
                    except CognitivePassHandoffStateError as exc:
                        add_issue(
                            "invalid_handoff_record", handoff_id,
                            f"Handoff record failed integrity validation: {exc}",
                        )
                        continue

                    raw_events = events_by_handoff.get(handoff_id, [])
                    if not raw_events:
                        untracked += 1
                        continue

                    decoded_events: list[StoredCognitivePassHandoffEvent] = []
                    malformed = False
                    for event_row in raw_events:
                        try:
                            decoded_events.append(self._decode_event_row(event_row))
                        except CognitivePassHandoffStateError as exc:
                            malformed = True
                            add_issue(
                                "malformed_event", handoff_id,
                                f"Audit event failed validation: {exc}",
                                event_row["event_id"],
                            )
                    if malformed:
                        continue

                    created_events = [e for e in decoded_events if e.event_type == "created"]
                    consumed_events = [e for e in decoded_events if e.event_type == "consumed"]
                    expired_events = [e for e in decoded_events if e.event_type == "expired"]

                    if len(created_events) > 1:
                        add_issue(
                            "duplicate_created_event", handoff_id,
                            "Handoff has more than one created audit event.",
                        )
                    if len(consumed_events) > 1 or len(expired_events) > 1:
                        add_issue(
                            "duplicate_terminal_event", handoff_id,
                            "Handoff has more than one terminal audit event.",
                        )

                    # Validate each event's semantic transition independently.
                    for event in decoded_events:
                        expected = {
                            "created": (None, "prepared"),
                            "consumed": ("prepared", "consumed"),
                            "expired": ("prepared", "expired"),
                        }.get(event.event_type)
                        if expected is None or (event.from_status, event.to_status) != expected:
                            add_issue(
                                "invalid_transition", handoff_id,
                                "Audit event contains an illegal lifecycle transition.",
                                event.event_id,
                            )

                    # The physical insertion sequence is authoritative for event
                    # order; timestamps must also be nondecreasing in that order.
                    previous_time: datetime | None = None
                    for event_row, event in zip(raw_events, decoded_events):
                        try:
                            occurred = _parse_timestamp(event.occurred_at)
                        except Exception:
                            # _decode_event_row normally catches this; retain a
                            # defensive branch in case its validation changes.
                            add_issue(
                                "invalid_event_timestamp", handoff_id,
                                "Audit event timestamp is invalid.", event.event_id,
                            )
                            continue
                        if previous_time is not None and occurred < previous_time:
                            add_issue(
                                "event_timestamp_regression", handoff_id,
                                "Audit event timestamp moves backwards in insertion order.",
                                event.event_id,
                            )
                        previous_time = occurred

                    if len(created_events) == 0:
                        add_issue(
                            "missing_created_event", handoff_id,
                            "Tracked handoff history has no created event.",
                        )
                    else:
                        created_event = created_events[0]
                        try:
                            event_created_at = _parse_timestamp(created_event.occurred_at)
                            handoff_created_at = _parse_timestamp(stored.created_at)
                            handoff_expires_at = _parse_timestamp(stored.handoff.expires_at)
                            if not (handoff_created_at <= event_created_at < handoff_expires_at):
                                add_issue(
                                    "event_timestamp_regression", handoff_id,
                                    "Created event timestamp is outside the handoff validity window.",
                                    created_event.event_id,
                                )
                        except Exception:
                            add_issue(
                                "invalid_event_timestamp", handoff_id,
                                "Created event timestamp is invalid.", created_event.event_id,
                            )

                    if stored.status == "prepared":
                        if consumed_events or expired_events:
                            add_issue(
                                "terminal_event_status_mismatch", handoff_id,
                                "Prepared handoff has a terminal audit event.",
                            )
                    elif stored.status == "consumed":
                        if len(consumed_events) == 0:
                            add_issue(
                                "missing_terminal_transition", handoff_id,
                                "Consumed handoff has no consumed audit event.",
                            )
                        if expired_events:
                            add_issue(
                                "terminal_event_status_mismatch", handoff_id,
                                "Consumed handoff also has an expired audit event.",
                            )
                        if consumed_events:
                            event = consumed_events[0]
                            if event.occurred_at != stored.consumed_at:
                                add_issue(
                                    "terminal_timestamp_mismatch", handoff_id,
                                    "Consumed event timestamp does not match consumed_at.",
                                    event.event_id,
                                )
                    elif stored.status == "expired":
                        if len(expired_events) == 0:
                            add_issue(
                                "missing_terminal_transition", handoff_id,
                                "Expired handoff has no expired audit event.",
                            )
                        if consumed_events:
                            add_issue(
                                "terminal_event_status_mismatch", handoff_id,
                                "Expired handoff also has a consumed audit event.",
                            )
                        if expired_events and expired_events[0].occurred_at != stored.updated_at:
                            add_issue(
                                "terminal_timestamp_mismatch", handoff_id,
                                "Expired event timestamp does not match updated_at.",
                                expired_events[0].event_id,
                            )

                    if len(created_events) == 1:
                        if not any(
                            issue.handoff_id == handoff_id
                            for issue in issues
                        ):
                            verified += 1

                connection.commit()
            except BaseException:
                connection.rollback()
                raise

        return CognitivePassHandoffAuditIntegrityReport(
            is_valid=not issues,
            checked_handoffs=checked,
            verified_handoffs=verified,
            untracked_handoffs=untracked,
            issues=tuple(issues),
        )


__all__ = [
    "SQLiteCognitivePassHandoffStore",
    "StoredCognitivePassHandoff",
    "StoredCognitivePassHandoffEvent",
    "CognitivePassHandoffAuditIssue",
    "CognitivePassHandoffAuditIntegrityReport",
    "CognitivePassHandoffStoreError",
    "CognitivePassHandoffNotFoundError",
    "CognitivePassHandoffAlreadyExistsError",
    "CognitivePassHandoffAlreadyConsumedError",
    "CognitivePassHandoffExpiredError",
    "CognitivePassHandoffStateError",
]
