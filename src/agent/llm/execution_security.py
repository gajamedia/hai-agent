from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import sqlite3
from threading import Lock
from typing import Any, Callable, Mapping, Protocol


class ExecutionSecurityError(RuntimeError):
    """Raised when an action is blocked by execution security policy."""


class ExecutionSecurityAuditError(RuntimeError):
    """Raised when a required security audit event cannot be persisted."""


@dataclass(frozen=True)
class ExecutionSecurityDecision:
    action: str
    position: int
    risk_score: float
    approval_required: bool
    allowed: bool
    reason: str


@dataclass(frozen=True)
class ExecutionSecurityAuditEvent:
    timestamp: str
    action: str
    position: int
    risk_score: float
    approval_required: bool
    allowed: bool
    reason: str
    event_type: str = "authorization"
    execution_success: bool | None = None


class ExecutionSecurityAuditSink(Protocol):
    def record(self, event: ExecutionSecurityAuditEvent) -> None:
        """Persist one security decision or raise on failure."""
        ...


class SQLiteExecutionSecurityAuditSink:
    """Append-only SQLite sink for security decisions.

    A new connection is opened per event so separate threads/processes do not
    share a connection. The sink stores only security metadata, not request
    bodies, world state, credentials, or LLM-generated free text.
    """

    def __init__(self, path: str | Path) -> None:
        if str(path).strip() == ":memory:":
            raise ValueError("SQLite audit path must be a persistent file path.")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS execution_security_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        action TEXT NOT NULL,
                        position INTEGER NOT NULL,
                        risk_score REAL NOT NULL,
                        approval_required INTEGER NOT NULL,
                        allowed INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        event_type TEXT NOT NULL DEFAULT 'authorization',
                        execution_success INTEGER NULL
                    )
                    """
                )
                columns = {
                    row[1]
                    for row in connection.execute(
                        "PRAGMA table_info(execution_security_audit)"
                    ).fetchall()
                }
                if "event_type" not in columns:
                    connection.execute(
                        "ALTER TABLE execution_security_audit "
                        "ADD COLUMN event_type TEXT NOT NULL DEFAULT 'authorization'"
                    )
                if "execution_success" not in columns:
                    connection.execute(
                        "ALTER TABLE execution_security_audit "
                        "ADD COLUMN execution_success INTEGER NULL"
                    )
        except sqlite3.Error as exc:
            raise ExecutionSecurityAuditError(
                f"Could not initialize security audit database: {exc}"
            ) from exc

    def record(self, event: ExecutionSecurityAuditEvent) -> None:
        try:
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                connection.execute(
                    """
                    INSERT INTO execution_security_audit
                    (timestamp, action, position, risk_score,
                     approval_required, allowed, reason, event_type,
                     execution_success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.timestamp,
                        event.action,
                        event.position,
                        event.risk_score,
                        int(event.approval_required),
                        int(event.allowed),
                        event.reason,
                        event.event_type,
                        None if event.execution_success is None else int(event.execution_success),
                    ),
                )
        except sqlite3.Error as exc:
            raise ExecutionSecurityAuditError(
                f"Could not persist security audit event: {exc}"
            ) from exc

    def read_all(self) -> tuple[ExecutionSecurityAuditEvent, ...]:
        try:
            with sqlite3.connect(str(self.path), timeout=10) as connection:
                rows = connection.execute(
                    """SELECT timestamp, action, position, risk_score,
                              approval_required, allowed, reason, event_type,
                              execution_success
                       FROM execution_security_audit ORDER BY id"""
                ).fetchall()
        except sqlite3.Error as exc:
            raise ExecutionSecurityAuditError(
                f"Could not read security audit events: {exc}"
            ) from exc
        return tuple(
            ExecutionSecurityAuditEvent(
                timestamp=row[0],
                action=row[1],
                position=int(row[2]),
                risk_score=float(row[3]),
                approval_required=bool(row[4]),
                allowed=bool(row[5]),
                reason=row[6],
                event_type=row[7],
                execution_success=None if row[8] is None else bool(row[8]),
            )
            for row in rows
        )


class ExecutionSecurityPolicy:
    """Enforce trusted risk configuration and trusted approval checks.

    Unknown actions are assigned maximum risk and therefore require approval.
    Request metadata and plan-provided risk values are never authorization.
    When ``audit_sink`` is configured, an event must be persisted successfully
    before authorization can be returned to the caller (fail-closed audit).
    """

    def __init__(
        self,
        *,
        action_risks: Mapping[str, float] | None = None,
        risk_threshold: float = 0.65,
        approval_required_actions: frozenset[str] | set[str] | tuple[str, ...] = (),
        approval_checker: Callable[[Any], bool] | None = None,
        audit_sink: ExecutionSecurityAuditSink | Callable[[ExecutionSecurityAuditEvent], None] | None = None,
    ) -> None:
        self._action_risks = self._validate_action_risks(action_risks or {})
        self.risk_threshold = self._bounded_score(risk_threshold, "risk_threshold")
        self.approval_required_actions = frozenset(
            str(action).strip()
            for action in approval_required_actions
            if str(action).strip()
        )
        if approval_checker is not None and not callable(approval_checker):
            raise TypeError("approval_checker must be callable when provided.")
        if audit_sink is not None and not (
            callable(getattr(audit_sink, "record", None)) or callable(audit_sink)
        ):
            raise TypeError("audit_sink must provide record(event) or be callable.")
        self.approval_checker = approval_checker
        self.audit_sink = audit_sink
        self._audit_log: list[ExecutionSecurityAuditEvent] = []
        self._audit_lock = Lock()

    @staticmethod
    def _bounded_score(value: Any, field_name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be a number between 0 and 1.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a number between 0 and 1.") from exc
        if not isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{field_name} must be a number between 0 and 1.")
        return number

    @classmethod
    def _validate_action_risks(cls, action_risks: Mapping[str, float]) -> dict[str, float]:
        validated: dict[str, float] = {}
        for action, risk in action_risks.items():
            name = str(action).strip()
            if not name:
                raise ValueError("Action risk registry cannot contain an empty action.")
            validated[name] = cls._bounded_score(risk, f"risk for action '{name}'")
        return validated

    @property
    def audit_log(self) -> tuple[ExecutionSecurityAuditEvent, ...]:
        with self._audit_lock:
            return tuple(self._audit_log)

    def _record(self, event: ExecutionSecurityAuditEvent) -> None:
        # Keep an in-process diagnostic copy even if durable storage fails.
        with self._audit_lock:
            self._audit_log.append(event)
        if self.audit_sink is None:
            return
        try:
            recorder = getattr(self.audit_sink, "record", None)
            if callable(recorder):
                recorder(event)
            else:
                self.audit_sink(event)  # type: ignore[misc]
        except Exception as exc:
            raise ExecutionSecurityAuditError(
                "Security audit persistence failed; execution is blocked."
            ) from exc

    def evaluate(self, request: Any) -> ExecutionSecurityDecision:
        action = str(getattr(request, "action", "")).strip()
        try:
            position = int(getattr(request, "position", 0))
        except (TypeError, ValueError, OverflowError):
            position = 0
        risk_score = self._action_risks.get(action, 1.0)
        approval_required = (
            action in self.approval_required_actions or risk_score >= self.risk_threshold
        )

        if not action or position < 1:
            allowed, reason = False, "invalid_execution_request"
        elif not approval_required:
            allowed, reason = True, "allowed_low_risk"
        elif self.approval_checker is None:
            allowed, reason = False, "approval_required_but_no_approval_checker"
        else:
            try:
                approved = self.approval_checker(request)
            except Exception:
                approved = False
                reason = "approval_check_failed"
            else:
                reason = "approved" if approved is True else "approval_missing_or_invalid"
            allowed = approved is True

        decision = ExecutionSecurityDecision(
            action=action,
            position=position,
            risk_score=risk_score,
            approval_required=approval_required,
            allowed=allowed,
            reason=reason,
        )
        self._record(
            ExecutionSecurityAuditEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=decision.action,
                position=decision.position,
                risk_score=decision.risk_score,
                approval_required=decision.approval_required,
                allowed=decision.allowed,
                reason=decision.reason,
            )
        )
        return decision

    def record_execution_outcome(
        self,
        request: Any,
        *,
        success: bool | None,
        reason: str | None = None,
    ) -> None:
        """Persist a known or unknown outcome of an attempted action.

        None means the outcome is unknown, not that execution failed.
        If durable outcome logging fails, callers must not blindly retry.
        """
        action = str(getattr(request, "action", "")).strip()
        try:
            position = int(getattr(request, "position", 0))
        except (TypeError, ValueError, OverflowError):
            position = 0
        risk_score = self._action_risks.get(action, 1.0)
        approval_required = (
            action in self.approval_required_actions or risk_score >= self.risk_threshold
        )
        event = ExecutionSecurityAuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            position=position,
            risk_score=risk_score,
            approval_required=approval_required,
            allowed=True,
            reason=reason or (
                "execution_succeeded"
                if success is True
                else "execution_failed"
                if success is False
                else "execution_outcome_unknown"
            ),
            event_type="execution_outcome",
            execution_success=success,
        )
        try:
            self._record(event)
        except ExecutionSecurityAuditError as exc:
            raise ExecutionSecurityAuditError(
                "Execution outcome could not be durably audited; the action may "
                "already have produced side effects. Do not retry blindly."
            ) from exc

    def authorize(self, request: Any) -> ExecutionSecurityDecision:
        decision = self.evaluate(request)
        if not decision.allowed:
            raise ExecutionSecurityError(
                f"Execution blocked for action '{decision.action}' at position "
                f"{decision.position}: {decision.reason}."
            )
        return decision


__all__ = [
    "ExecutionSecurityAuditError",
    "ExecutionSecurityAuditEvent",
    "ExecutionSecurityAuditSink",
    "ExecutionSecurityDecision",
    "ExecutionSecurityError",
    "ExecutionSecurityPolicy",
    "SQLiteExecutionSecurityAuditSink",
]
