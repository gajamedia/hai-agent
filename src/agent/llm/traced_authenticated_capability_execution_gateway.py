from __future__ import annotations

"""Audited, fail-closed trace boundary for authenticated capability reads.

This wrapper reuses the existing ExecutionTraceStore and authenticated gateway.
It never creates a second trace format and never stores request arguments,
credentials, returned business rows, adapter output, or raw exception messages.
"""

from typing import Any

from .capability_execution_contracts import CapabilityAuthorizationError
from .execution_trace_store import ExecutionTraceStore


class CapabilityExecutionTraceError(RuntimeError):
    """Base error for traced capability reads."""


class CapabilityTracePersistenceError(CapabilityExecutionTraceError):
    """Raised when a required lifecycle event cannot be persisted."""


class TracedAuthenticatedCapabilityExecutionGateway:
    """Persist lifecycle evidence around an authenticated read gateway.

    The ``started`` event is written before the gateway is called. Therefore a
    trace-store failure at start prevents the read entirely. Authorization
    outcomes are recorded as ``authorized`` or ``denied`` after the gateway
    returns/raises; ``completed`` follows ``authorized``. Other gateway errors
    produce ``failed``.

    The capability gateway is read-only. If a terminal trace append fails after
    the underlying read, this wrapper withholds the result and raises a trace
    persistence error. The read itself cannot be rolled back.

    ``last_error`` intentionally stores only the exception class name. Raw
    exception messages can contain paths, hostnames, SQL fragments, or other
    sensitive operational details and must not be retained as diagnostics.
    """

    def __init__(self, gateway: Any, store: ExecutionTraceStore) -> None:
        if gateway is None or not callable(getattr(gateway, "read", None)):
            raise TypeError("gateway must expose callable read(principal, request).")
        if store is None or not callable(getattr(store, "append", None)):
            raise TypeError("store must expose callable append(event).")
        self._gateway = gateway
        self._store = store
        self.persisted_count = 0
        self.error_count = 0
        self.last_error: str | None = None

    @staticmethod
    def _required_text(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _request_identity(principal: Any, request: Any) -> dict[str, str]:
        actor_id = TracedAuthenticatedCapabilityExecutionGateway._required_text(
            getattr(principal, "actor_id", None), "principal.actor_id"
        )
        target = TracedAuthenticatedCapabilityExecutionGateway._required_text(
            getattr(request, "target", None), "request.target"
        )
        capability_id = TracedAuthenticatedCapabilityExecutionGateway._required_text(
            getattr(request, "capability_id", None), "request.capability_id"
        )
        return {
            "actor_id": actor_id,
            "target": target,
            "capability_id": capability_id,
        }

    def _append(
        self,
        *,
        execution_id: str,
        identity: dict[str, str],
        event_type: str,
        outcome: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            **identity,
            "component": "authenticated_capability_execution_gateway",
        }
        if outcome is not None:
            payload["outcome"] = outcome
        event_id = f"capability-read::{execution_id}::{event_type}"
        try:
            self._store.append(
                {
                    "event_id": event_id,
                    "execution_id": execution_id,
                    "event_type": event_type,
                    "payload": payload,
                }
            )
            self.persisted_count += 1
            self.last_error = None
        except Exception as exc:
            self.error_count += 1
            # Keep diagnostics useful without retaining exception messages.
            self.last_error = type(exc).__name__
            raise CapabilityTracePersistenceError(
                f"Required capability trace event '{event_type}' could not be persisted."
            ) from None

    def read(self, principal: Any, request: Any, *, execution_id: str) -> Any:
        """Execute one authenticated read and persist its lifecycle evidence."""
        execution_id = self._required_text(execution_id, "execution_id")
        identity = self._request_identity(principal, request)

        # Fail closed before accessing the gateway.
        self._append(
            execution_id=execution_id,
            identity=identity,
            event_type="started",
        )

        try:
            result = self._gateway.read(principal, request)
        except CapabilityAuthorizationError:
            self._append(
                execution_id=execution_id,
                identity=identity,
                event_type="denied",
                outcome="authorization_denied",
            )
            raise
        except Exception as exc:
            self._append(
                execution_id=execution_id,
                identity=identity,
                event_type="failed",
                outcome=type(exc).__name__,
            )
            raise

        # The authenticated gateway returned only after its permission,
        # registry, and execution-security checks succeeded.
        self._append(
            execution_id=execution_id,
            identity=identity,
            event_type="authorized",
            outcome="allowed",
        )
        self._append(
            execution_id=execution_id,
            identity=identity,
            event_type="completed",
            outcome="read_completed",
        )
        return result


__all__ = [
    "CapabilityExecutionTraceError",
    "CapabilityTracePersistenceError",
    "TracedAuthenticatedCapabilityExecutionGateway",
]
