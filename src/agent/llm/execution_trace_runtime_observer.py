from __future__ import annotations

"""Bridge existing LLM plan runtime events into an ExecutionTraceStore.

The adapter is intentionally an observer only: it never invokes an action,
planner, executor, or replay callback. Attach it with
``runtime.add_event_observer(observer)`` and pass a stable ``execution_id`` in
``runtime.execute(..., metadata={"execution_id": ...})``.
"""

from collections.abc import Mapping
from typing import Any, Callable

from .execution_trace_store import ExecutionTraceStore


TraceObserverErrorHandler = Callable[[Exception], None]


class ExecutionTraceRuntimeObserver:
    """Persist runtime events with stable IDs and non-invasive error handling.

    By default, trace-store failures are recorded in ``last_error`` and
    ``error_count`` rather than raised into the execution loop. This avoids a
    persistence outage turning an already-attempted action into a runtime
    failure. Applications must inspect these fields or provide ``on_error``
    for operational alerting. Set ``raise_on_error=True`` only when the caller
    explicitly wants fail-closed observer semantics.

    ``execution_id`` must be present in each event's metadata. A missing ID is
    treated as an observable adapter error and that event is not persisted.
    Event IDs are deterministic per execution/phase/step, making identical
    observer retries idempotent in ``ExecutionTraceStore``.
    """

    def __init__(
        self,
        store: ExecutionTraceStore,
        *,
        execution_id_key: str = "execution_id",
        include_world_state: bool = False,
        raise_on_error: bool = False,
        on_error: TraceObserverErrorHandler | None = None,
    ) -> None:
        if store is None or not callable(getattr(store, "append", None)):
            raise TypeError("store must expose callable append(...).")
        if not isinstance(execution_id_key, str) or not execution_id_key.strip():
            raise ValueError("execution_id_key must be a non-empty string.")
        if on_error is not None and not callable(on_error):
            raise TypeError("on_error must be callable when provided.")
        self.store = store
        self.execution_id_key = execution_id_key.strip()
        self.include_world_state = bool(include_world_state)
        self.raise_on_error = bool(raise_on_error)
        self.on_error = on_error
        self.last_error: str | None = None
        self.error_count = 0
        self.persisted_count = 0
        self.last_persisted_sequence: int | None = None

    @staticmethod
    def _safe_value(value: Any, *, depth: int = 0) -> Any:
        """Convert values to bounded JSON-safe summaries without raw reprs."""
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            # JSON forbids NaN and infinities in the trace contract.
            if value != value or value in (float("inf"), float("-inf")):
                return {"type": "float", "value": "non_finite"}
            return value
        if depth >= 4:
            return {"type": type(value).__name__, "truncated": True}
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in list(value.items())[:100]:
                result[str(key)] = ExecutionTraceRuntimeObserver._safe_value(
                    item, depth=depth + 1
                )
            if len(value) > 100:
                result["_truncated_items"] = len(value) - 100
            return result
        if isinstance(value, (list, tuple)):
            values = [
                ExecutionTraceRuntimeObserver._safe_value(item, depth=depth + 1)
                for item in value[:100]
            ]
            if len(value) > 100:
                values.append({"_truncated_items": len(value) - 100})
            return values
        # Do not use repr(value): it may contain secrets or unstable memory IDs.
        return {"type": type(value).__name__}

    @staticmethod
    def _step_index(event: Any) -> int | str:
        step = getattr(event, "step", None)
        index = getattr(step, "index", None) if step is not None else None
        return index if isinstance(index, int) and not isinstance(index, bool) else "plan"

    @staticmethod
    def _goal_text(event: Any) -> str | None:
        plan = getattr(event, "plan", None)
        goal = getattr(plan, "goal", None)
        if isinstance(goal, str):
            return goal[:500]
        if goal is None:
            # Closed-loop coordinator events may expose goal_id rather than a
            # plan.goal string. Preserve the stable identifier, not arbitrary repr.
            goal_id = getattr(event, "goal_id", None)
            return goal_id[:250] if isinstance(goal_id, str) else None
        return type(goal).__name__

    @staticmethod
    def _generation(event: Any, metadata: Mapping[str, Any]) -> int | None:
        """Resolve a positive segment/generation number when the event has one.

        LLMPlanRuntimeEvent does not define generation directly, but the closed-loop
        coordinator forwards it in runtime metadata. Keep generation optional so
        legacy event IDs remain unchanged when no generation is available.
        """
        value = getattr(event, "generation", None)
        if value is None:
            value = metadata.get("generation")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
            return value
        return None

    @staticmethod
    def _runtime_result_summary(event: Any) -> dict[str, Any] | None:
        """Extract a small, non-executable summary from coordinator events."""
        result = getattr(event, "runtime_result", None)
        if result is None:
            result = getattr(event, "result", None)
        if result is None:
            return None
        summary: dict[str, Any] = {"type": type(result).__name__}
        for name in ("success", "completed", "stopped", "stop_reason", "stop_reason_code"):
            value = getattr(result, name, None)
            if isinstance(value, (str, bool, int, float)) or value is None:
                if value is not None:
                    summary[name] = ExecutionTraceRuntimeObserver._safe_value(value)
        records = getattr(result, "records", None)
        if isinstance(records, (tuple, list)):
            summary["record_count"] = len(records)
        return summary

    def _build_event(self, event: Any) -> dict[str, Any]:
        phase = getattr(event, "phase", None)
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError("runtime event must have a non-empty phase.")
        metadata = getattr(event, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("runtime event metadata must be a mapping.")
        execution_id = metadata.get(self.execution_id_key)
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError(
                f"runtime event metadata must contain non-empty {self.execution_id_key!r}."
            )
        execution_id = execution_id.strip()
        step = getattr(event, "step", None)
        step_index = self._step_index(event)
        generation = self._generation(event, metadata)
        record = getattr(event, "record", None)
        step_payload: dict[str, Any] | None = None
        if step is not None:
            action = getattr(step, "action", None)
            step_payload = {
                "index": step_index,
                "action": action[:250] if isinstance(action, str) else type(action).__name__,
            }
        if record is not None:
            if step_payload is None:
                step_payload = {"index": step_index}
            step_payload.update(
                {
                    "success": bool(getattr(record, "success", False)),
                    "skipped": bool(getattr(record, "skipped", False)),
                    "stop_requested": bool(getattr(record, "stop_requested", False)),
                    "error": (
                        str(getattr(record, "error"))[:1000]
                        if getattr(record, "error", None) is not None
                        else None
                    ),
                }
            )
        goal_id = getattr(event, "goal_id", None)
        # Keep coordinator goal identity in the persisted payload even when a
        # plan also exposes a human-readable goal string. This makes a reused
        # deterministic event_id with a different goal identity a detectable
        # integrity conflict rather than an apparently identical retry.
        if not isinstance(goal_id, str) or not goal_id.strip():
            goal_id = None
        else:
            goal_id = goal_id.strip()[:250]

        payload: dict[str, Any] = {
            "phase": phase,
            "goal": self._goal_text(event),
            "goal_id": goal_id,
            "step": step_payload,
            "stop_reason": getattr(event, "stop_reason", None),
            "next_step_index": getattr(event, "next_step_index", None),
            "metadata": self._safe_value(dict(metadata)),
        }
        if generation is not None:
            payload["generation"] = generation
        runtime_result = self._runtime_result_summary(event)
        if runtime_result is not None:
            payload["runtime_result"] = runtime_result
        if self.include_world_state:
            payload["world_state"] = self._safe_value(getattr(event, "world_state", None))
        # Generation-scoped IDs prevent collisions when one closed-loop execution
        # emits the same phase for multiple recovery/replanning segments. Preserve
        # the Step 37 ID contract for events with no generation.
        if generation is None:
            event_id = f"{execution_id}::{phase}::{step_index}"
        else:
            event_id = f"{execution_id}::{phase}::g{generation}::{step_index}"
        return {
            "event_id": event_id,
            "execution_id": execution_id,
            "event_type": phase,
            "payload": payload,
        }

    def __call__(self, event: Any) -> None:
        """Persist one runtime event; does not perform or replay any action."""
        try:
            persisted = self.store.append(self._build_event(event))
            self.persisted_count += 1
            self.last_error = None
            # Keep a useful reference for integrations/diagnostics without
            # retaining the entire runtime event or mutable caller objects.
            self.last_persisted_sequence = persisted.get("sequence")
        except Exception as exc:
            self.error_count += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.on_error is not None:
                try:
                    self.on_error(exc)
                except Exception:
                    # Error reporting must not mask the original trace failure.
                    pass
            if self.raise_on_error:
                raise


__all__ = ["ExecutionTraceRuntimeObserver", "TraceObserverErrorHandler"]
