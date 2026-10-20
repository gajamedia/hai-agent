from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .llm_plan_runtime_execution import (
    LLMPlanRuntimeEvent,
    LLMPlanRuntimeExecutionError,
    LLMPlanRuntimeExecutionReport,
    LLMPlanRuntimeExecutor,
)
from .llm_plan_runtime_state import (
    LLMPlanRuntimeState,
    LLMPlanRuntimeStateTracker,
)
from .planning import LLMPlan


class LLMPlanLiveRuntimeError(RuntimeError):
    """Base error for live runtime-state integration."""


class LLMPlanLiveRuntimeConfigurationError(LLMPlanLiveRuntimeError):
    """Raised when live runtime integration is configured incorrectly."""


class LLMPlanLiveRuntimeStateError(LLMPlanLiveRuntimeError):
    """Raised when runtime state cannot follow execution events."""


StateObserver = Callable[[LLMPlanRuntimeState], None]


@dataclass(frozen=True)
class LLMPlanLiveRuntimeExecutionResult:
    """Execution report together with live state snapshots."""

    report: LLMPlanRuntimeExecutionReport
    final_state: LLMPlanRuntimeState
    state_history: tuple[LLMPlanRuntimeState, ...] = field(default_factory=tuple)

    @property
    def success(self) -> bool:
        return self.report.success

    @property
    def completed(self) -> bool:
        return self.report.completed

    @property
    def stopped(self) -> bool:
        return self.report.stopped

    @property
    def next_step_index(self) -> int | None:
        return self.final_state.next_step_index

    @property
    def progress(self) -> float:
        return self.final_state.progress


class LLMPlanLiveRuntimeExecutor:
    """
    Integrate LLMPlanRuntimeExecutor with LLMPlanRuntimeStateTracker live.

    The existing runtime executor remains the sole execution engine. This
    adapter subscribes to its events and updates a state tracker immediately
    after each runtime transition.
    """

    def __init__(
        self,
        runtime_executor: LLMPlanRuntimeExecutor,
        *,
        state_tracker: LLMPlanRuntimeStateTracker | None = None,
        state_observer: StateObserver | None = None,
    ) -> None:
        if runtime_executor is None or not callable(
            getattr(runtime_executor, "execute", None)
        ):
            raise LLMPlanLiveRuntimeConfigurationError(
                "runtime_executor must expose callable execute(...)."
            )
        if not callable(getattr(runtime_executor, "add_event_observer", None)):
            raise LLMPlanLiveRuntimeConfigurationError(
                "runtime_executor must expose add_event_observer(...)."
            )
        if state_tracker is not None and not callable(
            getattr(state_tracker, "start", None)
        ):
            raise LLMPlanLiveRuntimeConfigurationError(
                "state_tracker must expose callable start(...)."
            )
        if state_observer is not None and not callable(state_observer):
            raise LLMPlanLiveRuntimeConfigurationError(
                "state_observer must be callable."
            )

        self.runtime_executor = runtime_executor
        self.state_tracker = state_tracker or LLMPlanRuntimeStateTracker()
        self.state_observer = state_observer
        self._lock = RLock()
        self._state_history: list[LLMPlanRuntimeState] = []
        self._last_result: LLMPlanLiveRuntimeExecutionResult | None = None
        self._active = False

    @classmethod
    def for_executor(
        cls,
        runtime_executor: LLMPlanRuntimeExecutor,
        **kwargs: Any,
    ) -> "LLMPlanLiveRuntimeExecutor":
        return cls(runtime_executor, **kwargs)

    @property
    def last_result(self) -> LLMPlanLiveRuntimeExecutionResult | None:
        return self._last_result

    @property
    def last_state(self) -> LLMPlanRuntimeState:
        return self.state_tracker.snapshot()

    @property
    def state_history(self) -> tuple[LLMPlanRuntimeState, ...]:
        return tuple(self._state_history)

    def _publish_state(self, state: LLMPlanRuntimeState) -> None:
        self._state_history.append(state)
        if self.state_observer is not None:
            self.state_observer(state)

    def _handle_event(self, event: LLMPlanRuntimeEvent) -> None:
        try:
            if event.phase == "started":
                state = self.state_tracker.start(
                    event.plan,
                    metadata=event.metadata,
                )
                self._publish_state(state)
                return

            if event.phase == "step_recorded":
                if event.record is None:
                    raise LLMPlanLiveRuntimeStateError(
                        "step_recorded event requires a record."
                    )
                state = self.state_tracker.record_step(event.record)
                self._publish_state(state)
                return

            if event.phase == "stopped":
                state = self.state_tracker.mark_stopped(
                    reason=event.stop_reason or "stopped",
                    next_step_index=event.next_step_index,
                    success=(
                        event.stop_reason == "goal_satisfied"
                        and event.metadata.get("goal_satisfied") is True
                    ),
                )
                self._publish_state(state)
                return

            if event.phase == "completed":
                state = self.state_tracker.mark_completed(
                    success=True,
                    final_world_state=event.world_state,
                    metadata=event.metadata,
                )
                self._publish_state(state)
                return

            if event.phase == "finished":
                # A run can finish after exhausting all steps while having
                # at least one failure when stop_on_step_failure=False.
                state = self.state_tracker.mark_completed(
                    success=False,
                    final_world_state=event.world_state,
                    metadata=event.metadata,
                )
                self._publish_state(state)
                return

        except Exception as exc:
            raise LLMPlanLiveRuntimeStateError(
                f"Failed to apply runtime event '{event.phase}': {exc}"
            ) from exc

    def execute(
        self,
        plan: LLMPlan,
        **kwargs: Any,
    ) -> LLMPlanLiveRuntimeExecutionResult:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")

        with self._lock:
            if self._active:
                raise LLMPlanLiveRuntimeConfigurationError(
                    "A live runtime execution is already active."
                )
            self._active = True
            self._state_history = []
            self._last_result = None
            self.state_tracker.reset()
            self.runtime_executor.add_event_observer(self._handle_event)
            try:
                report = self.runtime_executor.execute(plan, **kwargs)
                final_state = self.state_tracker.snapshot()
                if final_state.plan_goal != plan.goal:
                    raise LLMPlanLiveRuntimeStateError(
                        "Final runtime state does not belong to the executed plan."
                    )
                result = LLMPlanLiveRuntimeExecutionResult(
                    report=report,
                    final_state=final_state,
                    state_history=tuple(self._state_history),
                )
                self._last_result = result
                return result
            except LLMPlanRuntimeExecutionError:
                raise
            finally:
                self.runtime_executor.remove_event_observer(self._handle_event)
                self._active = False

    def execute_report_only(self, plan: LLMPlan, **kwargs: Any) -> LLMPlanRuntimeExecutionReport:
        """Compatibility helper returning only the underlying execution report."""
        return self.execute(plan, **kwargs).report


__all__ = [
    "LLMPlanLiveRuntimeError",
    "LLMPlanLiveRuntimeConfigurationError",
    "LLMPlanLiveRuntimeStateError",
    "LLMPlanLiveRuntimeExecutionResult",
    "LLMPlanLiveRuntimeExecutor",
]
