from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .planning import LLMPlan, LLMPlanStep
from .llm_plan_runtime_execution import (
    LLMPlanRuntimeExecutionReport,
    LLMPlanStepExecutionRecord,
)


class LLMPlanRuntimeStateError(RuntimeError):
    """Base error for runtime state tracking."""


class LLMPlanRuntimeStateConfigurationError(LLMPlanRuntimeStateError):
    """Raised when runtime state tracking is configured incorrectly."""


class LLMPlanRuntimeStateTransitionError(LLMPlanRuntimeStateError):
    """Raised when an invalid runtime-state transition is requested."""


class LLMPlanRuntimeStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class LLMPlanRuntimeState:
    """Immutable snapshot of the execution state of one LLM plan."""

    plan_goal: str | None
    planned_step_indices: tuple[int, ...]
    current_step_index: int | None
    next_step_index: int | None
    attempted_step_indices: tuple[int, ...]
    succeeded_step_indices: tuple[int, ...]
    failed_step_indices: tuple[int, ...]
    skipped_step_indices: tuple[int, ...]
    progress: float
    success_rate: float
    status: LLMPlanRuntimeStatus
    completed: bool
    stopped: bool
    success: bool
    stop_reason: str | None
    last_observation: Any = None
    last_result: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def planned_step_count(self) -> int:
        return len(self.planned_step_indices)

    @property
    def attempted_step_count(self) -> int:
        return len(self.attempted_step_indices)

    @property
    def succeeded_step_count(self) -> int:
        return len(self.succeeded_step_indices)

    @property
    def failed_step_count(self) -> int:
        return len(self.failed_step_indices)

    @property
    def remaining_step_indices(self) -> tuple[int, ...]:
        processed = set(self.attempted_step_indices) | set(self.skipped_step_indices)
        return tuple(
            index
            for index in self.planned_step_indices
            if index not in processed
        )

    @property
    def can_continue(self) -> bool:
        return (
            self.status is LLMPlanRuntimeStatus.RUNNING
            and self.next_step_index is not None
        )


class LLMPlanRuntimeStateTracker:
    """
    Track plan progress independently from action execution.

    The tracker owns no Agent, executor, or LLM state. It records the
    runtime facts supplied by the execution layer and exposes immutable
    snapshots suitable for diagnostics, persistence, resume decisions,
    and metacognitive context.
    """

    def __init__(self) -> None:
        self._plan: LLMPlan | None = None
        self._planned_indices: tuple[int, ...] = ()
        self._current_step_index: int | None = None
        self._next_step_index: int | None = None
        self._attempted: list[int] = []
        self._succeeded: list[int] = []
        self._failed: list[int] = []
        self._skipped: list[int] = []
        self._status = LLMPlanRuntimeStatus.IDLE
        self._completed = False
        self._stopped = False
        self._success = False
        self._stop_reason: str | None = None
        self._last_observation: Any = None
        self._last_result: Any = None
        self._metadata: dict[str, Any] = {}

    def reset(self) -> None:
        self.__init__()

    def start(
        self,
        plan: LLMPlan,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanRuntimeState:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")
        if not plan.steps:
            raise LLMPlanRuntimeStateTransitionError(
                "Cannot start a plan without steps."
            )

        indices = tuple(step.index for step in plan.steps)
        if len(set(indices)) != len(indices):
            raise LLMPlanRuntimeStateTransitionError(
                "Plan step indices must be unique."
            )

        self._plan = plan
        self._planned_indices = tuple(sorted(indices))
        self._current_step_index = None
        self._next_step_index = self._planned_indices[0]
        self._attempted = []
        self._succeeded = []
        self._failed = []
        self._skipped = []
        self._status = LLMPlanRuntimeStatus.RUNNING
        self._completed = False
        self._stopped = False
        self._success = False
        self._stop_reason = None
        self._last_observation = None
        self._last_result = None
        self._metadata = dict(metadata or {})
        self._metadata["planned_step_count"] = len(self._planned_indices)
        return self.snapshot()

    def _require_started(self) -> None:
        if self._plan is None or self._status is LLMPlanRuntimeStatus.IDLE:
            raise LLMPlanRuntimeStateTransitionError(
                "Runtime state has not been started."
            )

    def _validate_record_step(self, record: LLMPlanStepExecutionRecord) -> None:
        if not isinstance(record, LLMPlanStepExecutionRecord):
            raise TypeError(
                "record must be an LLMPlanStepExecutionRecord."
            )
        if record.step.index not in self._planned_indices:
            raise LLMPlanRuntimeStateTransitionError(
                f"Step index {record.step.index} is not part of the active plan."
            )
        if record.step.index in self._attempted or record.step.index in self._skipped:
            raise LLMPlanRuntimeStateTransitionError(
                f"Step index {record.step.index} has already been recorded."
            )
        if self._next_step_index is not None and record.step.index != self._next_step_index:
            raise LLMPlanRuntimeStateTransitionError(
                f"Step index {record.step.index} is out of order; expected {self._next_step_index}."
            )

    def record_step(
        self,
        record: LLMPlanStepExecutionRecord,
    ) -> LLMPlanRuntimeState:
        self._require_started()
        if self._status not in {
            LLMPlanRuntimeStatus.RUNNING,
        }:
            raise LLMPlanRuntimeStateTransitionError(
                "Cannot record a step after the plan has stopped or completed."
            )

        self._validate_record_step(record)
        index = record.step.index
        self._current_step_index = index
        self._attempted.append(index)
        if record.success:
            self._succeeded.append(index)
        else:
            self._failed.append(index)
        if record.skipped:
            self._skipped.append(index)

        self._last_observation = record.observation
        self._last_result = record.result

        remaining = [
            candidate
            for candidate in self._planned_indices
            if candidate not in self._attempted
            and candidate not in self._skipped
        ]
        self._next_step_index = remaining[0] if remaining else None

        self._metadata["last_recorded_step_index"] = index
        self._metadata["attempted_step_count"] = len(self._attempted)
        self._metadata["succeeded_step_count"] = len(self._succeeded)
        self._metadata["failed_step_count"] = len(self._failed)
        return self.snapshot()

    def mark_skipped(
        self,
        step_index: int,
        *,
        reason: str | None = None,
    ) -> LLMPlanRuntimeState:
        self._require_started()
        if step_index not in self._planned_indices:
            raise LLMPlanRuntimeStateTransitionError(
                f"Step index {step_index} is not part of the active plan."
            )
        if step_index in self._attempted or step_index in self._skipped:
            raise LLMPlanRuntimeStateTransitionError(
                f"Step index {step_index} has already been recorded."
            )

        self._skipped.append(step_index)
        remaining = [
            candidate
            for candidate in self._planned_indices
            if candidate not in self._attempted
            and candidate not in self._skipped
        ]
        self._next_step_index = remaining[0] if remaining else None
        if reason:
            self._metadata["last_skip_reason"] = reason
        return self.snapshot()

    def mark_stopped(
        self,
        *,
        reason: str,
        next_step_index: int | None = None,
        success: bool = False,
    ) -> LLMPlanRuntimeState:
        self._require_started()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        if self._status not in {
            LLMPlanRuntimeStatus.RUNNING,
        }:
            raise LLMPlanRuntimeStateTransitionError(
                "Plan is no longer running."
            )

        if next_step_index is not None and next_step_index not in self._planned_indices:
            raise LLMPlanRuntimeStateTransitionError(
                f"next_step_index {next_step_index} is not part of the active plan."
            )

        self._status = LLMPlanRuntimeStatus.STOPPED
        self._completed = False
        self._stopped = True
        self._success = bool(success)
        self._stop_reason = reason.strip()
        self._next_step_index = next_step_index
        return self.snapshot()

    def mark_completed(
        self,
        *,
        success: bool,
        final_world_state: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanRuntimeState:
        self._require_started()
        if self._status not in {
            LLMPlanRuntimeStatus.RUNNING,
        }:
            raise LLMPlanRuntimeStateTransitionError(
                "Plan is no longer running."
            )

        self._completed = True
        self._stopped = False
        self._success = bool(success)
        self._status = (
            LLMPlanRuntimeStatus.COMPLETED
            if success
            else LLMPlanRuntimeStatus.FAILED
        )
        self._next_step_index = None
        self._last_observation = final_world_state
        self._metadata.update(dict(metadata or {}))
        return self.snapshot()

    def apply_report(
        self,
        report: LLMPlanRuntimeExecutionReport,
    ) -> LLMPlanRuntimeState:
        if not isinstance(report, LLMPlanRuntimeExecutionReport):
            raise TypeError(
                "report must be an LLMPlanRuntimeExecutionReport."
            )

        self.start(report.plan, metadata=report.metadata)
        for record in report.records:
            self.record_step(record)

        if report.stopped:
            self.mark_stopped(
                reason=report.stop_reason or "stopped",
                next_step_index=report.next_step_index,
                success=(
                    report.stop_reason == "goal_satisfied"
                    and report.goal_satisfied is True
                ),
            )
        elif report.completed:
            self.mark_completed(
                success=report.success,
                final_world_state=report.final_world_state,
                metadata=report.metadata,
            )
        else:
            self._last_observation = report.final_world_state
            self._next_step_index = report.next_step_index
        return self.snapshot()

    def snapshot(self) -> LLMPlanRuntimeState:
        planned_count = len(self._planned_indices)
        attempted_count = len(self._attempted)
        progress = (
            attempted_count / planned_count
            if planned_count
            else 0.0
        )
        success_rate = (
            len(self._succeeded) / attempted_count
            if attempted_count
            else 0.0
        )
        return LLMPlanRuntimeState(
            plan_goal=self._plan.goal if self._plan is not None else None,
            planned_step_indices=self._planned_indices,
            current_step_index=self._current_step_index,
            next_step_index=self._next_step_index,
            attempted_step_indices=tuple(self._attempted),
            succeeded_step_indices=tuple(self._succeeded),
            failed_step_indices=tuple(self._failed),
            skipped_step_indices=tuple(self._skipped),
            progress=round(progress, 6),
            success_rate=round(success_rate, 6),
            status=self._status,
            completed=self._completed,
            stopped=self._stopped,
            success=self._success,
            stop_reason=self._stop_reason,
            last_observation=self._last_observation,
            last_result=self._last_result,
            metadata=dict(self._metadata),
        )


__all__ = [
    "LLMPlanRuntimeStateError",
    "LLMPlanRuntimeStateConfigurationError",
    "LLMPlanRuntimeStateTransitionError",
    "LLMPlanRuntimeStatus",
    "LLMPlanRuntimeState",
    "LLMPlanRuntimeStateTracker",
]
