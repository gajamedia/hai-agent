from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .plan_validation import LLMPlanValidationResult


class ControlledExecutionError(RuntimeError):
    """Base error raised by the controlled execution gate."""


@dataclass(frozen=True)
class ControlledStepResult:
    step_index: int
    action: str
    success: bool
    result: Any = None


@dataclass(frozen=True)
class ControlledExecutionReport:
    started: bool
    completed: bool
    stopped: bool
    stop_reason: str | None
    steps: tuple[ControlledStepResult, ...]


class ControlledPlanExecutor:
    """
    Execute only a previously validated LLM plan.

    The executor is intentionally callback-driven. It does not know how
    an Agent executes an action and therefore cannot silently bypass the
    existing Agent executor or mutate Agent state on its own.
    """

    def __init__(
        self,
        action_executor: Callable[[str, Any, int], Any],
    ) -> None:
        if not callable(action_executor):
            raise TypeError("action_executor must be callable.")
        self.action_executor = action_executor

    def execute(
        self,
        plan: Any,
        validation: LLMPlanValidationResult,
        *,
        stop_checker: Callable[[Any, int, Any], bool] | None = None,
    ) -> ControlledExecutionReport:
        if not validation.valid:
            raise ControlledExecutionError(
                "Cannot execute an invalid LLM plan."
            )

        if isinstance(plan, dict):
            steps = list(plan.get("steps", []))
        else:
            steps = list(getattr(plan, "steps", []))
        results: list[ControlledStepResult] = []

        for position, step in enumerate(steps, start=1):
            action = (
                step.get("action", step.get("action_name"))
                if isinstance(step, dict)
                else getattr(step, "action", getattr(step, "action_name", None))
            )
            if not action:
                raise ControlledExecutionError(
                    f"Validated plan contains an action-less step at position {position}."
                )

            try:
                outcome = self.action_executor(str(action), step, position)
            except Exception as exc:  # pragma: no cover - behavior asserted via result path
                result = ControlledStepResult(
                    step_index=position,
                    action=str(action),
                    success=False,
                    result=exc,
                )
                results.append(result)
                return ControlledExecutionReport(
                    started=True,
                    completed=False,
                    stopped=True,
                    stop_reason=f"executor_error:{type(exc).__name__}",
                    steps=tuple(results),
                )

            success = bool(outcome.get("success", False)) if isinstance(outcome, dict) else bool(outcome)
            result = ControlledStepResult(
                step_index=position,
                action=str(action),
                success=success,
                result=outcome,
            )
            results.append(result)

            if not success:
                return ControlledExecutionReport(
                    started=True,
                    completed=False,
                    stopped=True,
                    stop_reason="step_failed",
                    steps=tuple(results),
                )

            if stop_checker is not None and stop_checker(step, position, outcome):
                return ControlledExecutionReport(
                    started=True,
                    completed=False,
                    stopped=True,
                    stop_reason="stop_condition",
                    steps=tuple(results),
                )

        return ControlledExecutionReport(
            started=bool(results),
            completed=True,
            stopped=False,
            stop_reason=None,
            steps=tuple(results),
        )
