from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from .llm_plan_single_action_bridge import (
    LLMPlanSingleActionBridge,
    LLMPlanSingleActionExecutionRequest,
)
from .plan_validation import (
    LLMPlanValidationResult,
    LLMPlanValidator,
)
from .planning import LLMPlan, LLMPlanStep


class LLMPlanRuntimeExecutionError(RuntimeError):
    """Base error for runtime execution of an LLM-generated plan."""


class LLMPlanRuntimeConfigurationError(LLMPlanRuntimeExecutionError):
    """Raised when the runtime executor is configured incorrectly."""


class LLMPlanRuntimeValidationError(LLMPlanRuntimeExecutionError):
    """Raised when a plan is invalid and must not be executed."""

    def __init__(self, validation: LLMPlanValidationResult) -> None:
        self.validation = validation
        messages = "; ".join(issue.message for issue in validation.issues)
        super().__init__(messages or "LLM plan validation failed.")


@dataclass(frozen=True)
class LLMPlanStepExecutionRecord:
    """Trace for one attempted plan step."""

    step: LLMPlanStep
    request: LLMPlanSingleActionExecutionRequest | None
    result: Any = None
    success: bool = False
    observation: Any = None
    skipped: bool = False
    stop_requested: bool = False
    error: str | None = None


@dataclass(frozen=True)
class LLMPlanRuntimeExecutionReport:
    """Complete, immutable trace of one runtime plan execution."""

    plan: LLMPlan
    validation: LLMPlanValidationResult
    records: tuple[LLMPlanStepExecutionRecord, ...]
    completed: bool
    stopped: bool
    success: bool
    stop_reason: str | None
    final_world_state: Any
    next_step_index: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # None means no goal verifier was available; True/False is an observed verdict.
    goal_satisfied: bool | None = None

    @property
    def executed_step_indices(self) -> tuple[int, ...]:
        return tuple(
            record.step.index
            for record in self.records
            if not record.skipped
        )

    @property
    def failed_step_indices(self) -> tuple[int, ...]:
        return tuple(
            record.step.index
            for record in self.records
            if not record.skipped and not record.success
        )


StopConditionEvaluator = Callable[[str, Mapping[str, Any]], bool]
StepSuccessEvaluator = Callable[[LLMPlanStep, Any, Mapping[str, Any]], bool]
WorldStateObserver = Callable[[Any], Any]
GoalSatisfactionEvaluator = Callable[[Any, Any, Mapping[str, Any]], bool]


@dataclass(frozen=True)
class LLMPlanRuntimeEvent:
    """Immutable event emitted by the runtime execution loop."""

    phase: str
    plan: LLMPlan
    step: LLMPlanStep | None = None
    record: LLMPlanStepExecutionRecord | None = None
    world_state: Any = None
    result: Any = None
    stop_reason: str | None = None
    next_step_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


RuntimeEventObserver = Callable[[LLMPlanRuntimeEvent], None]


class _RuntimeEventObserverError(RuntimeError):
    """Distinguish telemetry/state observer failures from action failures."""



def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("description", "name", "goal"):
            if key in value:
                return str(value[key]).strip()
    for key in ("description", "name", "goal"):
        if hasattr(value, key):
            return str(getattr(value, key)).strip()
    return str(value).strip()


def _risk_score(value: Any) -> float:
    """
    Convert the LLMPlanStep textual risk field into a bounded numeric score.

    LLMPlanStep intentionally stores human-readable risk text while the
    existing provider-neutral validator expects a numeric risk. This adapter
    keeps those contracts separate without mutating the original plan.
    """

    if value is None or value == "":
        return 0.0

    if isinstance(value, bool):
        return 1.0

    try:
        numeric = float(value)
        if isfinite(numeric):
            return min(1.0, max(0.0, numeric))
    except (TypeError, ValueError):
        pass

    normalized = str(value).strip().casefold()
    labels = {
        "none": 0.0,
        "negligible": 0.1,
        "very low": 0.15,
        "low": 0.25,
        "medium": 0.5,
        "moderate": 0.5,
        "high": 0.75,
        "very high": 0.9,
        "critical": 1.0,
    }
    return labels.get(normalized, 1.0)


class _ValidationView:
    """Non-mutating validation projection for LLMPlan textual risk values."""

    def __init__(self, plan: LLMPlan) -> None:
        self.plan = plan
        self.goal = plan.goal
        self.confidence = plan.confidence
        self.steps = tuple(
            _ValidationStep(step)
            for step in plan.steps
        )


class _ValidationStep:
    def __init__(self, step: LLMPlanStep) -> None:
        self.index = step.index
        self.action = step.action
        self.risk = _risk_score(step.risk)
        self.success_criteria = step.success_criteria


class LLMPlanRuntimeExecutor:
    """
    Execute a validated LLM plan one step at a time.

    Runtime responsibilities:

    * validate the complete plan before any action is attempted;
    * execute steps strictly in plan order;
    * route every action through LLMPlanSingleActionBridge;
    * stop on failed action or an explicitly evaluated stop condition;
    * observe world state after each successful attempt when an observer exists;
    * return a complete execution trace.

    The runtime does not call an LLM, ``Agent.decide_and_act()``, or any
    second execution engine.
    """

    def __init__(
        self,
        bridge: LLMPlanSingleActionBridge,
        *,
        validator: LLMPlanValidator | None = None,
        stop_condition_evaluator: StopConditionEvaluator | None = None,
        step_success_evaluator: StepSuccessEvaluator | None = None,
        world_state_observer: WorldStateObserver | None = None,
        goal_satisfaction_evaluator: GoalSatisfactionEvaluator | None = None,
        stop_on_step_failure: bool = True,
        raise_on_step_error: bool = False,
        runtime_event_observer: RuntimeEventObserver | None = None,
    ) -> None:
        if bridge is None or not callable(getattr(bridge, "execute_step", None)):
            raise LLMPlanRuntimeConfigurationError(
                "bridge must expose callable execute_step(...)."
            )
        if validator is not None and not callable(getattr(validator, "validate", None)):
            raise LLMPlanRuntimeConfigurationError(
                "validator must expose callable validate(...)."
            )
        if stop_condition_evaluator is not None and not callable(stop_condition_evaluator):
            raise LLMPlanRuntimeConfigurationError(
                "stop_condition_evaluator must be callable."
            )
        if step_success_evaluator is not None and not callable(step_success_evaluator):
            raise LLMPlanRuntimeConfigurationError(
                "step_success_evaluator must be callable."
            )
        if world_state_observer is not None and not callable(world_state_observer):
            raise LLMPlanRuntimeConfigurationError(
                "world_state_observer must be callable."
            )
        if goal_satisfaction_evaluator is not None and not callable(goal_satisfaction_evaluator):
            raise LLMPlanRuntimeConfigurationError(
                "goal_satisfaction_evaluator must be callable."
            )

        self.bridge = bridge
        self.validator = validator or LLMPlanValidator()
        self.stop_condition_evaluator = stop_condition_evaluator
        self.step_success_evaluator = step_success_evaluator
        self.world_state_observer = world_state_observer
        self.goal_satisfaction_evaluator = goal_satisfaction_evaluator
        self.stop_on_step_failure = bool(stop_on_step_failure)
        self.raise_on_step_error = bool(raise_on_step_error)
        if runtime_event_observer is not None and not callable(runtime_event_observer):
            raise LLMPlanRuntimeConfigurationError(
                "runtime_event_observer must be callable."
            )
        self.runtime_event_observer = runtime_event_observer
        self._additional_event_observers: list[RuntimeEventObserver] = []
        self.last_report: LLMPlanRuntimeExecutionReport | None = None

    def _emit_event(self, event: LLMPlanRuntimeEvent) -> None:
        observers = []
        if self.runtime_event_observer is not None:
            observers.append(self.runtime_event_observer)
        observers.extend(tuple(self._additional_event_observers))
        for observer in observers:
            try:
                observer(event)
            except Exception as exc:
                # An observer failure is not an action execution failure.
                # Propagate it distinctly so the step is never recorded twice.
                raise _RuntimeEventObserverError(
                    f"Runtime event observer failed during '{event.phase}': {exc}"
                ) from exc

    def add_event_observer(self, observer: RuntimeEventObserver) -> None:
        if not callable(observer):
            raise TypeError("observer must be callable.")
        if observer not in self._additional_event_observers:
            self._additional_event_observers.append(observer)

    def remove_event_observer(self, observer: RuntimeEventObserver) -> None:
        try:
            self._additional_event_observers.remove(observer)
        except ValueError:
            pass

    @classmethod
    def for_bridge(
        cls,
        bridge: LLMPlanSingleActionBridge,
        **kwargs: Any,
    ) -> "LLMPlanRuntimeExecutor":
        return cls(bridge, **kwargs)

    def _available_actions(self, explicit: Any = None) -> Any:
        if explicit is not None:
            return explicit

        api = getattr(self.bridge, "api", None)
        agent = getattr(api, "agent", None)
        actions = getattr(agent, "actions", None)
        if actions is not None:
            return actions
        return None

    def _validate(
        self,
        plan: LLMPlan,
        *,
        goal: Any,
        available_actions: Any,
    ) -> LLMPlanValidationResult:
        validation_view = _ValidationView(plan)
        return self.validator.validate(
            validation_view,
            goal=goal,
            available_actions=available_actions,
        )

    def _evaluate_stop_conditions(
        self,
        plan: LLMPlan,
        *,
        phase: str,
        step: LLMPlanStep | None,
        records: Sequence[LLMPlanStepExecutionRecord],
        world_state: Any,
        last_result: Any,
    ) -> tuple[bool, str | None]:
        if self.stop_condition_evaluator is None:
            return False, None

        context = {
            "phase": phase,
            "plan": plan,
            "step": step,
            "records": tuple(records),
            "world_state": world_state,
            "last_result": last_result,
        }

        for condition in plan.stop_conditions:
            if self.stop_condition_evaluator(condition, context):
                return True, condition
        return False, None

    def _observe_world_state(
        self,
        *,
        environment: Any,
        current_world_state: Any,
        result: Any,
    ) -> Any:
        if self.world_state_observer is not None:
            return self.world_state_observer(environment)

        observe = getattr(environment, "observe", None)
        if callable(observe):
            return observe()

        value = getattr(result, "result", None)
        if isinstance(value, Mapping):
            for key in ("world_state", "observation"):
                if key in value:
                    return value[key]
        return current_world_state

    def _resolve_success(
        self,
        step: LLMPlanStep,
        result: Any,
        *,
        context: Mapping[str, Any],
    ) -> bool:
        if self.step_success_evaluator is not None:
            return bool(self.step_success_evaluator(step, result, context))
        return bool(getattr(result, "success", False))

    def _verify_goal_satisfaction(
        self,
        goal: Any,
        world_state: Any,
        *,
        plan: LLMPlan,
        records: Sequence[LLMPlanStepExecutionRecord],
        environment: Any,
    ) -> bool | None:
        """Verify the requested goal against the final observed state.

        An explicit evaluator takes precedence. Otherwise, Goal-like objects
        exposing ``is_satisfied(world_state)`` are verified automatically.
        Text-only goals retain legacy behavior and return ``None``.
        """
        context = {
            "plan": plan,
            "records": tuple(records),
            "world_state": world_state,
            "environment": environment,
        }
        if self.goal_satisfaction_evaluator is not None:
            return bool(self.goal_satisfaction_evaluator(goal, world_state, context))
        checker = getattr(goal, "is_satisfied", None)
        if callable(checker):
            return bool(checker(world_state))
        return None

    @staticmethod
    def _next_planned_step_index(
        plan: LLMPlan,
        current_step_index: int,
    ) -> int | None:
        """Return the next actual plan index, or None when no step remains."""
        remaining = sorted(
            step.index for step in plan.steps
            if step.index > current_step_index
        )
        return remaining[0] if remaining else None

    def execute(
        self,
        plan: LLMPlan,
        *,
        goal: Any = None,
        world_state: Any = None,
        environment: Any = None,
        strategies: Any = None,
        information_gain: Mapping[str, float] | None = None,
        action_risk: Mapping[str, float] | None = None,
        exploration_risk: float = 0.0,
        action_cost: Mapping[str, float] | None = None,
        experience_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        available_actions: Any = None,
    ) -> LLMPlanRuntimeExecutionReport:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")

        resolved_goal = plan.goal if goal is None else goal
        actions = self._available_actions(available_actions)
        validation = self._validate(
            plan,
            goal=resolved_goal,
            available_actions=actions,
        )
        if not validation.valid:
            raise LLMPlanRuntimeValidationError(validation)

        start_metadata = dict(metadata or {})
        start_metadata.update(
            {
                "runtime_executor": "llm_plan_runtime_execution",
                "planned_step_count": len(plan.steps),
            }
        )
        self._emit_event(
            LLMPlanRuntimeEvent(
                phase="started",
                plan=plan,
                world_state=world_state,
                metadata=start_metadata,
            )
        )

        records: list[LLMPlanStepExecutionRecord] = []
        current_world_state = world_state
        stop_reason: str | None = None
        goal_satisfied: bool | None = None
        stopped = False
        next_step_index: int | None = None
        last_result: Any = None

        for step in sorted(plan.steps, key=lambda item: item.index):
            # A verified goal is terminal: do not execute further plan steps
            # when the observed state already satisfies it.
            # Only perform the pre-action check when the caller supplied an
            # observed initial state. If it did not, let the first action run
            # and verify against the fresh environment observation afterward.
            goal_already_satisfied = (
                self._verify_goal_satisfaction(
                    resolved_goal,
                    current_world_state,
                    plan=plan,
                    records=records,
                    environment=environment,
                )
                if world_state is not None
                else None
            )
            if goal_already_satisfied is True:
                goal_satisfied = True
                stopped = True
                stop_reason = "goal_satisfied"
                next_step_index = step.index
                self._emit_event(
                    LLMPlanRuntimeEvent(
                        phase="stopped",
                        plan=plan,
                        step=step,
                        world_state=current_world_state,
                        result=last_result,
                        stop_reason=stop_reason,
                        next_step_index=next_step_index,
                        metadata=dict(metadata or {}),
                    )
                )
                break

            stop_requested, condition = self._evaluate_stop_conditions(
                plan,
                phase="before_step",
                step=step,
                records=records,
                world_state=current_world_state,
                last_result=last_result,
            )
            if stop_requested:
                stopped = True
                stop_reason = f"stop_condition:{condition}"
                next_step_index = step.index
                self._emit_event(
                    LLMPlanRuntimeEvent(
                        phase="stopped",
                        plan=plan,
                        step=step,
                        world_state=current_world_state,
                        result=last_result,
                        stop_reason=stop_reason,
                        next_step_index=next_step_index,
                        metadata=dict(metadata or {}),
                    )
                )
                break

            runtime_metadata = dict(metadata or {})
            runtime_metadata.update(
                {
                    "runtime_executor": "llm_plan_runtime_execution",
                    "llm_plan_step_index": step.index,
                    "llm_plan_step_risk_score": _risk_score(step.risk),
                }
            )

            effective_action_risk = dict(action_risk or {})
            if step.action not in effective_action_risk:
                effective_action_risk[step.action] = _risk_score(step.risk)

            request_trace: LLMPlanSingleActionExecutionRequest | None = None
            result: Any = None
            observation: Any = None

            try:
                result = self.bridge.execute_step(
                    plan,
                    step,
                    goal=resolved_goal,
                    world_state=current_world_state,
                    environment=environment,
                    strategies=strategies,
                    information_gain=information_gain,
                    action_risk=effective_action_risk,
                    exploration_risk=exploration_risk,
                    action_cost=action_cost,
                    experience_id=experience_id,
                    metadata=runtime_metadata,
                )

                # A feedback-blocked step was never sent to the underlying
                # action API. Record it as skipped and stop without emitting
                # "step_recorded", which is reserved for actual attempts.
                result_payload = getattr(result, "result", None)
                if (
                    isinstance(result_payload, Mapping)
                    and result_payload.get("blocked_by_runtime_feedback") is True
                ):
                    blocked_reason = str(
                        result_payload.get("reason") or "runtime_feedback_stop"
                    )
                    record = LLMPlanStepExecutionRecord(
                        step=step,
                        request=None,
                        result=result,
                        success=False,
                        observation=current_world_state,
                        skipped=True,
                        error=blocked_reason,
                    )
                    records.append(record)
                    last_result = result
                    stopped = True
                    stop_reason = "runtime_feedback_stop"
                    next_step_index = step.index

                    blocked_metadata = dict(runtime_metadata)
                    blocked_metadata.update(
                        {
                            "runtime_feedback_stop": True,
                            "runtime_feedback_reason": blocked_reason,
                            "blocked_by_runtime_feedback": True,
                        }
                    )
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="step_blocked",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            stop_reason=stop_reason,
                            next_step_index=next_step_index,
                            metadata=blocked_metadata,
                        )
                    )
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="stopped",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            stop_reason=stop_reason,
                            next_step_index=next_step_index,
                            metadata=blocked_metadata,
                        )
                    )
                    break

                request_trace = getattr(self.bridge, "last_execution_request", None)

                context = {
                    "plan": plan,
                    "step": step,
                    "records": tuple(records),
                    "world_state": current_world_state,
                    "environment": environment,
                }
                success = self._resolve_success(
                    step,
                    result,
                    context=context,
                )
                observation = self._observe_world_state(
                    environment=environment,
                    current_world_state=current_world_state,
                    result=result,
                )
                current_world_state = observation

                record = LLMPlanStepExecutionRecord(
                    step=step,
                    request=request_trace,
                    result=result,
                    success=success,
                    observation=observation,
                )
                records.append(record)
                last_result = result

                if not success and self.stop_on_step_failure:
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="step_recorded",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            next_step_index=self._next_planned_step_index(plan, step.index),
                            metadata=dict(runtime_metadata),
                        )
                    )
                    stopped = True
                    stop_reason = "step_failed"
                    next_step_index = self._next_planned_step_index(plan, step.index)
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="stopped",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            stop_reason=stop_reason,
                            next_step_index=next_step_index,
                            metadata=dict(runtime_metadata),
                        )
                    )
                    break

                # Verify the explicit goal against the freshly observed state
                # after every successful action. This is independent of the
                # planner's natural-language stop_conditions.
                goal_now_satisfied = self._verify_goal_satisfaction(
                    resolved_goal,
                    current_world_state,
                    plan=plan,
                    records=records,
                    environment=environment,
                )
                goal_satisfied = goal_now_satisfied
                remaining_step_index = self._next_planned_step_index(
                    plan, step.index
                )
                # A satisfied goal only stops execution early when there are
                # additional planned actions to skip. If this was the final
                # step, let the run complete normally (stop_reason=None).
                if goal_now_satisfied is True and remaining_step_index is not None:
                    stopped = True
                    stop_reason = "goal_satisfied"
                    next_step_index = remaining_step_index
                    records[-1] = LLMPlanStepExecutionRecord(
                        step=record.step,
                        request=record.request,
                        result=record.result,
                        success=record.success,
                        observation=record.observation,
                        skipped=record.skipped,
                        stop_requested=True,
                        error=record.error,
                    )
                    record = records[-1]
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="step_recorded",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            next_step_index=next_step_index,
                            metadata=dict(runtime_metadata),
                        )
                    )
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="stopped",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            stop_reason=stop_reason,
                            next_step_index=next_step_index,
                            metadata={
                                **dict(runtime_metadata),
                                "goal_satisfied": True,
                            },
                        )
                    )
                    break

                stop_requested, condition = self._evaluate_stop_conditions(
                    plan,
                    phase="after_step",
                    step=step,
                    records=records,
                    world_state=current_world_state,
                    last_result=last_result,
                )
                if stop_requested:
                    stopped = True
                    stop_reason = f"stop_condition:{condition}"
                    next_step_index = self._next_planned_step_index(plan, step.index)
                    records[-1] = LLMPlanStepExecutionRecord(
                        step=record.step,
                        request=record.request,
                        result=record.result,
                        success=record.success,
                        observation=record.observation,
                        skipped=record.skipped,
                        stop_requested=True,
                        error=record.error,
                    )
                    record = records[-1]
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="step_recorded",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            next_step_index=self._next_planned_step_index(plan, step.index),
                            metadata=dict(runtime_metadata),
                        )
                    )
                    self._emit_event(
                        LLMPlanRuntimeEvent(
                            phase="stopped",
                            plan=plan,
                            step=step,
                            record=record,
                            world_state=current_world_state,
                            result=result,
                            stop_reason=stop_reason,
                            next_step_index=next_step_index,
                            metadata=dict(runtime_metadata),
                        )
                    )
                    break

                self._emit_event(
                    LLMPlanRuntimeEvent(
                        phase="step_recorded",
                        plan=plan,
                        step=step,
                        record=record,
                        world_state=current_world_state,
                        result=result,
                        next_step_index=self._next_planned_step_index(plan, step.index),
                        metadata=dict(runtime_metadata),
                    )
                )

            except _RuntimeEventObserverError:
                # Do not convert telemetry/state-transition failures into a new
                # failed action record: the current step may already be recorded.
                raise
            except Exception as exc:
                record = LLMPlanStepExecutionRecord(
                    step=step,
                    request=request_trace,
                    result=result,
                    success=False,
                    observation=observation,
                    error=str(exc),
                )
                records.append(record)
                self._emit_event(
                    LLMPlanRuntimeEvent(
                        phase="step_recorded",
                        plan=plan,
                        step=step,
                        record=record,
                        world_state=current_world_state,
                        result=result,
                        next_step_index=step.index,
                        metadata=dict(runtime_metadata),
                    )
                )
                stopped = True
                stop_reason = "step_execution_error"
                next_step_index = step.index
                self._emit_event(
                    LLMPlanRuntimeEvent(
                        phase="stopped",
                        plan=plan,
                        step=step,
                        record=record,
                        world_state=current_world_state,
                        result=result,
                        stop_reason=stop_reason,
                        next_step_index=next_step_index,
                        metadata=dict(runtime_metadata),
                    )
                )
                if self.raise_on_step_error:
                    raise
                break

        completed = len(records) == len(plan.steps) and all(
            record.success and not record.skipped for record in records
        )
        if goal_satisfied is None:
            goal_satisfied = self._verify_goal_satisfaction(
                resolved_goal,
                current_world_state,
                plan=plan,
                records=records,
                environment=environment,
            )
        # Successful individual actions do not imply that the goal was reached.
        # An early stop is successful only when the explicit goal verifier
        # confirms the goal was reached. Ordinary stop conditions and failed
        # actions remain unsuccessful.
        success = (
            completed and not stopped and goal_satisfied is not False
        ) or (
            stopped
            and stop_reason == "goal_satisfied"
            and goal_satisfied is True
        )
        if goal_satisfied is False:
            success = False
        if completed:
            next_step_index = None

        report_metadata = dict(metadata or {})
        report_metadata.update(
            {
                "executed_step_count": sum(
                    1 for record in records if not record.skipped
                ),
                "planned_step_count": len(plan.steps),
                "goal_satisfied": goal_satisfied,
                "goal_verification_performed": goal_satisfied is not None,
            }
        )
        if completed and goal_satisfied is False and stop_reason is None:
            stop_reason = "goal_not_satisfied"
            report_metadata["goal_verification_failure"] = True

        report = LLMPlanRuntimeExecutionReport(
            plan=plan,
            validation=validation,
            records=tuple(records),
            completed=completed,
            stopped=stopped,
            success=success,
            stop_reason=stop_reason,
            final_world_state=current_world_state,
            next_step_index=next_step_index,
            metadata=report_metadata,
            goal_satisfied=goal_satisfied,
        )
        self.last_report = report
        if completed or not stopped:
            self._emit_event(
                LLMPlanRuntimeEvent(
                    phase="completed" if completed else "finished",
                    plan=plan,
                    step=(records[-1].step if records else None),
                    record=(records[-1] if records else None),
                    world_state=current_world_state,
                    result=last_result,
                    stop_reason=stop_reason,
                    next_step_index=next_step_index,
                    metadata=dict(report_metadata),
                )
            )
        return report


__all__ = [
    "LLMPlanRuntimeExecutionError",
    "LLMPlanRuntimeConfigurationError",
    "LLMPlanRuntimeValidationError",
    "LLMPlanStepExecutionRecord",
    "LLMPlanRuntimeExecutionReport",
    "LLMPlanRuntimeEvent",
    "RuntimeEventObserver",
    "GoalSatisfactionEvaluator",
    "LLMPlanRuntimeExecutor",
]
