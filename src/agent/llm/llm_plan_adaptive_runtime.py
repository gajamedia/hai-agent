from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .llm_plan_adaptive_replanning import (
    LLMPlanAdaptiveReplanner,
    LLMPlanAdaptiveReplanningResult,
)
from .llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback
from .planning import LLMPlan, LLMPlanStep


class LLMPlanAdaptiveRuntimeError(RuntimeError):
    """Base error for closed-loop adaptive plan runtime integration."""


class LLMPlanAdaptiveRuntimeConfigurationError(LLMPlanAdaptiveRuntimeError):
    """Raised when the adaptive runtime is misconfigured."""


class LLMPlanAdaptiveRuntimeExecutionError(LLMPlanAdaptiveRuntimeError):
    """Raised when an adaptive runtime segment cannot execute safely."""


class LLMPlanAdaptiveRuntimeSafetyError(LLMPlanAdaptiveRuntimeError):
    """Raised when a replacement plan repeats completed work unsafely."""


class LLMPlanAdaptiveRuntimeReplanningError(LLMPlanAdaptiveRuntimeError):
    """Raised when an adaptive replan cannot be applied."""


AdaptiveRuntimeFeedbackSource = Any
AdaptiveRuntimeEventObserver = Callable[["LLMPlanAdaptiveRuntimeEvent"], None]


@dataclass(frozen=True)
class LLMPlanAdaptiveRuntimeEvent:
    """Immutable event emitted by the closed-loop adaptive controller."""

    phase: str
    generation: int
    plan: LLMPlan
    segment_index: int
    report: Any = None
    feedback: LLMPlanRuntimeLearningFeedback | None = None
    replanning: LLMPlanAdaptiveReplanningResult | None = None
    world_state: Any = None
    next_step_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMPlanAdaptiveRuntimeExecutionReport:
    """Aggregate immutable trace across all adaptive runtime segments."""

    initial_plan: LLMPlan
    final_plan: LLMPlan
    segment_reports: tuple[Any, ...]
    replans: tuple[LLMPlanAdaptiveReplanningResult, ...]
    completed_successful_step_indices: tuple[int, ...]
    failed_step_indices: tuple[int, ...]
    completed: bool
    stopped: bool
    success: bool
    stop_reason: str | None
    final_world_state: Any
    next_step_index: int | None
    generation_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def replanned(self) -> bool:
        return bool(self.replans)

    @property
    def segment_count(self) -> int:
        return len(self.segment_reports)

    @property
    def plan_generations(self) -> tuple[LLMPlan, ...]:
        return tuple(
            [self.initial_plan]
            + [result.replacement_plan for result in self.replans if result.replacement_plan]
        )


_REPLAN_STOP_CONDITION = "__adaptive_runtime_replanning__"


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _step_signature(step: LLMPlanStep) -> tuple[Any, ...]:
    """Semantic signature used only for global completed-work protection."""

    return (
        _text(step.action).casefold(),
        _text(step.purpose).casefold(),
        tuple(_text(item).casefold() for item in step.preconditions),
        None if step.expected_observation is None else _text(step.expected_observation).casefold(),
        None if step.success_criteria is None else _text(step.success_criteria).casefold(),
    )


def _get_report(result: Any) -> Any:
    report = getattr(result, "report", None)
    return report if report is not None else result


def _get_feedback(
    source: AdaptiveRuntimeFeedbackSource,
    step_index: int,
) -> LLMPlanRuntimeLearningFeedback | None:
    if source is None:
        return None

    getter = getattr(source, "get_feedback", None)
    if callable(getter):
        feedback = getter(step_index)
        if feedback is not None:
            return feedback

    latest = getattr(source, "last_feedback", None)
    if latest is not None and int(getattr(latest, "step_index", -1)) == int(step_index):
        return latest

    if callable(source):
        feedback = source(step_index)
        if feedback is not None:
            source_step = int(getattr(feedback, "step_index", -1))
            if source_step == int(step_index):
                return feedback

    return None


class LLMPlanAdaptiveRuntime:
    """
    Run an LLM plan in a closed loop with runtime learning and adaptive replanning.

    The existing ``LLMPlanRuntimeExecutor`` (or ``LLMPlanLiveRuntimeExecutor``)
    remains responsible for executing individual segments. This controller only:

    * observes runtime ``step_recorded`` events;
    * reads provider-neutral runtime feedback already produced elsewhere;
    * requests a replacement plan through ``LLMPlanAdaptiveReplanner``;
    * stops at a safe step boundary before the next action;
    * resumes the same runtime engine with the replacement plan.

    It never executes actions itself and never calls Agent.decide_and_act().
    """

    def __init__(
        self,
        runtime_executor: Any,
        replanner: LLMPlanAdaptiveReplanner,
        *,
        feedback_source: AdaptiveRuntimeFeedbackSource | None = None,
        prevent_global_completed_step_repetition: bool = True,
        max_segments: int | None = None,
        event_observer: AdaptiveRuntimeEventObserver | None = None,
    ) -> None:
        if runtime_executor is None or not callable(getattr(runtime_executor, "execute", None)):
            raise LLMPlanAdaptiveRuntimeConfigurationError(
                "runtime_executor must expose callable execute(...)."
            )
        event_runtime = self._resolve_event_runtime(runtime_executor)
        if event_runtime is None:
            raise LLMPlanAdaptiveRuntimeConfigurationError(
                "runtime_executor must expose add_event_observer/remove_event_observer, "
                "or expose them through runtime_executor.runtime_executor."
            )
        if not isinstance(replanner, LLMPlanAdaptiveReplanner):
            if not callable(getattr(replanner, "replan", None)):
                raise LLMPlanAdaptiveRuntimeConfigurationError(
                    "replanner must expose callable replan(...)."
                )
        if feedback_source is not None and not (
            callable(feedback_source)
            or callable(getattr(feedback_source, "get_feedback", None))
            or hasattr(feedback_source, "last_feedback")
        ):
            raise LLMPlanAdaptiveRuntimeConfigurationError(
                "feedback_source must be callable or expose get_feedback()/last_feedback."
            )
        if max_segments is not None and max_segments < 1:
            raise LLMPlanAdaptiveRuntimeConfigurationError(
                "max_segments must be >= 1 when provided."
            )
        if event_observer is not None and not callable(event_observer):
            raise LLMPlanAdaptiveRuntimeConfigurationError(
                "event_observer must be callable."
            )

        self.runtime_executor = runtime_executor
        self._event_runtime = event_runtime
        self.replanner = replanner
        self.feedback_source = feedback_source
        self.prevent_global_completed_step_repetition = bool(
            prevent_global_completed_step_repetition
        )
        default_segments = int(getattr(replanner, "max_replans", 3)) + 1
        self.max_segments = default_segments if max_segments is None else int(max_segments)
        self.event_observer = event_observer

        self.last_result: LLMPlanAdaptiveRuntimeExecutionReport | None = None
        self.last_feedback: LLMPlanRuntimeLearningFeedback | None = None
        self.last_replanning: LLMPlanAdaptiveReplanningResult | None = None
        self._event_observers: list[AdaptiveRuntimeEventObserver] = []
        self._replan_requested = False
        self._requested_feedback: LLMPlanRuntimeLearningFeedback | None = None
        self._active_plan: LLMPlan | None = None
        self._active_generation = 1
        self._active_segment_index = 0
        self._historical_successful_steps: list[LLMPlanStep] = []
        self._lock_active = False

    @staticmethod
    def _resolve_event_runtime(runtime_executor: Any) -> Any:
        if callable(getattr(runtime_executor, "add_event_observer", None)) and callable(
            getattr(runtime_executor, "remove_event_observer", None)
        ):
            return runtime_executor
        nested = getattr(runtime_executor, "runtime_executor", None)
        if nested is not None and callable(getattr(nested, "add_event_observer", None)) and callable(
            getattr(nested, "remove_event_observer", None)
        ):
            return nested
        return None

    @classmethod
    def for_runtime(
        cls,
        runtime_executor: Any,
        replanner: LLMPlanAdaptiveReplanner,
        **kwargs: Any,
    ) -> "LLMPlanAdaptiveRuntime":
        return cls(runtime_executor, replanner, **kwargs)

    def add_event_observer(self, observer: AdaptiveRuntimeEventObserver) -> None:
        if not callable(observer):
            raise TypeError("observer must be callable.")
        if observer not in self._event_observers:
            self._event_observers.append(observer)

    def remove_event_observer(self, observer: AdaptiveRuntimeEventObserver) -> None:
        try:
            self._event_observers.remove(observer)
        except ValueError:
            pass

    def _emit(self, event: LLMPlanAdaptiveRuntimeEvent) -> None:
        observers: list[AdaptiveRuntimeEventObserver] = []
        if self.event_observer is not None:
            observers.append(self.event_observer)
        observers.extend(tuple(self._event_observers))
        for observer in observers:
            observer(event)

    def _runtime_stop_evaluator(self, original: Callable[..., bool] | None) -> Callable[[str, Mapping[str, Any]], bool]:
        def evaluate(condition: str, context: Mapping[str, Any]) -> bool:
            if condition == _REPLAN_STOP_CONDITION and self._replan_requested:
                return True
            if original is not None:
                return bool(original(condition, context))
            return False

        return evaluate

    def _prepare_segment_plan(self, plan: LLMPlan) -> LLMPlan:
        if _REPLAN_STOP_CONDITION in plan.stop_conditions:
            return plan
        return replace(
            plan,
            stop_conditions=tuple(plan.stop_conditions) + (_REPLAN_STOP_CONDITION,),
        )

    def _on_runtime_event(self, event: Any) -> None:
        phase = getattr(event, "phase", None)
        if phase != "step_recorded":
            return
        record = getattr(event, "record", None)
        if record is None:
            return
        step = getattr(record, "step", None)
        if step is None:
            return

        feedback = _get_feedback(self.feedback_source, int(step.index))
        if feedback is None:
            return

        self.last_feedback = feedback
        self._requested_feedback = feedback
        self._replan_requested = bool(self.replanner.evaluate_trigger(feedback).should_replan)
        if self._replan_requested:
            self._emit(
                LLMPlanAdaptiveRuntimeEvent(
                    phase="replan_requested",
                    generation=self._active_generation,
                    plan=self._active_plan or getattr(event, "plan", None),
                    segment_index=self._active_segment_index,
                    feedback=feedback,
                    world_state=getattr(event, "world_state", None),
                    next_step_index=getattr(event, "next_step_index", None),
                    metadata={
                        "trigger_reason": getattr(self.replanner.last_trigger, "reason", None),
                    },
                )
            )

    def _successful_steps(self, report: Any) -> tuple[LLMPlanStep, ...]:
        return tuple(
            record.step
            for record in getattr(report, "records", ())
            if not getattr(record, "skipped", False) and bool(getattr(record, "success", False))
        )

    def _failed_step_index(self, report: Any) -> int | None:
        records = tuple(getattr(report, "records", ()))
        if not records:
            return None
        last = records[-1]
        if bool(getattr(last, "success", False)):
            return None
        return int(last.step.index)

    def _next_step_index(self, report: Any) -> int:
        records = tuple(getattr(report, "records", ()))
        value = getattr(report, "next_step_index", None)
        if records:
            last = records[-1]
            last_index = int(last.step.index)
            if not bool(getattr(last, "success", False)):
                # A failed/error step is replaced by the new plan. Keep the
                # next runtime index strictly monotonic so feedback/state
                # from the failed generation cannot collide with the
                # replacement generation.
                report_next = last_index + 1 if value is None else int(value)
                return max(last_index + 1, report_next)
            if value is not None:
                return max(last_index + 1, int(value))
            return last_index + 1
        if value is not None:
            return max(1, int(value))
        return 1

    def _enforce_global_safety(self, replacement_plan: LLMPlan) -> None:
        if not self.prevent_global_completed_step_repetition:
            return
        completed_signatures = {
            _step_signature(step) for step in self._historical_successful_steps
        }
        for step in replacement_plan.steps:
            if _step_signature(step) in completed_signatures:
                raise LLMPlanAdaptiveRuntimeSafetyError(
                    "Replacement plan repeats a completed successful step from an earlier generation: "
                    f"action={step.action!r}, purpose={step.purpose!r}."
                )

    def _build_replan(
        self,
        plan: LLMPlan,
        report: Any,
        *,
        goal: Any,
        available_actions: Any,
        metadata: Mapping[str, Any],
    ) -> LLMPlanAdaptiveReplanningResult | None:
        feedback = self._requested_feedback
        if feedback is None or not self._replan_requested:
            return None

        successful = self._successful_steps(report)
        preserved_indices = tuple(step.index for step in successful)
        failed_index = self._failed_step_index(report)
        next_index = self._next_step_index(report)
        generation = self._active_generation + 1

        replan_metadata = dict(metadata)
        replan_metadata.update(
            {
                "stage": "24.14",
                "adaptive_runtime": True,
                "generation": generation,
                "segment_index": self._active_segment_index,
                "historical_successful_step_indices": tuple(
                    step.index for step in self._historical_successful_steps
                ),
                "historical_successful_step_count": len(self._historical_successful_steps),
            }
        )

        try:
            result = self.replanner.replan(
                plan,
                feedback=feedback,
                successful_step_indices=preserved_indices,
                goal=goal,
                world_state=getattr(report, "final_world_state", None),
                failed_step_index=failed_index,
                next_step_index=next_index,
                generation=generation,
                available_actions=available_actions,
                metadata=replan_metadata,
            )
        except Exception as exc:
            raise LLMPlanAdaptiveRuntimeReplanningError(str(exc)) from exc

        if result.replacement_plan is None:
            return result

        replacement = result.replacement_plan
        expected_indices = tuple(range(next_index, next_index + len(replacement.steps)))
        if tuple(step.index for step in replacement.steps) != expected_indices:
            raise LLMPlanAdaptiveRuntimeSafetyError(
                "Replacement plan step indices are not sequentially rebased from the next runtime index."
            )

        self._enforce_global_safety(replacement)
        return result

    def reset(self) -> None:
        self.last_result = None
        self.last_feedback = None
        self.last_replanning = None
        self._replan_requested = False
        self._requested_feedback = None
        self._active_plan = None
        self._active_generation = 1
        self._active_segment_index = 0
        self._historical_successful_steps = []
        self._lock_active = False

    def execute(
        self,
        plan: LLMPlan,
        *,
        goal: Any = None,
        world_state: Any = None,
        metadata: Mapping[str, Any] | None = None,
        available_actions: Any = None,
        **runtime_kwargs: Any,
    ) -> LLMPlanAdaptiveRuntimeExecutionReport:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")
        if self._lock_active:
            raise LLMPlanAdaptiveRuntimeConfigurationError(
                "An adaptive runtime execution is already active."
            )

        resolved_goal = plan.goal if goal is None else goal
        segment_plan = plan
        current_world_state = world_state
        segment_reports: list[Any] = []
        replans: list[LLMPlanAdaptiveReplanningResult] = []
        all_successful_indices: list[int] = []
        all_failed_indices: list[int] = []
        final_stop_reason: str | None = None
        final_next_step_index: int | None = None
        final_completed = False
        final_stopped = False
        final_success = False
        self.reset()
        self._lock_active = True
        original_stop_evaluator = getattr(self._event_runtime, "stop_condition_evaluator", None)

        def observer(event: Any) -> None:
            self._on_runtime_event(event)

        try:
            self._event_runtime.add_event_observer(observer)
            self._event_runtime.stop_condition_evaluator = self._runtime_stop_evaluator(
                original_stop_evaluator
            )

            for segment_index in range(self.max_segments):
                self._active_segment_index = segment_index
                self._active_plan = self._prepare_segment_plan(segment_plan)
                self._active_generation = segment_index + 1
                self._replan_requested = False
                self._requested_feedback = None

                runtime_metadata = dict(metadata or {})
                runtime_metadata.update(
                    {
                        "stage": "24.14",
                        "adaptive_runtime": True,
                        "adaptive_generation": self._active_generation,
                        "adaptive_segment_index": segment_index,
                    }
                )

                self._emit(
                    LLMPlanAdaptiveRuntimeEvent(
                        phase="segment_started",
                        generation=self._active_generation,
                        plan=self._active_plan,
                        segment_index=segment_index,
                        world_state=current_world_state,
                        metadata=runtime_metadata,
                    )
                )

                runtime_result = self.runtime_executor.execute(
                    self._active_plan,
                    goal=resolved_goal,
                    world_state=current_world_state,
                    available_actions=available_actions,
                    metadata=runtime_metadata,
                    **runtime_kwargs,
                )
                report = _get_report(runtime_result)
                segment_reports.append(runtime_result)
                current_world_state = getattr(report, "final_world_state", current_world_state)
                final_stop_reason = getattr(report, "stop_reason", None)
                final_next_step_index = getattr(report, "next_step_index", None)
                final_completed = bool(getattr(report, "completed", False))
                final_stopped = bool(getattr(report, "stopped", False))
                final_success = bool(getattr(report, "success", False))

                successful_steps = self._successful_steps(report)
                self._historical_successful_steps.extend(successful_steps)
                all_successful_indices.extend(step.index for step in successful_steps)
                all_failed_indices.extend(
                    int(record.step.index)
                    for record in getattr(report, "records", ())
                    if not getattr(record, "skipped", False)
                    and not bool(getattr(record, "success", False))
                )

                self._emit(
                    LLMPlanAdaptiveRuntimeEvent(
                        phase="segment_finished",
                        generation=self._active_generation,
                        plan=self._active_plan,
                        segment_index=segment_index,
                        report=runtime_result,
                        feedback=self.last_feedback,
                        world_state=current_world_state,
                        next_step_index=final_next_step_index,
                        metadata={
                            "completed": final_completed,
                            "stopped": final_stopped,
                            "success": final_success,
                            "stop_reason": final_stop_reason,
                        },
                    )
                )

                result = self._build_replan(
                    segment_plan,
                    report,
                    goal=resolved_goal,
                    available_actions=available_actions,
                    metadata=runtime_metadata,
                )

                if result is None or result.replacement_plan is None:
                    self.last_replanning = result
                    break

                replans.append(result)
                self.last_replanning = result
                self._emit(
                    LLMPlanAdaptiveRuntimeEvent(
                        phase="replanned",
                        generation=self._active_generation + 1,
                        plan=result.replacement_plan,
                        segment_index=segment_index,
                        report=runtime_result,
                        feedback=self._requested_feedback,
                        replanning=result,
                        world_state=current_world_state,
                        next_step_index=result.next_step_index,
                        metadata=result.metadata,
                    )
                )
                segment_plan = result.replacement_plan

                if segment_index + 1 >= self.max_segments:
                    raise LLMPlanAdaptiveRuntimeReplanningError(
                        "Adaptive runtime reached max_segments before the replacement plan could execute."
                    )

            else:
                raise LLMPlanAdaptiveRuntimeReplanningError(
                    "Adaptive runtime exhausted max_segments without reaching a terminal result."
                )

            final_plan = segment_plan
            aggregate_metadata = dict(metadata or {})
            aggregate_metadata.update(
                {
                    "stage": "24.14",
                    "adaptive_runtime": True,
                    "segment_count": len(segment_reports),
                    "replan_count": len(replans),
                    "successful_step_count": len(all_successful_indices),
                    "failed_step_count": len(all_failed_indices),
                }
            )
            aggregate = LLMPlanAdaptiveRuntimeExecutionReport(
                initial_plan=plan,
                final_plan=final_plan,
                segment_reports=tuple(segment_reports),
                replans=tuple(replans),
                completed=final_completed,
                stopped=final_stopped,
                success=final_success,
                stop_reason=final_stop_reason,
                final_world_state=current_world_state,
                next_step_index=final_next_step_index,
                completed_successful_step_indices=tuple(dict.fromkeys(all_successful_indices)),
                failed_step_indices=tuple(dict.fromkeys(all_failed_indices)),
                generation_count=len(segment_reports),
                metadata=aggregate_metadata,
            )
            self.last_result = aggregate
            self._emit(
                LLMPlanAdaptiveRuntimeEvent(
                    phase="completed" if aggregate.success else "finished",
                    generation=aggregate.generation_count,
                    plan=aggregate.final_plan,
                    segment_index=max(0, aggregate.segment_count - 1),
                    report=aggregate,
                    feedback=self.last_feedback,
                    replanning=self.last_replanning,
                    world_state=aggregate.final_world_state,
                    next_step_index=aggregate.next_step_index,
                    metadata=aggregate.metadata,
                )
            )
            return aggregate
        except LLMPlanAdaptiveRuntimeError:
            raise
        except Exception as exc:
            raise LLMPlanAdaptiveRuntimeExecutionError(str(exc)) from exc
        finally:
            try:
                self._event_runtime.remove_event_observer(observer)
            finally:
                self._event_runtime.stop_condition_evaluator = original_stop_evaluator
                self._active_plan = None
                self._lock_active = False


__all__ = [
    "LLMPlanAdaptiveRuntimeError",
    "LLMPlanAdaptiveRuntimeConfigurationError",
    "LLMPlanAdaptiveRuntimeExecutionError",
    "LLMPlanAdaptiveRuntimeSafetyError",
    "LLMPlanAdaptiveRuntimeReplanningError",
    "AdaptiveRuntimeFeedbackSource",
    "AdaptiveRuntimeEventObserver",
    "LLMPlanAdaptiveRuntimeEvent",
    "LLMPlanAdaptiveRuntimeExecutionReport",
    "LLMPlanAdaptiveRuntime",
]
