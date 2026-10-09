from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .goal_aware_adaptive_replanning import (
    GoalAwareAdaptiveReplanningController,
    GoalAwareAdaptiveReplanningResult,
)
from .goal_conflict_resolution import (
    GoalConflictRelation,
    GoalConflictResolutionResult,
    GoalConflictResolver,
)
from .goal_management import GoalLifecycleManager, GoalStatus, ManagedGoal
from .goal_progress_evaluation import (
    GoalAchievementEvaluation,
    GoalAchievementEvidence,
    GoalProgressAchievementEvaluator,
)
from .llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback
from .multi_goal_arbitration import (
    MultiGoalArbiter,
    MultiGoalArbitrationApplier,
    MultiGoalArbitrationDecision,
)
from .planning import LLMPlan


class ClosedLoopMultiGoalRuntimeError(RuntimeError):
    """Base error for closed-loop multi-goal runtime coordination."""


class ClosedLoopMultiGoalRuntimeConfigurationError(
    ClosedLoopMultiGoalRuntimeError
):
    """Raised when runtime dependencies or configuration are invalid."""


class ClosedLoopMultiGoalRuntimeInputError(ClosedLoopMultiGoalRuntimeError):
    """Raised when runtime inputs are invalid."""


class ClosedLoopMultiGoalRuntimeExecutionError(ClosedLoopMultiGoalRuntimeError):
    """Raised when the underlying execution engine fails."""


class ClosedLoopMultiGoalRuntimeCoordinationError(ClosedLoopMultiGoalRuntimeError):
    """Raised when goal arbitration/conflict coordination is inconsistent."""


class ClosedLoopMultiGoalRuntimeReplanningError(ClosedLoopMultiGoalRuntimeError):
    """Raised when a goal-aware replan is requested but cannot be produced."""


RuntimeExecutor = Any
UrgencyProvider = Callable[
    [tuple[ManagedGoal, ...], Mapping[str, Any]], Mapping[str, float]
]
RelationsProvider = Callable[
    [tuple[ManagedGoal, ...], Mapping[str, Any]], Iterable[GoalConflictRelation | Mapping[str, Any]]
]
EvidenceProvider = Callable[
    [ManagedGoal, Any, Any, Mapping[str, Any]], Iterable[GoalAchievementEvidence | Mapping[str, Any] | Any]
]
FeedbackProvider = Callable[
    [ManagedGoal, Any, GoalAchievementEvaluation | None, Mapping[str, Any]], Any
]
BlockedGoalProvider = Callable[
    [tuple[ManagedGoal, ...], Mapping[str, Any]], Iterable[str]
]
RuntimeObserver = Callable[["ClosedLoopMultiGoalRuntimeEvent"], None]


@dataclass(frozen=True)
class ClosedLoopMultiGoalRuntimeEvent:
    """Immutable event emitted by the coordinator."""

    phase: str
    generation: int
    goal_id: str | None = None
    plan: LLMPlan | None = None
    runtime_result: Any = None
    evaluation: GoalAchievementEvaluation | None = None
    arbitration: MultiGoalArbitrationDecision | None = None
    conflict_resolution: GoalConflictResolutionResult | None = None
    replanning: GoalAwareAdaptiveReplanningResult | None = None
    world_state: Any = None
    next_goal_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.phase).strip():
            raise ClosedLoopMultiGoalRuntimeInputError("phase must not be empty.")
        if self.generation < 1:
            raise ClosedLoopMultiGoalRuntimeInputError("generation must be >= 1.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ClosedLoopMultiGoalRuntimeSegment:
    """Record of one execution segment and its coordination outcome."""

    generation: int
    goal_id: str
    plan: LLMPlan
    runtime_result: Any
    evaluation: GoalAchievementEvaluation | None
    arbitration: MultiGoalArbitrationDecision | None
    conflict_resolution: GoalConflictResolutionResult | None
    replanning: GoalAwareAdaptiveReplanningResult | None
    switched_from_goal_id: str | None
    success: bool
    world_state: Any
    next_goal_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ClosedLoopMultiGoalRuntimeInputError("generation must be >= 1.")
        if not self.goal_id.strip():
            raise ClosedLoopMultiGoalRuntimeInputError("goal_id must not be empty.")
        if not isinstance(self.plan, LLMPlan):
            raise ClosedLoopMultiGoalRuntimeInputError("plan must be LLMPlan.")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ClosedLoopMultiGoalRuntimeExecutionReport:
    """Aggregate immutable result for one closed-loop multi-goal execution."""

    initial_goal_id: str
    final_goal_id: str | None
    initial_plan: LLMPlan
    final_plan: LLMPlan
    segments: tuple[ClosedLoopMultiGoalRuntimeSegment, ...]
    evaluations: Mapping[str, GoalAchievementEvaluation]
    completed_goal_ids: tuple[str, ...]
    failed_goal_ids: tuple[str, ...]
    final_world_state: Any
    completed: bool
    stopped: bool
    success: bool
    stop_reason: str | None
    goal_switch_count: int
    replan_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def goal_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(segment.goal_id for segment in self.segments))

    @property
    def all_goals_completed(self) -> bool:
        return bool(self.completed_goal_ids) and not self.failed_goal_ids


class ClosedLoopMultiGoalRuntime:
    """Coordinate multiple goals while delegating action execution to one existing runtime engine.

    This is an orchestration boundary, not a second execution engine. The supplied
    ``runtime_executor`` is always responsible for executing actions. This layer only
    coordinates goal selection, progress evaluation, conflict handling, and goal-aware
    replanning between execution segments.
    """

    def __init__(
        self,
        runtime_executor: RuntimeExecutor,
        goal_manager: GoalLifecycleManager,
        *,
        arbiter: MultiGoalArbiter | None = None,
        conflict_resolver: GoalConflictResolver | None = None,
        progress_evaluator: GoalProgressAchievementEvaluator | None = None,
        arbitration_applier: MultiGoalArbitrationApplier | None = None,
        goal_replanner: GoalAwareAdaptiveReplanningController | None = None,
        max_segments: int = 20,
        max_goal_switches: int = 10,
        auto_complete_achieved_goals: bool = True,
        stop_on_unresolved_conflict: bool = True,
        stop_on_execution_failure_without_replan: bool = True,
    ) -> None:
        if not callable(getattr(runtime_executor, "execute", None)):
            raise ClosedLoopMultiGoalRuntimeConfigurationError(
                "runtime_executor must expose callable execute(...)."
            )
        if not isinstance(goal_manager, GoalLifecycleManager):
            raise ClosedLoopMultiGoalRuntimeConfigurationError(
                "goal_manager must be GoalLifecycleManager."
            )
        if goal_replanner is not None and not callable(getattr(goal_replanner, "replan", None)):
            raise ClosedLoopMultiGoalRuntimeConfigurationError(
                "goal_replanner must expose callable replan(...)."
            )
        if max_segments < 1:
            raise ClosedLoopMultiGoalRuntimeConfigurationError("max_segments must be >= 1.")
        if max_goal_switches < 0:
            raise ClosedLoopMultiGoalRuntimeConfigurationError(
                "max_goal_switches must be >= 0."
            )

        self.runtime_executor = runtime_executor
        self.goal_manager = goal_manager
        self.arbiter = arbiter or MultiGoalArbiter()
        self.conflict_resolver = conflict_resolver or GoalConflictResolver()
        self.progress_evaluator = progress_evaluator or GoalProgressAchievementEvaluator()
        self.arbitration_applier = arbitration_applier or MultiGoalArbitrationApplier()
        self.goal_replanner = goal_replanner
        self.max_segments = int(max_segments)
        self.max_goal_switches = int(max_goal_switches)
        self.auto_complete_achieved_goals = bool(auto_complete_achieved_goals)
        self.stop_on_unresolved_conflict = bool(stop_on_unresolved_conflict)
        self.stop_on_execution_failure_without_replan = bool(
            stop_on_execution_failure_without_replan
        )

        self._observers: list[RuntimeObserver] = []
        self.last_result: ClosedLoopMultiGoalRuntimeExecutionReport | None = None
        self.last_evaluations: dict[str, GoalAchievementEvaluation] = {}
        self.last_arbitration: MultiGoalArbitrationDecision | None = None
        self.last_conflict_resolution: GoalConflictResolutionResult | None = None
        self.last_replanning: GoalAwareAdaptiveReplanningResult | None = None
        self._last_feedback_signature: tuple[Any, ...] | None = None

    def add_event_observer(self, observer: RuntimeObserver) -> None:
        if not callable(observer):
            raise ClosedLoopMultiGoalRuntimeConfigurationError(
                "observer must be callable."
            )
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_event_observer(self, observer: RuntimeObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def reset(self) -> None:
        self.last_result = None
        self.last_evaluations = {}
        self.last_arbitration = None
        self.last_conflict_resolution = None
        self.last_replanning = None
        self._last_feedback_signature = None

    def _emit(self, event: ClosedLoopMultiGoalRuntimeEvent) -> None:
        for observer in tuple(self._observers):
            observer(event)

    @staticmethod
    def _result_report(runtime_result: Any) -> Any:
        return getattr(runtime_result, "report", runtime_result)

    @staticmethod
    def _successful_step_indices(runtime_result: Any) -> tuple[int, ...]:
        report = ClosedLoopMultiGoalRuntime._result_report(runtime_result)
        return tuple(
            int(record.step.index)
            for record in getattr(report, "records", ())
            if not getattr(record, "skipped", False)
            and bool(getattr(record, "success", False))
        )

    @staticmethod
    def _failed_step_index(runtime_result: Any) -> int | None:
        report = ClosedLoopMultiGoalRuntime._result_report(runtime_result)
        records = tuple(getattr(report, "records", ()))
        for record in reversed(records):
            if not getattr(record, "skipped", False) and not bool(getattr(record, "success", False)):
                return int(record.step.index)
        return None

    @staticmethod
    def _next_step_index(runtime_result: Any) -> int:
        report = ClosedLoopMultiGoalRuntime._result_report(runtime_result)
        value = getattr(report, "next_step_index", None)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
        records = tuple(getattr(report, "records", ()))
        if records:
            return max(1, int(records[-1].step.index) + 1)
        return 1

    @staticmethod
    def _is_success(runtime_result: Any) -> bool:
        return bool(getattr(ClosedLoopMultiGoalRuntime._result_report(runtime_result), "success", False))

    @staticmethod
    def _final_world_state(runtime_result: Any, current: Any) -> Any:
        value = getattr(ClosedLoopMultiGoalRuntime._result_report(runtime_result), "final_world_state", None)
        return current if value is None else value

    @staticmethod
    def _is_complete(runtime_result: Any) -> bool:
        return bool(getattr(ClosedLoopMultiGoalRuntime._result_report(runtime_result), "completed", False))

    @staticmethod
    def _remaining_plan_indices(plan: LLMPlan, successful_indices: Sequence[int]) -> tuple[int, ...]:
        done = set(int(item) for item in successful_indices)
        return tuple(step.index for step in plan.steps if step.index not in done)

    @staticmethod
    def _feedback_signature(feedback: Any) -> tuple[Any, ...]:
        if feedback is None:
            return ()
        return (
            getattr(feedback, "step_index", None),
            getattr(feedback, "action", None),
            getattr(feedback, "diagnosis", None),
            getattr(feedback, "change_required", None),
            getattr(feedback, "continue_recommended", None),
            getattr(feedback, "progress_delta", None),
            getattr(feedback, "risk_level", None),
        )

    @staticmethod
    def _normalize_evidence(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, (GoalAchievementEvidence, Mapping)):
            return (value,)
        try:
            return tuple(value)
        except TypeError:
            return (value,)

    def _build_goal_context(
        self,
        *,
        urgency_provider: UrgencyProvider | None,
        relations_provider: RelationsProvider | None,
        blocked_goal_ids_provider: BlockedGoalProvider | None,
        metadata: Mapping[str, Any],
    ) -> tuple[MultiGoalArbitrationDecision, GoalConflictResolutionResult]:
        goals = self.goal_manager.list_goals()
        urgency = (
            dict(urgency_provider(goals, metadata))
            if urgency_provider is not None
            else {}
        )
        blocked = (
            tuple(blocked_goal_ids_provider(goals, metadata))
            if blocked_goal_ids_provider is not None
            else ()
        )
        relations = (
            tuple(relations_provider(goals, metadata))
            if relations_provider is not None
            else ()
        )

        arbitration = self.arbiter.arbitrate_manager(
            self.goal_manager,
            urgency=urgency,
            evaluations=self.last_evaluations,
            blocked_goal_ids=blocked,
        )
        conflict = self.conflict_resolver.resolve_manager(
            self.goal_manager,
            relations=relations,
            urgency=urgency,
            arbitration=arbitration,
            blocked_goal_ids=blocked,
        )

        effective_blocked = set(str(item).strip() for item in blocked)
        effective_blocked.update(conflict.deferred_goal_ids)
        effective_blocked.update(conflict.blocked_goal_ids)
        if (
            arbitration.selected_goal_id is not None
            and arbitration.selected_goal_id in effective_blocked
        ):
            arbitration = self.arbiter.arbitrate_manager(
                self.goal_manager,
                urgency=urgency,
                evaluations=self.last_evaluations,
                blocked_goal_ids=effective_blocked,
            )
            conflict = self.conflict_resolver.resolve_manager(
                self.goal_manager,
                relations=relations,
                urgency=urgency,
                arbitration=arbitration,
                blocked_goal_ids=effective_blocked,
            )

        self.last_arbitration = arbitration
        self.last_conflict_resolution = conflict
        return arbitration, conflict

    def _apply_arbitration(
        self,
        arbitration: MultiGoalArbitrationDecision,
    ) -> ManagedGoal | None:
        if arbitration.selected_goal_id is None:
            return None
        try:
            return self.arbitration_applier.apply(self.goal_manager, arbitration)
        except Exception as exc:
            raise ClosedLoopMultiGoalRuntimeCoordinationError(str(exc)) from exc

    def _evaluate_goal(
        self,
        goal: ManagedGoal,
        runtime_result: Any,
        world_state: Any,
        *,
        evidence_provider: EvidenceProvider | None,
        metadata: Mapping[str, Any],
    ) -> GoalAchievementEvaluation | None:
        if evidence_provider is None:
            return None
        evidence = evidence_provider(goal, runtime_result, world_state, metadata)
        normalized = self._normalize_evidence(evidence)
        evaluation = self.progress_evaluator.evaluate(goal, normalized)
        self.last_evaluations[goal.goal_id] = evaluation
        if evaluation.progress_measured and not goal.status.terminal:
            progress = max(goal.progress, evaluation.measured_progress)
            self.goal_manager.update_progress(goal.goal_id, progress)
        if (
            self.auto_complete_achieved_goals
            and evaluation.achieved
            and self.goal_manager.get_goal(goal.goal_id).status == GoalStatus.ACTIVE
        ):
            self.goal_manager.complete(goal.goal_id)
        return evaluation

    @staticmethod
    def _all_goals_terminal(manager: GoalLifecycleManager) -> bool:
        goals = manager.list_goals()
        return bool(goals) and all(goal.status.terminal for goal in goals)

    @staticmethod
    def _all_goals_completed(manager: GoalLifecycleManager) -> bool:
        goals = manager.list_goals()
        return bool(goals) and all(goal.status == GoalStatus.COMPLETED for goal in goals)

    @staticmethod
    def _validate_callback(value: Any, name: str) -> None:
        if value is not None and not callable(value):
            raise ClosedLoopMultiGoalRuntimeInputError(f"{name} must be callable when provided.")

    def execute(
        self,
        plan: LLMPlan,
        *,
        initial_goal_id: str | None = None,
        world_state: Any = None,
        available_actions: Any = None,
        urgency_provider: UrgencyProvider | None = None,
        relations_provider: RelationsProvider | None = None,
        evidence_provider: EvidenceProvider | None = None,
        feedback_provider: FeedbackProvider | None = None,
        blocked_goal_ids_provider: BlockedGoalProvider | None = None,
        metadata: Mapping[str, Any] | None = None,
        **runtime_kwargs: Any,
    ) -> ClosedLoopMultiGoalRuntimeExecutionReport:
        if not isinstance(plan, LLMPlan):
            raise ClosedLoopMultiGoalRuntimeInputError("plan must be an LLMPlan.")
        self._validate_callback(urgency_provider, "urgency_provider")
        self._validate_callback(relations_provider, "relations_provider")
        self._validate_callback(evidence_provider, "evidence_provider")
        self._validate_callback(feedback_provider, "feedback_provider")
        self._validate_callback(blocked_goal_ids_provider, "blocked_goal_ids_provider")
        if not self.goal_manager.list_goals():
            raise ClosedLoopMultiGoalRuntimeCoordinationError("Goal manager contains no goals.")
        if self.goal_replanner is None and len(self.goal_manager.list_goals(include_terminal=False)) > 1:
            # Multi-goal execution still works without a replanner when no switch is needed,
            # but an actual goal switch requiring a different plan must be rejected safely.
            allow_switch_without_replan = False
        else:
            allow_switch_without_replan = True

        self.reset()
        base_metadata = {"stage": "25.8", "closed_loop_multi_goal_runtime": True, **dict(metadata or {})}

        if initial_goal_id is not None:
            current_goal = self.goal_manager.set_current_goal(str(initial_goal_id).strip())
        elif self.goal_manager.current_goal_id is not None:
            current_goal = self.goal_manager.get_goal(self.goal_manager.current_goal_id)
        else:
            selection = self.goal_manager.select_next_goal()
            current_goal = (
                self.goal_manager.get_goal(selection.selected_goal_id)
                if selection.selected_goal_id is not None
                else None
            )
        if current_goal is None or current_goal.status.terminal:
            raise ClosedLoopMultiGoalRuntimeCoordinationError("No active goal is available for execution.")
        if plan.goal.strip().casefold() != current_goal.description.strip().casefold():
            raise ClosedLoopMultiGoalRuntimeInputError(
                "Initial plan goal must match the selected goal description."
            )

        initial_goal = current_goal
        segment_plan = plan
        current_world_state = world_state
        segments: list[ClosedLoopMultiGoalRuntimeSegment] = []
        completed_ids: list[str] = []
        failed_ids: list[str] = []
        goal_switch_count = 0
        replan_count = 0
        stop_reason: str | None = None
        stopped = False
        completed = False
        success = False

        for generation in range(1, self.max_segments + 1):
            current_goal = self.goal_manager.get_goal(self.goal_manager.current_goal_id)
            arbitration, conflict = self._build_goal_context(
                urgency_provider=urgency_provider,
                relations_provider=relations_provider,
                blocked_goal_ids_provider=blocked_goal_ids_provider,
                metadata=base_metadata,
            )
            self._emit(
                ClosedLoopMultiGoalRuntimeEvent(
                    phase="coordinated",
                    generation=generation,
                    goal_id=current_goal.goal_id,
                    plan=segment_plan,
                    arbitration=arbitration,
                    conflict_resolution=conflict,
                    world_state=current_world_state,
                    next_goal_id=arbitration.selected_goal_id,
                    metadata=base_metadata,
                )
            )

            if conflict.unresolved_conflict_ids and self.stop_on_unresolved_conflict:
                stop_reason = "unresolved_goal_conflict"
                stopped = True
                break

            selected_goal = self._apply_arbitration(arbitration)
            if selected_goal is None:
                stop_reason = "no_eligible_goal"
                stopped = True
                break
            if selected_goal.goal_id != current_goal.goal_id:
                previous_goal_id = current_goal.goal_id
                goal_switch_count += 1
                if goal_switch_count > self.max_goal_switches:
                    stop_reason = "max_goal_switches_reached"
                    stopped = True
                    break
                if not allow_switch_without_replan and self.goal_replanner is None:
                    stop_reason = "goal_switch_requires_replanner"
                    stopped = True
                    break
                current_goal = selected_goal
                self._emit(
                    ClosedLoopMultiGoalRuntimeEvent(
                        phase="goal_switched",
                        generation=generation,
                        goal_id=current_goal.goal_id,
                        plan=segment_plan,
                        arbitration=arbitration,
                        conflict_resolution=conflict,
                        world_state=current_world_state,
                        next_goal_id=current_goal.goal_id,
                        metadata={**base_metadata, "previous_goal_id": previous_goal_id},
                    )
                )

            # A segment can only execute a plan for its selected goal.
            if segment_plan.goal.strip().casefold() != current_goal.description.strip().casefold():
                if self.goal_replanner is None:
                    stop_reason = "goal_changed_without_replanner"
                    stopped = True
                    break
                try:
                    replan_feedback = None
                    replanning = self.goal_replanner.replan(
                        segment_plan,
                        current_goal,
                        available_goals=self.goal_manager.list_goals(),
                        arbitration=arbitration,
                        conflict_resolution=conflict,
                        feedback=replan_feedback,
                        successful_step_indices=(),
                        world_state=current_world_state,
                        failed_step_index=None,
                        next_step_index=1,
                        generation=generation,
                        available_actions=available_actions,
                        original_goal_id=arbitration.previous_goal_id,
                        metadata={**base_metadata, "goal_switch": True},
                    )
                except TypeError as exc:
                    # Goal-aware controller normally requires feedback only when a runtime
                    # trigger exists; the controller itself accepts the goal-change path.
                    raise ClosedLoopMultiGoalRuntimeReplanningError(str(exc)) from exc
                if replanning.replacement_plan is None:
                    stop_reason = "goal_change_replan_not_produced"
                    stopped = True
                    break
                segment_plan = replanning.replacement_plan
                replan_count += 1
                self.last_replanning = replanning
                self._emit(
                    ClosedLoopMultiGoalRuntimeEvent(
                        phase="replanned",
                        generation=generation,
                        goal_id=current_goal.goal_id,
                        plan=segment_plan,
                        arbitration=arbitration,
                        conflict_resolution=conflict,
                        replanning=replanning,
                        world_state=current_world_state,
                        metadata={**base_metadata, "trigger": "goal_switch"},
                    )
                )

            self._emit(
                ClosedLoopMultiGoalRuntimeEvent(
                    phase="segment_started",
                    generation=generation,
                    goal_id=current_goal.goal_id,
                    plan=segment_plan,
                    world_state=current_world_state,
                    metadata=base_metadata,
                )
            )
            try:
                runtime_result = self.runtime_executor.execute(
                    segment_plan,
                    goal=current_goal.description,
                    world_state=current_world_state,
                    available_actions=available_actions,
                    metadata={**base_metadata, "generation": generation, "goal_id": current_goal.goal_id},
                    **runtime_kwargs,
                )
            except Exception as exc:
                raise ClosedLoopMultiGoalRuntimeExecutionError(str(exc)) from exc

            current_world_state = self._final_world_state(runtime_result, current_world_state)
            segment_success = self._is_success(runtime_result)
            evaluation = self._evaluate_goal(
                current_goal,
                runtime_result,
                current_world_state,
                evidence_provider=evidence_provider,
                metadata={**base_metadata, "generation": generation, "goal_id": current_goal.goal_id},
            )
            current_after_evaluation = self.goal_manager.get_goal(current_goal.goal_id)
            if current_after_evaluation.status == GoalStatus.COMPLETED:
                if current_goal.goal_id not in completed_ids:
                    completed_ids.append(current_goal.goal_id)
            elif current_after_evaluation.status == GoalStatus.FAILED:
                if current_goal.goal_id not in failed_ids:
                    failed_ids.append(current_goal.goal_id)

            feedback = (
                feedback_provider(
                    current_after_evaluation,
                    runtime_result,
                    evaluation,
                    {**base_metadata, "generation": generation, "goal_id": current_goal.goal_id},
                )
                if feedback_provider is not None
                else None
            )
            if feedback is not None and not isinstance(feedback, LLMPlanRuntimeLearningFeedback):
                raise ClosedLoopMultiGoalRuntimeInputError(
                    "feedback_provider must return LLMPlanRuntimeLearningFeedback or None."
                )
            feedback_signature = self._feedback_signature(feedback)
            fresh_feedback = feedback is not None and feedback_signature != self._last_feedback_signature
            if fresh_feedback:
                self._last_feedback_signature = feedback_signature

            segment_replanning: GoalAwareAdaptiveReplanningResult | None = None
            next_goal_id: str | None = None
            switched_from_goal_id: str | None = None

            arbitration_after, conflict_after = self._build_goal_context(
                urgency_provider=urgency_provider,
                relations_provider=relations_provider,
                blocked_goal_ids_provider=blocked_goal_ids_provider,
                metadata={**base_metadata, "post_execution": True},
            )
            selected_after = self._apply_arbitration(arbitration_after)
            if selected_after is not None:
                next_goal_id = selected_after.goal_id
            if next_goal_id != current_goal.goal_id and next_goal_id is not None:
                switched_from_goal_id = current_goal.goal_id
                goal_switch_count += 1
                if goal_switch_count > self.max_goal_switches:
                    stop_reason = "max_goal_switches_reached"
                    stopped = True

            self._emit(
                ClosedLoopMultiGoalRuntimeEvent(
                    phase="segment_finished",
                    generation=generation,
                    goal_id=current_goal.goal_id,
                    plan=segment_plan,
                    runtime_result=runtime_result,
                    evaluation=evaluation,
                    arbitration=arbitration_after,
                    conflict_resolution=conflict_after,
                    world_state=current_world_state,
                    next_goal_id=next_goal_id,
                    metadata={
                        **base_metadata,
                        "success": segment_success,
                        "feedback_present": feedback is not None,
                    },
                )
            )

            segment = ClosedLoopMultiGoalRuntimeSegment(
                generation=generation,
                goal_id=current_goal.goal_id,
                plan=segment_plan,
                runtime_result=runtime_result,
                evaluation=evaluation,
                arbitration=arbitration_after,
                conflict_resolution=conflict_after,
                replanning=None,
                switched_from_goal_id=switched_from_goal_id,
                success=segment_success,
                world_state=current_world_state,
                next_goal_id=next_goal_id,
                metadata={"feedback": feedback},
            )

            if stopped and stop_reason == "max_goal_switches_reached":
                segments.append(segment)
                break

            # Terminal success is determined only by goal lifecycle, not by a single
            # action success. This avoids treating successful execution of one plan
            # segment as proof that the entire multi-goal workload is complete.
            if self._all_goals_completed(self.goal_manager):
                completed = True
                stopped = False
                success = True
                segments.append(segment)
                break

            if self._all_goals_terminal(self.goal_manager):
                completed = True
                stopped = False
                success = False
                stop_reason = "not_all_goals_completed"
                segments.append(segment)
                break

            should_replan = False
            if fresh_feedback and self.goal_replanner is not None:
                trigger = self.goal_replanner.adaptive_replanner.evaluate_trigger(feedback)
                should_replan = bool(trigger.should_replan)
            if next_goal_id is not None and next_goal_id != current_goal.goal_id and not stopped:
                should_replan = True

            if should_replan:
                if self.goal_replanner is None:
                    stop_reason = "replan_required_without_replanner"
                    stopped = True
                    segments.append(segment)
                    break
                next_goal = self.goal_manager.get_goal(next_goal_id or current_goal.goal_id)
                successful_indices = self._successful_step_indices(runtime_result)
                failed_index = self._failed_step_index(runtime_result)
                next_index = self._next_step_index(runtime_result)
                try:
                    segment_replanning = self.goal_replanner.replan(
                        segment_plan,
                        next_goal,
                        available_goals=self.goal_manager.list_goals(),
                        arbitration=arbitration_after,
                        conflict_resolution=conflict_after,
                        feedback=feedback,
                        successful_step_indices=successful_indices,
                        world_state=current_world_state,
                        failed_step_index=failed_index,
                        next_step_index=next_index,
                        generation=generation + 1,
                        available_actions=available_actions,
                        original_goal_id=current_goal.goal_id,
                        metadata={**base_metadata, "post_execution_replan": True},
                    )
                except Exception as exc:
                    raise ClosedLoopMultiGoalRuntimeReplanningError(str(exc)) from exc
                if segment_replanning.replacement_plan is None:
                    stop_reason = "replacement_plan_not_produced"
                    stopped = True
                    segments.append(
                        ClosedLoopMultiGoalRuntimeSegment(
                            **{**segment.__dict__, "replanning": segment_replanning}
                        )
                    )
                    break
                replan_count += 1
                self.last_replanning = segment_replanning
                segment = ClosedLoopMultiGoalRuntimeSegment(
                    **{**segment.__dict__, "replanning": segment_replanning}
                )
                segment_plan = segment_replanning.replacement_plan
                self.goal_manager.set_current_goal(next_goal.goal_id)
                self._emit(
                    ClosedLoopMultiGoalRuntimeEvent(
                        phase="replanned",
                        generation=generation + 1,
                        goal_id=next_goal.goal_id,
                        plan=segment_plan,
                        runtime_result=runtime_result,
                        evaluation=evaluation,
                        arbitration=arbitration_after,
                        conflict_resolution=conflict_after,
                        replanning=segment_replanning,
                        world_state=current_world_state,
                        next_goal_id=next_goal.goal_id,
                        metadata=base_metadata,
                    )
                )
                segments.append(segment)
                continue

            segments.append(segment)
            if not segment_success and feedback is None and self.stop_on_execution_failure_without_replan:
                stop_reason = "execution_failure_without_replan"
                stopped = True
                break

            if segment_success:
                stop_reason = "segment_succeeded_without_goal_completion"
            else:
                stop_reason = "segment_finished_without_replan"
            stopped = True
            break
        else:
            stop_reason = "max_segments_reached"
            stopped = True

        final_goal_id = self.goal_manager.current_goal_id
        snapshot = self.goal_manager.snapshot()
        report = ClosedLoopMultiGoalRuntimeExecutionReport(
            initial_goal_id=initial_goal.goal_id,
            final_goal_id=final_goal_id,
            initial_plan=plan,
            final_plan=segment_plan,
            segments=tuple(segments),
            evaluations=dict(self.last_evaluations),
            completed_goal_ids=tuple(completed_ids),
            failed_goal_ids=tuple(failed_ids),
            final_world_state=current_world_state,
            completed=completed,
            stopped=stopped,
            success=success,
            stop_reason=stop_reason,
            goal_switch_count=goal_switch_count,
            replan_count=replan_count,
            metadata={
                **base_metadata,
                "goal_count": len(snapshot.goals),
                "terminal_goal_count": sum(goal.status.terminal for goal in snapshot.goals),
            },
        )
        self.last_result = report
        self._emit(
            ClosedLoopMultiGoalRuntimeEvent(
                phase="completed" if report.success else "finished",
                generation=len(segments) or 1,
                goal_id=final_goal_id,
                plan=segment_plan,
                world_state=current_world_state,
                next_goal_id=final_goal_id,
                metadata=report.metadata,
            )
        )
        return report


__all__ = [
    "ClosedLoopMultiGoalRuntimeError",
    "ClosedLoopMultiGoalRuntimeConfigurationError",
    "ClosedLoopMultiGoalRuntimeInputError",
    "ClosedLoopMultiGoalRuntimeExecutionError",
    "ClosedLoopMultiGoalRuntimeCoordinationError",
    "ClosedLoopMultiGoalRuntimeReplanningError",
    "ClosedLoopMultiGoalRuntimeEvent",
    "ClosedLoopMultiGoalRuntimeSegment",
    "ClosedLoopMultiGoalRuntimeExecutionReport",
    "ClosedLoopMultiGoalRuntime",
]
