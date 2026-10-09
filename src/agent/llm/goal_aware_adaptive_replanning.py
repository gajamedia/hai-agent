from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from .context import AgentContext, AgentContextBuilder
from .goal_aware_llm_planning import (
    GoalAwarePlanningContext,
    GoalAwarePlanningContextBuilder,
    GoalAwareLLMPlanner,
)
from .goal_conflict_resolution import GoalConflictResolutionResult
from .goal_management import GoalStatus, ManagedGoal
from .llm_plan_adaptive_replanning import (
    LLMPlanAdaptiveReplanningResult,
    LLMPlanAdaptiveReplanner,
    LLMPlanReplanningRequest,
)
from .llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback
from .llm_planner_adaptive_replanning import default_risk_resolver
from .multi_goal_arbitration import MultiGoalArbitrationDecision
from .planning import LLMPlan, LLMPlanStep
from .reasoning import LLMReasoning


class GoalAwareAdaptiveReplanningError(RuntimeError):
    """Base error for goal-aware adaptive replanning."""


class GoalAwareAdaptiveReplanningConfigurationError(
    GoalAwareAdaptiveReplanningError
):
    """Raised when controller/provider configuration is invalid."""


class GoalAwareAdaptiveReplanningInputError(GoalAwareAdaptiveReplanningError):
    """Raised when goal-aware replanning inputs are invalid."""


class GoalAwareAdaptiveReplanningConsistencyError(
    GoalAwareAdaptiveReplanningError
):
    """Raised when goal-management and runtime plan state disagree."""


GoalAwareContextFactory = Callable[
    [LLMPlanReplanningRequest, GoalAwarePlanningContext], AgentContext
]
GoalAwareReasoningFactory = Callable[
    [LLMPlanReplanningRequest, GoalAwarePlanningContext], LLMReasoning
]


@dataclass(frozen=True)
class GoalAwareAdaptiveReplanningDecision:
    """Immutable explanation of why a goal-aware replan is required."""

    should_replan: bool
    reason: str
    active_goal_id: str
    original_goal: str
    goal_changed: bool
    arbitration_switched: bool
    arbitration_preempted: bool
    conflict_requires_replanning: bool
    runtime_replan_requested: bool
    unresolved_conflict_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.active_goal_id.strip():
            raise GoalAwareAdaptiveReplanningInputError(
                "active_goal_id must not be empty."
            )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.unresolved_conflict_count < 0:
            raise GoalAwareAdaptiveReplanningInputError(
                "unresolved_conflict_count must be >= 0."
            )


@dataclass(frozen=True)
class GoalAwareAdaptiveReplanningResult:
    """Trace for one goal-aware replan evaluation and optional replacement plan."""

    decision: GoalAwareAdaptiveReplanningDecision
    planning: GoalAwarePlanningContext
    feedback: LLMPlanRuntimeLearningFeedback | None
    effective_feedback: LLMPlanRuntimeLearningFeedback | None
    adaptive_result: LLMPlanAdaptiveReplanningResult | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def replanned(self) -> bool:
        return self.adaptive_result is not None and self.adaptive_result.replanned

    @property
    def replacement_plan(self) -> LLMPlan | None:
        if self.adaptive_result is None:
            return None
        return self.adaptive_result.replacement_plan

    @property
    def validation(self) -> Any:
        if self.adaptive_result is None:
            return None
        return self.adaptive_result.validation

    @property
    def llm_plan_requested(self) -> bool:
        return self.replanned or (
            self.adaptive_result is not None
            and self.adaptive_result.request is not None
        )


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _normalize_goal(value: ManagedGoal | Mapping[str, Any]) -> ManagedGoal:
    if isinstance(value, ManagedGoal):
        return value
    if not isinstance(value, Mapping):
        raise GoalAwareAdaptiveReplanningInputError(
            "Goal must be ManagedGoal or mapping."
        )
    status = value.get("status", GoalStatus.PENDING)
    if not isinstance(status, GoalStatus):
        status = GoalStatus(str(status))
    return ManagedGoal(
        goal_id=str(value["goal_id"]),
        description=str(value["description"]),
        priority=value.get("priority", 0.5),
        parent_id=value.get("parent_id"),
        status=status,
        progress=value.get("progress", 0.0),
        payload=value.get("payload"),
        metadata=value.get("metadata") or {},
        sequence=int(value.get("sequence", 0)),
    )


def _normalize_goals(values: Sequence[ManagedGoal | Mapping[str, Any]]) -> tuple[ManagedGoal, ...]:
    return tuple(_normalize_goal(value) for value in values)


def _normalize_plan_risks(plan: LLMPlan) -> LLMPlan:
    changed = False
    steps: list[LLMPlanStep] = []
    for step in plan.steps:
        risk = step.risk
        if risk is None or isinstance(risk, (int, float)):
            normalized = risk
        else:
            normalized = default_risk_resolver(risk)
            changed = True
        steps.append(
            replace(
                step,
                risk=normalized,
            )
        )
    return plan if not changed else replace(plan, steps=tuple(steps))


def _default_context_factory(
    request: LLMPlanReplanningRequest,
    planning: GoalAwarePlanningContext,
) -> AgentContext:
    builder = AgentContextBuilder()
    management = planning.to_dict()
    return builder.build(
        goal=management["selected_goal"],
        state={
            "runtime_generation": request.generation,
            "next_step_index": request.next_step_index,
            "failed_step_index": request.failed_step_index,
            "feedback_step_index": request.feedback.step_index,
            "goal_changed": True,
        },
        world_model=request.world_state,
        relevant_memory=tuple(
            {"type": "preserved_successful_step", "step": step.to_dict()}
            for step in request.preserved_steps
        )
        + tuple(
            {"type": "remaining_original_step", "step": step.to_dict()}
            for step in request.remaining_steps
        ),
        strategy={
            "mode": "goal_aware_adaptive_replanning",
            "goal_management": management,
            "feedback": request.feedback.metadata,
        },
        metadata={"goal_management": management},
    )


def _default_reasoning_factory(
    request: LLMPlanReplanningRequest,
    planning: GoalAwarePlanningContext,
) -> LLMReasoning:
    feedback = request.feedback
    return LLMReasoning(
        summary=(
            f"Replan for active goal {planning.selected_goal.goal_id} after "
            f"runtime feedback at step {feedback.step_index}."
        ),
        conclusions=(
            f"Active goal: {planning.selected_goal.description}",
            f"Diagnosis: {feedback.diagnosis}",
            f"Lesson: {feedback.lesson}",
        ),
        alternatives=(
            "The previous plan may be unsuitable for the current goal context.",
        ),
        risks=(
            f"Runtime risk level: {feedback.risk_level}",
        ),
        information_gaps=(
            feedback.next_step,
        ),
        recommended_direction="Replan only the remaining work for the selected goal.",
        confidence=max(0.0, min(1.0, float(feedback.confidence))),
    )


class GoalAwareAdaptiveReplanningProvider:
    """Provider adapter combining Stage 25 goal context with Stage 23 LLM planning.

    The provider receives one active coordination snapshot before a replan.
    It never executes actions and never mutates the goal manager.
    """

    def __init__(
        self,
        goal_planner: GoalAwareLLMPlanner,
        *,
        context_factory: GoalAwareContextFactory | None = None,
        reasoning_factory: GoalAwareReasoningFactory | None = None,
        risk_resolver: Callable[[str | None], float] | None = None,
    ) -> None:
        if not callable(getattr(goal_planner, "plan_for_goal", None)):
            raise GoalAwareAdaptiveReplanningConfigurationError(
                "goal_planner must expose callable plan_for_goal(...)."
            )
        if context_factory is not None and not callable(context_factory):
            raise GoalAwareAdaptiveReplanningConfigurationError(
                "context_factory must be callable."
            )
        if reasoning_factory is not None and not callable(reasoning_factory):
            raise GoalAwareAdaptiveReplanningConfigurationError(
                "reasoning_factory must be callable."
            )
        if risk_resolver is not None and not callable(risk_resolver):
            raise GoalAwareAdaptiveReplanningConfigurationError(
                "risk_resolver must be callable."
            )

        self.goal_planner = goal_planner
        self.context_factory = context_factory or _default_context_factory
        self.reasoning_factory = reasoning_factory or _default_reasoning_factory
        self.risk_resolver = risk_resolver or default_risk_resolver
        self._planning: GoalAwarePlanningContext | None = None
        self._available_goals: tuple[ManagedGoal, ...] = ()
        self._arbitration: MultiGoalArbitrationDecision | None = None
        self._conflict_resolution: GoalConflictResolutionResult | None = None
        self.last_request: LLMPlanReplanningRequest | None = None
        self.last_context: AgentContext | None = None
        self.last_reasoning: LLMReasoning | None = None
        self.last_plan: LLMPlan | None = None

    def set_planning_context(self, planning: GoalAwarePlanningContext) -> None:
        if not isinstance(planning, GoalAwarePlanningContext):
            raise GoalAwareAdaptiveReplanningInputError(
                "planning must be GoalAwarePlanningContext."
            )
        self._planning = planning
        self._available_goals = (planning.selected_goal,)
        self._arbitration = None
        self._conflict_resolution = None

    def set_coordination_context(
        self,
        planning: GoalAwarePlanningContext,
        *,
        available_goals: Sequence[ManagedGoal | Mapping[str, Any]] = (),
        arbitration: MultiGoalArbitrationDecision | None = None,
        conflict_resolution: GoalConflictResolutionResult | None = None,
    ) -> None:
        if not isinstance(planning, GoalAwarePlanningContext):
            raise GoalAwareAdaptiveReplanningInputError(
                "planning must be GoalAwarePlanningContext."
            )
        self._planning = planning
        normalized = _normalize_goals(available_goals)
        if not any(item.goal_id == planning.selected_goal.goal_id for item in normalized):
            normalized = (planning.selected_goal,) + normalized
        self._available_goals = normalized
        self._arbitration = arbitration
        self._conflict_resolution = conflict_resolution

    @property
    def planning_context(self) -> GoalAwarePlanningContext | None:
        return self._planning

    def __call__(self, request: LLMPlanReplanningRequest) -> LLMPlan:
        if not isinstance(request, LLMPlanReplanningRequest):
            raise GoalAwareAdaptiveReplanningInputError(
                "request must be LLMPlanReplanningRequest."
            )
        if self._planning is None:
            raise GoalAwareAdaptiveReplanningInputError(
                "Planning context must be configured before replan."
            )
        selected_goal = self._planning.selected_goal
        if _text(request.goal).casefold() != selected_goal.description.strip().casefold():
            raise GoalAwareAdaptiveReplanningConsistencyError(
                "Replanning request goal does not match selected goal description."
            )

        try:
            context = self.context_factory(request, self._planning)
            reasoning = self.reasoning_factory(request, self._planning)
        except Exception as exc:
            raise GoalAwareAdaptiveReplanningError(
                f"Failed to build goal-aware planning inputs: {exc}"
            ) from exc
        if not isinstance(context, AgentContext):
            raise GoalAwareAdaptiveReplanningInputError(
                "context_factory must return AgentContext."
            )
        if not isinstance(reasoning, LLMReasoning):
            raise GoalAwareAdaptiveReplanningInputError(
                "reasoning_factory must return LLMReasoning."
            )

        self.last_request = request
        self.last_context = context
        self.last_reasoning = reasoning
        plan = self.goal_planner.plan_for_goal(
            reasoning=reasoning,
            context=context,
            selected_goal=selected_goal,
            available_goals=self._planning_goals(),
            arbitration=self._planning_arbitration(),
            conflict_resolution=self._planning_conflict_resolution(),
            metadata={
                **dict(request.metadata),
                "stage": "25.7",
                "operation": "goal_aware_adaptive_replanning",
                "goal_id": selected_goal.goal_id,
            },
        )
        if not isinstance(plan, LLMPlan):
            raise GoalAwareAdaptiveReplanningError(
                "goal_planner.plan_for_goal(...) must return LLMPlan."
            )
        if plan.goal.strip().casefold() != selected_goal.description.strip().casefold():
            raise GoalAwareAdaptiveReplanningConsistencyError(
                "LLM replacement plan goal does not match selected goal."
            )

        normalized_steps: list[LLMPlanStep] = []
        changed = False
        for step in plan.steps:
            risk = step.risk
            if risk is not None and not isinstance(risk, (int, float)):
                risk = self.risk_resolver(risk)
                changed = True
            normalized_steps.append(replace(step, risk=risk))
        result = plan if not changed else replace(plan, steps=tuple(normalized_steps))
        self.last_plan = result
        return result

    def _planning_goals(self) -> tuple[ManagedGoal, ...]:
        return self._available_goals if self._planning else ()

    def _planning_arbitration(self) -> MultiGoalArbitrationDecision | None:
        return self._arbitration

    def _planning_conflict_resolution(self) -> GoalConflictResolutionResult | None:
        return self._conflict_resolution

    def reset(self) -> None:
        self._planning = None
        self._available_goals = ()
        self._arbitration = None
        self._conflict_resolution = None
        self.last_request = None
        self.last_context = None
        self.last_reasoning = None
        self.last_plan = None


class GoalAwareAdaptiveReplanningController:
    """Coordinate goal management, conflict arbitration, and adaptive replanning."""

    def __init__(
        self,
        adaptive_replanner: LLMPlanAdaptiveReplanner,
        provider: GoalAwareAdaptiveReplanningProvider,
    ) -> None:
        if not callable(getattr(adaptive_replanner, "replan", None)):
            raise GoalAwareAdaptiveReplanningConfigurationError(
                "adaptive_replanner must expose callable replan(...)."
            )
        if not isinstance(provider, GoalAwareAdaptiveReplanningProvider):
            raise GoalAwareAdaptiveReplanningConfigurationError(
                "provider must be GoalAwareAdaptiveReplanningProvider."
            )
        self.adaptive_replanner = adaptive_replanner
        self.provider = provider
        self.last_result: GoalAwareAdaptiveReplanningResult | None = None

    def evaluate(
        self,
        original_plan: LLMPlan,
        selected_goal: ManagedGoal | Mapping[str, Any],
        *,
        arbitration: MultiGoalArbitrationDecision | None = None,
        conflict_resolution: GoalConflictResolutionResult | None = None,
        feedback: LLMPlanRuntimeLearningFeedback | None = None,
        original_goal_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GoalAwareAdaptiveReplanningDecision:
        if not isinstance(original_plan, LLMPlan):
            raise GoalAwareAdaptiveReplanningInputError(
                "original_plan must be LLMPlan."
            )
        goal = _normalize_goal(selected_goal)
        if goal.status.terminal:
            raise GoalAwareAdaptiveReplanningConsistencyError(
                "Selected goal cannot be terminal for replanning."
            )
        if arbitration is not None and not isinstance(arbitration, MultiGoalArbitrationDecision):
            raise GoalAwareAdaptiveReplanningInputError(
                "arbitration must be MultiGoalArbitrationDecision."
            )
        if conflict_resolution is not None and not isinstance(
            conflict_resolution, GoalConflictResolutionResult
        ):
            raise GoalAwareAdaptiveReplanningInputError(
                "conflict_resolution must be GoalConflictResolutionResult."
            )

        description_changed = (
            original_plan.goal.strip().casefold() != goal.description.strip().casefold()
        )
        id_changed = original_goal_id is not None and str(original_goal_id).strip() != goal.goal_id
        goal_changed = description_changed or id_changed
        switched = bool(arbitration and arbitration.switched)
        preempted = bool(arbitration and arbitration.preempted)

        conflict_requires = False
        unresolved_count = 0
        if conflict_resolution is not None:
            unresolved_count = len(conflict_resolution.unresolved_conflict_ids)
            winners = set(conflict_resolution.winner_goal_ids)
            involved = any(
                goal.goal_id in decision.goal_ids
                for decision in conflict_resolution.decisions
            )
            if goal.goal_id in conflict_resolution.blocked_goal_ids:
                raise GoalAwareAdaptiveReplanningConsistencyError(
                    "Selected goal is blocked by conflict resolution."
                )
            conflict_requires = bool(
                involved and goal.goal_id not in winners
            )

        runtime_triggered = False
        if feedback is not None:
            runtime_triggered = bool(
                self.adaptive_replanner.evaluate_trigger(feedback).should_replan
            )

        reasons: list[str] = []
        if goal_changed:
            reasons.append("goal_changed")
        if preempted:
            reasons.append("goal_preempted")
        elif switched:
            reasons.append("goal_switched")
        if conflict_requires:
            reasons.append("goal_conflict_requires_replan")
        if runtime_triggered:
            reasons.append("runtime_feedback_trigger")

        return GoalAwareAdaptiveReplanningDecision(
            should_replan=bool(reasons),
            reason="|".join(reasons) if reasons else "no_replan_trigger",
            active_goal_id=goal.goal_id,
            original_goal=original_plan.goal,
            goal_changed=goal_changed,
            arbitration_switched=switched,
            arbitration_preempted=preempted,
            conflict_requires_replanning=conflict_requires,
            runtime_replan_requested=runtime_triggered,
            unresolved_conflict_count=unresolved_count,
            metadata={**dict(metadata or {}), "selected_goal": goal.to_dict() if hasattr(goal, "to_dict") else goal.description},
        )

    @staticmethod
    def _synthesize_goal_change_feedback(
        selected_goal: ManagedGoal,
        *,
        step_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlanRuntimeLearningFeedback:
        return LLMPlanRuntimeLearningFeedback(
            step_index=max(0, int(step_index)),
            action="goal_transition",
            success=None,
            progress_delta=0.0,
            evidence_quality=0.0,
            confidence=0.0,
            diagnosis="goal_changed",
            lesson=(
                f"Active goal changed to {selected_goal.description}; the remaining "
                "plan must be recomputed for the new goal."
            ),
            next_step="Replan the remaining work for the selected goal.",
            change_required=True,
            continue_recommended=False,
            exploration_recommended=False,
            risk_level="medium",
            risk_score=0.50,
            information_gain=0.0,
            metadata={
                **dict(metadata or {}),
                "synthetic": True,
                "trigger": "goal_change",
                "goal_id": selected_goal.goal_id,
            },
        )

    def replan(
        self,
        original_plan: LLMPlan,
        selected_goal: ManagedGoal | Mapping[str, Any],
        *,
        available_goals: Sequence[ManagedGoal | Mapping[str, Any]] = (),
        arbitration: MultiGoalArbitrationDecision | None = None,
        conflict_resolution: GoalConflictResolutionResult | None = None,
        feedback: LLMPlanRuntimeLearningFeedback | None = None,
        successful_step_indices: Sequence[int] = (),
        world_state: Any = None,
        failed_step_index: int | None = None,
        next_step_index: int | None = None,
        generation: int | None = None,
        available_actions: Any = None,
        original_goal_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GoalAwareAdaptiveReplanningResult:
        goal = _normalize_goal(selected_goal)
        goals = _normalize_goals(available_goals)
        if not any(item.goal_id == goal.goal_id for item in goals):
            goals = (goal,) + goals
        if not goals:
            goals = (goal,)
        decision = self.evaluate(
            original_plan,
            goal,
            arbitration=arbitration,
            conflict_resolution=conflict_resolution,
            feedback=feedback,
            original_goal_id=original_goal_id,
            metadata=metadata,
        )

        planning_metadata = dict(metadata or {})
        if arbitration is not None:
            planning_metadata["_arbitration"] = arbitration
        if conflict_resolution is not None:
            planning_metadata["_conflict_resolution"] = conflict_resolution

        planning_builder = GoalAwarePlanningContextBuilder()
        planning = planning_builder.build(
            goal,
            available_goals=goals,
            arbitration=arbitration,
            conflict_resolution=conflict_resolution,
            metadata=planning_metadata,
        )
        self.provider.set_coordination_context(
            planning,
            available_goals=goals,
            arbitration=arbitration,
            conflict_resolution=conflict_resolution,
        )

        effective_feedback = feedback
        trigger_step = (
            feedback.step_index if feedback is not None else (failed_step_index or 0)
        )
        if decision.should_replan and not decision.runtime_replan_requested:
            effective_feedback = self._synthesize_goal_change_feedback(
                goal,
                step_index=trigger_step,
                metadata={"decision_reason": decision.reason},
            )

        adaptive_result = None
        if decision.should_replan:
            adaptive_result = self.adaptive_replanner.replan(
                original_plan,
                feedback=effective_feedback,
                successful_step_indices=successful_step_indices,
                goal=goal.description,
                world_state=world_state,
                failed_step_index=failed_step_index,
                next_step_index=next_step_index,
                generation=generation,
                available_actions=available_actions,
                metadata={
                    **dict(metadata or {}),
                    "stage": "25.7",
                    "integration_stage": "25.7",
                    "goal_aware_replanning": True,
                    "active_goal_id": goal.goal_id,
                    "goal_change": decision.goal_changed,
                    "arbitration_switched": decision.arbitration_switched,
                    "arbitration_preempted": decision.arbitration_preempted,
                    "conflict_requires_replanning": decision.conflict_requires_replanning,
                    "decision_reason": decision.reason,
                },
            )

        result = GoalAwareAdaptiveReplanningResult(
            decision=decision,
            planning=planning,
            feedback=feedback,
            effective_feedback=effective_feedback,
            adaptive_result=adaptive_result,
            metadata={
                **dict(metadata or {}),
                "stage": "25.7",
            },
        )
        self.last_result = result
        return result

    def reset(self) -> None:
        self.provider.reset()
        self.last_result = None
