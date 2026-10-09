from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from .context import AgentContext
from .goal_conflict_resolution import GoalConflictResolutionResult
from .goal_management import GoalStatus, ManagedGoal
from .multi_goal_arbitration import MultiGoalArbitrationDecision
from .planning import LLMPlan, LLMPlanner
from .reasoning import LLMReasoning


class GoalAwareLLMPlanningError(RuntimeError):
    """Base error for goal-aware LLM planning."""


class GoalAwareLLMPlanningConfigurationError(GoalAwareLLMPlanningError):
    """Raised for invalid planner configuration."""


class GoalAwareLLMPlanningInputError(GoalAwareLLMPlanningError):
    """Raised when goal-planning inputs cannot be normalized safely."""


class GoalAwareLLMPlanningConsistencyError(GoalAwareLLMPlanningError):
    """Raised when arbitration/conflict/goal inputs are inconsistent."""


@dataclass(frozen=True)
class GoalAwarePlanningContext:
    """Immutable coordination snapshot supplied to goal-aware planning."""

    selected_goal: ManagedGoal
    available_goal_ids: tuple[str, ...] = ()
    competing_goal_ids: tuple[str, ...] = ()
    deferred_goal_ids: tuple[str, ...] = ()
    blocked_goal_ids: tuple[str, ...] = ()
    unresolved_conflict_ids: tuple[str, ...] = ()
    arbitration_reason: str | None = None
    arbitration_score: float | None = None
    arbitration_switched: bool = False
    arbitration_preempted: bool = False
    conflict_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selected_goal, ManagedGoal):
            raise GoalAwareLLMPlanningInputError(
                "selected_goal must be ManagedGoal."
            )
        for name in (
            "available_goal_ids",
            "competing_goal_ids",
            "deferred_goal_ids",
            "blocked_goal_ids",
            "unresolved_conflict_ids",
        ):
            object.__setattr__(self, name, _normalize_ids(getattr(self, name)))

        if self.arbitration_score is not None:
            try:
                score = float(self.arbitration_score)
            except (TypeError, ValueError) as exc:
                raise GoalAwareLLMPlanningInputError(
                    "arbitration_score must be numeric."
                ) from exc
            if not 0.0 <= score <= 1.0:
                raise GoalAwareLLMPlanningInputError(
                    "arbitration_score must be between 0.0 and 1.0."
                )
            object.__setattr__(self, "arbitration_score", round(score, 12))

        try:
            conflict_count = int(self.conflict_count)
        except (TypeError, ValueError) as exc:
            raise GoalAwareLLMPlanningInputError(
                "conflict_count must be an integer."
            ) from exc
        if conflict_count < 0:
            raise GoalAwareLLMPlanningInputError("conflict_count must be >= 0.")
        object.__setattr__(self, "conflict_count", conflict_count)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def has_conflict(self) -> bool:
        return self.conflict_count > 0 or bool(self.unresolved_conflict_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_goal": {
                "goal_id": self.selected_goal.goal_id,
                "description": self.selected_goal.description,
                "priority": self.selected_goal.priority,
                "parent_id": self.selected_goal.parent_id,
                "status": self.selected_goal.status.value,
                "progress": self.selected_goal.progress,
                "metadata": dict(self.selected_goal.metadata),
            },
            "available_goal_ids": list(self.available_goal_ids),
            "competing_goal_ids": list(self.competing_goal_ids),
            "deferred_goal_ids": list(self.deferred_goal_ids),
            "blocked_goal_ids": list(self.blocked_goal_ids),
            "unresolved_conflict_ids": list(self.unresolved_conflict_ids),
            "arbitration_reason": self.arbitration_reason,
            "arbitration_score": self.arbitration_score,
            "arbitration_switched": self.arbitration_switched,
            "arbitration_preempted": self.arbitration_preempted,
            "conflict_count": self.conflict_count,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GoalAwareLLMPlanningRequest:
    """Provider-neutral request for goal-aware planning."""

    reasoning: LLMReasoning
    context: AgentContext
    planning: GoalAwarePlanningContext
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reasoning, LLMReasoning):
            raise GoalAwareLLMPlanningInputError("reasoning must be LLMReasoning.")
        if not isinstance(self.context, AgentContext):
            raise GoalAwareLLMPlanningInputError("context must be AgentContext.")
        if not isinstance(self.planning, GoalAwarePlanningContext):
            raise GoalAwareLLMPlanningInputError(
                "planning must be GoalAwarePlanningContext."
            )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class GoalAwarePlanningContextBuilder:
    """Build immutable goal coordination context from existing Stage 25 outputs."""

    def build(
        self,
        selected_goal: ManagedGoal | Mapping[str, Any],
        *,
        available_goals: Sequence[ManagedGoal | Mapping[str, Any]] = (),
        arbitration: MultiGoalArbitrationDecision | None = None,
        conflict_resolution: GoalConflictResolutionResult | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GoalAwarePlanningContext:
        goal = _normalize_goal(selected_goal)
        goals = tuple(_normalize_goal(item) for item in available_goals)
        available_ids = tuple(item.goal_id for item in goals) or (goal.goal_id,)

        if arbitration is not None and not isinstance(arbitration, MultiGoalArbitrationDecision):
            raise GoalAwareLLMPlanningInputError(
                "arbitration must be MultiGoalArbitrationDecision."
            )
        if conflict_resolution is not None and not isinstance(
            conflict_resolution, GoalConflictResolutionResult
        ):
            raise GoalAwareLLMPlanningInputError(
                "conflict_resolution must be GoalConflictResolutionResult."
            )

        candidate_ids: tuple[str, ...] = ()
        if arbitration is not None:
            if arbitration.selected_goal_id is not None and arbitration.selected_goal_id != goal.goal_id:
                raise GoalAwareLLMPlanningConsistencyError(
                    "Selected goal does not match arbitration.selected_goal_id."
                )
            candidate_ids = tuple(
                item.goal_id
                for item in arbitration.candidates
                if item.goal_id != goal.goal_id and item.eligible
            )

        deferred_ids: tuple[str, ...] = ()
        blocked_ids: tuple[str, ...] = ()
        unresolved_ids: tuple[str, ...] = ()
        conflict_count = 0

        if conflict_resolution is not None:
            conflict_count = conflict_resolution.conflict_count
            deferred_ids = tuple(
                item
                for item in conflict_resolution.deferred_goal_ids
                if item != goal.goal_id
            )
            blocked_ids = tuple(
                item
                for item in conflict_resolution.blocked_goal_ids
                if item != goal.goal_id
            )
            unresolved_ids = tuple(conflict_resolution.unresolved_conflict_ids)
            winners = set(conflict_resolution.winner_goal_ids)
            involved = any(
                goal.goal_id in decision.goal_ids
                for decision in conflict_resolution.decisions
            )
            if goal.goal_id in set(conflict_resolution.blocked_goal_ids):
                raise GoalAwareLLMPlanningConsistencyError(
                    "Selected goal cannot be blocked by conflict resolution."
                )
            if involved and conflict_count and goal.goal_id not in winners:
                raise GoalAwareLLMPlanningConsistencyError(
                    "Selected conflicting goal is not a conflict-resolution winner."
                )

        competing = tuple(
            dict.fromkeys(
                item
                for item in (*candidate_ids, *deferred_ids)
                if item != goal.goal_id
            )
        )

        return GoalAwarePlanningContext(
            selected_goal=goal,
            available_goal_ids=available_ids,
            competing_goal_ids=competing,
            deferred_goal_ids=deferred_ids,
            blocked_goal_ids=blocked_ids,
            unresolved_conflict_ids=unresolved_ids,
            arbitration_reason=arbitration.reason if arbitration else None,
            arbitration_score=arbitration.selected_score if arbitration else None,
            arbitration_switched=bool(arbitration.switched) if arbitration else False,
            arbitration_preempted=bool(arbitration.preempted) if arbitration else False,
            conflict_count=conflict_count,
            metadata=metadata,
        )

    def enrich_context(
        self,
        context: AgentContext,
        planning: GoalAwarePlanningContext,
    ) -> AgentContext:
        if not isinstance(context, AgentContext):
            raise GoalAwareLLMPlanningInputError("context must be AgentContext.")
        if not isinstance(planning, GoalAwarePlanningContext):
            raise GoalAwareLLMPlanningInputError(
                "planning must be GoalAwarePlanningContext."
            )

        payload = planning.to_dict()
        goal_payload = dict(payload["selected_goal"])
        goal_payload["management"] = {
            "arbitration_reason": planning.arbitration_reason,
            "arbitration_score": planning.arbitration_score,
            "arbitration_switched": planning.arbitration_switched,
            "arbitration_preempted": planning.arbitration_preempted,
            "available_goal_ids": list(planning.available_goal_ids),
            "competing_goal_ids": list(planning.competing_goal_ids),
            "deferred_goal_ids": list(planning.deferred_goal_ids),
            "blocked_goal_ids": list(planning.blocked_goal_ids),
            "unresolved_conflict_ids": list(planning.unresolved_conflict_ids),
            "conflict_count": planning.conflict_count,
        }

        metadata = dict(context.metadata or {})
        existing = metadata.get("goal_management")
        if isinstance(existing, Mapping):
            management = {**dict(existing), **payload}
        else:
            management = payload
        metadata["goal_management"] = management
        return replace(context, goal=goal_payload, metadata=metadata)


class GoalAwareLLMPlanner:
    """Thin adapter over the canonical Stage 23.6 LLMPlanner."""

    def __init__(
        self,
        planner: LLMPlanner,
        *,
        context_builder: GoalAwarePlanningContextBuilder | None = None,
    ) -> None:
        if not isinstance(planner, LLMPlanner):
            raise GoalAwareLLMPlanningConfigurationError(
                "planner must be LLMPlanner."
            )
        self.planner = planner
        self.context_builder = context_builder or GoalAwarePlanningContextBuilder()
        self.last_plan: LLMPlan | None = None

    @property
    def last_response(self) -> Any:
        return self.planner.last_response

    def build_request(
        self,
        *,
        reasoning: LLMReasoning,
        context: AgentContext,
        selected_goal: ManagedGoal | Mapping[str, Any],
        available_goals: Sequence[ManagedGoal | Mapping[str, Any]] = (),
        arbitration: MultiGoalArbitrationDecision | None = None,
        conflict_resolution: GoalConflictResolutionResult | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GoalAwareLLMPlanningRequest:
        planning = self.context_builder.build(
            selected_goal,
            available_goals=available_goals,
            arbitration=arbitration,
            conflict_resolution=conflict_resolution,
            metadata=metadata,
        )
        enriched = self.context_builder.enrich_context(context, planning)
        return GoalAwareLLMPlanningRequest(
            reasoning=reasoning,
            context=enriched,
            planning=planning,
            metadata=dict(metadata or {}),
        )

    def plan(self, request: GoalAwareLLMPlanningRequest) -> LLMPlan:
        if not isinstance(request, GoalAwareLLMPlanningRequest):
            raise GoalAwareLLMPlanningInputError(
                "request must be GoalAwareLLMPlanningRequest."
            )

        metadata = {
            "operation": "goal_aware_planning",
            "goal_id": request.planning.selected_goal.goal_id,
            "goal_priority": request.planning.selected_goal.priority,
            "goal_progress": request.planning.selected_goal.progress,
            "goal_management": request.planning.to_dict(),
            **dict(request.metadata),
        }
        plan = self.planner.plan(
            request.reasoning,
            request.context,
            metadata=metadata,
        )
        if plan.goal.strip() != request.planning.selected_goal.description.strip():
            raise GoalAwareLLMPlanningConsistencyError(
                "LLM plan goal does not match the selected goal description."
            )
        self.last_plan = plan
        return plan

    def plan_for_goal(
        self,
        *,
        reasoning: LLMReasoning,
        context: AgentContext,
        selected_goal: ManagedGoal | Mapping[str, Any],
        available_goals: Sequence[ManagedGoal | Mapping[str, Any]] = (),
        arbitration: MultiGoalArbitrationDecision | None = None,
        conflict_resolution: GoalConflictResolutionResult | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LLMPlan:
        return self.plan(
            self.build_request(
                reasoning=reasoning,
                context=context,
                selected_goal=selected_goal,
                available_goals=available_goals,
                arbitration=arbitration,
                conflict_resolution=conflict_resolution,
                metadata=metadata,
            )
        )

    def clear_trace(self) -> None:
        self.last_plan = None
        self.planner.clear_trace()


def _normalize_goal(value: ManagedGoal | Mapping[str, Any]) -> ManagedGoal:
    if isinstance(value, ManagedGoal):
        return value
    if not isinstance(value, Mapping):
        raise GoalAwareLLMPlanningInputError(
            "Each goal must be ManagedGoal or mapping."
        )
    try:
        status = value.get("status", GoalStatus.PENDING)
        if not isinstance(status, GoalStatus):
            status = GoalStatus(str(status))
        return ManagedGoal(
            goal_id=str(value["goal_id"]),
            description=str(value["description"]),
            priority=float(value.get("priority", 0.5)),
            parent_id=value.get("parent_id"),
            status=status,
            progress=float(value.get("progress", 0.0)),
            payload=value.get("payload"),
            metadata=value.get("metadata") or {},
            sequence=int(value.get("sequence", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GoalAwareLLMPlanningInputError("Invalid goal mapping.") from exc


def _normalize_ids(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


__all__ = [
    "GoalAwareLLMPlanningConsistencyError",
    "GoalAwareLLMPlanningConfigurationError",
    "GoalAwarePlanningContextBuilder",
    "GoalAwareLLMPlanningError",
    "GoalAwareLLMPlanningInputError",
    "GoalAwareLLMPlanningRequest",
    "GoalAwareLLMPlanner",
    "GoalAwarePlanningContext",
]
