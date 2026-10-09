from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from .agent_goal_management_integration import AgentGoalManagementIntegrator
from .goal_management import (
    GoalLifecycleError,
    GoalLifecycleManager,
    GoalManagementError,
    GoalStatus,
    ManagedGoal,
)
from .goal_proposal import GoalCandidate


class GoalAcceptanceError(Exception):
    """Base error for goal acceptance."""


class GoalAcceptanceConfigurationError(GoalAcceptanceError):
    """Raised for invalid acceptance policy or integration configuration."""


class GoalAcceptanceInputError(GoalAcceptanceError):
    """Raised when an acceptance candidate cannot be evaluated."""


class GoalAcceptanceApplicationError(GoalAcceptanceError):
    """Raised when an accepted candidate cannot be applied to the goal manager."""


class GoalAcceptanceReason(str, Enum):
    ACCEPTED = "accepted"
    BELOW_PROPOSAL_SCORE = "below_proposal_score"
    BELOW_FEASIBILITY = "below_feasibility"
    DUPLICATE_GOAL_ID = "duplicate_goal_id"
    DUPLICATE_DESCRIPTION = "duplicate_description"
    PARENT_NOT_FOUND = "parent_not_found"
    PARENT_CANCELLED = "parent_cancelled"


@dataclass(frozen=True)
class GoalAcceptancePolicy:
    """Deterministic acceptance gate for immutable GoalCandidate objects."""

    minimum_proposal_score: float = 0.35
    minimum_feasibility: float = 0.25
    allow_duplicate_descriptions: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_proposal_score",
            _bounded_number(self.minimum_proposal_score, "minimum_proposal_score"),
        )
        object.__setattr__(
            self,
            "minimum_feasibility",
            _bounded_number(self.minimum_feasibility, "minimum_feasibility"),
        )
        object.__setattr__(self, "allow_duplicate_descriptions", bool(self.allow_duplicate_descriptions))


@dataclass(frozen=True)
class GoalAcceptanceDecision:
    """Side-effect-free acceptance decision."""

    candidate: GoalCandidate
    accepted: bool
    reason: GoalAcceptanceReason
    conflicts: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def should_accept(self) -> bool:
        return self.accepted

    def __post_init__(self) -> None:
        if self.accepted and self.reason is not GoalAcceptanceReason.ACCEPTED:
            raise ValueError("accepted decisions must use reason=ACCEPTED")
        if not self.accepted and self.reason is GoalAcceptanceReason.ACCEPTED:
            raise ValueError("rejected decisions cannot use reason=ACCEPTED")
        object.__setattr__(self, "reason", _coerce_reason(self.reason))
        object.__setattr__(self, "conflicts", tuple(str(v) for v in self.conflicts))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class GoalAcceptanceResult:
    """Result of applying one candidate through the Agent goal boundary."""

    decision: GoalAcceptanceDecision
    goal: Optional[ManagedGoal]
    current_goal_id: Optional[str]
    projected_to_agent: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision.accepted and self.goal is not None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class GoalAcceptanceEngine:
    """Evaluates GoalCandidate objects against an existing GoalLifecycleManager.

    Evaluation is side-effect free. It never inserts, activates, or mutates goals.
    """

    def __init__(self, *, policy: Optional[GoalAcceptancePolicy] = None) -> None:
        self.policy = policy or GoalAcceptancePolicy()

    def evaluate(
        self,
        candidate: GoalCandidate,
        manager: GoalLifecycleManager,
    ) -> GoalAcceptanceDecision:
        candidate = self._require_candidate(candidate)
        manager = self._require_manager(manager)
        conflicts: list[str] = []

        if candidate.proposal_score < self.policy.minimum_proposal_score:
            conflicts.append(GoalAcceptanceReason.BELOW_PROPOSAL_SCORE.value)
            return self._reject(candidate, GoalAcceptanceReason.BELOW_PROPOSAL_SCORE, conflicts)

        if candidate.feasibility < self.policy.minimum_feasibility:
            conflicts.append(GoalAcceptanceReason.BELOW_FEASIBILITY.value)
            return self._reject(candidate, GoalAcceptanceReason.BELOW_FEASIBILITY, conflicts)

        existing_goals = manager.list_goals(include_terminal=True)
        existing_ids = {goal.goal_id for goal in existing_goals}
        if candidate.candidate_id in existing_ids:
            conflicts.append(GoalAcceptanceReason.DUPLICATE_GOAL_ID.value)
            return self._reject(candidate, GoalAcceptanceReason.DUPLICATE_GOAL_ID, conflicts)

        if not self.policy.allow_duplicate_descriptions:
            candidate_key = _normalize_text(candidate.description)
            for goal in existing_goals:
                if _normalize_text(goal.description) == candidate_key:
                    conflicts.append(goal.goal_id)
                    return self._reject(candidate, GoalAcceptanceReason.DUPLICATE_DESCRIPTION, conflicts)

        if candidate.parent_goal_id is not None:
            try:
                parent = manager.get_goal(candidate.parent_goal_id)
            except Exception:
                conflicts.append(candidate.parent_goal_id)
                return self._reject(candidate, GoalAcceptanceReason.PARENT_NOT_FOUND, conflicts)
            if parent.status is GoalStatus.CANCELLED:
                conflicts.append(parent.goal_id)
                return self._reject(candidate, GoalAcceptanceReason.PARENT_CANCELLED, conflicts)

        return GoalAcceptanceDecision(
            candidate=candidate,
            accepted=True,
            reason=GoalAcceptanceReason.ACCEPTED,
            metadata={
                "proposal_score": candidate.proposal_score,
                "feasibility": candidate.feasibility,
                "parent_goal_id": candidate.parent_goal_id,
            },
        )

    def evaluate_many(
        self,
        candidates: Iterable[GoalCandidate],
        manager: GoalLifecycleManager,
    ) -> tuple[GoalAcceptanceDecision, ...]:
        candidate_items = tuple(candidates)
        return tuple(self.evaluate(candidate, manager) for candidate in candidate_items)

    def _reject(
        self,
        candidate: GoalCandidate,
        reason: GoalAcceptanceReason,
        conflicts: Iterable[str],
    ) -> GoalAcceptanceDecision:
        return GoalAcceptanceDecision(
            candidate=candidate,
            accepted=False,
            reason=reason,
            conflicts=tuple(conflicts),
            metadata={"proposal_score": candidate.proposal_score},
        )

    @staticmethod
    def _require_candidate(candidate: GoalCandidate) -> GoalCandidate:
        if not isinstance(candidate, GoalCandidate):
            raise GoalAcceptanceInputError("candidate must be a GoalCandidate.")
        return candidate

    @staticmethod
    def _require_manager(manager: GoalLifecycleManager) -> GoalLifecycleManager:
        if not isinstance(manager, GoalLifecycleManager):
            raise GoalAcceptanceInputError("manager must be a GoalLifecycleManager.")
        return manager


class AgentGoalAcceptanceIntegrator:
    """Accept GoalCandidate objects into the existing Agent goal boundary.

    The existing GoalLifecycleManager remains the source of truth and
    AgentGoalManagementIntegrator remains the projection boundary.
    """

    def __init__(
        self,
        agent: Any,
        *,
        goal_integrator: Optional[AgentGoalManagementIntegrator] = None,
        manager: Optional[GoalLifecycleManager] = None,
        acceptance_engine: Optional[GoalAcceptanceEngine] = None,
        project_to_agent: bool = True,
        auto_select: bool = False,
    ) -> None:
        if agent is None:
            raise GoalAcceptanceConfigurationError("agent must not be None.")
        if goal_integrator is not None and manager is not None:
            if goal_integrator.manager is not manager:
                raise GoalAcceptanceConfigurationError(
                    "goal_integrator and manager must reference the same manager."
                )
        if goal_integrator is None:
            goal_integrator = AgentGoalManagementIntegrator.for_agent(
                agent,
                manager=manager,
                project_to_agent=project_to_agent,
            )
        if not isinstance(goal_integrator, AgentGoalManagementIntegrator):
            raise GoalAcceptanceConfigurationError(
                "goal_integrator must be AgentGoalManagementIntegrator."
            )
        self.agent = agent
        self.goal_integrator = goal_integrator
        self.manager = goal_integrator.manager
        self.acceptance_engine = acceptance_engine or GoalAcceptanceEngine()
        self.auto_select = bool(auto_select)
        self.last_result: Optional[GoalAcceptanceResult] = None

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        *,
        manager: Optional[GoalLifecycleManager] = None,
        acceptance_engine: Optional[GoalAcceptanceEngine] = None,
        project_to_agent: bool = True,
        auto_select: bool = False,
    ) -> "AgentGoalAcceptanceIntegrator":
        return cls(
            agent,
            manager=manager,
            acceptance_engine=acceptance_engine,
            project_to_agent=project_to_agent,
            auto_select=auto_select,
        )

    def evaluate(self, candidate: GoalCandidate) -> GoalAcceptanceDecision:
        return self.acceptance_engine.evaluate(candidate, self.manager)

    def accept(self, candidate: GoalCandidate) -> GoalAcceptanceResult:
        decision = self.evaluate(candidate)
        if not decision.accepted:
            result = GoalAcceptanceResult(
                decision=decision,
                goal=None,
                current_goal_id=self.manager.current_goal_id,
                projected_to_agent=self.goal_integrator.project_to_agent,
                metadata={"applied": False},
            )
            self.last_result = result
            return result

        try:
            goal = self.goal_integrator.add_goal(**candidate.manager_kwargs())
            if self.auto_select:
                self.goal_integrator.select_next_goal()
                goal = self.manager.get_goal(goal.goal_id)
        except (GoalManagementError, GoalLifecycleError, ValueError) as exc:
            raise GoalAcceptanceApplicationError(
                f"Unable to accept candidate {candidate.candidate_id!r}."
            ) from exc

        result = GoalAcceptanceResult(
            decision=decision,
            goal=goal,
            current_goal_id=self.manager.current_goal_id,
            projected_to_agent=self.goal_integrator.project_to_agent,
            metadata={"applied": True, "auto_select": self.auto_select},
        )
        self.last_result = result
        return result

    def accept_many(
        self,
        candidates: Iterable[GoalCandidate],
        *,
        atomic: bool = False,
    ) -> tuple[GoalAcceptanceResult, ...]:
        candidate_items = tuple(candidates)
        if not atomic:
            return tuple(self.accept(candidate) for candidate in candidate_items)

        decisions = self.acceptance_engine.evaluate_many(candidate_items, self.manager)
        if any(not decision.accepted for decision in decisions):
            return tuple(
                GoalAcceptanceResult(
                    decision=decision,
                    goal=None,
                    current_goal_id=self.manager.current_goal_id,
                    projected_to_agent=self.goal_integrator.project_to_agent,
                    metadata={"applied": False, "atomic": True},
                )
                for decision in decisions
            )

        results: list[GoalAcceptanceResult] = []
        added_ids: list[str] = []
        try:
            for candidate in candidate_items:
                result = self.accept(candidate)
                results.append(result)
                if result.goal is not None:
                    added_ids.append(result.goal.goal_id)
        except Exception as exc:
            # GoalLifecycleManager has no public transactional API, so avoid
            # claiming rollback semantics. The atomic pre-check only guarantees
            # validation-before-mutation; application errors remain surfaced.
            raise GoalAcceptanceApplicationError(
                f"Atomic goal acceptance failed after applying {added_ids!r}."
            ) from exc
        return tuple(results)

    @property
    def current_goal(self) -> Optional[ManagedGoal]:
        return self.goal_integrator.current_goal

    def snapshot(self):
        return self.goal_integrator.snapshot()


_MISSING = object()


def _coerce_reason(value: GoalAcceptanceReason | str) -> GoalAcceptanceReason:
    if isinstance(value, GoalAcceptanceReason):
        return value
    try:
        return GoalAcceptanceReason(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown acceptance reason: {value!r}.") from exc


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _bounded_number(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return number


__all__ = [
    "AgentGoalAcceptanceIntegrator",
    "GoalAcceptanceApplicationError",
    "GoalAcceptanceConfigurationError",
    "GoalAcceptanceDecision",
    "GoalAcceptanceEngine",
    "GoalAcceptanceError",
    "GoalAcceptanceInputError",
    "GoalAcceptancePolicy",
    "GoalAcceptanceReason",
    "GoalAcceptanceResult",
]
