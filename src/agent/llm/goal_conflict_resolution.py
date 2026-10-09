from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_management import GoalStatus, GoalLifecycleManager, ManagedGoal
from .multi_goal_arbitration import (
    MultiGoalArbitrationCandidate,
    MultiGoalArbitrationDecision,
)


class GoalConflictResolutionError(Exception):
    """Base error for goal conflict detection and resolution."""


class GoalConflictResolutionConfigurationError(GoalConflictResolutionError):
    """Raised when conflict policy configuration is invalid."""


class GoalConflictResolutionInputError(GoalConflictResolutionError):
    """Raised when a conflict input cannot be normalized safely."""


class GoalConflictResolutionApplicationError(GoalConflictResolutionError):
    """Raised when a conflict decision cannot be applied safely."""


@dataclass(frozen=True)
class GoalConflictPolicy:
    """Deterministic policy used to resolve a set of conflicting goals.

    Higher priority wins first; arbitration score, urgency, progress and
    insertion order are deterministic tie-breakers when available.
    """

    priority_weight: float = 0.60
    arbitration_weight: float = 0.25
    urgency_weight: float = 0.10
    progress_weight: float = 0.05
    allow_same_goal_conflict: bool = False

    def __post_init__(self) -> None:
        values = {
            "priority_weight": self.priority_weight,
            "arbitration_weight": self.arbitration_weight,
            "urgency_weight": self.urgency_weight,
            "progress_weight": self.progress_weight,
        }
        total = 0.0
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise GoalConflictResolutionConfigurationError(
                    f"{name} must be numeric."
                ) from exc
            if number < 0.0:
                raise GoalConflictResolutionConfigurationError(
                    f"{name} must be non-negative."
                )
            object.__setattr__(self, name, number)
            total += number
        if total <= 0.0:
            raise GoalConflictResolutionConfigurationError(
                "At least one conflict-resolution weight must be positive."
            )
        for name in values:
            object.__setattr__(self, name, getattr(self, name) / total)


@dataclass(frozen=True)
class GoalConflictRelation:
    """Normalized conflict relation between two distinct goals."""

    goal_id: str
    conflicting_goal_id: str
    reason: str = "explicit_conflict"
    severity: float = 1.0
    source: str = "explicit"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        first = str(self.goal_id).strip()
        second = str(self.conflicting_goal_id).strip()
        if not first or not second:
            raise GoalConflictResolutionInputError(
                "Conflict relation goal IDs must not be empty."
            )
        if first == second:
            raise GoalConflictResolutionInputError(
                "A goal cannot conflict with itself."
            )
        try:
            severity = float(self.severity)
        except (TypeError, ValueError) as exc:
            raise GoalConflictResolutionInputError(
                "Conflict severity must be numeric."
            ) from exc
        if not 0.0 <= severity <= 1.0:
            raise GoalConflictResolutionInputError(
                "Conflict severity must be between 0.0 and 1.0."
            )
        object.__setattr__(self, "goal_id", first)
        object.__setattr__(self, "conflicting_goal_id", second)
        object.__setattr__(self, "reason", str(self.reason).strip() or "explicit_conflict")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "source", str(self.source).strip() or "explicit")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted((self.goal_id, self.conflicting_goal_id)))


@dataclass(frozen=True)
class GoalConflictCandidate:
    """Candidate-level evidence used for a conflict-resolution decision."""

    goal_id: str
    priority: float
    urgency: float
    progress: float
    arbitration_score: float
    conflict_score: float
    eligible: bool
    status: GoalStatus
    sequence: int
    reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True)
class GoalConflictResolutionDecision:
    """Side-effect-free resolution result for one conflict component."""

    conflict_id: str
    goal_ids: tuple[str, ...]
    winner_goal_id: Optional[str]
    deferred_goal_ids: tuple[str, ...]
    blocked_goal_ids: tuple[str, ...]
    resolved: bool
    conflict_severity: float
    reason: str
    candidates: tuple[GoalConflictCandidate, ...]
    relations: tuple[GoalConflictRelation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_winner(self) -> bool:
        return self.winner_goal_id is not None

    @property
    def should_defer(self) -> bool:
        return bool(self.deferred_goal_ids)


@dataclass(frozen=True)
class GoalConflictResolutionResult:
    """Complete conflict-resolution result across all detected components."""

    decisions: tuple[GoalConflictResolutionDecision, ...]
    winner_goal_ids: tuple[str, ...]
    deferred_goal_ids: tuple[str, ...]
    blocked_goal_ids: tuple[str, ...]
    unresolved_conflict_ids: tuple[str, ...]
    conflict_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return not self.unresolved_conflict_ids

    @property
    def has_conflicts(self) -> bool:
        return self.conflict_count > 0


class GoalConflictResolver:
    """Detects conflict components and deterministically chooses winners."""

    def __init__(self, *, policy: Optional[GoalConflictPolicy] = None) -> None:
        self.policy = policy or GoalConflictPolicy()

    def resolve(
        self,
        goals: Iterable[ManagedGoal | Mapping[str, Any]],
        *,
        relations: Iterable[GoalConflictRelation | Mapping[str, Any]] = (),
        urgency: Optional[Mapping[str, float]] = None,
        arbitration: Optional[MultiGoalArbitrationDecision] = None,
        blocked_goal_ids: Iterable[str] = (),
    ) -> GoalConflictResolutionResult:
        normalized_goals = tuple(self._normalize_goal(item) for item in goals)
        goal_map = {goal.goal_id: goal for goal in normalized_goals}
        relation_map = self._normalize_relations(relations, goal_map)
        urgency_map = self._normalize_urgency(urgency or {})
        preblocked = {str(item).strip() for item in blocked_goal_ids}
        arbitration_scores = self._arbitration_scores(arbitration)

        components = self._components(tuple(goal_map), relation_map)
        decisions: list[GoalConflictResolutionDecision] = []
        winners: list[str] = []
        deferred: list[str] = []
        blocked: list[str] = sorted(preblocked.intersection(goal_map))
        unresolved: list[str] = []

        for component in components:
            component_relations = tuple(
                relation for relation in relation_map.values()
                if relation.goal_id in component and relation.conflicting_goal_id in component
            )
            candidates = tuple(
                self._build_candidate(
                    goal_map[goal_id],
                    urgency=urgency_map.get(goal_id, 0.0),
                    arbitration_score=arbitration_scores.get(goal_id, 0.0),
                    blocked=goal_id in preblocked,
                    relation_count=sum(
                        goal_id in (item.goal_id, item.conflicting_goal_id)
                        for item in component_relations
                    ),
                )
                for goal_id in component
            )
            severity = max((item.severity for item in component_relations), default=1.0)
            eligible = [item for item in candidates if item.eligible]
            conflict_id = self._conflict_id(component)

            if len(eligible) <= 1:
                winner = eligible[0].goal_id if eligible else None
                deferred_ids = tuple(
                    sorted(item.goal_id for item in candidates if item.goal_id != winner and not item.eligible)
                )
                reason = "conflict_resolved_by_eligibility"
                resolved = bool(winner) or not candidates
                if candidates and not eligible:
                    reason = "no_eligible_conflict_winner"
                    resolved = False
            else:
                ranked = sorted(eligible, key=self._ranking_key)
                winner = ranked[0].goal_id
                deferred_ids = tuple(item.goal_id for item in ranked[1:])
                reason = "higher_conflict_resolution_score"
                resolved = True

            if winner:
                winners.append(winner)
            deferred.extend(deferred_ids)
            unresolved_conflict = not resolved
            if unresolved_conflict:
                unresolved.append(conflict_id)

            decisions.append(
                GoalConflictResolutionDecision(
                    conflict_id=conflict_id,
                    goal_ids=tuple(component),
                    winner_goal_id=winner,
                    deferred_goal_ids=tuple(sorted(deferred_ids)),
                    blocked_goal_ids=tuple(sorted(item.goal_id for item in candidates if not item.eligible and item.goal_id in preblocked)),
                    resolved=resolved,
                    conflict_severity=round(severity, 12),
                    reason=reason,
                    candidates=candidates,
                    relations=component_relations,
                    metadata={"component_size": len(component)},
                )
            )

        return GoalConflictResolutionResult(
            decisions=tuple(decisions),
            winner_goal_ids=tuple(dict.fromkeys(winners)),
            deferred_goal_ids=tuple(dict.fromkeys(deferred)),
            blocked_goal_ids=tuple(dict.fromkeys(blocked)),
            unresolved_conflict_ids=tuple(unresolved),
            conflict_count=len(decisions),
            metadata={"relation_count": len(relation_map)},
        )

    def resolve_manager(
        self,
        manager: GoalLifecycleManager,
        *,
        relations: Iterable[GoalConflictRelation | Mapping[str, Any]] = (),
        urgency: Optional[Mapping[str, float]] = None,
        arbitration: Optional[MultiGoalArbitrationDecision] = None,
        blocked_goal_ids: Iterable[str] = (),
    ) -> GoalConflictResolutionResult:
        if not isinstance(manager, GoalLifecycleManager):
            raise GoalConflictResolutionInputError(
                "manager must be GoalLifecycleManager."
            )
        return self.resolve(
            manager.list_goals(),
            relations=relations,
            urgency=urgency,
            arbitration=arbitration,
            blocked_goal_ids=blocked_goal_ids,
        )

    @staticmethod
    def _normalize_goal(value: ManagedGoal | Mapping[str, Any]) -> ManagedGoal:
        if isinstance(value, ManagedGoal):
            return value
        if not isinstance(value, Mapping):
            raise GoalConflictResolutionInputError(
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
            raise GoalConflictResolutionInputError("Invalid goal mapping.") from exc

    @staticmethod
    def _normalize_relations(
        relations: Iterable[GoalConflictRelation | Mapping[str, Any]],
        goal_map: Mapping[str, ManagedGoal],
    ) -> dict[tuple[str, str], GoalConflictRelation]:
        normalized: dict[tuple[str, str], GoalConflictRelation] = {}
        for value in relations:
            if isinstance(value, GoalConflictRelation):
                relation = value
            elif isinstance(value, Mapping):
                try:
                    relation = GoalConflictRelation(**dict(value))
                except (TypeError, ValueError) as exc:
                    raise GoalConflictResolutionInputError(
                        "Invalid conflict relation mapping."
                    ) from exc
            else:
                raise GoalConflictResolutionInputError(
                    "Each relation must be GoalConflictRelation or mapping."
                )
            if relation.goal_id not in goal_map or relation.conflicting_goal_id not in goal_map:
                raise GoalConflictResolutionInputError(
                    f"Conflict relation references unknown goal: {relation.pair_key!r}."
                )
            normalized[relation.pair_key] = relation
        return normalized

    @staticmethod
    def _normalize_urgency(values: Mapping[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise GoalConflictResolutionInputError(
                    f"Urgency for {key!r} must be numeric."
                ) from exc
            if not 0.0 <= number <= 1.0:
                raise GoalConflictResolutionInputError(
                    f"Urgency for {key!r} must be between 0.0 and 1.0."
                )
            normalized[str(key).strip()] = number
        return normalized

    @staticmethod
    def _arbitration_scores(
        arbitration: Optional[MultiGoalArbitrationDecision],
    ) -> dict[str, float]:
        if arbitration is None:
            return {}
        scores: dict[str, float] = {}
        for candidate in arbitration.candidates:
            raw = getattr(candidate, "score", None)
            if raw is None:
                raw = getattr(candidate, "arbitration_score", 0.0)
            try:
                scores[candidate.goal_id] = max(0.0, min(1.0, float(raw)))
            except (TypeError, ValueError) as exc:
                raise GoalConflictResolutionInputError(
                    f"Arbitration score for {candidate.goal_id!r} must be numeric."
                ) from exc
        return scores

    def _build_candidate(
        self,
        goal: ManagedGoal,
        *,
        urgency: float,
        arbitration_score: float,
        blocked: bool,
        relation_count: int,
    ) -> GoalConflictCandidate:
        if blocked:
            eligible = False
            reasons = ("blocked_goal",)
        elif goal.status.terminal:
            eligible = False
            reasons = ("terminal_goal",)
        else:
            eligible = True
            reasons = ()
        conflict_score = _bounded(
            self.policy.priority_weight * goal.priority
            + self.policy.arbitration_weight * arbitration_score
            + self.policy.urgency_weight * urgency
            + self.policy.progress_weight * goal.progress
        )
        metadata = {"relation_count": relation_count}
        return GoalConflictCandidate(
            goal_id=goal.goal_id,
            priority=round(goal.priority, 12),
            urgency=round(urgency, 12),
            progress=round(goal.progress, 12),
            arbitration_score=round(arbitration_score, 12),
            conflict_score=round(conflict_score, 12),
            eligible=eligible,
            status=goal.status,
            sequence=goal.sequence,
            reasons=reasons,
            metadata=metadata,
        )

    @staticmethod
    def _ranking_key(candidate: GoalConflictCandidate) -> tuple[float, float, float, float, int]:
        return (
            -candidate.conflict_score,
            -candidate.priority,
            -candidate.urgency,
            -candidate.progress,
            candidate.sequence,
        )

    @staticmethod
    def _components(
        goal_ids: tuple[str, ...],
        relations: Mapping[tuple[str, str], GoalConflictRelation],
    ) -> tuple[tuple[str, ...], ...]:
        if not relations:
            return ()
        adjacency = {goal_id: set() for goal_id in goal_ids}
        for relation in relations.values():
            adjacency[relation.goal_id].add(relation.conflicting_goal_id)
            adjacency[relation.conflicting_goal_id].add(relation.goal_id)
        seen: set[str] = set()
        components: list[tuple[str, ...]] = []
        for goal_id in sorted(goal_ids):
            if goal_id in seen or not adjacency[goal_id]:
                continue
            stack = [goal_id]
            component: list[str] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                component.append(current)
                stack.extend(sorted(adjacency[current], reverse=True))
            components.append(tuple(sorted(component)))
        return tuple(components)

    @staticmethod
    def _conflict_id(goal_ids: tuple[str, ...]) -> str:
        return "conflict:" + ":".join(sorted(goal_ids))


class GoalConflictResolutionApplier:
    """Applies only safe lifecycle effects from a conflict decision.

    The applier never cancels or fails goals as a side effect of conflict
    resolution. Deferred goals remain intact; only an explicitly selected
    winner can be made current.
    """

    def apply(
        self,
        manager: GoalLifecycleManager,
        decision: GoalConflictResolutionDecision,
        *,
        set_current: bool = False,
    ) -> Optional[ManagedGoal]:
        if not isinstance(manager, GoalLifecycleManager):
            raise GoalConflictResolutionApplicationError(
                "manager must be GoalLifecycleManager."
            )
        if not isinstance(decision, GoalConflictResolutionDecision):
            raise GoalConflictResolutionApplicationError(
                "decision must be GoalConflictResolutionDecision."
            )
        if not decision.resolved:
            raise GoalConflictResolutionApplicationError(
                f"Cannot apply unresolved conflict: {decision.conflict_id}."
            )
        if decision.winner_goal_id is None or not set_current:
            return None
        if decision.winner_goal_id in decision.blocked_goal_ids:
            raise GoalConflictResolutionApplicationError(
                "A blocked goal cannot become current."
            )
        try:
            return manager.set_current_goal(decision.winner_goal_id)
        except Exception as exc:
            raise GoalConflictResolutionApplicationError(
                f"Unable to make winner current: {decision.winner_goal_id!r}."
            ) from exc


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
