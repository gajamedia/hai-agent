from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_management import GoalStatus, GoalLifecycleManager, ManagedGoal
from .goal_progress_evaluation import GoalAchievementEvaluation


class MultiGoalArbitrationError(Exception):
    """Base exception for multi-goal arbitration."""


class MultiGoalArbitrationConfigurationError(MultiGoalArbitrationError):
    """Raised for invalid arbitration configuration."""


class MultiGoalArbitrationInputError(MultiGoalArbitrationError):
    """Raised when arbitration inputs cannot be normalized safely."""


class MultiGoalArbitrationApplicationError(MultiGoalArbitrationError):
    """Raised when a chosen goal cannot be applied to a goal manager."""


@dataclass(frozen=True)
class MultiGoalArbitrationWeights:
    """Normalized weights for deterministic multi-goal scoring."""

    priority: float = 0.50
    urgency: float = 0.25
    progress_need: float = 0.15
    achievement_gap: float = 0.10

    def __post_init__(self) -> None:
        values = {
            "priority": self.priority,
            "urgency": self.urgency,
            "progress_need": self.progress_need,
            "achievement_gap": self.achievement_gap,
        }
        total = 0.0
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise MultiGoalArbitrationConfigurationError(
                    f"{name} must be numeric."
                ) from exc
            if number < 0.0:
                raise MultiGoalArbitrationConfigurationError(
                    f"{name} must be non-negative."
                )
            object.__setattr__(self, name, number)
            total += number
        if total <= 0.0:
            raise MultiGoalArbitrationConfigurationError(
                "At least one arbitration weight must be positive."
            )
        for name in values:
            object.__setattr__(self, name, getattr(self, name) / total)


@dataclass(frozen=True)
class MultiGoalArbitrationThresholds:
    """Safety thresholds used when switching the current goal."""

    preempt_score_margin: float = 0.05
    minimum_urgency: float = 0.0

    def __post_init__(self) -> None:
        for name in ("preempt_score_margin", "minimum_urgency"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise MultiGoalArbitrationConfigurationError(
                    f"{name} must be numeric."
                ) from exc
            if not 0.0 <= value <= 1.0:
                raise MultiGoalArbitrationConfigurationError(
                    f"{name} must be between 0.0 and 1.0."
                )
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MultiGoalArbitrationCandidate:
    goal_id: str
    description: str
    status: GoalStatus
    sequence: int
    priority: float
    urgency: float
    progress: float
    progress_need: float
    achievement_score: float
    achievement_gap: float
    score: float
    eligible: bool
    exclusion_reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class MultiGoalArbitrationDecision:
    selected_goal_id: Optional[str]
    previous_goal_id: Optional[str]
    selected_score: float
    previous_score: float
    score_margin: float
    switched: bool
    preempted: bool
    reason: str
    ranked_goal_ids: tuple[str, ...]
    eligible_goal_ids: tuple[str, ...]
    candidates: tuple[MultiGoalArbitrationCandidate, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_selection(self) -> bool:
        return self.selected_goal_id is not None

    @property
    def should_switch(self) -> bool:
        return self.switched


class MultiGoalArbiter:
    """Deterministic, provider-neutral arbitration across multiple goals."""

    def __init__(
        self,
        *,
        weights: Optional[MultiGoalArbitrationWeights] = None,
        thresholds: Optional[MultiGoalArbitrationThresholds] = None,
    ) -> None:
        self.weights = weights or MultiGoalArbitrationWeights()
        self.thresholds = thresholds or MultiGoalArbitrationThresholds()

    def arbitrate(
        self,
        goals: Iterable[ManagedGoal | Mapping[str, Any]],
        *,
        current_goal_id: Optional[str] = None,
        urgency: Optional[Mapping[str, float]] = None,
        evaluations: Optional[
            Mapping[str, GoalAchievementEvaluation | Mapping[str, Any]]
        ] = None,
        blocked_goal_ids: Iterable[str] = (),
    ) -> MultiGoalArbitrationDecision:
        urgency_map = self._normalize_urgency(urgency or {})
        blocked = {str(item).strip() for item in blocked_goal_ids}
        normalized = [self._normalize_goal(goal) for goal in goals]
        candidate_records: list[MultiGoalArbitrationCandidate] = []

        for goal in normalized:
            goal_urgency = urgency_map.get(goal.goal_id, 0.0)
            evaluation = self._get_evaluation(evaluations or {}, goal.goal_id)
            achievement_score = self._evaluation_score(evaluation)
            progress_need = _bounded(1.0 - goal.progress)
            achievement_gap = _bounded(1.0 - achievement_score)
            score = _bounded(
                self.weights.priority * goal.priority
                + self.weights.urgency * goal_urgency
                + self.weights.progress_need * progress_need
                + self.weights.achievement_gap * achievement_gap
            )

            exclusion_reason = None
            eligible = True
            if goal.status.terminal:
                eligible = False
                exclusion_reason = "terminal_goal"
            elif goal.goal_id in blocked:
                eligible = False
                exclusion_reason = "blocked_goal"
            elif goal_urgency < self.thresholds.minimum_urgency:
                eligible = False
                exclusion_reason = "below_minimum_urgency"

            candidate_records.append(
                MultiGoalArbitrationCandidate(
                    goal_id=goal.goal_id,
                    description=goal.description,
                    status=goal.status,
                    sequence=goal.sequence,
                    priority=round(goal.priority, 12),
                    urgency=round(goal_urgency, 12),
                    progress=round(goal.progress, 12),
                    progress_need=round(progress_need, 12),
                    achievement_score=round(achievement_score, 12),
                    achievement_gap=round(achievement_gap, 12),
                    score=round(score, 12),
                    eligible=eligible,
                    exclusion_reason=exclusion_reason,
                    metadata=dict(goal.metadata),
                )
            )

        eligible = [item for item in candidate_records if item.eligible]
        eligible.sort(key=lambda item: self._ranking_key(item))
        ranked_ids = tuple(item.goal_id for item in eligible)
        all_candidates = tuple(
            sorted(candidate_records, key=lambda item: item.sequence)
        )
        current = next(
            (item for item in candidate_records if item.goal_id == current_goal_id),
            None,
        )

        if not eligible:
            return MultiGoalArbitrationDecision(
                selected_goal_id=None,
                previous_goal_id=current_goal_id,
                selected_score=0.0,
                previous_score=round(current.score, 12) if current else 0.0,
                score_margin=0.0,
                switched=False,
                preempted=False,
                reason="no_eligible_goals",
                ranked_goal_ids=(),
                eligible_goal_ids=(),
                candidates=all_candidates,
            )

        top = eligible[0]
        if current_goal_id is None:
            return self._decision(
                selected=top,
                previous=None,
                reason="highest_arbitration_score",
                switched=False,
                preempted=False,
                ranked_ids=ranked_ids,
                eligible_ids=ranked_ids,
                candidates=all_candidates,
            )

        if current is None or not current.eligible:
            return self._decision(
                selected=top,
                previous=current,
                reason="current_goal_not_eligible",
                switched=True,
                preempted=False,
                ranked_ids=ranked_ids,
                eligible_ids=ranked_ids,
                candidates=all_candidates,
            )

        if top.goal_id == current.goal_id:
            return self._decision(
                selected=current,
                previous=current,
                reason="continue_current_goal",
                switched=False,
                preempted=False,
                ranked_ids=ranked_ids,
                eligible_ids=ranked_ids,
                candidates=all_candidates,
            )

        margin = top.score - current.score
        if margin > self.thresholds.preempt_score_margin:
            return self._decision(
                selected=top,
                previous=current,
                reason="higher_arbitration_score_preempts_current",
                switched=True,
                preempted=True,
                ranked_ids=ranked_ids,
                eligible_ids=ranked_ids,
                candidates=all_candidates,
            )

        return self._decision(
            selected=current,
            previous=current,
            reason="current_goal_retained_by_preemption_margin",
            switched=False,
            preempted=False,
            ranked_ids=ranked_ids,
            eligible_ids=ranked_ids,
            candidates=all_candidates,
        )

    def arbitrate_manager(
        self,
        manager: GoalLifecycleManager,
        *,
        urgency: Optional[Mapping[str, float]] = None,
        evaluations: Optional[
            Mapping[str, GoalAchievementEvaluation | Mapping[str, Any]]
        ] = None,
        blocked_goal_ids: Iterable[str] = (),
    ) -> MultiGoalArbitrationDecision:
        if not isinstance(manager, GoalLifecycleManager):
            raise MultiGoalArbitrationInputError(
                "manager must be GoalLifecycleManager."
            )
        snapshot = manager.snapshot()
        return self.arbitrate(
            snapshot.goals,
            current_goal_id=snapshot.current_goal_id,
            urgency=urgency,
            evaluations=evaluations,
            blocked_goal_ids=blocked_goal_ids,
        )

    @staticmethod
    def _ranking_key(candidate: MultiGoalArbitrationCandidate) -> tuple[float, float, float, int]:
        return (
            -candidate.score,
            -candidate.priority,
            -candidate.urgency,
            candidate.sequence,
        )

    def _decision(
        self,
        *,
        selected: MultiGoalArbitrationCandidate,
        previous: Optional[MultiGoalArbitrationCandidate],
        reason: str,
        switched: bool,
        preempted: bool,
        ranked_ids: tuple[str, ...],
        eligible_ids: tuple[str, ...],
        candidates: tuple[MultiGoalArbitrationCandidate, ...],
    ) -> MultiGoalArbitrationDecision:
        previous_score = previous.score if previous else 0.0
        return MultiGoalArbitrationDecision(
            selected_goal_id=selected.goal_id,
            previous_goal_id=previous.goal_id if previous else None,
            selected_score=round(selected.score, 12),
            previous_score=round(previous_score, 12),
            score_margin=round(selected.score - previous_score, 12),
            switched=switched,
            preempted=preempted,
            reason=reason,
            ranked_goal_ids=ranked_ids,
            eligible_goal_ids=eligible_ids,
            candidates=candidates,
            metadata={
                "score_margin_threshold": self.thresholds.preempt_score_margin,
                "weights": {
                    "priority": self.weights.priority,
                    "urgency": self.weights.urgency,
                    "progress_need": self.weights.progress_need,
                    "achievement_gap": self.weights.achievement_gap,
                },
            },
        )

    @staticmethod
    def _normalize_goal(value: ManagedGoal | Mapping[str, Any]) -> ManagedGoal:
        if isinstance(value, ManagedGoal):
            return value
        if not isinstance(value, Mapping):
            raise MultiGoalArbitrationInputError(
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
            raise MultiGoalArbitrationInputError("Invalid goal mapping.") from exc

    @staticmethod
    def _normalize_urgency(values: Mapping[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise MultiGoalArbitrationInputError(
                    f"Urgency for {key!r} must be numeric."
                ) from exc
            if not 0.0 <= number <= 1.0:
                raise MultiGoalArbitrationInputError(
                    f"Urgency for {key!r} must be between 0.0 and 1.0."
                )
            normalized[str(key).strip()] = number
        return normalized

    @staticmethod
    def _get_evaluation(
        evaluations: Mapping[str, GoalAchievementEvaluation | Mapping[str, Any]],
        goal_id: str,
    ) -> Optional[Any]:
        return evaluations.get(goal_id)

    @staticmethod
    def _evaluation_score(value: Optional[Any]) -> float:
        if value is None:
            return 0.0
        if isinstance(value, Mapping):
            raw = value.get("achievement_score", 0.0)
        else:
            raw = getattr(value, "achievement_score", 0.0)
        try:
            return _bounded(float(raw))
        except (TypeError, ValueError):
            raise MultiGoalArbitrationInputError(
                "achievement_score must be numeric."
            )


class MultiGoalArbitrationApplier:
    """Applies a previously computed arbitration decision to GoalLifecycleManager."""

    def apply(
        self,
        manager: GoalLifecycleManager,
        decision: MultiGoalArbitrationDecision,
    ) -> Optional[ManagedGoal]:
        if not isinstance(manager, GoalLifecycleManager):
            raise MultiGoalArbitrationApplicationError(
                "manager must be GoalLifecycleManager."
            )
        if not isinstance(decision, MultiGoalArbitrationDecision):
            raise MultiGoalArbitrationApplicationError(
                "decision must be MultiGoalArbitrationDecision."
            )
        try:
            return manager.set_current_goal(decision.selected_goal_id)
        except Exception as exc:
            raise MultiGoalArbitrationApplicationError(
                f"Unable to apply selected goal: {decision.selected_goal_id!r}."
            ) from exc


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
