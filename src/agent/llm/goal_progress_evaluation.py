from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_management import GoalStatus, ManagedGoal


class GoalEvaluationError(Exception):
    """Base exception for goal progress and achievement evaluation."""


class GoalEvaluationConfigurationError(GoalEvaluationError):
    """Raised when evaluator configuration is invalid."""


class GoalEvaluationInputError(GoalEvaluationError):
    """Raised when an evaluation input cannot be normalized safely."""


class GoalEvaluationEvidenceError(GoalEvaluationError):
    """Raised when evidence is malformed or unusable."""


@dataclass(frozen=True)
class GoalAchievementEvidence:
    """Normalized provider-neutral evidence about a goal."""

    source: str = "unknown"
    success: Optional[bool] = None
    progress_delta: Optional[float] = None
    progress: Optional[float] = None
    confidence: float = 0.5
    quality: float = 0.5
    achieved: Optional[bool] = None
    observation: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = str(self.source).strip() or "unknown"
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))
        object.__setattr__(self, "quality", _bounded(self.quality, "quality"))
        if self.progress_delta is not None:
            try:
                delta = float(self.progress_delta)
            except (TypeError, ValueError) as exc:
                raise ValueError("progress_delta must be numeric.") from exc
            if delta < 0.0:
                raise ValueError("progress_delta must be non-negative.")
            object.__setattr__(self, "progress_delta", delta)
        if self.progress is not None:
            object.__setattr__(self, "progress", _bounded(self.progress, "progress"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class GoalAchievementEvaluation:
    """Immutable evaluation result; does not mutate a goal manager."""

    goal_id: str
    goal_description: str
    previous_progress: float
    measured_progress: float
    progress_delta: float
    evidence_count: int
    evidence_quality: float
    success_evidence: float
    achievement_evidence: float
    achievement_score: float
    achieved: bool
    progress_measured: bool
    confidence: float
    recommended_status: str
    reason: str
    evidence: tuple[GoalAchievementEvidence, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def should_complete(self) -> bool:
        return self.achieved

    @property
    def should_update_progress(self) -> bool:
        return self.progress_measured and self.measured_progress != self.previous_progress


@dataclass(frozen=True)
class GoalEvaluationThresholds:
    """Bounded thresholds and weights used by the deterministic evaluator."""

    achievement_threshold: float = 0.8
    minimum_evidence_quality: float = 0.35
    success_weight: float = 0.45
    progress_weight: float = 0.35
    explicit_achievement_weight: float = 0.20
    confidence_floor: float = 0.0
    measurement_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "achievement_threshold",
            "minimum_evidence_quality",
            "confidence_floor",
        ):
            object.__setattr__(self, name, _bounded(getattr(self, name), name))
        for name in (
            "success_weight",
            "progress_weight",
            "explicit_achievement_weight",
        ):
            object.__setattr__(self, name, _non_negative(getattr(self, name), name))
        try:
            tolerance = float(self.measurement_tolerance)
        except (TypeError, ValueError) as exc:
            raise GoalEvaluationConfigurationError(
                "measurement_tolerance must be numeric."
            ) from exc
        if tolerance < 0.0:
            raise GoalEvaluationConfigurationError(
                "measurement_tolerance must be non-negative."
            )
        object.__setattr__(self, "measurement_tolerance", tolerance)
        weight_sum = (
            self.success_weight
            + self.progress_weight
            + self.explicit_achievement_weight
        )
        if weight_sum <= 0.0:
            raise GoalEvaluationConfigurationError("At least one evidence weight must be positive.")
        object.__setattr__(self, "success_weight", self.success_weight / weight_sum)
        object.__setattr__(self, "progress_weight", self.progress_weight / weight_sum)
        object.__setattr__(
            self,
            "explicit_achievement_weight",
            self.explicit_achievement_weight / weight_sum,
        )


class GoalProgressAchievementEvaluator:
    """Deterministic, side-effect-free evaluator for goal achievement.

    Evidence is normalized first, then combined into a bounded achievement
    score. The evaluator never changes the goal manager; callers decide whether
    to update progress or transition lifecycle state.
    """

    def __init__(
        self,
        *,
        thresholds: Optional[GoalEvaluationThresholds] = None,
    ) -> None:
        self.thresholds = thresholds or GoalEvaluationThresholds()

    def evaluate(
        self,
        goal: ManagedGoal | Mapping[str, Any],
        evidence: Iterable[GoalAchievementEvidence | Mapping[str, Any] | Any] = (),
    ) -> GoalAchievementEvaluation:
        normalized_goal = _normalize_goal(goal)
        normalized_evidence = tuple(self._normalize_evidence(item) for item in evidence)
        previous_progress = _bounded(normalized_goal.progress, "goal.progress")

        measured_progress_values = [
            item.progress
            for item in normalized_evidence
            if item.progress is not None
        ]
        measured_delta_values = [
            item.progress_delta
            for item in normalized_evidence
            if item.progress_delta is not None
        ]
        progress_measured = bool(measured_progress_values or measured_delta_values)

        if measured_progress_values:
            measured_progress = max(measured_progress_values)
        elif measured_delta_values:
            measured_progress = min(1.0, max(0.0, previous_progress + sum(measured_delta_values)))
        else:
            measured_progress = previous_progress

        delta = measured_progress - previous_progress
        if abs(delta) <= self.thresholds.measurement_tolerance:
            delta = 0.0

        usable = [
            item
            for item in normalized_evidence
            if item.quality >= self.thresholds.minimum_evidence_quality
            and item.confidence >= self.thresholds.confidence_floor
        ]
        evidence_quality = _weighted_average(
            [(item.quality, item.confidence) for item in normalized_evidence]
        )

        success_evidence = _weighted_signal(
            usable,
            lambda item: 1.0 if item.success is True else 0.0,
        )
        explicit_achievement = _weighted_signal(
            usable,
            lambda item: 1.0 if item.achieved is True else 0.0,
        )
        progress_signal = measured_progress if progress_measured else 0.0

        score = (
            self.thresholds.success_weight * success_evidence
            + self.thresholds.progress_weight * progress_signal
            + self.thresholds.explicit_achievement_weight * explicit_achievement
        )
        score = _bounded(score, "achievement_score")

        evidence_sufficient = bool(usable)
        achieved = (
            normalized_goal.progress >= self.thresholds.achievement_threshold
            or (
                evidence_sufficient
                and progress_signal >= self.thresholds.achievement_threshold
            )
            or (
                evidence_sufficient
                and score >= self.thresholds.achievement_threshold
                and explicit_achievement > 0.0
            )
        )

        if achieved:
            reason = "goal_achievement_threshold_met"
            recommended_status = "completed"
        elif not normalized_evidence:
            reason = "no_achievement_evidence"
            recommended_status = "unchanged"
        elif not evidence_sufficient:
            reason = "insufficient_evidence_quality"
            recommended_status = "needs_more_evidence"
        elif delta > 0.0:
            reason = "measurable_progress_but_not_achieved"
            recommended_status = "active"
        elif any(item.success is False for item in usable):
            reason = "failure_evidence_without_achievement"
            recommended_status = "active"
        else:
            reason = "insufficient_achievement_signal"
            recommended_status = "active"

        confidence = _bounded(
            0.30 + 0.70 * evidence_quality if usable else 0.30,
            "confidence",
        )
        if achieved and not usable:
            confidence = max(confidence, 0.60)

        return GoalAchievementEvaluation(
            goal_id=normalized_goal.goal_id,
            goal_description=normalized_goal.description,
            previous_progress=round(previous_progress, 12),
            measured_progress=round(measured_progress, 12),
            progress_delta=round(delta, 12),
            evidence_count=len(normalized_evidence),
            evidence_quality=round(evidence_quality, 12),
            success_evidence=round(success_evidence, 12),
            achievement_evidence=round(explicit_achievement, 12),
            achievement_score=round(score, 12),
            achieved=achieved,
            progress_measured=progress_measured,
            confidence=round(confidence, 12),
            recommended_status=recommended_status,
            reason=reason,
            evidence=normalized_evidence,
            metadata={"evidence_sufficient": evidence_sufficient},
        )

    def _normalize_evidence(
        self,
        value: GoalAchievementEvidence | Mapping[str, Any] | Any,
    ) -> GoalAchievementEvidence:
        if isinstance(value, GoalAchievementEvidence):
            return value
        if isinstance(value, Mapping):
            data = dict(value)
        else:
            data = {
                field: getattr(value, field)
                for field in (
                    "source",
                    "success",
                    "progress_delta",
                    "progress",
                    "confidence",
                    "quality",
                    "achieved",
                    "observation",
                    "metadata",
                )
                if hasattr(value, field)
            }
        if not data:
            raise GoalEvaluationEvidenceError("Evidence must be a mapping, evidence object, or object with supported attributes.")
        if "progress_delta" in data and data["progress_delta"] is not None:
            try:
                delta = float(data["progress_delta"])
            except (TypeError, ValueError) as exc:
                raise GoalEvaluationEvidenceError("progress_delta must be numeric.") from exc
            if delta < 0.0:
                raise GoalEvaluationEvidenceError("progress_delta must be non-negative.")
            data["progress_delta"] = delta
        try:
            return GoalAchievementEvidence(**data)
        except (TypeError, ValueError) as exc:
            raise GoalEvaluationEvidenceError("Invalid goal achievement evidence.") from exc


def _normalize_status(value: Any) -> GoalStatus:
    if isinstance(value, GoalStatus):
        return value
    try:
        return GoalStatus(str(value))
    except ValueError as exc:
        raise GoalEvaluationInputError(f"Invalid goal status: {value!r}.") from exc


def _normalize_goal(value: ManagedGoal | Mapping[str, Any]) -> ManagedGoal:
    if isinstance(value, ManagedGoal):
        return value
    if not isinstance(value, Mapping):
        raise GoalEvaluationInputError("goal must be ManagedGoal or mapping.")
    try:
        return ManagedGoal(
            goal_id=str(value["goal_id"]),
            description=str(value["description"]),
            priority=float(value.get("priority", 0.5)),
            parent_id=value.get("parent_id"),
            status=_normalize_status(value.get("status", GoalStatus.PENDING)),
            progress=float(value.get("progress", 0.0)),
            payload=value.get("payload"),
            metadata=value.get("metadata") or {},
            sequence=int(value.get("sequence", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GoalEvaluationInputError("Invalid goal input.") from exc


def _weighted_average(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    denominator = sum(weight for _, weight in values)
    if denominator <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in values) / denominator


def _weighted_signal(
    evidence: list[GoalAchievementEvidence],
    signal,
) -> float:
    if not evidence:
        return 0.0
    return _weighted_average([(signal(item), item.quality * item.confidence) for item in evidence])


def _non_negative(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GoalEvaluationConfigurationError(f"{name} must be numeric.") from exc
    if number < 0.0:
        raise GoalEvaluationConfigurationError(f"{name} must be non-negative.")
    return number


def _bounded(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GoalEvaluationConfigurationError(f"{name} must be numeric.") from exc
    if not 0.0 <= number <= 1.0:
        raise GoalEvaluationConfigurationError(f"{name} must be between 0.0 and 1.0.")
    return number
