from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


class GoalUtilityEvaluationError(Exception):
    """Base error for goal utility evaluation."""


class GoalUtilityEvaluationConfigurationError(GoalUtilityEvaluationError):
    """Raised for invalid evaluator configuration."""


class GoalUtilityEvaluationInputError(GoalUtilityEvaluationError):
    """Raised for invalid goal utility inputs."""


@dataclass(frozen=True)
class GoalUtilityWeights:
    value: float = 0.30
    urgency: float = 0.15
    importance: float = 0.15
    progress_need: float = 0.15
    effort_cost: float = 0.10
    risk_cost: float = 0.10
    opportunity_cost: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.value,
            self.urgency,
            self.importance,
            self.progress_need,
            self.effort_cost,
            self.risk_cost,
            self.opportunity_cost,
        )
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values):
            raise GoalUtilityEvaluationConfigurationError("weights must be numeric")
        if any(v < 0 or v > 1 for v in values):
            raise GoalUtilityEvaluationConfigurationError("weights must be in [0, 1]")
        if sum(values) <= 0:
            raise GoalUtilityEvaluationConfigurationError("at least one weight must be positive")

    @property
    def total(self) -> float:
        return sum(
            (
                self.value,
                self.urgency,
                self.importance,
                self.progress_need,
                self.effort_cost,
                self.risk_cost,
                self.opportunity_cost,
            )
        )


@dataclass(frozen=True)
class GoalUtilityInput:
    goal_id: str
    description: str = ""
    value: float = 0.0
    urgency: float = 0.0
    importance: float = 0.0
    progress: float = 0.0
    estimated_effort: float = 0.0
    risk: float = 0.0
    opportunity_cost: float = 0.0
    confidence: float = 1.0
    evidence_quality: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.goal_id).strip():
            raise GoalUtilityEvaluationInputError("goal_id must not be empty")
        if not isinstance(self.metadata, Mapping):
            raise GoalUtilityEvaluationInputError("metadata must be a mapping")
        for name in (
            "value",
            "urgency",
            "importance",
            "progress",
            "estimated_effort",
            "risk",
            "opportunity_cost",
            "confidence",
            "evidence_quality",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise GoalUtilityEvaluationInputError(f"{name} must be numeric")
            if not 0 <= float(value) <= 1:
                raise GoalUtilityEvaluationInputError(f"{name} must be in [0, 1]")

    @property
    def progress_need(self) -> float:
        return max(0.0, min(1.0, 1.0 - float(self.progress)))


@dataclass(frozen=True)
class GoalUtilityEvaluation:
    goal_id: str
    description: str
    benefit_score: float
    cost_score: float
    expected_utility: float
    utility_score: float
    confidence_adjusted_utility: float
    recommended: bool
    recommendation: str
    rationale: str
    components: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def should_prioritize(self) -> bool:
        return self.recommended

    @property
    def net_utility(self) -> float:
        return self.expected_utility


class GoalUtilityEvaluator:
    """Pure evaluator for goal value, cost, risk, and expected utility."""

    def __init__(
        self,
        *,
        weights: GoalUtilityWeights | None = None,
        recommendation_threshold: float = 0.50,
        minimum_confidence: float = 0.0,
    ) -> None:
        self.weights = weights or GoalUtilityWeights()
        self.recommendation_threshold = self._bounded_config(
            recommendation_threshold, "recommendation_threshold"
        )
        self.minimum_confidence = self._bounded_config(
            minimum_confidence, "minimum_confidence"
        )

    @staticmethod
    def _bounded_config(value: float, name: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GoalUtilityEvaluationConfigurationError(f"{name} must be numeric")
        value = float(value)
        if not 0 <= value <= 1:
            raise GoalUtilityEvaluationConfigurationError(f"{name} must be in [0, 1]")
        return value

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def evaluate(self, goal: GoalUtilityInput | Mapping[str, Any] | Any) -> GoalUtilityEvaluation:
        normalized = self._normalize(goal)
        w = self.weights
        benefit_raw = (
            w.value * normalized.value
            + w.urgency * normalized.urgency
            + w.importance * normalized.importance
            + w.progress_need * normalized.progress_need
        )
        benefit_weight = w.value + w.urgency + w.importance + w.progress_need
        cost_raw = (
            w.effort_cost * normalized.estimated_effort
            + w.risk_cost * normalized.risk
            + w.opportunity_cost * normalized.opportunity_cost
        )
        cost_weight = w.effort_cost + w.risk_cost + w.opportunity_cost
        benefit_score = self._clamp(benefit_raw / benefit_weight) if benefit_weight else 0.0
        cost_score = self._clamp(cost_raw / cost_weight) if cost_weight else 0.0
        expected_utility = round(benefit_score - cost_score, 12)
        utility_score = round(self._clamp((expected_utility + 1.0) / 2.0), 12)
        confidence_factor = self._clamp(normalized.confidence * normalized.evidence_quality)
        confidence_adjusted_utility = round(expected_utility * confidence_factor, 12)
        recommended = (
            confidence_adjusted_utility >= (self.recommendation_threshold * 2.0 - 1.0)
            and normalized.confidence >= self.minimum_confidence
        )
        if normalized.confidence < self.minimum_confidence:
            recommendation = "insufficient_confidence"
        elif recommended:
            recommendation = "prioritize"
        else:
            recommendation = "defer"
        rationale = (
            f"benefit={benefit_score:.6f}; cost={cost_score:.6f}; "
            f"expected_utility={expected_utility:.6f}; "
            f"confidence_factor={confidence_factor:.6f}"
        )
        components = {
            "value": float(normalized.value),
            "urgency": float(normalized.urgency),
            "importance": float(normalized.importance),
            "progress_need": float(normalized.progress_need),
            "estimated_effort": float(normalized.estimated_effort),
            "risk": float(normalized.risk),
            "opportunity_cost": float(normalized.opportunity_cost),
            "confidence": float(normalized.confidence),
            "evidence_quality": float(normalized.evidence_quality),
        }
        return GoalUtilityEvaluation(
            goal_id=normalized.goal_id,
            description=normalized.description,
            benefit_score=round(benefit_score, 12),
            cost_score=round(cost_score, 12),
            expected_utility=expected_utility,
            utility_score=utility_score,
            confidence_adjusted_utility=confidence_adjusted_utility,
            recommended=recommended,
            recommendation=recommendation,
            rationale=rationale,
            components=components,
            metadata=dict(normalized.metadata),
        )

    def evaluate_many(self, goals: list[GoalUtilityInput | Mapping[str, Any] | Any]) -> list[GoalUtilityEvaluation]:
        results = [self.evaluate(goal) for goal in goals]
        return sorted(results, key=lambda item: (item.confidence_adjusted_utility, item.utility_score), reverse=True)

    def _normalize(self, goal: GoalUtilityInput | Mapping[str, Any] | Any) -> GoalUtilityInput:
        if isinstance(goal, GoalUtilityInput):
            return goal
        if isinstance(goal, Mapping):
            data = dict(goal)
        else:
            data = {
                name: getattr(goal, name)
                for name in (
                    "goal_id",
                    "description",
                    "value",
                    "urgency",
                    "importance",
                    "progress",
                    "estimated_effort",
                    "risk",
                    "opportunity_cost",
                    "confidence",
                    "evidence_quality",
                    "metadata",
                )
                if hasattr(goal, name)
            }
        goal_id = data.get("goal_id", data.get("id", ""))
        description = data.get("description", data.get("goal", ""))
        metadata = data.get("metadata", {})
        return GoalUtilityInput(
            goal_id=str(goal_id),
            description=str(description or ""),
            value=self._number(data.get("value", data.get("expected_value", 0.0)), "value"),
            urgency=self._number(data.get("urgency", 0.0), "urgency"),
            importance=self._number(data.get("importance", data.get("priority_score", 0.0)), "importance"),
            progress=self._number(data.get("progress", data.get("measured_progress", 0.0)), "progress"),
            estimated_effort=self._number(data.get("estimated_effort", data.get("effort", 0.0)), "estimated_effort"),
            risk=self._number(data.get("risk", data.get("risk_score", 0.0)), "risk"),
            opportunity_cost=self._number(data.get("opportunity_cost", 0.0), "opportunity_cost"),
            confidence=self._number(data.get("confidence", 1.0), "confidence"),
            evidence_quality=self._number(data.get("evidence_quality", 1.0), "evidence_quality"),
            metadata=metadata,
        )

    @staticmethod
    def _number(value: Any, name: str) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise GoalUtilityEvaluationInputError(f"{name} must be numeric")
        return float(value)
