from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class IntrinsicMotivationError(Exception):
    """Base error for intrinsic motivation evaluation."""


class IntrinsicMotivationConfigurationError(
    IntrinsicMotivationError
):
    """Raised for invalid evaluator configuration."""


class IntrinsicMotivationInputError(
    IntrinsicMotivationError
):
    """Raised for invalid intrinsic motivation input."""


@dataclass(frozen=True)
class IntrinsicMotivationWeights:
    novelty: float = 0.20
    curiosity: float = 0.20
    exploration_value: float = 0.20
    information_gain: float = 0.15
    uncertainty: float = 0.10
    learning_value: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.novelty,
            self.curiosity,
            self.exploration_value,
            self.information_gain,
            self.uncertainty,
            self.learning_value,
        )

        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            for value in values
        ):
            raise IntrinsicMotivationConfigurationError(
                "weights must be numeric"
            )

        if any(
            value < 0 or value > 1
            for value in values
        ):
            raise IntrinsicMotivationConfigurationError(
                "weights must be in [0, 1]"
            )

        if sum(values) <= 0:
            raise IntrinsicMotivationConfigurationError(
                "at least one weight must be positive"
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.novelty,
                self.curiosity,
                self.exploration_value,
                self.information_gain,
                self.uncertainty,
                self.learning_value,
            )
        )


@dataclass(frozen=True)
class IntrinsicMotivationInput:
    goal_id: str
    description: str = ""

    novelty: float = 0.0
    curiosity: float = 0.0
    exploration_value: float = 0.0
    information_gain: float = 0.0
    uncertainty: float = 0.0
    learning_value: float = 0.0

    confidence: float = 1.0
    evidence_quality: float = 1.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not str(self.goal_id).strip():
            raise IntrinsicMotivationInputError(
                "goal_id must not be empty"
            )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise IntrinsicMotivationInputError(
                "metadata must be a mapping"
            )

        for name in (
            "novelty",
            "curiosity",
            "exploration_value",
            "information_gain",
            "uncertainty",
            "learning_value",
            "confidence",
            "evidence_quality",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise IntrinsicMotivationInputError(
                    f"{name} must be numeric"
                )

            if not 0 <= float(value) <= 1:
                raise IntrinsicMotivationInputError(
                    f"{name} must be in [0, 1]"
                )


@dataclass(frozen=True)
class IntrinsicMotivationEvaluation:
    goal_id: str
    description: str

    intrinsic_score: float
    confidence_adjusted_score: float

    recommended: bool
    recommendation: str
    rationale: str

    components: Mapping[str, float] = field(
        default_factory=dict
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    @property
    def should_explore(self) -> bool:
        return self.recommended

    @property
    def exploration_score(self) -> float:
        return self.intrinsic_score

    @property
    def adjusted_exploration_score(self) -> float:
        return self.confidence_adjusted_score


class IntrinsicMotivationEvaluator:
    """
    Pure evaluator for intrinsic motivation.

    The evaluator measures whether a goal is valuable
    because it improves exploration, curiosity,
    information acquisition, or learning.
    """

    def __init__(
        self,
        *,
        weights: IntrinsicMotivationWeights | None = None,
        exploration_threshold: float = 0.50,
        minimum_confidence: float = 0.0,
    ) -> None:
        self.weights = (
            weights
            or IntrinsicMotivationWeights()
        )

        self.exploration_threshold = (
            self._bounded_config(
                exploration_threshold,
                "exploration_threshold",
            )
        )

        self.minimum_confidence = (
            self._bounded_config(
                minimum_confidence,
                "minimum_confidence",
            )
        )

    @staticmethod
    def _bounded_config(
        value: float,
        name: str,
    ) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise IntrinsicMotivationConfigurationError(
                f"{name} must be numeric"
            )

        value = float(value)

        if not 0 <= value <= 1:
            raise IntrinsicMotivationConfigurationError(
                f"{name} must be in [0, 1]"
            )

        return value

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(1.0, float(value)),
        )

    def evaluate(
        self,
        goal: (
            IntrinsicMotivationInput
            | Mapping[str, Any]
            | Any
        ),
    ) -> IntrinsicMotivationEvaluation:
        normalized = self._normalize(goal)

        weights = self.weights

        weighted_score = (
            weights.novelty
            * normalized.novelty
            + weights.curiosity
            * normalized.curiosity
            + weights.exploration_value
            * normalized.exploration_value
            + weights.information_gain
            * normalized.information_gain
            + weights.uncertainty
            * normalized.uncertainty
            + weights.learning_value
            * normalized.learning_value
        )

        intrinsic_score = self._clamp(
            weighted_score / weights.total
        )

        confidence_factor = self._clamp(
            normalized.confidence
            * normalized.evidence_quality
        )

        confidence_adjusted_score = round(
            intrinsic_score
            * confidence_factor,
            12,
        )

        recommended = (
            confidence_adjusted_score
            >= self.exploration_threshold
            and normalized.confidence
            >= self.minimum_confidence
        )

        if (
            normalized.confidence
            < self.minimum_confidence
        ):
            recommendation = (
                "insufficient_confidence"
            )
        elif recommended:
            recommendation = "explore"
        else:
            recommendation = "defer"

        rationale = (
            f"novelty={normalized.novelty:.6f}; "
            f"curiosity={normalized.curiosity:.6f}; "
            f"exploration_value="
            f"{normalized.exploration_value:.6f}; "
            f"information_gain="
            f"{normalized.information_gain:.6f}; "
            f"uncertainty="
            f"{normalized.uncertainty:.6f}; "
            f"learning_value="
            f"{normalized.learning_value:.6f}; "
            f"intrinsic_score="
            f"{intrinsic_score:.6f}; "
            f"confidence_factor="
            f"{confidence_factor:.6f}"
        )

        components = {
            "novelty": float(
                normalized.novelty
            ),
            "curiosity": float(
                normalized.curiosity
            ),
            "exploration_value": float(
                normalized.exploration_value
            ),
            "information_gain": float(
                normalized.information_gain
            ),
            "uncertainty": float(
                normalized.uncertainty
            ),
            "learning_value": float(
                normalized.learning_value
            ),
            "confidence": float(
                normalized.confidence
            ),
            "evidence_quality": float(
                normalized.evidence_quality
            ),
        }

        return IntrinsicMotivationEvaluation(
            goal_id=normalized.goal_id,
            description=normalized.description,
            intrinsic_score=round(
                intrinsic_score,
                12,
            ),
            confidence_adjusted_score=(
                confidence_adjusted_score
            ),
            recommended=recommended,
            recommendation=recommendation,
            rationale=rationale,
            components=components,
            metadata=dict(
                normalized.metadata
            ),
        )

    def evaluate_many(
        self,
        goals: list[
            IntrinsicMotivationInput
            | Mapping[str, Any]
            | Any
        ],
    ) -> list[IntrinsicMotivationEvaluation]:
        results = [
            self.evaluate(goal)
            for goal in goals
        ]

        return sorted(
            results,
            key=lambda item: (
                item.confidence_adjusted_score,
                item.intrinsic_score,
            ),
            reverse=True,
        )

    def _normalize(
        self,
        goal: (
            IntrinsicMotivationInput
            | Mapping[str, Any]
            | Any
        ),
    ) -> IntrinsicMotivationInput:
        if isinstance(
            goal,
            IntrinsicMotivationInput,
        ):
            return goal

        if isinstance(goal, Mapping):
            data = dict(goal)

        else:
            data = {
                name: getattr(goal, name)
                for name in (
                    "goal_id",
                    "description",
                    "novelty",
                    "curiosity",
                    "exploration_value",
                    "information_gain",
                    "uncertainty",
                    "learning_value",
                    "confidence",
                    "evidence_quality",
                    "metadata",
                )
                if hasattr(goal, name)
            }

        goal_id = data.get(
            "goal_id",
            data.get("id", ""),
        )

        description = data.get(
            "description",
            data.get("goal", ""),
        )

        metadata = data.get(
            "metadata",
            {},
        )

        return IntrinsicMotivationInput(
            goal_id=str(goal_id),
            description=str(
                description or ""
            ),
            novelty=self._number(
                data.get(
                    "novelty",
                    data.get(
                        "novelty_score",
                        0.0,
                    ),
                ),
                "novelty",
            ),
            curiosity=self._number(
                data.get(
                    "curiosity",
                    data.get(
                        "curiosity_score",
                        0.0,
                    ),
                ),
                "curiosity",
            ),
            exploration_value=self._number(
                data.get(
                    "exploration_value",
                    data.get(
                        "exploration_score",
                        0.0,
                    ),
                ),
                "exploration_value",
            ),
            information_gain=self._number(
                data.get(
                    "information_gain",
                    data.get(
                        "information_gain_score",
                        0.0,
                    ),
                ),
                "information_gain",
            ),
            uncertainty=self._number(
                data.get(
                    "uncertainty",
                    data.get(
                        "uncertainty_score",
                        0.0,
                    ),
                ),
                "uncertainty",
            ),
            learning_value=self._number(
                data.get(
                    "learning_value",
                    data.get(
                        "learning_score",
                        0.0,
                    ),
                ),
                "learning_value",
            ),
            confidence=self._number(
                data.get(
                    "confidence",
                    1.0,
                ),
                "confidence",
            ),
            evidence_quality=self._number(
                data.get(
                    "evidence_quality",
                    1.0,
                ),
                "evidence_quality",
            ),
            metadata=metadata,
        )

    @staticmethod
    def _number(
        value: Any,
        name: str,
    ) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise IntrinsicMotivationInputError(
                f"{name} must be numeric"
            )

        value = float(value)

        if not 0 <= value <= 1:
            raise IntrinsicMotivationInputError(
                f"{name} must be in [0, 1]"
            )

        return value


__all__ = [
    "IntrinsicMotivationError",
    "IntrinsicMotivationConfigurationError",
    "IntrinsicMotivationInputError",
    "IntrinsicMotivationWeights",
    "IntrinsicMotivationInput",
    "IntrinsicMotivationEvaluation",
    "IntrinsicMotivationEvaluator",
]