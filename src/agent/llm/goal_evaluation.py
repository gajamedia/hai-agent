from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_proposal import GoalCandidate
from .goal_utility_evaluation import (
    GoalUtilityEvaluation,
    GoalUtilityEvaluator,
    GoalUtilityInput,
)
from .intrinsic_motivation import (
    IntrinsicMotivationEvaluation,
    IntrinsicMotivationEvaluator,
    IntrinsicMotivationInput,
)


class GoalEvaluationError(Exception):
    """Base error for goal evaluation."""


class GoalEvaluationConfigurationError(GoalEvaluationError):
    """Raised for invalid goal evaluation configuration."""


class GoalEvaluationInputError(GoalEvaluationError):
    """Raised for invalid goal evaluation input."""


@dataclass(frozen=True)
class GoalEvaluationWeights:
    """
    Weights used to combine utility and intrinsic motivation.

    The final combined score is normalized by the total weight.
    """

    utility: float = 0.70
    intrinsic: float = 0.30

    def __post_init__(self) -> None:
        values = (
            self.utility,
            self.intrinsic,
        )

        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            for value in values
        ):
            raise GoalEvaluationConfigurationError(
                "evaluation weights must be numeric"
            )

        if any(value < 0 or value > 1 for value in values):
            raise GoalEvaluationConfigurationError(
                "evaluation weights must be in [0, 1]"
            )

        if sum(values) <= 0:
            raise GoalEvaluationConfigurationError(
                "at least one evaluation weight must be positive"
            )

    @property
    def total(self) -> float:
        return float(self.utility + self.intrinsic)


@dataclass(frozen=True)
class GoalEvaluationResult:
    """
    Immutable evaluation result for one GoalCandidate.

    The original candidate is never mutated.
    """

    candidate: GoalCandidate
    utility: GoalUtilityEvaluation
    intrinsic: IntrinsicMotivationEvaluation

    combined_score: float
    confidence_score: float

    recommended: bool
    recommendation: str

    components: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def goal_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def description(self) -> str:
        return self.candidate.description

    @property
    def utility_score(self) -> float:
        return self.utility.utility_score

    @property
    def expected_utility(self) -> float:
        return self.utility.expected_utility

    @property
    def intrinsic_score(self) -> float:
        return self.intrinsic.intrinsic_score

    @property
    def confidence_adjusted_utility(self) -> float:
        return self.utility.confidence_adjusted_utility

    @property
    def confidence_adjusted_intrinsic(self) -> float:
        return self.intrinsic.confidence_adjusted_score

    @property
    def should_prioritize(self) -> bool:
        return self.recommended

    @property
    def should_explore(self) -> bool:
        return self.intrinsic.should_explore

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, GoalCandidate):
            raise GoalEvaluationInputError(
                "candidate must be a GoalCandidate"
            )

        if not isinstance(
            self.utility,
            GoalUtilityEvaluation,
        ):
            raise GoalEvaluationInputError(
                "utility must be a GoalUtilityEvaluation"
            )

        if not isinstance(
            self.intrinsic,
            IntrinsicMotivationEvaluation,
        ):
            raise GoalEvaluationInputError(
                "intrinsic must be an IntrinsicMotivationEvaluation"
            )

        for name in (
            "combined_score",
            "confidence_score",
        ):
            value = getattr(self, name)

            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise GoalEvaluationInputError(
                    f"{name} must be numeric"
                )

            if not 0 <= float(value) <= 1:
                raise GoalEvaluationInputError(
                    f"{name} must be in [0, 1]"
                )

        object.__setattr__(
            self,
            "components",
            dict(self.components or {}),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata or {}),
        )


@dataclass(frozen=True)
class GoalEvaluationBatch:
    """Immutable deterministic collection of evaluated goals."""

    results: tuple[GoalEvaluationResult, ...]
    rejected_candidate_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def best_result(self) -> Optional[GoalEvaluationResult]:
        return self.results[0] if self.results else None

    @property
    def best_candidate(self) -> Optional[GoalCandidate]:
        if not self.results:
            return None

        return self.results[0].candidate

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def recommended_results(self) -> tuple[GoalEvaluationResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.recommended
        )

    @property
    def exploration_results(self) -> tuple[GoalEvaluationResult, ...]:
        return tuple(
            result
            for result in self.results
            if result.intrinsic.should_explore
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "results",
            tuple(self.results),
        )

        object.__setattr__(
            self,
            "rejected_candidate_ids",
            tuple(
                str(candidate_id)
                for candidate_id in self.rejected_candidate_ids
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata or {}),
        )


class GoalEvaluationEngine:
    """
    Enrich GoalCandidate objects with utility and intrinsic evaluation.

    This class is deliberately side-effect free.

    It does NOT:
    - mutate GoalCandidate
    - mutate Agent
    - insert goals
    - activate goals
    - cancel goals
    - call GoalLifecycleManager
    - execute actions
    """

    def __init__(
        self,
        *,
        utility_evaluator: GoalUtilityEvaluator | None = None,
        intrinsic_evaluator: IntrinsicMotivationEvaluator | None = None,
        weights: GoalEvaluationWeights | None = None,
        recommendation_threshold: float = 0.50,
        minimum_confidence: float = 0.0,
    ) -> None:
        if utility_evaluator is not None and not isinstance(
            utility_evaluator,
            GoalUtilityEvaluator,
        ):
            raise GoalEvaluationConfigurationError(
                "utility_evaluator must be a GoalUtilityEvaluator"
            )

        if intrinsic_evaluator is not None and not isinstance(
            intrinsic_evaluator,
            IntrinsicMotivationEvaluator,
        ):
            raise GoalEvaluationConfigurationError(
                "intrinsic_evaluator must be an "
                "IntrinsicMotivationEvaluator"
            )

        self.utility_evaluator = (
            utility_evaluator
            or GoalUtilityEvaluator()
        )

        self.intrinsic_evaluator = (
            intrinsic_evaluator
            or IntrinsicMotivationEvaluator()
        )

        self.weights = (
            weights
            or GoalEvaluationWeights()
        )

        self.recommendation_threshold = (
            self._bounded_config(
                recommendation_threshold,
                "recommendation_threshold",
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
            raise GoalEvaluationConfigurationError(
                f"{name} must be numeric"
            )

        value = float(value)

        if not 0 <= value <= 1:
            raise GoalEvaluationConfigurationError(
                f"{name} must be in [0, 1]"
            )

        return value

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(1.0, float(value)),
        )

    @staticmethod
    def _metadata_number(
        metadata: Mapping[str, Any],
        keys: tuple[str, ...],
        default: float,
    ) -> float:
        for key in keys:
            if key not in metadata:
                continue

            value = metadata[key]

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                return max(
                    0.0,
                    min(1.0, float(value)),
                )

        return default

    def build_utility_input(
        self,
        candidate: GoalCandidate,
    ) -> GoalUtilityInput:
        candidate = self._require_candidate(candidate)

        metadata = candidate.metadata

        value = self._metadata_number(
            metadata,
            (
                "value",
                "utility_value",
                "goal_value",
                "expected_value",
            ),
            candidate.expected_value,
        )

        importance = self._metadata_number(
            metadata,
            (
                "importance",
                "goal_importance",
            ),
            candidate.expected_value,
        )

        progress = self._metadata_number(
            metadata,
            (
                "progress",
                "goal_progress",
            ),
            0.0,
        )

        estimated_effort = self._metadata_number(
            metadata,
            (
                "estimated_effort",
                "effort",
                "effort_cost",
            ),
            max(
                0.0,
                min(
                    1.0,
                    1.0 - candidate.feasibility,
                ),
            ),
        )

        risk = self._metadata_number(
            metadata,
            (
                "risk",
                "risk_cost",
            ),
            max(
                0.0,
                min(
                    1.0,
                    1.0 - candidate.feasibility,
                ),
            ),
        )

        opportunity_cost = self._metadata_number(
            metadata,
            (
                "opportunity_cost",
                "opportunity",
            ),
            0.0,
        )

        confidence = self._metadata_number(
            metadata,
            (
                "confidence",
                "goal_confidence",
            ),
            candidate.evidence_quality,
        )

        evidence_quality = self._metadata_number(
            metadata,
            (
                "evidence_quality",
            ),
            candidate.evidence_quality,
        )

        return GoalUtilityInput(
            goal_id=candidate.candidate_id,
            description=candidate.description,
            value=value,
            urgency=candidate.urgency,
            importance=importance,
            progress=progress,
            estimated_effort=estimated_effort,
            risk=risk,
            opportunity_cost=opportunity_cost,
            confidence=confidence,
            evidence_quality=evidence_quality,
            metadata=dict(metadata),
        )

    def build_intrinsic_input(
        self,
        candidate: GoalCandidate,
    ) -> IntrinsicMotivationInput:
        candidate = self._require_candidate(candidate)

        metadata = candidate.metadata

        novelty = self._metadata_number(
            metadata,
            (
                "novelty",
                "intrinsic_novelty",
            ),
            candidate.novelty,
        )

        curiosity = self._metadata_number(
            metadata,
            (
                "curiosity",
                "curiosity_value",
            ),
            novelty,
        )

        exploration_value = self._metadata_number(
            metadata,
            (
                "exploration_value",
                "exploration",
            ),
            novelty,
        )

        information_gain = self._metadata_number(
            metadata,
            (
                "information_gain",
                "info_gain",
            ),
            candidate.expected_value,
        )

        uncertainty = self._metadata_number(
            metadata,
            (
                "uncertainty",
                "knowledge_uncertainty",
            ),
            novelty,
        )

        learning_value = self._metadata_number(
            metadata,
            (
                "learning_value",
                "learning",
            ),
            information_gain,
        )

        confidence = self._metadata_number(
            metadata,
            (
                "confidence",
                "goal_confidence",
            ),
            candidate.evidence_quality,
        )

        evidence_quality = self._metadata_number(
            metadata,
            (
                "evidence_quality",
            ),
            candidate.evidence_quality,
        )

        return IntrinsicMotivationInput(
            goal_id=candidate.candidate_id,
            description=candidate.description,
            novelty=novelty,
            curiosity=curiosity,
            exploration_value=exploration_value,
            information_gain=information_gain,
            uncertainty=uncertainty,
            learning_value=learning_value,
            confidence=confidence,
            evidence_quality=evidence_quality,
            metadata=dict(metadata),
        )

    def evaluate(
        self,
        candidate: GoalCandidate,
    ) -> GoalEvaluationResult:
        candidate = self._require_candidate(candidate)

        utility_input = self.build_utility_input(
            candidate
        )

        intrinsic_input = self.build_intrinsic_input(
            candidate
        )

        utility = self.utility_evaluator.evaluate(
            utility_input
        )

        intrinsic = self.intrinsic_evaluator.evaluate(
            intrinsic_input
        )

        weights = self.weights

        utility_component = self._clamp(
            utility.utility_score
        )

        intrinsic_component = self._clamp(
            intrinsic.confidence_adjusted_score
        )

        combined_score = round(
            (
                weights.utility * utility_component
                + weights.intrinsic * intrinsic_component
            )
            / weights.total,
            12,
        )

        confidence_score = round(
            self._clamp(
                (
                    utility_input.confidence
                    + intrinsic_input.confidence
                )
                / 2.0
            ),
            12,
        )

        recommended = (
            combined_score
            >= self.recommendation_threshold
            and confidence_score
            >= self.minimum_confidence
        )

        if confidence_score < self.minimum_confidence:
            recommendation = "insufficient_confidence"
        elif recommended:
            recommendation = "prioritize"
        elif intrinsic.should_explore:
            recommendation = "explore"
        else:
            recommendation = "defer"

        components = {
            "utility_score": utility_component,
            "intrinsic_score": intrinsic_component,
            "utility_weight": float(weights.utility),
            "intrinsic_weight": float(weights.intrinsic),
            "confidence_score": confidence_score,
        }

        metadata = dict(candidate.metadata)

        metadata.update(
            {
                "goal_evaluation": True,
                "utility_expected": utility.expected_utility,
                "utility_confidence_adjusted": (
                    utility.confidence_adjusted_utility
                ),
                "intrinsic_score": intrinsic.intrinsic_score,
                "intrinsic_confidence_adjusted": (
                    intrinsic.confidence_adjusted_score
                ),
                "combined_evaluation_score": combined_score,
                "evaluation_recommendation": recommendation,
            }
        )

        return GoalEvaluationResult(
            candidate=candidate,
            utility=utility,
            intrinsic=intrinsic,
            combined_score=combined_score,
            confidence_score=confidence_score,
            recommended=recommended,
            recommendation=recommendation,
            components=components,
            metadata=metadata,
        )

    def evaluate_many(
        self,
        candidates: Iterable[GoalCandidate],
        *,
        max_results: int | None = None,
    ) -> GoalEvaluationBatch:
        if max_results is not None:
            if (
                not isinstance(max_results, int)
                or isinstance(max_results, bool)
            ):
                raise GoalEvaluationInputError(
                    "max_results must be an integer or None"
                )

            if max_results <= 0:
                raise GoalEvaluationInputError(
                    "max_results must be greater than zero"
                )

        candidate_items = tuple(candidates)

        results: list[GoalEvaluationResult] = []
        rejected: list[str] = []

        for candidate in candidate_items:
            try:
                result = self.evaluate(candidate)
            except (
                GoalEvaluationError,
                ValueError,
                TypeError,
            ):
                candidate_id = getattr(
                    candidate,
                    "candidate_id",
                    None,
                )

                if candidate_id is not None:
                    rejected.append(
                        str(candidate_id)
                    )

                continue

            results.append(result)

        results.sort(
            key=lambda result: (
                result.combined_score,
                result.confidence_score,
                result.utility.confidence_adjusted_utility,
                result.intrinsic.confidence_adjusted_score,
                result.candidate.proposal_score,
            ),
            reverse=True,
        )

        if max_results is not None:
            results = results[:max_results]

        return GoalEvaluationBatch(
            results=tuple(results),
            rejected_candidate_ids=tuple(rejected),
            metadata={
                "input_count": len(candidate_items),
                "evaluated_count": len(results),
                "rejected_count": len(rejected),
                "recommended_count": sum(
                    1
                    for result in results
                    if result.recommended
                ),
            },
        )

    @staticmethod
    def _require_candidate(
        candidate: GoalCandidate,
    ) -> GoalCandidate:
        if not isinstance(candidate, GoalCandidate):
            raise GoalEvaluationInputError(
                "candidate must be a GoalCandidate"
            )

        return candidate


__all__ = [
    "GoalEvaluationError",
    "GoalEvaluationConfigurationError",
    "GoalEvaluationInputError",
    "GoalEvaluationWeights",
    "GoalEvaluationResult",
    "GoalEvaluationBatch",
    "GoalEvaluationEngine",
]