from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional

from .goal_proposal import GoalOrigin, GoalSignal


class MemoryGoalGenerationError(Exception):
    """Base error for memory/world-gap goal generation."""


class MemoryGoalGenerationConfigurationError(
    MemoryGoalGenerationError
):
    """Raised for invalid generator configuration."""


class MemoryGoalGenerationInputError(
    MemoryGoalGenerationError
):
    """Raised for invalid memory/world-gap input."""


class MemoryGapType(str, Enum):
    MISSING_KNOWLEDGE = "missing_knowledge"
    UNCERTAIN_KNOWLEDGE = "uncertain_knowledge"
    UNRESOLVED_STATE = "unresolved_state"
    MISSING_RELATION = "missing_relation"
    INCOMPLETE_WORLD_MODEL = "incomplete_world_model"


@dataclass(frozen=True)
class MemoryGap:
    """
    Immutable representation of a gap detected in memory
    or the world model.
    """

    gap_id: str
    gap_type: MemoryGapType
    description: str

    importance: float = 0.5
    urgency: float = 0.5
    uncertainty: float = 0.5
    novelty: float = 0.5
    feasibility: float = 0.5
    evidence_quality: float = 0.5

    expected_value: Optional[float] = None

    target: Optional[str] = None
    source: Optional[str] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        gap_id = str(
            self.gap_id
        ).strip()

        if not gap_id:
            raise MemoryGoalGenerationInputError(
                "gap_id must not be empty"
            )

        description = str(
            self.description
        ).strip()

        if not description:
            raise MemoryGoalGenerationInputError(
                "description must not be empty"
            )

        object.__setattr__(
            self,
            "gap_id",
            gap_id,
        )

        object.__setattr__(
            self,
            "description",
            description,
        )

        object.__setattr__(
            self,
            "gap_type",
            _coerce_gap_type(
                self.gap_type
            ),
        )

        for field_name in (
            "importance",
            "urgency",
            "uncertainty",
            "novelty",
            "feasibility",
            "evidence_quality",
        ):
            value = _bounded_number(
                getattr(
                    self,
                    field_name,
                ),
                field_name,
            )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        if self.expected_value is not None:
            object.__setattr__(
                self,
                "expected_value",
                _bounded_number(
                    self.expected_value,
                    "expected_value",
                ),
            )

        if self.target is not None:
            target = str(
                self.target
            ).strip()

            object.__setattr__(
                self,
                "target",
                target or None,
            )

        if self.source is not None:
            source = str(
                self.source
            ).strip()

            object.__setattr__(
                self,
                "source",
                source or None,
            )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise MemoryGoalGenerationInputError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


@dataclass(frozen=True)
class MemoryGapGoalGenerationWeights:
    """
    Weights used to convert a memory gap into
    a goal-generation score.
    """

    importance: float = 0.25
    urgency: float = 0.20
    uncertainty: float = 0.15
    novelty: float = 0.10
    feasibility: float = 0.15
    evidence_quality: float = 0.15

    def __post_init__(self) -> None:
        values = (
            self.importance,
            self.urgency,
            self.uncertainty,
            self.novelty,
            self.feasibility,
            self.evidence_quality,
        )

        if any(
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
            for value in values
        ):
            raise MemoryGoalGenerationConfigurationError(
                "weights must be numeric"
            )

        if any(
            value < 0
            or value > 1
            for value in values
        ):
            raise MemoryGoalGenerationConfigurationError(
                "weights must be in [0, 1]"
            )

        if sum(values) <= 0:
            raise MemoryGoalGenerationConfigurationError(
                "at least one weight must be positive"
            )

    @property
    def total(self) -> float:
        return sum(
            (
                self.importance,
                self.urgency,
                self.uncertainty,
                self.novelty,
                self.feasibility,
                self.evidence_quality,
            )
        )


@dataclass(frozen=True)
class MemoryGapGoalGenerationResult:
    """
    Immutable result of converting a memory gap
    into a GoalSignal.
    """

    gap_id: str
    gap_type: MemoryGapType
    signal: Optional[GoalSignal]

    generated: bool
    reason: str

    score: float = 0.0

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        gap_id = str(
            self.gap_id
        ).strip()

        if not gap_id:
            raise MemoryGoalGenerationInputError(
                "gap_id must not be empty"
            )

        if not isinstance(
            self.generated,
            bool,
        ):
            raise MemoryGoalGenerationInputError(
                "generated must be bool"
            )

        reason = str(
            self.reason
        ).strip()

        if not reason:
            raise MemoryGoalGenerationInputError(
                "reason must not be empty"
            )

        score = _bounded_number(
            self.score,
            "score",
        )

        if self.signal is not None:
            if not isinstance(
                self.signal,
                GoalSignal,
            ):
                raise MemoryGoalGenerationInputError(
                    "signal must be GoalSignal or None"
                )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise MemoryGoalGenerationInputError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "gap_id",
            gap_id,
        )

        object.__setattr__(
            self,
            "gap_type",
            _coerce_gap_type(
                self.gap_type
            ),
        )

        object.__setattr__(
            self,
            "reason",
            reason,
        )

        object.__setattr__(
            self,
            "score",
            score,
        )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


@dataclass(frozen=True)
class MemoryGapGoalGenerationBatch:
    """
    Immutable batch result.
    """

    results: tuple[
        MemoryGapGoalGenerationResult,
        ...
    ]

    signals: tuple[
        GoalSignal,
        ...
    ]

    rejected_gap_ids: tuple[str, ...] = ()

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "results",
            tuple(self.results),
        )

        object.__setattr__(
            self,
            "signals",
            tuple(self.signals),
        )

        object.__setattr__(
            self,
            "rejected_gap_ids",
            tuple(
                str(value)
                for value in self.rejected_gap_ids
            ),
        )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise MemoryGoalGenerationInputError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def count(self) -> int:
        return len(self.signals)

    @property
    def best_signal(
        self,
    ) -> Optional[GoalSignal]:
        if not self.signals:
            return None

        return self.signals[0]


class MemoryGapGoalGenerator:
    """
    Converts memory/world-model gaps into GoalSignal objects.

    This component is intentionally side-effect free.

    It does NOT:
    - mutate Agent state
    - mutate memory
    - insert goals
    - call GoalLifecycleManager
    - execute actions
    - call an LLM
    - persist anything

    It only transforms detected gaps into GoalSignal objects.
    """

    _DEFAULT_TYPE_BONUS = {
        MemoryGapType.MISSING_KNOWLEDGE: 0.10,
        MemoryGapType.UNCERTAIN_KNOWLEDGE: 0.10,
        MemoryGapType.UNRESOLVED_STATE: 0.05,
        MemoryGapType.MISSING_RELATION: 0.05,
        MemoryGapType.INCOMPLETE_WORLD_MODEL: 0.10,
    }

    def __init__(
        self,
        *,
        weights: Optional[
            MemoryGapGoalGenerationWeights
        ] = None,
        minimum_score: float = 0.35,
        max_signals: int = 10,
        type_bonus: Optional[
            Mapping[
                MemoryGapType | str,
                float,
            ]
        ] = None,
    ) -> None:
        self.weights = (
            weights
            or MemoryGapGoalGenerationWeights()
        )

        self.minimum_score = _bounded_number(
            minimum_score,
            "minimum_score",
        )

        if (
            not isinstance(
                max_signals,
                int,
            )
            or isinstance(
                max_signals,
                bool,
            )
            or max_signals <= 0
        ):
            raise MemoryGoalGenerationConfigurationError(
                "max_signals must be a positive integer"
            )

        self.max_signals = max_signals

        raw_bonus = dict(
            type_bonus
            or self._DEFAULT_TYPE_BONUS
        )

        normalized_bonus: dict[
            MemoryGapType,
            float,
        ] = {}

        for key, value in raw_bonus.items():
            gap_type = _coerce_gap_type(
                key
            )

            normalized_bonus[
                gap_type
            ] = _bounded_number(
                value,
                f"type_bonus[{gap_type.value}]",
            )

        self.type_bonus = normalized_bonus

    def score_gap(
        self,
        gap: (
            MemoryGap
            | Mapping[str, Any]
            | Any
        ),
    ) -> float:
        """
        Calculate a bounded score for a memory gap.
        """

        normalized = self._normalize_gap(
            gap
        )

        weights = self.weights

        base_score = (
            weights.importance
            * normalized.importance
            + weights.urgency
            * normalized.urgency
            + weights.uncertainty
            * normalized.uncertainty
            + weights.novelty
            * normalized.novelty
            + weights.feasibility
            * normalized.feasibility
            + weights.evidence_quality
            * normalized.evidence_quality
        ) / weights.total

        bonus = self.type_bonus.get(
            normalized.gap_type,
            0.0,
        )

        return round(
            _clamp(
                base_score + bonus
            ),
            12,
        )

    def generate(
        self,
        gap: (
            MemoryGap
            | Mapping[str, Any]
            | Any
        ),
    ) -> MemoryGapGoalGenerationResult:
        """
        Generate one GoalSignal from one MemoryGap.
        """

        normalized = self._normalize_gap(
            gap
        )

        score = self.score_gap(
            normalized
        )

        if score < self.minimum_score:
            return MemoryGapGoalGenerationResult(
                gap_id=normalized.gap_id,
                gap_type=normalized.gap_type,
                signal=None,
                generated=False,
                reason="BELOW_MINIMUM_SCORE",
                score=score,
                metadata={
                    "minimum_score": self.minimum_score,
                },
            )

        signal = self._build_signal(
            normalized,
            score,
        )

        return MemoryGapGoalGenerationResult(
            gap_id=normalized.gap_id,
            gap_type=normalized.gap_type,
            signal=signal,
            generated=True,
            reason="GOAL_SIGNAL_GENERATED",
            score=score,
            metadata={
                "minimum_score": self.minimum_score,
                "goal_origin": signal.origin.value,
            },
        )

    def generate_many(
        self,
        gaps: Iterable[
            MemoryGap
            | Mapping[str, Any]
            | Any
        ],
        *,
        max_signals: Optional[int] = None,
    ) -> MemoryGapGoalGenerationBatch:
        """
        Generate GoalSignals from multiple memory gaps.

        Results are ranked by generation score.
        """

        limit = (
            self.max_signals
            if max_signals is None
            else max_signals
        )

        if (
            not isinstance(
                limit,
                int,
            )
            or isinstance(
                limit,
                bool,
            )
            or limit <= 0
        ):
            raise MemoryGoalGenerationConfigurationError(
                "max_signals must be a positive integer"
            )

        normalized = tuple(
            self._normalize_gap(gap)
            for gap in gaps
        )

        seen_ids: set[str] = set()

        results: list[
            MemoryGapGoalGenerationResult
        ] = []

        for gap in normalized:
            if gap.gap_id in seen_ids:
                results.append(
                    MemoryGapGoalGenerationResult(
                        gap_id=gap.gap_id,
                        gap_type=gap.gap_type,
                        signal=None,
                        generated=False,
                        reason="DUPLICATE_GAP_ID",
                        score=0.0,
                    )
                )
                continue

            seen_ids.add(
                gap.gap_id
            )

            results.append(
                self.generate(gap)
            )

        generated = [
            result
            for result in results
            if result.generated
            and result.signal is not None
        ]

        generated.sort(
            key=lambda item: (
                -item.score,
                item.gap_id,
            )
        )

        selected = generated[:limit]

        overflow = generated[limit:]

        signals = tuple(
            result.signal
            for result in selected
            if result.signal is not None
        )

        rejected_ids = [
            result.gap_id
            for result in results
            if not result.generated
        ]

        rejected_ids.extend(
            result.gap_id
            for result in overflow
        )

        return MemoryGapGoalGenerationBatch(
            results=tuple(results),
            signals=signals,
            rejected_gap_ids=tuple(
                rejected_ids
            ),
            metadata={
                "input_count": len(
                    normalized
                ),
                "generated_count": len(
                    generated
                ),
                "selected_count": len(
                    signals
                ),
                "rejected_count": len(
                    rejected_ids
                ),
                "minimum_score": self.minimum_score,
            },
        )

    def _build_signal(
        self,
        gap: MemoryGap,
        score: float,
    ) -> GoalSignal:
        """
        Convert a MemoryGap into GoalSignal.
        """

        if gap.expected_value is not None:
            expected_value = gap.expected_value
        else:
            expected_value = _clamp(
                (
                    gap.importance
                    + gap.uncertainty
                    + gap.novelty
                )
                / 3.0
            )

        description = self._goal_description(
            gap
        )

        metadata = dict(
            gap.metadata
        )

        metadata.update(
            {
                "memory_gap_id": gap.gap_id,
                "memory_gap_type": gap.gap_type.value,
                "generation_score": score,
                "source": (
                    gap.source
                    or "memory_world_gap"
                ),
                "target": gap.target,
                "uncertainty": gap.uncertainty,
                "information_gain": gap.uncertainty,
                "learning_value": gap.importance,
                "exploration_value": gap.novelty,
            }
        )

        origin = (
            GoalOrigin.MEMORY
            if gap.gap_type
            in (
                MemoryGapType.MISSING_KNOWLEDGE,
                MemoryGapType.UNCERTAIN_KNOWLEDGE,
            )
            else GoalOrigin.WORLD_GAP
        )

        return GoalSignal(
            description=description,
            origin=origin,
            urgency=gap.urgency,
            expected_value=expected_value,
            feasibility=gap.feasibility,
            novelty=gap.novelty,
            evidence_quality=gap.evidence_quality,
            payload={
                "gap_id": gap.gap_id,
                "gap_type": gap.gap_type.value,
                "target": gap.target,
                "source": gap.source,
            },
            metadata=metadata,
            candidate_id=(
                f"memory-gap-{gap.gap_id}"
            ),
        )

    @staticmethod
    def _goal_description(
        gap: MemoryGap,
    ) -> str:
        prefixes = {
            MemoryGapType.MISSING_KNOWLEDGE:
                "Learn",
            MemoryGapType.UNCERTAIN_KNOWLEDGE:
                "Resolve uncertainty about",
            MemoryGapType.UNRESOLVED_STATE:
                "Resolve",
            MemoryGapType.MISSING_RELATION:
                "Determine the relationship for",
            MemoryGapType.INCOMPLETE_WORLD_MODEL:
                "Complete understanding of",
        }

        prefix = prefixes[
            gap.gap_type
        ]

        if gap.target:
            return (
                f"{prefix} "
                f"{gap.target}: "
                f"{gap.description}"
            )

        return (
            f"{prefix}: "
            f"{gap.description}"
        )

    def _normalize_gap(
        self,
        gap: (
            MemoryGap
            | Mapping[str, Any]
            | Any
        ),
    ) -> MemoryGap:
        """
        Normalize MemoryGap input.

        Supported inputs:
        - MemoryGap
        - Mapping/dict
        - Generic object with matching attributes

        Supported aliases:
        - id -> gap_id
        - gap -> description
        - priority -> importance
        - uncertainty_score -> uncertainty
        - novelty_score -> novelty
        """

        if isinstance(
            gap,
            MemoryGap,
        ):
            return gap

        if isinstance(
            gap,
            Mapping,
        ):
            data = dict(gap)

        else:
            data = {
                name: getattr(
                    gap,
                    name,
                )
                for name in (
                    "gap_id",
                    "gap_type",
                    "description",
                    "importance",
                    "urgency",
                    "uncertainty",
                    "novelty",
                    "feasibility",
                    "evidence_quality",
                    "expected_value",
                    "target",
                    "source",
                    "metadata",
                    "id",
                    "gap",
                    "priority",
                    "uncertainty_score",
                    "novelty_score",
                )
                if hasattr(
                    gap,
                    name,
                )
            }

        gap_id = data.get(
            "gap_id",
            data.get(
                "id",
                "",
            ),
        )

        description = data.get(
            "description",
            data.get(
                "gap",
                "",
            ),
        )

        importance = data.get(
            "importance",
            data.get(
                "priority",
                0.5,
            ),
        )

        urgency = data.get(
            "urgency",
            0.5,
        )

        uncertainty = data.get(
            "uncertainty",
            data.get(
                "uncertainty_score",
                0.5,
            ),
        )

        novelty = data.get(
            "novelty",
            data.get(
                "novelty_score",
                0.5,
            ),
        )

        feasibility = data.get(
            "feasibility",
            0.5,
        )

        evidence_quality = data.get(
            "evidence_quality",
            0.5,
        )

        expected_value_raw = data.get(
            "expected_value"
        )

        expected_value = (
            None
            if expected_value_raw is None
            else _number(
                expected_value_raw,
                "expected_value",
            )
        )

        return MemoryGap(
            gap_id=str(
                gap_id
            ),
            gap_type=data.get(
                "gap_type",
                MemoryGapType.MISSING_KNOWLEDGE,
            ),
            description=str(
                description
            ),
            importance=_number(
                importance,
                "importance",
            ),
            urgency=_number(
                urgency,
                "urgency",
            ),
            uncertainty=_number(
                uncertainty,
                "uncertainty",
            ),
            novelty=_number(
                novelty,
                "novelty",
            ),
            feasibility=_number(
                feasibility,
                "feasibility",
            ),
            evidence_quality=_number(
                evidence_quality,
                "evidence_quality",
            ),
            expected_value=expected_value,
            target=data.get(
                "target"
            ),
            source=data.get(
                "source"
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )


def _coerce_gap_type(
    value: MemoryGapType | str,
) -> MemoryGapType:
    if isinstance(
        value,
        MemoryGapType,
    ):
        return value

    try:
        return MemoryGapType(
            str(value)
            .strip()
            .lower()
        )
    except ValueError as exc:
        raise MemoryGoalGenerationInputError(
            f"unsupported gap type: {value!r}"
        ) from exc


def _number(
    value: Any,
    name: str,
) -> float:
    """
    Validate and normalize numeric external input.

    This helper is intentionally separate from
    _bounded_number so _normalize_gap() can clearly
    validate values coming from dictionaries or objects.
    """

    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise MemoryGoalGenerationInputError(
            f"{name} must be numeric"
        )

    value = float(
        value
    )

    if not 0 <= value <= 1:
        raise MemoryGoalGenerationInputError(
            f"{name} must be in [0, 1]"
        )

    return value


def _bounded_number(
    value: Any,
    name: str,
) -> float:
    """
    Validate and normalize a numeric value in [0, 1].
    """

    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(
            value,
            bool,
        )
    ):
        raise MemoryGoalGenerationInputError(
            f"{name} must be numeric"
        )

    value = float(
        value
    )

    if not 0 <= value <= 1:
        raise MemoryGoalGenerationInputError(
            f"{name} must be in [0, 1]"
        )

    return value


def _clamp(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


__all__ = [
    "MemoryGoalGenerationError",
    "MemoryGoalGenerationConfigurationError",
    "MemoryGoalGenerationInputError",
    "MemoryGapType",
    "MemoryGap",
    "MemoryGapGoalGenerationWeights",
    "MemoryGapGoalGenerationResult",
    "MemoryGapGoalGenerationBatch",
    "MemoryGapGoalGenerator",
]