from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from .goal_proposal import (
    GoalCandidate,
    GoalProposalBatch,
    GoalProposalEngine,
    GoalSignal,
)
from .memory_goal_generation import (
    MemoryGap,
    MemoryGapGoalGenerationBatch,
    MemoryGapGoalGenerationResult,
    MemoryGapGoalGenerator,
)


class GoalGenerationPipelineError(Exception):
    """Base error for goal generation pipeline."""


class GoalGenerationPipelineConfigurationError(
    GoalGenerationPipelineError
):
    """Raised for invalid pipeline configuration."""


class GoalGenerationPipelineInputError(
    GoalGenerationPipelineError
):
    """Raised for invalid pipeline input."""


@dataclass(frozen=True)
class GoalGenerationPipelineResult:
    """
    Immutable result of the memory-gap -> signal -> candidate pipeline.
    """

    memory_generation: MemoryGapGoalGenerationBatch
    proposal: GoalProposalBatch

    signals: tuple[GoalSignal, ...]
    candidates: tuple[GoalCandidate, ...]

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.memory_generation,
            MemoryGapGoalGenerationBatch,
        ):
            raise GoalGenerationPipelineInputError(
                "memory_generation must be "
                "MemoryGapGoalGenerationBatch"
            )

        if not isinstance(
            self.proposal,
            GoalProposalBatch,
        ):
            raise GoalGenerationPipelineInputError(
                "proposal must be GoalProposalBatch"
            )

        object.__setattr__(
            self,
            "signals",
            tuple(self.signals),
        )

        object.__setattr__(
            self,
            "candidates",
            tuple(self.candidates),
        )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise GoalGenerationPipelineInputError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def best_candidate(
        self,
    ) -> Optional[GoalCandidate]:
        return (
            self.candidates[0]
            if self.candidates
            else None
        )

    @property
    def best_signal(
        self,
    ) -> Optional[GoalSignal]:
        return (
            self.signals[0]
            if self.signals
            else None
        )

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def signal_count(self) -> int:
        return len(self.signals)

    @property
    def has_candidates(self) -> bool:
        return bool(self.candidates)

    @property
    def has_signals(self) -> bool:
        return bool(self.signals)


class GoalGenerationPipeline:
    """
    Orchestrates:

        MemoryGap
            ->
        MemoryGapGoalGenerator
            ->
        GoalSignal
            ->
        GoalProposalEngine
            ->
        GoalCandidate

    This class is intentionally side-effect free.

    It does NOT:

    - mutate Agent state
    - mutate memory
    - insert goals
    - call GoalLifecycleManager
    - call GoalAcceptanceEngine
    - execute actions
    - call an LLM
    - persist data
    """

    def __init__(
        self,
        *,
        memory_generator: Optional[
            MemoryGapGoalGenerator
        ] = None,
        proposal_engine: Optional[
            GoalProposalEngine
        ] = None,
    ) -> None:
        self.memory_generator = (
            memory_generator
            or MemoryGapGoalGenerator()
        )

        self.proposal_engine = (
            proposal_engine
            or GoalProposalEngine()
        )

        if not isinstance(
            self.memory_generator,
            MemoryGapGoalGenerator,
        ):
            raise GoalGenerationPipelineConfigurationError(
                "memory_generator must be "
                "MemoryGapGoalGenerator"
            )

        if not isinstance(
            self.proposal_engine,
            GoalProposalEngine,
        ):
            raise GoalGenerationPipelineConfigurationError(
                "proposal_engine must be "
                "GoalProposalEngine"
            )

    def generate(
        self,
        gaps: Iterable[
            MemoryGap
            | Mapping[str, Any]
            | Any
        ],
        *,
        existing_descriptions: Iterable[str] = (),
        existing_candidate_ids: Iterable[str] = (),
        max_signals: Optional[int] = None,
        max_candidates: Optional[int] = None,
    ) -> GoalGenerationPipelineResult:
        """
        Generate goal candidates from memory/world-model gaps.
        """

        memory_generation = (
            self.memory_generator.generate_many(
                gaps,
                max_signals=max_signals,
            )
        )

        signals = memory_generation.signals

        proposal = self.proposal_engine.propose(
            signals,
            existing_descriptions=(
                existing_descriptions
            ),
            existing_candidate_ids=(
                existing_candidate_ids
            ),
            max_candidates=max_candidates,
        )

        candidates = proposal.candidates

        return GoalGenerationPipelineResult(
            memory_generation=memory_generation,
            proposal=proposal,
            signals=signals,
            candidates=candidates,
            metadata={
                "input_gap_count": (
                    memory_generation.metadata.get(
                        "input_count",
                        0,
                    )
                ),
                "generated_signal_count": (
                    memory_generation.metadata.get(
                        "generated_count",
                        0,
                    )
                ),
                "selected_signal_count": len(
                    signals
                ),
                "candidate_count": len(
                    candidates
                ),
                "memory_rejected_count": len(
                    memory_generation.rejected_gap_ids
                ),
                "proposal_rejected_count": len(
                    proposal.rejected_descriptions
                ),
                "proposal_duplicate_count": len(
                    proposal.duplicate_descriptions
                ),
            },
        )

    def generate_from_signals(
        self,
        signals: Iterable[
            GoalSignal
            | Mapping[str, Any]
            | Any
        ],
        *,
        existing_descriptions: Iterable[str] = (),
        existing_candidate_ids: Iterable[str] = (),
        max_candidates: Optional[int] = None,
    ) -> GoalGenerationPipelineResult:
        """
        Run the proposal stage directly from GoalSignal objects.

        This method is useful for non-memory providers such as:
        - intrinsic motivation
        - external events
        - recovery
        - maintenance
        - world-state detectors
        """

        raw_signals = tuple(signals)

        normalized_signals = tuple(
            self.proposal_engine._normalize_signal(
                signal
            )
            for signal in raw_signals
        )

        proposal = self.proposal_engine.propose(
            normalized_signals,
            existing_descriptions=(
                existing_descriptions
            ),
            existing_candidate_ids=(
                existing_candidate_ids
            ),
            max_candidates=max_candidates,
        )

        candidates = proposal.candidates

        memory_generation = (
            MemoryGapGoalGenerationBatch(
                results=(),
                signals=normalized_signals,
                rejected_gap_ids=(),
                metadata={
                    "input_count": len(
                        normalized_signals
                    ),
                    "generated_count": len(
                        normalized_signals
                    ),
                    "selected_count": len(
                        normalized_signals
                    ),
                    "rejected_count": 0,
                    "source": "direct_signals",
                },
            )
        )

        return GoalGenerationPipelineResult(
            memory_generation=memory_generation,
            proposal=proposal,
            signals=normalized_signals,
            candidates=candidates,
            metadata={
                "input_gap_count": 0,
                "generated_signal_count": len(
                    normalized_signals
                ),
                "selected_signal_count": len(
                    normalized_signals
                ),
                "candidate_count": len(
                    candidates
                ),
                "memory_rejected_count": 0,
                "proposal_rejected_count": len(
                    proposal.rejected_descriptions
                ),
                "proposal_duplicate_count": len(
                    proposal.duplicate_descriptions
                ),
                "source": "direct_signals",
            },
        )

    def generate_one(
        self,
        gap: (
            MemoryGap
            | Mapping[str, Any]
            | Any
        ),
        *,
        existing_descriptions: Iterable[str] = (),
        existing_candidate_ids: Iterable[str] = (),
    ) -> GoalGenerationPipelineResult:
        """
        Convenience method for one memory gap.
        """

        return self.generate(
            [gap],
            existing_descriptions=(
                existing_descriptions
            ),
            existing_candidate_ids=(
                existing_candidate_ids
            ),
            max_signals=1,
            max_candidates=1,
        )


__all__ = [
    "GoalGenerationPipelineError",
    "GoalGenerationPipelineConfigurationError",
    "GoalGenerationPipelineInputError",
    "GoalGenerationPipelineResult",
    "GoalGenerationPipeline",
]