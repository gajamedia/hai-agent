from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.llm.goal_generation_pipeline import (
    GoalGenerationPipeline,
    GoalGenerationPipelineConfigurationError,
    GoalGenerationPipelineInputError,
    GoalGenerationPipelineResult,
)
from agent.llm.goal_proposal import (
    GoalOrigin,
    GoalProposalEngine,
    GoalSignal,
)
from agent.llm.memory_goal_generation import (
    MemoryGap,
    MemoryGapGoalGenerator,
    MemoryGapGoalGenerationBatch,
    MemoryGapType,
)


def make_gap(
    gap_id: str = "gap-1",
    *,
    description: str = "unknown state",
    importance: float = 0.9,
    urgency: float = 0.8,
    uncertainty: float = 0.9,
    novelty: float = 0.7,
    feasibility: float = 0.8,
    evidence_quality: float = 0.8,
    expected_value: float | None = 0.85,
) -> MemoryGap:
    return MemoryGap(
        gap_id=gap_id,
        gap_type=MemoryGapType.MISSING_KNOWLEDGE,
        description=description,
        importance=importance,
        urgency=urgency,
        uncertainty=uncertainty,
        novelty=novelty,
        feasibility=feasibility,
        evidence_quality=evidence_quality,
        expected_value=expected_value,
        target="target-system",
        source="test",
        metadata={
            "source_type": "test",
        },
    )


class TestGoalGenerationPipelineResult:
    def test_result_is_immutable_container(self):
        pipeline = GoalGenerationPipeline()

        result = pipeline.generate(
            [make_gap()]
        )

        assert isinstance(
            result,
            GoalGenerationPipelineResult,
        )

    def test_best_candidate(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.best_candidate is not None
        assert (
            result.best_candidate
            == result.candidates[0]
        )

    def test_best_signal(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.best_signal is not None
        assert (
            result.best_signal
            == result.signals[0]
        )

    def test_empty_result(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=1.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.signal_count == 0
        assert result.candidate_count == 0
        assert result.best_signal is None
        assert result.best_candidate is None


class TestGoalGenerationPipeline:
    def test_default_components(self):
        pipeline = GoalGenerationPipeline()

        assert isinstance(
            pipeline.memory_generator,
            MemoryGapGoalGenerator,
        )

        assert isinstance(
            pipeline.proposal_engine,
            GoalProposalEngine,
        )

    def test_custom_components(self):
        memory_generator = (
            MemoryGapGoalGenerator(
                minimum_score=0.0
            )
        )

        proposal_engine = (
            GoalProposalEngine(
                minimum_score=0.0
            )
        )

        pipeline = GoalGenerationPipeline(
            memory_generator=memory_generator,
            proposal_engine=proposal_engine,
        )

        assert (
            pipeline.memory_generator
            is memory_generator
        )

        assert (
            pipeline.proposal_engine
            is proposal_engine
        )

    def test_invalid_memory_generator_rejected(self):
        with pytest.raises(
            GoalGenerationPipelineConfigurationError
        ):
            GoalGenerationPipeline(
                memory_generator=object()
            )

    def test_invalid_proposal_engine_rejected(self):
        with pytest.raises(
            GoalGenerationPipelineConfigurationError
        ):
            GoalGenerationPipeline(
                proposal_engine=object()
            )

    def test_memory_gap_becomes_signal_and_candidate(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.signal_count == 1
        assert result.candidate_count == 1

        signal = result.signals[0]
        candidate = result.candidates[0]

        assert isinstance(
            signal,
            GoalSignal,
        )

        assert signal.origin is GoalOrigin.MEMORY

        assert candidate.origin is GoalOrigin.MEMORY

        assert (
            candidate.description
            == signal.description
        )

    def test_memory_gap_metadata_survives_pipeline(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [make_gap()]
        )

        candidate = result.best_candidate

        assert candidate is not None

        assert (
            candidate.metadata[
                "memory_gap_id"
            ]
            == "gap-1"
        )

        assert (
            candidate.metadata[
                "memory_gap_type"
            ]
            == "missing_knowledge"
        )

        assert (
            candidate.metadata[
                "source"
            ]
            == "test"
        )

    def test_existing_description_is_forwarded(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [make_gap()],
            existing_descriptions=[
                "Learn target-system: unknown state"
            ],
        )

        assert result.candidate_count == 0

    def test_existing_candidate_id_is_forwarded(self):
        gap = make_gap()

        signal = (
            MemoryGapGoalGenerator(
                minimum_score=0.0
            )
            .generate(gap)
            .signal
        )

        assert signal is not None

        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [gap],
            existing_candidate_ids=[
                signal.candidate_id
            ],
        )

        assert result.candidate_count == 0

    def test_max_signals_is_forwarded(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0,
                max_signals=10,
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [
                make_gap("g1"),
                make_gap("g2"),
                make_gap("g3"),
            ],
            max_signals=2,
        )

        assert result.signal_count == 2

    def test_max_candidates_is_forwarded(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [
                make_gap(
                    "g1",
                    description="first",
                ),
                make_gap(
                    "g2",
                    description="second",
                ),
                make_gap(
                    "g3",
                    description="third",
                ),
            ],
            max_candidates=1,
        )

        assert result.candidate_count == 1

    def test_generate_one(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate_one(
            make_gap()
        )

        assert result.signal_count == 1
        assert result.candidate_count == 1

    def test_generate_accepts_mapping_gap(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [
                {
                    "id": "g1",
                    "description": "Unknown fact",
                    "importance": 0.8,
                    "urgency": 0.8,
                    "uncertainty": 0.8,
                    "novelty": 0.8,
                    "feasibility": 0.8,
                    "evidence_quality": 0.8,
                }
            ]
        )

        assert result.candidate_count == 1

    def test_generate_accepts_object_gap(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        gap = SimpleNamespace(
            gap_id="g1",
            gap_type=(
                MemoryGapType.MISSING_KNOWLEDGE
            ),
            description="Unknown fact",
            importance=0.8,
            urgency=0.8,
            uncertainty=0.8,
            novelty=0.8,
            feasibility=0.8,
            evidence_quality=0.8,
            expected_value=0.8,
            target="system",
            source="memory",
            metadata={},
        )

        result = pipeline.generate(
            [gap]
        )

        assert result.candidate_count == 1

    def test_direct_signal_generation(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        signal = GoalSignal(
            description="Explore unknown state",
            origin=GoalOrigin.INTRINSIC,
            urgency=0.8,
            expected_value=0.8,
            feasibility=0.8,
            novelty=0.9,
            evidence_quality=0.7,
        )

        result = (
            pipeline.generate_from_signals(
                [signal]
            )
        )

        assert result.signal_count == 1
        assert result.candidate_count == 1
        assert (
            result.best_candidate.origin
            is GoalOrigin.INTRINSIC
        )

    def test_direct_signal_mapping(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        result = (
            pipeline.generate_from_signals(
                [
                    {
                        "description": "Explore",
                        "origin": "intrinsic",
                        "urgency": 0.8,
                        "expected_value": 0.8,
                        "feasibility": 0.8,
                        "novelty": 0.8,
                        "evidence_quality": 0.8,
                    }
                ]
            )
        )

        assert result.candidate_count == 1
        assert (
            result.best_candidate.origin
            is GoalOrigin.INTRINSIC
        )

    def test_direct_signal_object(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        signal = SimpleNamespace(
            description="Explore",
            origin=GoalOrigin.INTRINSIC,
            urgency=0.8,
            expected_value=0.8,
            feasibility=0.8,
            novelty=0.8,
            evidence_quality=0.8,
        )

        result = (
            pipeline.generate_from_signals(
                [signal]
            )
        )

        assert result.candidate_count == 1

    def test_direct_signal_generator_is_supported(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        result = (
            pipeline.generate_from_signals(
                (
                    GoalSignal(
                        description=value
                    )
                    for value in (
                        "A",
                        "B",
                    )
                )
            )
        )

        assert result.signal_count == 2
        assert result.candidate_count == 2

    def test_duplicate_signals_are_handled_by_proposal_engine(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        signal_a = GoalSignal(
            description="Same goal"
        )

        signal_b = GoalSignal(
            description="Same goal"
        )

        result = (
            pipeline.generate_from_signals(
                [
                    signal_a,
                    signal_b,
                ]
            )
        )

        assert result.candidate_count == 1
        assert (
            result.proposal.duplicate_descriptions
            == ("Same goal",)
        )

    def test_low_proposal_score_is_rejected(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.99
            ),
        )

        gap = make_gap(
            importance=0.1,
            urgency=0.1,
            uncertainty=0.1,
            novelty=0.1,
            feasibility=0.1,
            evidence_quality=0.1,
            expected_value=0.1,
        )

        result = pipeline.generate(
            [gap]
        )

        assert result.signal_count == 1
        assert result.candidate_count == 0
        assert (
            len(
                result.proposal.rejected_descriptions
            )
            == 1
        )

    def test_memory_generation_rejection_is_preserved(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=1.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.signal_count == 0
        assert result.candidate_count == 0
        assert (
            result.memory_generation
            .rejected_gap_ids
            == ("gap-1",)
        )

    def test_multiple_gaps_are_ranked(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [
                make_gap(
                    "low",
                    description="Low",
                    importance=0.4,
                    urgency=0.4,
                    uncertainty=0.4,
                    novelty=0.4,
                    feasibility=0.4,
                    evidence_quality=0.4,
                    expected_value=0.4,
                ),
                make_gap(
                    "high",
                    description="High",
                    importance=1.0,
                    urgency=1.0,
                    uncertainty=1.0,
                    novelty=1.0,
                    feasibility=1.0,
                    evidence_quality=1.0,
                    expected_value=1.0,
                ),
            ]
        )

        assert result.candidate_count == 2

        assert (
            result.candidates[0].description
            == "Learn target-system: High"
        )

    def test_pipeline_does_not_create_lifecycle_goals(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [make_gap()]
        )

        assert result.candidate_count == 1

        # The pipeline result only contains signals/candidates.
        # No lifecycle manager or agent state exists here.
        assert not hasattr(
            result,
            "manager",
        )

    def test_pipeline_preserves_candidate_payload(self):
        gap = make_gap()

        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [gap]
        )

        candidate = result.best_candidate

        assert candidate is not None

        assert candidate.payload[
            "gap_id"
        ] == "gap-1"

    def test_metadata_counts(self):
        pipeline = GoalGenerationPipeline(
            memory_generator=MemoryGapGoalGenerator(
                minimum_score=0.0
            ),
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            ),
        )

        result = pipeline.generate(
            [
                make_gap(
                    "g1",
                    description="One",
                ),
                make_gap(
                    "g2",
                    description="Two",
                ),
            ]
        )

        assert (
            result.metadata[
                "input_gap_count"
            ]
            == 2
        )

        assert (
            result.metadata[
                "generated_signal_count"
            ]
            == 2
        )

        assert (
            result.metadata[
                "selected_signal_count"
            ]
            == 2
        )

        assert (
            result.metadata[
                "candidate_count"
            ]
            == 2
        )

    def test_direct_signal_metadata(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        result = (
            pipeline.generate_from_signals(
                [
                    GoalSignal("A"),
                    GoalSignal("B"),
                ]
            )
        )

        assert (
            result.metadata["source"]
            == "direct_signals"
        )

    def test_result_collections_are_tuples(self):
        pipeline = GoalGenerationPipeline(
            proposal_engine=GoalProposalEngine(
                minimum_score=0.0
            )
        )

        result = (
            pipeline.generate_from_signals(
                [GoalSignal("A")]
            )
        )

        assert isinstance(
            result.signals,
            tuple,
        )

        assert isinstance(
            result.candidates,
            tuple,
        )