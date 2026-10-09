from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent.llm.goal_proposal import GoalOrigin

from agent.llm.memory_goal_generation import (
    MemoryGap,
    MemoryGapGoalGenerationBatch,
    MemoryGapGoalGenerationResult,
    MemoryGapGoalGenerationWeights,
    MemoryGapGoalGenerator,
    MemoryGapType,
    MemoryGoalGenerationConfigurationError,
    MemoryGoalGenerationInputError,
)


def make_gap(**overrides):
    data = {
        "gap_id": "gap-1",
        "gap_type": MemoryGapType.MISSING_KNOWLEDGE,
        "description": "the target state",
        "importance": 0.8,
        "urgency": 0.7,
        "uncertainty": 0.8,
        "novelty": 0.7,
        "feasibility": 0.8,
        "evidence_quality": 0.8,
        "expected_value": 0.8,
        "target": "target-system",
        "source": "semantic-memory",
        "metadata": {
            "source_type": "test",
        },
    }

    data.update(overrides)

    return MemoryGap(**data)


def test_default_weights_total_is_one():
    assert (
        MemoryGapGoalGenerationWeights().total
        == pytest.approx(1.0)
    )


def test_weights_accept_valid_values():
    weights = MemoryGapGoalGenerationWeights(
        importance=1,
        urgency=0,
        uncertainty=0,
        novelty=0,
        feasibility=0,
        evidence_quality=0,
    )

    assert weights.total == 1


def test_weights_reject_negative():
    with pytest.raises(
        MemoryGoalGenerationConfigurationError
    ):
        MemoryGapGoalGenerationWeights(
            importance=-0.1
        )


def test_weights_reject_above_one():
    with pytest.raises(
        MemoryGoalGenerationConfigurationError
    ):
        MemoryGapGoalGenerationWeights(
            importance=1.1
        )


def test_weights_require_positive_total():
    with pytest.raises(
        MemoryGoalGenerationConfigurationError
    ):
        MemoryGapGoalGenerationWeights(
            importance=0,
            urgency=0,
            uncertainty=0,
            novelty=0,
            feasibility=0,
            evidence_quality=0,
        )


def test_gap_accepts_valid_values():
    gap = make_gap()

    assert gap.gap_id == "gap-1"
    assert (
        gap.gap_type
        is MemoryGapType.MISSING_KNOWLEDGE
    )


def test_gap_rejects_empty_id():
    with pytest.raises(
        MemoryGoalGenerationInputError
    ):
        make_gap(gap_id=" ")


def test_gap_rejects_empty_description():
    with pytest.raises(
        MemoryGoalGenerationInputError
    ):
        make_gap(description=" ")


@pytest.mark.parametrize(
    "field",
    [
        "importance",
        "urgency",
        "uncertainty",
        "novelty",
        "feasibility",
        "evidence_quality",
        "expected_value",
    ],
)
def test_gap_rejects_out_of_range(field):
    values = make_gap().__dict__.copy()

    values[field] = 1.1

    with pytest.raises(
        MemoryGoalGenerationInputError
    ):
        MemoryGap(**values)


def test_gap_metadata_must_be_mapping():
    with pytest.raises(
        MemoryGoalGenerationInputError
    ):
        make_gap(metadata=[])


def test_gap_metadata_is_copied():
    metadata = {
        "x": 1,
    }

    gap = make_gap(
        metadata=metadata
    )

    metadata["y"] = 2

    assert gap.metadata == {
        "x": 1
    }


def test_gap_type_string_is_supported():
    gap = make_gap(
        gap_type="uncertain_knowledge"
    )

    assert (
        gap.gap_type
        is MemoryGapType.UNCERTAIN_KNOWLEDGE
    )


def test_invalid_gap_type_rejected():
    with pytest.raises(
        MemoryGoalGenerationInputError
    ):
        make_gap(
            gap_type="invalid"
        )


def test_gap_is_frozen():
    gap = make_gap()

    with pytest.raises(
        FrozenInstanceError
    ):
        gap.gap_id = "x"


def test_generator_default_configuration():
    generator = (
        MemoryGapGoalGenerator()
    )

    assert (
        generator.minimum_score
        == 0.35
    )

    assert (
        generator.max_signals
        == 10
    )


def test_generator_rejects_bad_threshold():
    with pytest.raises(
        MemoryGoalGenerationInputError
    ):
        MemoryGapGoalGenerator(
            minimum_score=1.1
        )


def test_generator_rejects_bad_max_signals():
    with pytest.raises(
        MemoryGoalGenerationConfigurationError
    ):
        MemoryGapGoalGenerator(
            max_signals=0
        )


def test_score_gap_is_bounded():
    score = (
        MemoryGapGoalGenerator()
        .score_gap(make_gap())
    )

    assert 0 <= score <= 1


def test_high_quality_gap_generates_signal():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.generated is True
    assert result.signal is not None


def test_low_score_gap_is_rejected():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=1
        )
    )

    result = generator.generate(
        make_gap(
            importance=0,
            urgency=0,
            uncertainty=0,
            novelty=0,
            feasibility=0,
            evidence_quality=0,
        )
    )

    assert result.generated is False
    assert result.signal is None
    assert (
        result.reason
        == "BELOW_MINIMUM_SCORE"
    )


def test_generated_signal_is_goal_signal():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.signal is not None

    assert result.signal.origin in (
        GoalOrigin.MEMORY,
        GoalOrigin.WORLD_GAP,
    )


def test_missing_knowledge_uses_memory_origin():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.MISSING_KNOWLEDGE
                )
            )
        )
    )

    assert (
        result.signal.origin
        is GoalOrigin.MEMORY
    )


def test_uncertain_knowledge_uses_memory_origin():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.UNCERTAIN_KNOWLEDGE
                )
            )
        )
    )

    assert (
        result.signal.origin
        is GoalOrigin.MEMORY
    )


@pytest.mark.parametrize(
    "gap_type",
    [
        MemoryGapType.UNRESOLVED_STATE,
        MemoryGapType.MISSING_RELATION,
        MemoryGapType.INCOMPLETE_WORLD_MODEL,
    ],
)
def test_world_gaps_use_world_gap_origin(
    gap_type
):
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=gap_type
            )
        )
    )

    assert (
        result.signal.origin
        is GoalOrigin.WORLD_GAP
    )


def test_generated_description_contains_target():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.signal is not None

    assert (
        "target-system"
        in result.signal.description
    )


def test_generated_candidate_id_is_deterministic():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.signal is not None

    assert (
        result.signal.candidate_id
        == "memory-gap-gap-1"
    )


def test_gap_id_is_in_payload():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.signal is not None

    assert (
        result.signal.payload["gap_id"]
        == "gap-1"
    )


def test_gap_type_is_in_payload():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.signal is not None

    assert (
        result.signal.payload["gap_type"]
        == "missing_knowledge"
    )


def test_metadata_is_preserved():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                metadata={
                    "custom": "value"
                }
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.metadata["custom"]
        == "value"
    )


def test_generation_metadata_is_added():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert result.signal is not None

    assert (
        result.signal.metadata[
            "memory_gap_id"
        ]
        == "gap-1"
    )

    assert (
        result.signal.metadata[
            "memory_gap_type"
        ]
        == "missing_knowledge"
    )


def test_uncertainty_becomes_information_gain():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                uncertainty=0.9
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.metadata[
            "information_gain"
        ]
        == 0.9
    )


def test_novelty_becomes_exploration_value():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                novelty=0.9
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.metadata[
            "exploration_value"
        ]
        == 0.9
    )


def test_importance_becomes_learning_value():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                importance=0.9
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.metadata[
            "learning_value"
        ]
        == 0.9
    )


def test_explicit_expected_value_is_preserved():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                expected_value=0.91
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.expected_value
        == 0.91
    )


def test_expected_value_can_be_derived():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                expected_value=None,
                importance=0.9,
                uncertainty=0.6,
                novelty=0.3,
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.expected_value
        == pytest.approx(0.6)
    )


def test_evaluate_many_generates_multiple_signals():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    batch = generator.generate_many(
        [
            make_gap(
                gap_id="a",
                importance=1,
                urgency=1,
            ),
            make_gap(
                gap_id="b",
                importance=0.8,
                urgency=0.8,
            ),
        ]
    )

    assert batch.count == 2


def test_empty_batch():
    batch = (
        MemoryGapGoalGenerator()
        .generate_many([])
    )

    assert batch.count == 0
    assert batch.signals == ()


def test_batch_is_sorted_by_score():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    batch = generator.generate_many(
        [
            make_gap(
                gap_id="low",
                importance=0.1,
                urgency=0.1,
                uncertainty=0.1,
                novelty=0.1,
                feasibility=0.1,
                evidence_quality=0.1,
            ),
            make_gap(
                gap_id="high",
                importance=1,
                urgency=1,
                uncertainty=1,
                novelty=1,
                feasibility=1,
                evidence_quality=1,
            ),
        ]
    )

    assert (
        batch.signals[0].candidate_id
        == "memory-gap-high"
    )


def test_batch_limit():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0,
            max_signals=1,
        )
    )

    batch = generator.generate_many(
        [
            make_gap(gap_id="a"),
            make_gap(gap_id="b"),
        ]
    )

    assert batch.count == 1


def test_custom_batch_limit():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0,
            max_signals=10,
        )
    )

    batch = generator.generate_many(
        [
            make_gap(gap_id="a"),
            make_gap(gap_id="b"),
        ],
        max_signals=1,
    )

    assert batch.count == 1


def test_duplicate_gap_ids_are_rejected():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    batch = generator.generate_many(
        [
            make_gap(gap_id="same"),
            make_gap(gap_id="same"),
        ]
    )

    assert batch.count == 1

    assert any(
        result.reason
        == "DUPLICATE_GAP_ID"
        for result in batch.results
    )


def test_rejected_gap_ids_are_reported():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=1
        )
    )

    batch = generator.generate_many(
        [
            make_gap(
                importance=0,
                urgency=0,
                uncertainty=0,
                novelty=0,
                feasibility=0,
                evidence_quality=0,
            )
        ]
    )

    assert "gap-1" in (
        batch.rejected_gap_ids
    )


def test_batch_metadata_input_count():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    batch = generator.generate_many(
        [
            make_gap(gap_id="a"),
            make_gap(gap_id="b"),
        ]
    )

    assert (
        batch.metadata["input_count"]
        == 2
    )


def test_batch_metadata_selected_count():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    batch = generator.generate_many(
        [
            make_gap(gap_id="a"),
            make_gap(gap_id="b"),
        ]
    )

    assert (
        batch.metadata["selected_count"]
        == 2
    )


def test_mapping_input_is_supported():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    result = generator.generate(
        make_gap().__dict__
    )

    assert result.generated is True


def test_object_input_is_supported():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    obj = SimpleNamespace(
        **make_gap().__dict__
    )

    result = generator.generate(
        obj
    )

    assert result.generated is True


def test_id_alias_is_supported():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    result = generator.generate(
        {
            "id": "g9",
            "description": "Unknown fact",
        }
    )

    assert result.gap_id == "g9"


def test_gap_alias_is_supported():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    result = generator.generate(
        {
            "id": "g9",
            "gap": "Unknown fact",
        }
    )

    assert result.generated is True


def test_uncertainty_score_alias():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    result = generator.generate(
        {
            "id": "g9",
            "description": "Unknown",
            "uncertainty_score": 0.9,
        }
    )

    assert result.signal is not None

    assert (
        result.signal.metadata[
            "uncertainty"
        ]
        == 0.9
    )


def test_novelty_score_alias():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    result = generator.generate(
        {
            "id": "g9",
            "description": "Unknown",
            "novelty_score": 0.9,
        }
    )

    assert result.signal is not None

    assert (
        result.signal.novelty
        == 0.9
    )


def test_custom_weights_change_score():
    default = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .score_gap(
            make_gap(
                importance=1,
                urgency=0,
            )
        )
    )

    custom = (
        MemoryGapGoalGenerator(
            minimum_score=0,
            weights=(
                MemoryGapGoalGenerationWeights(
                    importance=0,
                    urgency=1,
                    uncertainty=0,
                    novelty=0,
                    feasibility=0,
                    evidence_quality=0,
                )
            ),
        )
        .score_gap(
            make_gap(
                importance=1,
                urgency=0,
            )
        )
    )

    assert default != custom


def test_custom_type_bonus_changes_score():
    normal = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .score_gap(
            make_gap(
                gap_type=(
                    MemoryGapType.MISSING_KNOWLEDGE
                )
            )
        )
    )

    custom = (
        MemoryGapGoalGenerator(
            minimum_score=0,
            type_bonus={
                MemoryGapType.MISSING_KNOWLEDGE: 0,
            },
        )
        .score_gap(
            make_gap(
                gap_type=(
                    MemoryGapType.MISSING_KNOWLEDGE
                )
            )
        )
    )

    assert normal > custom


def test_result_is_frozen():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        result.gap_id = "x"


def test_batch_is_frozen():
    batch = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate_many(
            [make_gap()]
        )
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        batch.signals = ()


def test_result_metadata_is_not_shared():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                metadata={
                    "x": 1
                }
            )
        )
    )

    result.metadata["changed"] = True

    assert (
        "changed"
        not in result.metadata
        or "changed"
        not in make_gap(
            metadata={"x": 1}
        ).metadata
    )


def test_generator_does_not_mutate_gap():
    gap = make_gap()

    before = gap.__dict__.copy()

    (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(gap)
    )

    assert gap.__dict__ == before


def test_generator_is_reusable():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    first = generator.generate(
        make_gap(gap_id="a")
    )

    second = generator.generate(
        make_gap(gap_id="b")
    )

    assert first.gap_id == "a"
    assert second.gap_id == "b"


def test_result_contains_score():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    assert 0 <= result.score <= 1


def test_generated_signal_preserves_urgency():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                urgency=0.91
            )
        )
    )

    assert result.signal is not None
    assert (
        result.signal.urgency
        == 0.91
    )


def test_generated_signal_preserves_feasibility():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                feasibility=0.91
            )
        )
    )

    assert result.signal is not None
    assert (
        result.signal.feasibility
        == 0.91
    )


def test_generated_signal_preserves_evidence_quality():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                evidence_quality=0.91
            )
        )
    )

    assert result.signal is not None
    assert (
        result.signal.evidence_quality
        == 0.91
    )


def test_no_llm_or_agent_dependency():
    generator = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
    )

    result = generator.generate(
        make_gap()
    )

    assert result.generated is True
    assert result.signal is not None


def test_signal_can_be_passed_to_proposal_engine():
    from agent.llm.goal_proposal import (
        GoalProposalEngine,
    )

    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(make_gap())
    )

    proposal = (
        GoalProposalEngine(
            minimum_score=0
        )
        .propose(
            [result.signal]
        )
    )

    assert proposal.count == 1
    assert (
        proposal.best_candidate
        is not None
    )


def test_missing_knowledge_goal_description():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.MISSING_KNOWLEDGE
                ),
                description="server state",
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.description
        == "Learn target-system: server state"
    )


def test_uncertain_knowledge_goal_description():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.UNCERTAIN_KNOWLEDGE
                ),
                description="server state",
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.description
        == (
            "Resolve uncertainty about "
            "target-system: server state"
        )
    )


def test_unresolved_state_goal_description():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.UNRESOLVED_STATE
                ),
                description="server state",
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.description
        == "Resolve target-system: server state"
    )


def test_missing_relation_goal_description():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.MISSING_RELATION
                ),
                description="dependency",
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.description
        == (
            "Determine the relationship for "
            "target-system: dependency"
        )
    )


def test_incomplete_world_model_description():
    result = (
        MemoryGapGoalGenerator(
            minimum_score=0
        )
        .generate(
            make_gap(
                gap_type=(
                    MemoryGapType.INCOMPLETE_WORLD_MODEL
                ),
                description="system topology",
            )
        )
    )

    assert result.signal is not None

    assert (
        result.signal.description
        == (
            "Complete understanding of "
            "target-system: system topology"
        )
    )