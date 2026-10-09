from __future__ import annotations

import pytest

from agent.llm.goal_evaluation import (
    GoalEvaluationBatch,
    GoalEvaluationConfigurationError,
    GoalEvaluationEngine,
    GoalEvaluationInputError,
    GoalEvaluationResult,
    GoalEvaluationWeights,
)

from agent.llm.goal_proposal import (
    GoalCandidate,
    GoalOrigin,
)


def make_candidate(
    candidate_id: str = "goal-1",
    *,
    proposal_score: float = 0.80,
    urgency: float = 0.80,
    expected_value: float = 0.80,
    feasibility: float = 0.80,
    novelty: float = 0.60,
    evidence_quality: float = 0.90,
    metadata=None,
):
    return GoalCandidate(
        candidate_id=candidate_id,
        description=f"Goal {candidate_id}",
        origin=GoalOrigin.WORLD_GAP,
        priority=proposal_score,
        proposal_score=proposal_score,
        urgency=urgency,
        expected_value=expected_value,
        feasibility=feasibility,
        novelty=novelty,
        evidence_quality=evidence_quality,
        metadata=metadata or {},
    )


def test_default_weights():
    weights = GoalEvaluationWeights()

    assert weights.utility == 0.70
    assert weights.intrinsic == 0.30
    assert weights.total == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"utility": -0.1, "intrinsic": 0.5},
        {"utility": 0.5, "intrinsic": -0.1},
        {"utility": 1.1, "intrinsic": 0.5},
        {"utility": 0.5, "intrinsic": 1.1},
        {"utility": 0.0, "intrinsic": 0.0},
        {"utility": True, "intrinsic": 0.5},
    ],
)
def test_invalid_weights(kwargs):
    with pytest.raises(GoalEvaluationConfigurationError):
        GoalEvaluationWeights(**kwargs)


def test_engine_defaults():
    engine = GoalEvaluationEngine()

    assert engine.utility_evaluator is not None
    assert engine.intrinsic_evaluator is not None
    assert engine.weights.utility == 0.70
    assert engine.weights.intrinsic == 0.30


def test_custom_weights():
    engine = GoalEvaluationEngine(
        weights=GoalEvaluationWeights(
            utility=0.60,
            intrinsic=0.40,
        )
    )

    assert engine.weights.utility == 0.60
    assert engine.weights.intrinsic == 0.40


def test_invalid_utility_evaluator():
    with pytest.raises(GoalEvaluationConfigurationError):
        GoalEvaluationEngine(
            utility_evaluator=object()
        )


def test_invalid_intrinsic_evaluator():
    with pytest.raises(GoalEvaluationConfigurationError):
        GoalEvaluationEngine(
            intrinsic_evaluator=object()
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("recommendation_threshold", -0.1),
        ("recommendation_threshold", 1.1),
        ("minimum_confidence", -0.1),
        ("minimum_confidence", 1.1),
    ],
)
def test_invalid_engine_configuration(name, value):
    kwargs = {name: value}

    with pytest.raises(GoalEvaluationConfigurationError):
        GoalEvaluationEngine(**kwargs)


def test_build_utility_input_uses_candidate_fields():
    candidate = make_candidate()

    engine = GoalEvaluationEngine()

    utility_input = engine.build_utility_input(
        candidate
    )

    assert utility_input.goal_id == candidate.candidate_id
    assert utility_input.description == candidate.description
    assert utility_input.value == candidate.expected_value
    assert utility_input.urgency == candidate.urgency
    assert utility_input.evidence_quality == candidate.evidence_quality


def test_build_intrinsic_input_uses_candidate_fields():
    candidate = make_candidate(
        novelty=0.77
    )

    engine = GoalEvaluationEngine()

    intrinsic_input = engine.build_intrinsic_input(
        candidate
    )

    assert intrinsic_input.goal_id == candidate.candidate_id
    assert intrinsic_input.description == candidate.description
    assert intrinsic_input.novelty == 0.77
    assert intrinsic_input.evidence_quality == candidate.evidence_quality


def test_metadata_overrides_utility_inputs():
    candidate = make_candidate(
        metadata={
            "value": 0.95,
            "importance": 0.91,
            "progress": 0.25,
            "estimated_effort": 0.20,
            "risk": 0.10,
            "opportunity_cost": 0.15,
            "confidence": 0.88,
        }
    )

    engine = GoalEvaluationEngine()

    utility_input = engine.build_utility_input(
        candidate
    )

    assert utility_input.value == 0.95
    assert utility_input.importance == 0.91
    assert utility_input.progress == 0.25
    assert utility_input.estimated_effort == 0.20
    assert utility_input.risk == 0.10
    assert utility_input.opportunity_cost == 0.15
    assert utility_input.confidence == 0.88


def test_metadata_overrides_intrinsic_inputs():
    candidate = make_candidate(
        metadata={
            "novelty": 0.90,
            "curiosity": 0.80,
            "exploration_value": 0.85,
            "information_gain": 0.95,
            "uncertainty": 0.75,
            "learning_value": 0.92,
            "confidence": 0.88,
        }
    )

    engine = GoalEvaluationEngine()

    intrinsic_input = engine.build_intrinsic_input(
        candidate
    )

    assert intrinsic_input.novelty == 0.90
    assert intrinsic_input.curiosity == 0.80
    assert intrinsic_input.exploration_value == 0.85
    assert intrinsic_input.information_gain == 0.95
    assert intrinsic_input.uncertainty == 0.75
    assert intrinsic_input.learning_value == 0.92
    assert intrinsic_input.confidence == 0.88


def test_evaluate_returns_result():
    candidate = make_candidate()

    engine = GoalEvaluationEngine()

    result = engine.evaluate(candidate)

    assert isinstance(
        result,
        GoalEvaluationResult,
    )

    assert result.candidate is candidate
    assert 0 <= result.combined_score <= 1
    assert 0 <= result.confidence_score <= 1
    assert result.goal_id == candidate.candidate_id


def test_evaluation_is_side_effect_free():
    metadata = {
        "value": 0.9,
        "curiosity": 0.8,
    }

    candidate = make_candidate(
        metadata=metadata
    )

    original_metadata = dict(
        candidate.metadata
    )

    engine = GoalEvaluationEngine()

    result = engine.evaluate(candidate)

    assert candidate.metadata == original_metadata
    assert result.candidate is candidate


def test_evaluation_metadata_contains_scores():
    candidate = make_candidate()

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert (
        result.metadata["goal_evaluation"]
        is True
    )

    assert (
        "combined_evaluation_score"
        in result.metadata
    )

    assert (
        "utility_expected"
        in result.metadata
    )

    assert (
        "intrinsic_score"
        in result.metadata
    )


def test_high_quality_candidate_is_recommended():
    candidate = make_candidate(
        proposal_score=0.95,
        urgency=0.95,
        expected_value=0.95,
        feasibility=0.95,
        novelty=0.90,
        evidence_quality=0.95,
    )

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.combined_score >= 0.50
    assert result.recommended is True
    assert result.recommendation == "prioritize"


def test_low_value_high_cost_candidate_can_be_deferred():
    candidate = make_candidate(
        proposal_score=0.30,
        urgency=0.10,
        expected_value=0.05,
        feasibility=0.05,
        novelty=0.05,
        evidence_quality=0.90,
        metadata={
            "value": 0.05,
            "importance": 0.05,
            "estimated_effort": 0.95,
            "risk": 0.95,
            "opportunity_cost": 0.95,
        },
    )

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.recommendation in {
        "defer",
        "explore",
    }


def test_intrinsic_exploration_can_be_detected():
    candidate = make_candidate(
        expected_value=0.40,
        novelty=0.95,
        evidence_quality=0.90,
        metadata={
            "novelty": 0.95,
            "curiosity": 0.95,
            "exploration_value": 0.95,
            "information_gain": 0.95,
            "uncertainty": 0.90,
            "learning_value": 0.95,
            "confidence": 0.90,
        },
    )

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.intrinsic.should_explore is True
    assert result.intrinsic_score >= 0.50


def test_confidence_adjustment():
    high = make_candidate(
        candidate_id="high-confidence",
        evidence_quality=1.0,
        metadata={"confidence": 1.0},
    )

    low = make_candidate(
        candidate_id="low-confidence",
        evidence_quality=0.20,
        metadata={"confidence": 0.20},
    )

    engine = GoalEvaluationEngine()

    high_result = engine.evaluate(high)
    low_result = engine.evaluate(low)

    assert (
        high_result.confidence_score
        > low_result.confidence_score
    )


def test_evaluate_many():
    candidates = (
        make_candidate(
            "goal-1",
            proposal_score=0.70,
        ),
        make_candidate(
            "goal-2",
            proposal_score=0.90,
            expected_value=0.95,
        ),
        make_candidate(
            "goal-3",
            proposal_score=0.60,
            expected_value=0.60,
        ),
    )

    batch = GoalEvaluationEngine().evaluate_many(
        candidates
    )

    assert isinstance(
        batch,
        GoalEvaluationBatch,
    )

    assert batch.count == 3


def test_evaluate_many_is_ranked():
    candidates = (
        make_candidate(
            "low",
            expected_value=0.30,
            feasibility=0.30,
            urgency=0.30,
            novelty=0.20,
        ),
        make_candidate(
            "high",
            expected_value=0.95,
            feasibility=0.95,
            urgency=0.95,
            novelty=0.95,
        ),
    )

    batch = GoalEvaluationEngine().evaluate_many(
        candidates
    )

    assert batch.best_candidate is not None
    assert (
        batch.best_candidate.candidate_id
        == "high"
    )


def test_evaluate_many_accepts_generator():
    candidates = (
        make_candidate(str(index))
        for index in range(5)
    )

    batch = GoalEvaluationEngine().evaluate_many(
        candidates
    )

    assert batch.count == 5
    assert batch.metadata["input_count"] == 5


def test_max_results():
    candidates = tuple(
        make_candidate(str(index))
        for index in range(5)
    )

    batch = GoalEvaluationEngine().evaluate_many(
        candidates,
        max_results=2,
    )

    assert batch.count == 2
    assert batch.metadata["evaluated_count"] == 2


def test_invalid_max_results_zero():
    with pytest.raises(GoalEvaluationInputError):
        GoalEvaluationEngine().evaluate_many(
            (),
            max_results=0,
        )


def test_invalid_max_results_negative():
    with pytest.raises(GoalEvaluationInputError):
        GoalEvaluationEngine().evaluate_many(
            (),
            max_results=-1,
        )


def test_invalid_max_results_type():
    with pytest.raises(GoalEvaluationInputError):
        GoalEvaluationEngine().evaluate_many(
            (),
            max_results=True,
        )


def test_invalid_candidate_rejected_in_batch():
    candidate = make_candidate("valid")

    batch = GoalEvaluationEngine().evaluate_many(
        (
            candidate,
            object(),
        )
    )

    assert batch.count == 1
    assert (
        "valid"
        not in batch.rejected_candidate_ids
    )


def test_recommended_results():
    candidates = (
        make_candidate(
            "high",
            expected_value=0.95,
            feasibility=0.95,
            urgency=0.95,
            evidence_quality=0.95,
        ),
        make_candidate(
            "low",
            expected_value=0.05,
            feasibility=0.05,
            urgency=0.05,
            novelty=0.05,
            evidence_quality=0.20,
        ),
    )

    batch = GoalEvaluationEngine().evaluate_many(
        candidates
    )

    assert all(
        result.recommended
        for result in batch.recommended_results
    )


def test_exploration_results():
    candidate = make_candidate(
        novelty=0.95,
        metadata={
            "curiosity": 0.95,
            "exploration_value": 0.95,
            "information_gain": 0.95,
            "learning_value": 0.95,
            "uncertainty": 0.90,
        },
    )

    batch = GoalEvaluationEngine().evaluate_many(
        (candidate,)
    )

    assert len(batch.exploration_results) == 1


def test_batch_properties_on_empty_batch():
    batch = GoalEvaluationBatch(
        results=()
    )

    assert batch.count == 0
    assert batch.best_result is None
    assert batch.best_candidate is None
    assert batch.recommended_results == ()
    assert batch.exploration_results == ()


def test_result_alias_properties():
    candidate = make_candidate()

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.goal_id == candidate.candidate_id
    assert result.description == candidate.description
    assert (
        result.utility_score
        == result.utility.utility_score
    )
    assert (
        result.expected_utility
        == result.utility.expected_utility
    )
    assert (
        result.intrinsic_score
        == result.intrinsic.intrinsic_score
    )


def test_candidate_metadata_is_preserved():
    candidate = make_candidate(
        metadata={
            "source": "memory",
            "memory_gap_id": "gap-1",
            "custom": "value",
        }
    )

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.metadata["source"] == "memory"
    assert (
        result.metadata["memory_gap_id"]
        == "gap-1"
    )
    assert result.metadata["custom"] == "value"


def test_result_is_frozen():
    candidate = make_candidate()

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    with pytest.raises(
        Exception
    ):
        result.combined_score = 0.0


def test_different_weighting_changes_score():
    candidate = make_candidate(
        metadata={
            "value": 0.90,
            "importance": 0.90,
            "novelty": 0.90,
            "curiosity": 0.90,
            "exploration_value": 0.90,
            "information_gain": 0.90,
            "learning_value": 0.90,
        }
    )

    utility_heavy = GoalEvaluationEngine(
        weights=GoalEvaluationWeights(
            utility=0.90,
            intrinsic=0.10,
        )
    )

    intrinsic_heavy = GoalEvaluationEngine(
        weights=GoalEvaluationWeights(
            utility=0.10,
            intrinsic=0.90,
        )
    )

    first = utility_heavy.evaluate(candidate)
    second = intrinsic_heavy.evaluate(candidate)

    assert 0 <= first.combined_score <= 1
    assert 0 <= second.combined_score <= 1


def test_goal_candidate_is_not_replaced():
    candidate = make_candidate()

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.candidate is candidate


def test_empty_candidates():
    batch = GoalEvaluationEngine().evaluate_many(
        ()
    )

    assert batch.count == 0
    assert batch.metadata["input_count"] == 0


def test_candidate_payload_is_preserved():
    payload = {
        "source": "memory-gap",
        "target": "database",
    }

    candidate = GoalCandidate(
        candidate_id="payload-goal",
        description="Inspect database",
        origin=GoalOrigin.MEMORY,
        priority=0.8,
        proposal_score=0.8,
        urgency=0.8,
        expected_value=0.8,
        feasibility=0.8,
        novelty=0.7,
        evidence_quality=0.9,
        payload=payload,
    )

    result = GoalEvaluationEngine().evaluate(
        candidate
    )

    assert result.candidate.payload == payload


def test_goal_evaluation_does_not_call_lifecycle():
    candidate = make_candidate()

    engine = GoalEvaluationEngine()

    result = engine.evaluate(candidate)

    assert result.goal_id == candidate.candidate_id