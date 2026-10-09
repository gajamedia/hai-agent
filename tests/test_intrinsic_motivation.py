from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from agent.llm.intrinsic_motivation import (
    IntrinsicMotivationConfigurationError,
    IntrinsicMotivationEvaluation,
    IntrinsicMotivationInput,
    IntrinsicMotivationInputError,
    IntrinsicMotivationEvaluator,
    IntrinsicMotivationWeights,
)


def make_goal(**overrides):
    data = {
        "goal_id": "g1",
        "description": "Explore unknown situation",
        "novelty": 0.8,
        "curiosity": 0.8,
        "exploration_value": 0.8,
        "information_gain": 0.7,
        "uncertainty": 0.6,
        "learning_value": 0.8,
        "confidence": 0.9,
        "evidence_quality": 0.8,
        "metadata": {
            "source": "test",
        },
    }

    data.update(overrides)

    return IntrinsicMotivationInput(
        **data
    )


def test_default_weights_total_is_one():
    assert (
        IntrinsicMotivationWeights().total
        == pytest.approx(1.0)
    )


def test_weights_accept_valid_values():
    weights = IntrinsicMotivationWeights(
        novelty=1,
        curiosity=0,
        exploration_value=0,
        information_gain=0,
        uncertainty=0,
        learning_value=0,
    )

    assert weights.total == 1


@pytest.mark.parametrize(
    "field",
    [
        "novelty",
        "curiosity",
        "exploration_value",
        "information_gain",
        "uncertainty",
        "learning_value",
        "confidence",
        "evidence_quality",
    ],
)
def test_input_rejects_out_of_range(field):
    values = make_goal().__dict__.copy()

    values[field] = 1.1

    with pytest.raises(
        IntrinsicMotivationInputError
    ):
        IntrinsicMotivationInput(
            **values
        )


def test_input_rejects_negative_values():
    with pytest.raises(
        IntrinsicMotivationInputError
    ):
        make_goal(novelty=-0.1)


def test_input_rejects_empty_id():
    with pytest.raises(
        IntrinsicMotivationInputError
    ):
        make_goal(goal_id="   ")


def test_input_rejects_non_mapping_metadata():
    with pytest.raises(
        IntrinsicMotivationInputError
    ):
        make_goal(metadata=[])


def test_weights_reject_negative():
    with pytest.raises(
        IntrinsicMotivationConfigurationError
    ):
        IntrinsicMotivationWeights(
            novelty=-0.1
        )


def test_weights_reject_above_one():
    with pytest.raises(
        IntrinsicMotivationConfigurationError
    ):
        IntrinsicMotivationWeights(
            novelty=1.1
        )


def test_weights_require_positive_total():
    with pytest.raises(
        IntrinsicMotivationConfigurationError
    ):
        IntrinsicMotivationWeights(
            novelty=0,
            curiosity=0,
            exploration_value=0,
            information_gain=0,
            uncertainty=0,
            learning_value=0,
        )


@pytest.mark.parametrize(
    "field",
    [
        "exploration_threshold",
        "minimum_confidence",
    ],
)
def test_evaluator_rejects_bad_config(field):
    kwargs = {
        field: 1.1
    }

    with pytest.raises(
        IntrinsicMotivationConfigurationError
    ):
        IntrinsicMotivationEvaluator(
            **kwargs
        )


def test_evaluator_returns_expected_fields():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    assert result.goal_id == "g1"
    assert (
        result.description
        == "Explore unknown situation"
    )

    assert 0 <= result.intrinsic_score <= 1
    assert (
        0
        <= result.confidence_adjusted_score
        <= 1
    )

    assert isinstance(
        result.recommended,
        bool,
    )


def test_result_is_correct_type():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    assert isinstance(
        result,
        IntrinsicMotivationEvaluation,
    )


def test_high_intrinsic_motivation_is_recommended():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                novelty=1,
                curiosity=1,
                exploration_value=1,
                information_gain=1,
                uncertainty=1,
                learning_value=1,
                confidence=1,
                evidence_quality=1,
            )
        )
    )

    assert result.recommended is True
    assert result.recommendation == "explore"


def test_low_intrinsic_motivation_is_deferred():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                novelty=0,
                curiosity=0,
                exploration_value=0,
                information_gain=0,
                uncertainty=0,
                learning_value=0,
            )
        )
    )

    assert result.recommended is False
    assert result.recommendation == "defer"


def test_confidence_gate_can_block_exploration():
    result = (
        IntrinsicMotivationEvaluator(
            minimum_confidence=0.9
        )
        .evaluate(
            make_goal(
                confidence=0.8
            )
        )
    )

    assert (
        result.recommendation
        == "insufficient_confidence"
    )

    assert result.recommended is False


def test_confidence_adjustment_reduces_score():
    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                confidence=1,
                evidence_quality=1,
            )
        )
    )

    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                confidence=0.5,
                evidence_quality=0.5,
            )
        )
    )

    assert (
        low.confidence_adjusted_score
        <= high.confidence_adjusted_score
    )


def test_zero_confidence_zeroes_adjusted_score():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                confidence=0
            )
        )
    )

    assert (
        result.confidence_adjusted_score
        == 0
    )


def test_zero_evidence_zeroes_adjusted_score():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                evidence_quality=0
            )
        )
    )

    assert (
        result.confidence_adjusted_score
        == 0
    )


def test_components_include_all_dimensions():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    expected = {
        "novelty",
        "curiosity",
        "exploration_value",
        "information_gain",
        "uncertainty",
        "learning_value",
        "confidence",
        "evidence_quality",
    }

    assert (
        expected
        == set(result.components)
    )


def test_custom_weights_change_evaluation():
    default = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                novelty=1,
                learning_value=0,
            )
        )
    )

    custom = (
        IntrinsicMotivationEvaluator(
            weights=IntrinsicMotivationWeights(
                novelty=0,
                curiosity=0,
                exploration_value=0,
                information_gain=0,
                uncertainty=0,
                learning_value=1,
            )
        )
        .evaluate(
            make_goal(
                novelty=1,
                learning_value=0,
            )
        )
    )

    assert (
        default.intrinsic_score
        != custom.intrinsic_score
    )


def test_zero_novelty_weight_is_supported():
    result = (
        IntrinsicMotivationEvaluator(
            weights=IntrinsicMotivationWeights(
                novelty=0,
                curiosity=1,
                exploration_value=0,
                information_gain=0,
                uncertainty=0,
                learning_value=0,
            )
        )
        .evaluate(
            make_goal(
                novelty=1,
                curiosity=0,
            )
        )
    )

    assert result.intrinsic_score == 0


def test_mapping_input_is_supported():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal().__dict__
        )
    )

    assert result.goal_id == "g1"


def test_object_input_is_supported():
    obj = SimpleNamespace(
        **make_goal().__dict__
    )

    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(obj)
    )

    assert result.goal_id == "g1"


def test_id_alias_is_supported():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9"
            }
        )
    )

    assert result.goal_id == "g9"


def test_goal_alias_is_supported():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "goal": "Explore",
            }
        )
    )

    assert result.description == "Explore"


def test_novelty_score_alias_is_supported():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "novelty_score": 0.7,
            }
        )
    )

    assert (
        result.components["novelty"]
        == 0.7
    )


def test_curiosity_score_alias_is_supported():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "curiosity_score": 0.7,
            }
        )
    )

    assert (
        result.components["curiosity"]
        == 0.7
    )


def test_exploration_score_alias_is_supported():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "exploration_score": 0.7,
            }
        )
    )

    assert (
        result.components[
            "exploration_value"
        ]
        == 0.7
    )


def test_information_gain_score_alias():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "information_gain_score": 0.7,
            }
        )
    )

    assert (
        result.components[
            "information_gain"
        ]
        == 0.7
    )


def test_uncertainty_score_alias():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "uncertainty_score": 0.7,
            }
        )
    )

    assert (
        result.components[
            "uncertainty"
        ]
        == 0.7
    )


def test_learning_score_alias():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "id": "g9",
                "learning_score": 0.7,
            }
        )
    )

    assert (
        result.components[
            "learning_value"
        ]
        == 0.7
    )


def test_evaluate_many_sorts_by_score():
    goals = [
        make_goal(
            goal_id="a",
            novelty=0.1,
            curiosity=0.1,
            exploration_value=0.1,
            information_gain=0.1,
            uncertainty=0.1,
            learning_value=0.1,
        ),
        make_goal(
            goal_id="b",
            novelty=1,
            curiosity=1,
            exploration_value=1,
            information_gain=1,
            uncertainty=1,
            learning_value=1,
        ),
        make_goal(
            goal_id="c",
            novelty=0.5,
            curiosity=0.5,
            exploration_value=0.5,
            information_gain=0.5,
            uncertainty=0.5,
            learning_value=0.5,
        ),
    ]

    results = (
        IntrinsicMotivationEvaluator()
        .evaluate_many(goals)
    )

    assert [
        item.goal_id
        for item in results
    ] == [
        "b",
        "c",
        "a",
    ]


def test_evaluate_many_empty():
    assert (
        IntrinsicMotivationEvaluator()
        .evaluate_many([])
        == []
    )


def test_threshold_is_respected():
    result = (
        IntrinsicMotivationEvaluator(
            exploration_threshold=0.99
        )
        .evaluate(
            make_goal(
                novelty=1,
                curiosity=1,
                exploration_value=1,
                information_gain=1,
                uncertainty=1,
                learning_value=1,
                confidence=1,
                evidence_quality=1,
            )
        )
    )

    assert result.recommended is True


def test_threshold_one_requires_maximum_score():
    result = (
        IntrinsicMotivationEvaluator(
            exploration_threshold=1
        )
        .evaluate(
            make_goal()
        )
    )

    assert result.recommended is False


def test_minimum_confidence_zero_does_not_block():
    result = (
        IntrinsicMotivationEvaluator(
            minimum_confidence=0
        )
        .evaluate(
            make_goal(
                confidence=0
            )
        )
    )

    assert (
        result.recommendation
        != "insufficient_confidence"
    )


def test_novelty_increases_score():
    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                novelty=0
            )
        )
    )

    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                novelty=1
            )
        )
    )

    assert (
        high.intrinsic_score
        > low.intrinsic_score
    )


def test_curiosity_increases_score():
    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                curiosity=0
            )
        )
    )

    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                curiosity=1
            )
        )
    )

    assert (
        high.intrinsic_score
        > low.intrinsic_score
    )


def test_exploration_value_increases_score():
    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                exploration_value=0
            )
        )
    )

    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                exploration_value=1
            )
        )
    )

    assert (
        high.intrinsic_score
        > low.intrinsic_score
    )


def test_information_gain_increases_score():
    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                information_gain=0
            )
        )
    )

    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                information_gain=1
            )
        )
    )

    assert (
        high.intrinsic_score
        > low.intrinsic_score
    )


def test_uncertainty_increases_exploration_score():
    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                uncertainty=0
            )
        )
    )

    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                uncertainty=1
            )
        )
    )

    assert (
        high.intrinsic_score
        > low.intrinsic_score
    )


def test_learning_value_increases_score():
    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                learning_value=0
            )
        )
    )

    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                learning_value=1
            )
        )
    )

    assert (
        high.intrinsic_score
        > low.intrinsic_score
    )


def test_metadata_is_copied():
    goal = make_goal()

    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(goal)
    )

    result.metadata["changed"] = True

    assert (
        "changed"
        not in goal.metadata
    )


def test_result_is_frozen():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        result.goal_id = "x"


def test_input_is_frozen():
    goal = make_goal()

    with pytest.raises(
        FrozenInstanceError
    ):
        goal.goal_id = "x"


def test_weights_are_frozen():
    weights = (
        IntrinsicMotivationWeights()
    )

    with pytest.raises(
        FrozenInstanceError
    ):
        weights.novelty = 1


def test_boolean_numeric_input_rejected():
    with pytest.raises(
        IntrinsicMotivationInputError
    ):
        make_goal(
            novelty=True
        )


def test_boolean_weight_rejected():
    with pytest.raises(
        IntrinsicMotivationConfigurationError
    ):
        IntrinsicMotivationWeights(
            novelty=True
        )


def test_non_numeric_mapping_rejected():
    with pytest.raises(
        IntrinsicMotivationInputError
    ):
        (
            IntrinsicMotivationEvaluator()
            .evaluate(
                {
                    "goal_id": "g",
                    "novelty": "high",
                }
            )
        )


def test_missing_values_use_zero():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "goal_id": "g"
            }
        )
    )

    assert (
        result.components["novelty"]
        == 0
    )

    assert (
        result.components["curiosity"]
        == 0
    )


def test_missing_confidence_defaults_to_one():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            {
                "goal_id": "g",
                "novelty": 1,
            }
        )
    )

    assert (
        result.components["confidence"]
        == 1
    )


def test_rationale_is_human_readable():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    assert "novelty=" in result.rationale
    assert "curiosity=" in result.rationale
    assert "intrinsic_score=" in result.rationale


def test_should_explore_matches_recommended():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    assert (
        result.should_explore
        == result.recommended
    )


def test_exploration_score_alias():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    assert (
        result.exploration_score
        == result.intrinsic_score
    )


def test_adjusted_exploration_score_alias():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(make_goal())
    )

    assert (
        result.adjusted_exploration_score
        == result.confidence_adjusted_score
    )


def test_evaluator_is_reusable():
    evaluator = (
        IntrinsicMotivationEvaluator()
    )

    first = evaluator.evaluate(
        make_goal(goal_id="a")
    )

    second = evaluator.evaluate(
        make_goal(goal_id="b")
    )

    assert first.goal_id == "a"
    assert second.goal_id == "b"


def test_evaluator_is_side_effect_free():
    goal = make_goal()

    before = goal.__dict__.copy()

    (
        IntrinsicMotivationEvaluator()
        .evaluate(goal)
    )

    assert goal.__dict__ == before


def test_evaluate_many_does_not_mutate_inputs():
    goals = [
        make_goal(goal_id="a"),
        make_goal(goal_id="b"),
    ]

    before = [
        goal.__dict__.copy()
        for goal in goals
    ]

    (
        IntrinsicMotivationEvaluator()
        .evaluate_many(goals)
    )

    assert [
        goal.__dict__
        for goal in goals
    ] == before


def test_neutral_goal_is_deferred():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                novelty=0.5,
                curiosity=0.5,
                exploration_value=0.5,
                information_gain=0.5,
                uncertainty=0.5,
                learning_value=0.5,
            )
        )
    )

    assert result.intrinsic_score == pytest.approx(
        0.5
    )

    assert (
        result.confidence_adjusted_score
        == pytest.approx(0.36)
    )

    assert result.recommended is False
    assert result.recommendation == "defer"


def test_low_confidence_can_reduce_exploration():
    high = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                confidence=1,
                evidence_quality=1,
            )
        )
    )

    low = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                confidence=0.2,
                evidence_quality=0.2,
            )
        )
    )

    assert (
        low.confidence_adjusted_score
        < high.confidence_adjusted_score
    )


def test_custom_learning_weight():
    evaluator = (
        IntrinsicMotivationEvaluator(
            weights=IntrinsicMotivationWeights(
                novelty=0,
                curiosity=0,
                exploration_value=0,
                information_gain=0,
                uncertainty=0,
                learning_value=1,
            )
        )
    )

    result = evaluator.evaluate(
        make_goal(
            learning_value=1,
            novelty=0,
            curiosity=0,
            exploration_value=0,
            information_gain=0,
            uncertainty=0,
        )
    )

    assert result.intrinsic_score == 1


def test_metadata_preserved():
    result = (
        IntrinsicMotivationEvaluator()
        .evaluate(
            make_goal(
                metadata={
                    "source": "test",
                    "kind": "exploration",
                }
            )
        )
    )

    assert result.metadata == {
        "source": "test",
        "kind": "exploration",
    }
