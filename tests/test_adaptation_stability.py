import pytest

from agent.learning.adaptation_stability import (
    AdaptationStabilityController,
)


def test_no_change_is_stable():
    controller = (
        AdaptationStabilityController()
    )

    result = controller.update(
        previous_bias=0.10,
        proposed_bias=0.10,
    )

    assert result.adapted is False
    assert result.held is True
    assert result.applied_bias == 0.10
    assert result.change == 0.0
    assert result.is_stable


def test_tiny_change_is_ignored():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.05,
        )
    )

    result = controller.update(
        previous_bias=0.10,
        proposed_bias=0.12,
    )

    assert result.held is True
    assert result.adapted is False
    assert result.applied_bias == 0.10
    assert result.change == 0.0


def test_change_above_threshold_is_applied():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.02,
        )
    )

    result = controller.update(
        previous_bias=0.10,
        proposed_bias=0.15,
    )

    assert result.adapted is True
    assert result.applied_bias == 0.15
    assert result.change == 0.05


def test_positive_change_is_limited():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            maximum_adaptation_step=0.05,
        )
    )

    result = controller.update(
        previous_bias=0.00,
        proposed_bias=0.25,
    )

    assert result.adapted is True
    assert result.applied_bias == 0.05
    assert result.change == 0.05
    assert result.clamped is True


def test_negative_change_is_limited():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            maximum_adaptation_step=0.05,
        )
    )

    result = controller.update(
        previous_bias=0.00,
        proposed_bias=-0.25,
    )

    assert result.adapted is True
    assert result.applied_bias == -0.05
    assert result.change == -0.05
    assert result.clamped is True


def test_positive_bias_does_not_flip_for_small_negative_proposal():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            hysteresis=0.05,
        )
    )

    result = controller.update(
        previous_bias=0.10,
        proposed_bias=-0.03,
    )

    assert result.held is True
    assert result.adapted is False
    assert result.applied_bias == 0.10


def test_negative_bias_does_not_flip_for_small_positive_proposal():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            hysteresis=0.05,
        )
    )

    result = controller.update(
        previous_bias=-0.10,
        proposed_bias=0.03,
    )

    assert result.held is True
    assert result.adapted is False
    assert result.applied_bias == -0.10


def test_positive_bias_can_flip_after_hysteresis():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            hysteresis=0.05,
            maximum_adaptation_step=0.20,
        )
    )

    result = controller.update(
        previous_bias=0.10,
        proposed_bias=-0.10,
    )

    assert result.adapted is True
    assert result.applied_bias == -0.10


def test_negative_bias_can_flip_after_hysteresis():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            hysteresis=0.05,
            maximum_adaptation_step=0.20,
        )
    )

    result = controller.update(
        previous_bias=-0.10,
        proposed_bias=0.10,
    )

    assert result.adapted is True
    assert result.applied_bias == 0.10


def test_bias_is_clamped_to_maximum():
    controller = (
        AdaptationStabilityController(
            maximum_bias=0.25,
        )
    )

    assert (
        controller.normalize(1.0)
        == 0.25
    )

    assert (
        controller.normalize(-1.0)
        == -0.25
    )


def test_update_clamps_previous_bias():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            maximum_adaptation_step=0.10,
            maximum_bias=0.25,
        )
    )

    result = controller.update(
        previous_bias=1.0,
        proposed_bias=0.30,
    )

    assert (
        result.previous_bias
        == 0.25
    )


def test_update_clamps_proposed_bias():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            maximum_adaptation_step=0.10,
            maximum_bias=0.25,
        )
    )

    result = controller.update(
        previous_bias=0.0,
        proposed_bias=1.0,
    )

    assert (
        result.proposed_bias
        == 0.25
    )


def test_direction_is_positive():
    controller = (
        AdaptationStabilityController()
    )

    result = controller.update(
        previous_bias=0.0,
        proposed_bias=0.10,
    )

    assert result.direction == "positive"


def test_direction_is_negative():
    controller = (
        AdaptationStabilityController()
    )

    result = controller.update(
        previous_bias=0.0,
        proposed_bias=-0.10,
    )

    assert result.direction == "negative"


def test_direction_is_neutral():
    controller = (
        AdaptationStabilityController()
    )

    result = controller.update(
        previous_bias=0.0,
        proposed_bias=0.0,
    )

    assert result.direction == "neutral"


def test_multiple_steps_converge_without_large_jump():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            maximum_adaptation_step=0.05,
        )
    )

    first = controller.update(
        previous_bias=0.0,
        proposed_bias=0.20,
    )

    second = controller.update(
        previous_bias=first.applied_bias,
        proposed_bias=0.20,
    )

    third = controller.update(
        previous_bias=second.applied_bias,
        proposed_bias=0.20,
    )

    assert first.applied_bias == 0.05
    assert second.applied_bias == 0.10
    assert third.applied_bias == 0.15


def test_small_negative_change_after_positive_bias_is_held():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            hysteresis=0.05,
        )
    )

    result = controller.update(
        previous_bias=0.20,
        proposed_bias=0.18,
    )

    assert result.held is True
    assert result.applied_bias == 0.20


def test_small_positive_change_after_negative_bias_is_held():
    controller = (
        AdaptationStabilityController(
            minimum_change=0.01,
            hysteresis=0.05,
        )
    )

    result = controller.update(
        previous_bias=-0.20,
        proposed_bias=-0.18,
    )

    assert result.held is True
    assert result.applied_bias == -0.20


def test_invalid_minimum_change_is_rejected():
    with pytest.raises(ValueError):
        AdaptationStabilityController(
            minimum_change=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptationStabilityController(
            minimum_change=1.1,
        )


def test_invalid_hysteresis_is_rejected():
    with pytest.raises(ValueError):
        AdaptationStabilityController(
            hysteresis=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptationStabilityController(
            hysteresis=1.1,
        )


def test_hysteresis_cannot_exceed_maximum_bias():
    with pytest.raises(
        ValueError,
        match="hysteresis cannot exceed maximum_bias",
    ):
        AdaptationStabilityController(
            hysteresis=0.30,
            maximum_bias=0.25,
        )


def test_invalid_maximum_adaptation_step_is_rejected():
    with pytest.raises(ValueError):
        AdaptationStabilityController(
            maximum_adaptation_step=0.0,
        )

    with pytest.raises(ValueError):
        AdaptationStabilityController(
            maximum_adaptation_step=1.1,
        )


def test_invalid_maximum_bias_is_rejected():
    with pytest.raises(ValueError):
        AdaptationStabilityController(
            maximum_bias=-0.1,
        )

    with pytest.raises(ValueError):
        AdaptationStabilityController(
            maximum_bias=1.1,
        )
