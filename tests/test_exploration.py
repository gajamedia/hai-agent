from agent.cognition.exploration import (
    ExplorationDecision,
    ExplorationDecisionMaker,
)


def test_high_uncertainty_and_high_information_gain_explores():

    maker = ExplorationDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.3,
    )

    assert isinstance(
        decision,
        ExplorationDecision,
    )

    assert decision.explore is True

    assert decision.uncertainty == 0.9
    assert decision.expected_information_gain == 0.9
    assert decision.risk == 0.3

    assert (
        decision.reason
        == "High uncertainty with useful information gain."
    )


def test_low_uncertainty_prefers_exploitation():

    maker = ExplorationDecisionMaker()

    decision = maker.decide(
        uncertainty=0.2,
        expected_information_gain=0.9,
        risk=0.2,
    )

    assert decision.explore is False

    assert (
        decision.reason
        == "Uncertainty is sufficiently low."
    )


def test_low_information_gain_does_not_explore():

    maker = ExplorationDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.2,
        risk=0.2,
    )

    assert decision.explore is False

    assert (
        decision.reason
        == "Expected information gain is too low."
    )


def test_high_exploration_risk_blocks_exploration():

    maker = ExplorationDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.9,
    )

    assert decision.explore is False

    assert (
        decision.reason
        == "Exploration risk is too high."
    )


def test_values_are_clamped():

    maker = ExplorationDecisionMaker()

    decision = maker.decide(
        uncertainty=2.0,
        expected_information_gain=3.0,
        risk=-1.0,
    )

    assert decision.uncertainty == 1.0

    assert (
        decision.expected_information_gain
        == 1.0
    )

    assert decision.risk == 0.0

    assert decision.explore is True


def test_custom_thresholds_are_supported():

    maker = ExplorationDecisionMaker(
        uncertainty_threshold=0.8,
        information_gain_threshold=0.8,
        maximum_exploration_risk=0.4,
    )

    low = maker.decide(
        uncertainty=0.7,
        expected_information_gain=0.9,
        risk=0.2,
    )

    assert low.explore is False

    high = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.3,
    )

    assert high.explore is True


def test_exploration_decision_is_immutable():

    maker = ExplorationDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.2,
    )

    try:
        decision.explore = False
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "ExplorationDecision should be immutable."
        )
