import pytest

from agent.cognition.agent_decision import (
    AgentDecision,
    AgentDecisionMaker,
    DecisionMode,
)


def test_high_uncertainty_selects_explore():

    maker = AgentDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.2,
    )

    assert isinstance(
        decision,
        AgentDecision,
    )

    assert (
        decision.mode
        == DecisionMode.EXPLORE
    )

    assert decision.should_explore
    assert not decision.should_exploit


def test_low_uncertainty_selects_exploit():

    maker = AgentDecisionMaker()

    decision = maker.decide(
        uncertainty=0.2,
        expected_information_gain=0.9,
        risk=0.2,
    )

    assert (
        decision.mode
        == DecisionMode.EXPLOIT
    )

    assert decision.should_exploit
    assert not decision.should_explore


def test_high_risk_blocks_exploration():

    maker = AgentDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.9,
    )

    assert (
        decision.mode
        == DecisionMode.EXPLOIT
    )

    assert (
        decision.reason
        == "Exploration risk is too high."
    )


def test_low_information_gain_selects_exploit():

    maker = AgentDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.2,
        risk=0.2,
    )

    assert (
        decision.mode
        == DecisionMode.EXPLOIT
    )


def test_custom_thresholds_are_supported():

    maker = AgentDecisionMaker(
        exploration_threshold=0.8,
        maximum_exploration_risk=0.4,
    )

    decision = maker.decide(
        uncertainty=0.8,
        expected_information_gain=0.9,
        risk=0.3,
    )

    assert (
        decision.mode
        == DecisionMode.EXPLORE
    )


def test_values_are_clamped():

    maker = AgentDecisionMaker()

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

    assert (
        decision.mode
        == DecisionMode.EXPLORE
    )


def test_decision_is_immutable():

    maker = AgentDecisionMaker()

    decision = maker.decide(
        uncertainty=0.9,
        expected_information_gain=0.9,
        risk=0.2,
    )

    try:
        decision.mode = DecisionMode.EXPLOIT
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "AgentDecision should be immutable."
        )


def test_decision_contains_all_signal_values():

    maker = AgentDecisionMaker()

    decision = maker.decide(
        uncertainty=0.7,
        expected_information_gain=0.8,
        risk=0.2,
    )

    assert (
        decision.uncertainty
        == pytest.approx(0.7)
    )

    assert (
        decision.expected_information_gain
        == pytest.approx(0.8)
    )

    assert (
        decision.risk
        == pytest.approx(0.2)
    )
