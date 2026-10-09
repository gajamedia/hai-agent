import pytest

from agent.cognition.causal_risk import (
    CausalRiskAssessor,
)
from agent.memory.causal import CausalMemory


def create_memory():

    return CausalMemory(
        content=(
            "Taking the key enables "
            "unlocking the door."
        ),
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
        confidence=0.5,
    )


def test_causal_memory_can_be_assessed():

    memory = create_memory()

    memory.reinforce(
        ["episode-1"]
    )

    memory.reinforce(
        ["episode-2"]
    )

    memory.reinforce(
        ["episode-3"]
    )

    assessor = CausalRiskAssessor()

    assessment = assessor.assess(
        memory
    )

    assert assessment.evidence == 3

    assert (
        assessment.success_probability
        > assessment.failure_probability
    )

    assert (
        assessment.risk_score
        < 0.5
    )


def test_causal_contradictions_increase_risk():

    memory = create_memory()

    memory.reinforce(
        ["episode-1"]
    )

    assessor = CausalRiskAssessor()

    low_risk = assessor.assess(
        memory
    )

    memory.contradict()
    memory.contradict()

    high_risk = assessor.assess(
        memory
    )

    assert (
        high_risk.risk_score
        > low_risk.risk_score
    )


def test_causal_memory_with_strong_evidence_has_lower_risk():

    weak = create_memory()

    weak.reinforce(
        ["episode-1"]
    )

    strong = create_memory()

    for index in range(20):
        strong.reinforce(
            [f"episode-{index}"]
        )

    assessor = CausalRiskAssessor()

    weak_assessment = assessor.assess(
        weak
    )

    strong_assessment = assessor.assess(
        strong
    )

    assert (
        strong_assessment.uncertainty
        < weak_assessment.uncertainty
    )

    assert (
        strong_assessment.risk_score
        < weak_assessment.risk_score
    )


def test_causal_memory_without_evidence_is_high_risk():

    memory = create_memory()

    assessor = CausalRiskAssessor()

    assessment = assessor.assess(
        memory
    )

    assert assessment.evidence == 0

    assert assessment.is_high_risk


def test_none_causal_memory_is_rejected():

    assessor = CausalRiskAssessor()

    with pytest.raises(
        ValueError,
        match="causal_memory cannot be None",
    ):
        assessor.assess(None)
