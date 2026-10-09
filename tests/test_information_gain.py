import pytest

from agent.cognition.information_gain import (
    InformationGainAssessment,
    InformationGainEstimator,
)
from agent.memory.procedural import ProceduralMemory
from agent.memory.causal import CausalMemory


def test_no_evidence_has_maximum_uncertainty():

    estimator = InformationGainEstimator()

    assessment = (
        estimator.estimate_from_evidence(
            evidence=0
        )
    )

    assert isinstance(
        assessment,
        InformationGainAssessment,
    )

    assert assessment.evidence == 0

    assert (
        assessment.uncertainty
        == pytest.approx(1.0)
    )

    assert (
        assessment.expected_information_gain
        == pytest.approx(1.0)
    )


def test_more_evidence_reduces_uncertainty():

    estimator = InformationGainEstimator()

    low = (
        estimator.estimate_from_evidence(
            evidence=1
        )
    )

    high = (
        estimator.estimate_from_evidence(
            evidence=100
        )
    )

    assert (
        high.uncertainty
        < low.uncertainty
    )

    assert (
        high.expected_information_gain
        < low.expected_information_gain
    )


def test_information_gain_follows_uncertainty():

    estimator = InformationGainEstimator()

    assessment = (
        estimator.estimate_from_evidence(
            evidence=10
        )
    )

    assert (
        assessment.expected_information_gain
        == pytest.approx(
            assessment.uncertainty
        )
    )


def test_outcomes_are_converted_to_evidence():

    estimator = InformationGainEstimator()

    assessment = (
        estimator.estimate_from_outcomes(
            success_count=7,
            failure_count=3,
        )
    )

    assert assessment.evidence == 10


def test_zero_outcomes_have_maximum_information_gain():

    estimator = InformationGainEstimator()

    assessment = (
        estimator.estimate_from_outcomes(
            success_count=0,
            failure_count=0,
        )
    )

    assert assessment.evidence == 0

    assert (
        assessment.expected_information_gain
        == pytest.approx(1.0)
    )


def test_negative_evidence_is_rejected():

    estimator = InformationGainEstimator()

    with pytest.raises(
        ValueError,
        match="evidence cannot be negative",
    ):
        estimator.estimate_from_evidence(
            evidence=-1
        )


def test_negative_success_count_is_rejected():

    estimator = InformationGainEstimator()

    with pytest.raises(
        ValueError,
        match="success_count cannot be negative",
    ):
        estimator.estimate_from_outcomes(
            success_count=-1,
            failure_count=0,
        )


def test_negative_failure_count_is_rejected():

    estimator = InformationGainEstimator()

    with pytest.raises(
        ValueError,
        match="failure_count cannot be negative",
    ):
        estimator.estimate_from_outcomes(
            success_count=0,
            failure_count=-1,
        )

def test_procedural_memory_can_provide_information_gain():

    memory = ProceduralMemory(
        content="Open door procedure.",
        skill="open_the_door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    memory.record_success(
        ["experience-1"]
    )

    estimator = InformationGainEstimator()

    assessment = (
        estimator.estimate_from_procedural_memory(
            memory
        )
    )

    assert assessment.evidence == 1
    assert assessment.uncertainty > 0.0
    assert assessment.expected_information_gain > 0.0


def test_more_procedural_evidence_reduces_information_gain():

    memory = ProceduralMemory(
        content="Open door procedure.",
        skill="open_the_door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    estimator = InformationGainEstimator()

    before = (
        estimator.estimate_from_procedural_memory(
            memory
        )
    )

    for index in range(20):
        memory.record_success(
            [f"experience-{index}"]
        )

    after = (
        estimator.estimate_from_procedural_memory(
            memory
        )
    )

    assert (
        after.evidence
        > before.evidence
    )

    assert (
        after.uncertainty
        < before.uncertainty
    )

    assert (
        after.expected_information_gain
        < before.expected_information_gain
    )


def test_causal_memory_can_provide_information_gain():

    memory = CausalMemory(
        content=(
            "Taking the key enables "
            "unlocking the door."
        ),
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )

    memory.reinforce(
        ["episode-1"]
    )

    estimator = InformationGainEstimator()

    assessment = (
        estimator.estimate_from_causal_memory(
            memory
        )
    )

    assert assessment.evidence == 1
    assert assessment.uncertainty > 0.0


# def test_causal_contradictions_increase_information_gain():

    memory = CausalMemory(
        content=(
            "Taking the key enables "
            "unlocking the door."
        ),
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )

    memory.reinforce(
        ["episode-1"]
    )

    estimator = InformationGainEstimator()

    stable = (
        estimator.estimate_from_causal_memory(
            memory
        )
    )

    memory.contradict()

    uncertain = (
        estimator.estimate_from_causal_memory(
            memory
        )
    )

    assert uncertain.evidence == 1

    assert (
        uncertain.uncertainty
        > stable.uncertainty
    )

    assert (
        uncertain.expected_information_gain
        > stable.expected_information_gain
    )

def test_none_procedural_memory_is_rejected():

    estimator = InformationGainEstimator()

    try:
        estimator.estimate_from_procedural_memory(
            None
        )
    except ValueError as exc:
        assert str(exc) == (
            "procedural_memory cannot be None."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_none_causal_memory_is_rejected():

    estimator = InformationGainEstimator()

    try:
        estimator.estimate_from_causal_memory(
            None
        )
    except ValueError as exc:
        assert str(exc) == (
            "causal_memory cannot be None."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_negative_causal_evidence_is_rejected():

    class InvalidCausalMemory:
        evidence_count = -1
        contradiction_count = 0

    estimator = InformationGainEstimator()

    try:
        estimator.estimate_from_causal_memory(
            InvalidCausalMemory()
        )
    except ValueError as exc:
        assert str(exc) == (
            "evidence_count cannot be negative."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )

def test_strong_causal_evidence_has_lower_information_gain():

    weak = CausalMemory(
        content="Taking the key enables unlocking.",
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )

    strong = CausalMemory(
        content="Taking the key enables unlocking.",
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )

    weak.reinforce(
        ["episode-1"]
    )

    for index in range(20):
        strong.reinforce(
            [f"episode-{index}"]
        )

    estimator = InformationGainEstimator()

    weak_assessment = (
        estimator.estimate_from_causal_memory(
            weak
        )
    )

    strong_assessment = (
        estimator.estimate_from_causal_memory(
            strong
        )
    )

    assert (
        strong_assessment.uncertainty
        < weak_assessment.uncertainty
    )

    assert (
        strong_assessment.expected_information_gain
        < weak_assessment.expected_information_gain
    )


def test_contradiction_can_make_existing_knowledge_uncertain():

    memory = CausalMemory(
        content=(
            "Taking the key enables "
            "unlocking the door."
        ),
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )

    for index in range(10):
        memory.reinforce(
            [f"episode-{index}"]
        )

    estimator = InformationGainEstimator()

    stable = (
        estimator.estimate_from_causal_memory(
            memory
        )
    )

    for index in range(10):
        memory.contradict()

    unstable = (
        estimator.estimate_from_causal_memory(
            memory
        )
    )

    assert (
        unstable.uncertainty
        > stable.uncertainty
    )

    assert (
        unstable.expected_information_gain
        > stable.expected_information_gain
    )
