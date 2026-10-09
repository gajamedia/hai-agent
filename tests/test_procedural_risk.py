import pytest

from agent.cognition.procedural_risk import (
    ProceduralRiskAssessor,
)
from agent.memory.procedural import (
    ProceduralMemory,
)


def create_memory():

    return ProceduralMemory(
        content="Procedure for opening the door.",
        skill="open_the_door",
        steps=[
            "take_key",
            "unlock_door",
        ],
        confidence=0.5,
        source="procedural_learning",
        tags=[
            "procedural",
            "learned",
        ],
    )


def test_procedural_memory_can_be_assessed():

    memory = create_memory()

    memory.record_success(
        ["experience-1"]
    )

    memory.record_success(
        ["experience-2"]
    )

    memory.record_success(
        ["experience-3"]
    )

    assessor = ProceduralRiskAssessor()

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


def test_procedural_failures_increase_risk():

    memory = create_memory()

    memory.record_success(
        ["experience-1"]
    )

    memory.record_failure(
        ["experience-2"]
    )

    memory.record_failure(
        ["experience-3"]
    )

    assessor = ProceduralRiskAssessor()

    assessment = assessor.assess(
        memory
    )

    assert assessment.evidence == 3

    assert (
        assessment.failure_probability
        > assessment.success_probability
    )

    assert (
        assessment.risk_score
        > 0.5
    )


def test_no_procedural_history_is_high_risk():

    memory = create_memory()

    assessor = ProceduralRiskAssessor()

    assessment = assessor.assess(
        memory
    )

    assert assessment.evidence == 0

    assert assessment.is_high_risk


def test_procedural_risk_assessment_uses_current_counts():

    memory = create_memory()

    memory.record_success(
        ["experience-1"]
    )

    assessor = ProceduralRiskAssessor()

    first = assessor.assess(
        memory
    )

    memory.record_failure(
        ["experience-2"]
    )

    memory.record_failure(
        ["experience-3"]
    )

    second = assessor.assess(
        memory
    )

    assert (
        second.risk_score
        > first.risk_score
    )


def test_none_procedural_memory_is_rejected():

    assessor = ProceduralRiskAssessor()

    with pytest.raises(
        ValueError,
        match="procedural_memory cannot be None",
    ):
        assessor.assess(None)
