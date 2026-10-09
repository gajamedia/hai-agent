import pytest

from agent.learning.unified_capability_evidence import (
    UnifiedCapabilityEvidenceBuilder,
    UnifiedCapabilityEvidenceResult,
)


def test_normalize_capability():
    builder = UnifiedCapabilityEvidenceBuilder()

    assert (
        builder.normalize_capability(
            "Decision Making"
        )
        == "decision_making"
    )


def test_normalize_capability_handles_hyphen():
    builder = UnifiedCapabilityEvidenceBuilder()

    assert (
        builder.normalize_capability(
            "goal-achievement"
        )
        == "goal_achievement"
    )


def test_normalize_capability_rejects_none():
    builder = UnifiedCapabilityEvidenceBuilder()

    with pytest.raises(ValueError):
        builder.normalize_capability(
            None
        )


def test_normalize_capability_rejects_empty():
    builder = UnifiedCapabilityEvidenceBuilder()

    with pytest.raises(ValueError):
        builder.normalize_capability(
            ""
        )


def test_empty_sources_produce_empty_result():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build()

    assert isinstance(
        result,
        UnifiedCapabilityEvidenceResult,
    )

    assert result.capabilities == {}
    assert result.capability_count == 0
    assert not result.has_evidence
    assert result.overlap_count == 0


def test_experience_evidence_is_preserved():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        experience_evidence={
            "planning": {
                "attempts": 5,
                "successes": 4,
                "failures": 1,
            }
        }
    )

    assert result.capabilities == {
        "planning": {
            "attempts": 5,
            "successes": 4,
            "failures": 1,
        }
    }

    assert (
        result.experience_capability_count
        == 1
    )


def test_telemetry_evidence_is_preserved():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 3,
                "successes": 2,
                "failures": 1,
            }
        }
    )

    assert result.capabilities == {
        "execution": {
            "attempts": 3,
            "successes": 2,
            "failures": 1,
        }
    }

    assert (
        result.telemetry_capability_count
        == 1
    )


def test_sources_are_combined_when_capabilities_are_distinct():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 3,
                "successes": 3,
                "failures": 0,
            }
        },
        experience_evidence={
            "planning": {
                "attempts": 5,
                "successes": 4,
                "failures": 1,
            }
        },
    )

    assert result.capabilities == {
        "planning": {
            "attempts": 5,
            "successes": 4,
            "failures": 1,
        },
        "execution": {
            "attempts": 3,
            "successes": 3,
            "failures": 0,
        },
    }

    assert result.capability_count == 2
    assert result.overlap_count == 0


def test_telemetry_wins_on_overlap():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 3,
                "successes": 1,
                "failures": 2,
            }
        },
        experience_evidence={
            "execution": {
                "attempts": 10,
                "successes": 10,
                "failures": 0,
            }
        },
    )

    assert result.capabilities[
        "execution"
    ] == {
        "attempts": 3,
        "successes": 1,
        "failures": 2,
    }

    assert result.overlap_count == 1
    assert result.has_overlap


def test_overlap_is_not_double_counted():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 2,
                "successes": 1,
                "failures": 1,
            }
        },
        experience_evidence={
            "execution": {
                "attempts": 100,
                "successes": 100,
                "failures": 0,
            }
        },
    )

    assert (
        result.capabilities[
            "execution"
        ]["attempts"]
        == 2
    )


def test_multiple_overlaps_are_counted():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 2,
                "successes": 2,
                "failures": 0,
            },
            "exploration": {
                "attempts": 3,
                "successes": 2,
                "failures": 1,
            },
        },
        experience_evidence={
            "execution": {
                "attempts": 10,
                "successes": 9,
                "failures": 1,
            },
            "exploration": {
                "attempts": 20,
                "successes": 10,
                "failures": 10,
            },
            "planning": {
                "attempts": 4,
                "successes": 3,
                "failures": 1,
            },
        },
    )

    assert result.overlap_count == 2
    assert result.capability_count == 3

    assert result.capabilities[
        "execution"
    ]["attempts"] == 2

    assert result.capabilities[
        "exploration"
    ]["attempts"] == 3

    assert result.capabilities[
        "planning"
    ]["attempts"] == 4


def test_capability_names_are_normalized():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "Decision Making": {
                "attempts": 2,
                "successes": 2,
                "failures": 0,
            }
        },
        experience_evidence={
            "decision-making": {
                "attempts": 9,
                "successes": 5,
                "failures": 4,
            }
        },
    )

    assert result.capability_count == 1
    assert result.overlap_count == 1

    assert (
        result.capabilities[
            "decision_making"
        ]["attempts"]
        == 2
    )


def test_records_are_copied():
    builder = UnifiedCapabilityEvidenceBuilder()

    source = {
        "execution": {
            "attempts": 2,
            "successes": 2,
            "failures": 0,
        }
    }

    result = builder.build(
        telemetry_evidence=source
    )

    result.capabilities[
        "execution"
    ]["attempts"] = 999

    assert (
        source[
            "execution"
        ]["attempts"]
        == 2
    )


def test_invalid_evidence_type_is_rejected():
    builder = UnifiedCapabilityEvidenceBuilder()

    with pytest.raises(TypeError):
        builder.build(
            telemetry_evidence=[]
        )


def test_invalid_record_type_is_rejected():
    builder = UnifiedCapabilityEvidenceBuilder()

    with pytest.raises(TypeError):
        builder.build(
            telemetry_evidence={
                "execution": []
            }
        )


def test_negative_attempts_are_rejected():
    builder = UnifiedCapabilityEvidenceBuilder()

    with pytest.raises(ValueError):
        builder.build(
            telemetry_evidence={
                "execution": {
                    "attempts": -1,
                    "successes": 0,
                    "failures": 0,
                }
            }
        )


def test_non_integer_successes_are_rejected():
    builder = UnifiedCapabilityEvidenceBuilder()

    with pytest.raises(TypeError):
        builder.build(
            telemetry_evidence={
                "execution": {
                    "attempts": 1,
                    "successes": 1.5,
                    "failures": 0,
                }
            }
        )


def test_telemetry_priority_flag_is_recorded():
    builder = UnifiedCapabilityEvidenceBuilder(
        telemetry_priority=True
    )

    result = builder.build()

    assert result.telemetry_priority is True


def test_experience_priority_can_be_selected():
    builder = UnifiedCapabilityEvidenceBuilder(
        telemetry_priority=False
    )

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 2,
                "successes": 1,
                "failures": 1,
            }
        },
        experience_evidence={
            "execution": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        },
    )

    assert result.capabilities[
        "execution"
    ] == {
        "attempts": 10,
        "successes": 8,
        "failures": 2,
    }


def test_result_reports_source_counts():
    builder = UnifiedCapabilityEvidenceBuilder()

    result = builder.build(
        telemetry_evidence={
            "execution": {
                "attempts": 2,
                "successes": 1,
                "failures": 1,
            },
            "exploration": {
                "attempts": 1,
                "successes": 1,
                "failures": 0,
            },
        },
        experience_evidence={
            "execution": {
                "attempts": 5,
                "successes": 5,
                "failures": 0,
            },
            "planning": {
                "attempts": 4,
                "successes": 3,
                "failures": 1,
            },
        },
    )

    assert (
        result.telemetry_capability_count
        == 2
    )

    assert (
        result.experience_capability_count
        == 2
    )

    assert result.overlap_count == 1
