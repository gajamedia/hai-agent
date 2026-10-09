import pytest

from agent.learning.capability_telemetry import (
    CapabilityTelemetry,
    CapabilityTelemetryEvent,
    CapabilityTelemetrySnapshot,
)


def test_normalize_capability():
    telemetry = CapabilityTelemetry()

    assert (
        telemetry.normalize_capability(
            "Decision Making"
        )
        == "decision_making"
    )


def test_normalize_capability_handles_hyphen():
    telemetry = CapabilityTelemetry()

    assert (
        telemetry.normalize_capability(
            "goal-achievement"
        )
        == "goal_achievement"
    )


def test_normalize_capability_rejects_none():
    telemetry = CapabilityTelemetry()

    with pytest.raises(ValueError):
        telemetry.normalize_capability(
            None
        )


def test_normalize_capability_rejects_empty():
    telemetry = CapabilityTelemetry()

    with pytest.raises(ValueError):
        telemetry.normalize_capability(
            ""
        )


def test_record_success():
    telemetry = CapabilityTelemetry()

    event = telemetry.record(
        "execution",
        True,
        source="decide_and_act",
    )

    assert isinstance(
        event,
        CapabilityTelemetryEvent,
    )

    assert event.capability == "execution"
    assert event.success is True
    assert event.source == "decide_and_act"
    assert event.event_id == 1

    assert telemetry.get(
        "execution"
    ) == {
        "attempts": 1,
        "successes": 1,
        "failures": 0,
    }


def test_record_failure():
    telemetry = CapabilityTelemetry()

    telemetry.record(
        "execution",
        False,
    )

    assert telemetry.get(
        "execution"
    ) == {
        "attempts": 1,
        "successes": 0,
        "failures": 1,
    }


def test_record_attempt_without_outcome():
    telemetry = CapabilityTelemetry()

    event = telemetry.record_attempt(
        "planning"
    )

    assert event.success is None

    assert telemetry.get(
        "planning"
    ) == {
        "attempts": 1,
        "successes": 0,
        "failures": 0,
    }


def test_record_success_helper():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "execution"
    )

    assert telemetry.count(
        "execution"
    ) == 1

    assert telemetry.total_successes() == 1


def test_record_failure_helper():
    telemetry = CapabilityTelemetry()

    telemetry.record_failure(
        "execution"
    )

    assert telemetry.count(
        "execution"
    ) == 1

    assert telemetry.total_failures() == 1


def test_multiple_events_are_aggregated():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "execution"
    )
    telemetry.record_failure(
        "execution"
    )
    telemetry.record_success(
        "execution"
    )

    assert telemetry.get(
        "execution"
    ) == {
        "attempts": 3,
        "successes": 2,
        "failures": 1,
    }


def test_multiple_capabilities_are_independent():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "planning"
    )
    telemetry.record_failure(
        "execution"
    )

    assert telemetry.evidence() == {
        "planning": {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        },
        "execution": {
            "attempts": 1,
            "successes": 0,
            "failures": 1,
        },
    }


def test_evidence_returns_copy():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "execution"
    )

    evidence = telemetry.evidence()

    evidence["execution"]["attempts"] = 999

    assert (
        telemetry.count(
            "execution"
        )
        == 1
    )


def test_get_returns_copy():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "execution"
    )

    evidence = telemetry.get(
        "execution"
    )

    assert evidence is not None

    evidence["attempts"] = 999

    assert (
        telemetry.count(
            "execution"
        )
        == 1
    )


def test_unknown_capability_count_is_zero():
    telemetry = CapabilityTelemetry()

    assert (
        telemetry.count(
            "unknown"
        )
        == 0
    )


def test_events_are_recorded_in_order():
    telemetry = CapabilityTelemetry()

    first = telemetry.record_success(
        "planning"
    )

    second = telemetry.record_failure(
        "execution"
    )

    third = telemetry.record_success(
        "goal_achievement"
    )

    assert [
        event.event_id
        for event in telemetry.events()
    ] == [
        first.event_id,
        second.event_id,
        third.event_id,
    ]

    assert (
        telemetry.event_count()
        == 3
    )


def test_snapshot_contains_aggregate_totals():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "planning"
    )

    telemetry.record_success(
        "execution"
    )

    telemetry.record_failure(
        "execution"
    )

    snapshot = telemetry.snapshot()

    assert isinstance(
        snapshot,
        CapabilityTelemetrySnapshot,
    )

    assert snapshot.capability_count == 2
    assert snapshot.event_count == 3
    assert snapshot.total_attempts == 3
    assert snapshot.total_successes == 2
    assert snapshot.total_failures == 1

    assert snapshot.capabilities == {
        "planning": {
            "attempts": 1,
            "successes": 1,
            "failures": 0,
        },
        "execution": {
            "attempts": 2,
            "successes": 1,
            "failures": 1,
        },
    }


def test_contains_normalized_capability():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "Decision Making"
    )

    assert (
        "decision_making"
        in telemetry
    )


def test_clear_resets_all_state():
    telemetry = CapabilityTelemetry()

    telemetry.record_success(
        "execution"
    )
    telemetry.record_failure(
        "planning"
    )

    telemetry.clear()

    assert telemetry.event_count() == 0
    assert telemetry.capability_count() == 0
    assert telemetry.total_attempts() == 0
    assert telemetry.total_successes() == 0
    assert telemetry.total_failures() == 0
    assert telemetry.evidence() == {}

    event = telemetry.record_success(
        "execution"
    )

    assert event.event_id == 1


def test_invalid_success_type_is_rejected():
    telemetry = CapabilityTelemetry()

    with pytest.raises(TypeError):
        telemetry.record(
            "execution",
            "true",
        )


def test_invalid_source_is_rejected():
    telemetry = CapabilityTelemetry()

    with pytest.raises(ValueError):
        telemetry.record(
            "execution",
            True,
            source="",
        )
