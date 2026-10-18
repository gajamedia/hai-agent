import pytest

from agent.llm.recovery_policy import (
    FailureCategory, FailureContext, RecoveryAction,
    classify_failure, decide_recovery,
)


@pytest.mark.parametrize(("kwargs", "expected"), [
    ({"error_kind": "timeout"}, FailureCategory.TRANSIENT),
    ({"error_kind": "connection-reset"}, FailureCategory.TRANSIENT),
    ({"error_kind": "invalid_input"}, FailureCategory.PERMANENT),
    ({"retryable_hint": True}, FailureCategory.TRANSIENT),
    ({"retryable_hint": False}, FailureCategory.PERMANENT),
    ({"state_conflict": True}, FailureCategory.STATE_CONFLICT),
    ({"verification_failed": True, "state_conflict": True}, FailureCategory.VERIFICATION_FAILURE),
    ({}, FailureCategory.UNKNOWN),
])
def test_failure_classification(kwargs, expected):
    assert classify_failure(**kwargs) == expected


def test_transient_idempotent_retries_with_bounded_attempt():
    decision = decide_recovery(FailureContext(FailureCategory.TRANSIENT, attempt=1, max_attempts=3, idempotent=True))
    assert decision.action == RecoveryAction.RETRY
    assert decision.next_attempt == 2


def test_transient_non_idempotent_is_stopped():
    decision = decide_recovery(FailureContext(FailureCategory.TRANSIENT, idempotent=False))
    assert decision.action == RecoveryAction.STOP


def test_retry_limit_stops():
    decision = decide_recovery(FailureContext(FailureCategory.TRANSIENT, attempt=3, max_attempts=3, idempotent=True))
    assert decision.action == RecoveryAction.STOP


def test_partial_side_effect_requires_manual_review():
    decision = decide_recovery(FailureContext(FailureCategory.TRANSIENT, idempotent=True, state_changed=True, state_verified=False))
    assert decision.action == RecoveryAction.MANUAL_REVIEW


def test_permanent_failure_stops():
    decision = decide_recovery(FailureContext(FailureCategory.PERMANENT))
    assert decision.action == RecoveryAction.STOP


def test_unknown_fails_closed():
    decision = decide_recovery(FailureContext(FailureCategory.UNKNOWN))
    assert decision.action == RecoveryAction.MANUAL_REVIEW


def test_state_conflict_requires_recovery_plan():
    unsafe = decide_recovery(FailureContext(FailureCategory.STATE_CONFLICT))
    safe = decide_recovery(FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True))
    assert unsafe.action == RecoveryAction.MANUAL_REVIEW
    assert safe.action == RecoveryAction.RECOVER


def test_verification_failure_after_unverified_change_requires_review():
    decision = decide_recovery(FailureContext(FailureCategory.VERIFICATION_FAILURE, state_changed=True, state_verified=False))
    assert decision.action == RecoveryAction.MANUAL_REVIEW


def test_missing_approval_stops_before_recovery():
    decision = decide_recovery(FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True, approval_required=True, approval_granted=False))
    assert decision.action == RecoveryAction.STOP


def test_invalid_attempt_counters_stop():
    assert decide_recovery(FailureContext(FailureCategory.TRANSIENT, attempt=0)).action == RecoveryAction.STOP
