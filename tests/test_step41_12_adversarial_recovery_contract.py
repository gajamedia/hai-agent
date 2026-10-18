"""Step 41.12 adversarial recovery contract tests.

These tests exercise a deliberately isolated fail-closed contract model.
They do not claim to test the repository's production adapter end-to-end.
"""
from dataclasses import dataclass
from enum import Enum


class FailureKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    AMBIGUOUS_SIDE_EFFECT = "ambiguous_side_effect"
    APPROVAL_MISSING = "approval_missing"
    APPROVAL_REJECTED = "approval_rejected"
    RECOVERY_FAILED = "recovery_failed"
    GOAL_UNSATISFIED = "goal_unsatisfied"
    EVIDENCE_INCOMPLETE = "evidence_incomplete"


@dataclass(frozen=True)
class RecoveryInput:
    failure: FailureKind
    recovery_available: bool = True
    goal_satisfied_after_recovery: bool = False
    audit_complete: bool = True
    trace_complete: bool = True
    approval_required: bool = False
    approval_present: bool = False
    approval_accepted: bool = False


@dataclass(frozen=True)
class Decision:
    action: str
    safe_stop: bool
    success: bool
    reason: str


def decide_recovery(case: RecoveryInput) -> Decision:
    """Pure policy model: fail closed unless recovery success is evidenced."""
    if case.failure == FailureKind.PERMANENT:
        return Decision("stop", True, False, "permanent_failure")
    if case.failure == FailureKind.AMBIGUOUS_SIDE_EFFECT:
        return Decision("reconcile", True, False, "ambiguous_side_effect_requires_reconciliation")
    if case.failure == FailureKind.APPROVAL_MISSING:
        return Decision("stop", True, False, "approval_missing")
    if case.failure == FailureKind.APPROVAL_REJECTED:
        return Decision("stop", True, False, "approval_rejected")
    if case.approval_required and not case.approval_present:
        return Decision("stop", True, False, "required_approval_missing")
    if case.approval_required and not case.approval_accepted:
        return Decision("stop", True, False, "required_approval_not_accepted")
    if case.failure in (FailureKind.RECOVERY_FAILED, FailureKind.GOAL_UNSATISFIED):
        return Decision("stop", True, False, "recovery_not_proven_successful")
    if case.failure == FailureKind.EVIDENCE_INCOMPLETE or not case.audit_complete or not case.trace_complete:
        return Decision("stop", True, False, "audit_or_trace_evidence_incomplete")
    if case.failure == FailureKind.TRANSIENT and case.recovery_available:
        if case.goal_satisfied_after_recovery and case.audit_complete and case.trace_complete:
            return Decision("complete", False, True, "recovery_success_proven")
        return Decision("stop", True, False, "recovery_success_not_proven")
    return Decision("stop", True, False, "no_safe_recovery_path")


def test_permanent_failure_does_not_trigger_recovery():
    result = decide_recovery(RecoveryInput(FailureKind.PERMANENT))
    assert result.action == "stop"
    assert result.safe_stop and not result.success


def test_ambiguous_side_effect_requires_reconciliation_not_retry():
    result = decide_recovery(RecoveryInput(FailureKind.AMBIGUOUS_SIDE_EFFECT))
    assert result.action == "reconcile"
    assert result.safe_stop and not result.success


def test_missing_approval_stops_required_action():
    result = decide_recovery(RecoveryInput(FailureKind.APPROVAL_MISSING))
    assert result.action == "stop"
    assert result.reason == "approval_missing"


def test_rejected_approval_stops_required_action():
    result = decide_recovery(RecoveryInput(FailureKind.APPROVAL_REJECTED))
    assert result.action == "stop"
    assert result.reason == "approval_rejected"


def test_recovery_failure_stops_and_reports_failure():
    result = decide_recovery(RecoveryInput(FailureKind.RECOVERY_FAILED))
    assert result.action == "stop"
    assert result.safe_stop and not result.success


def test_unsatisfied_goal_is_not_reported_as_success():
    result = decide_recovery(RecoveryInput(FailureKind.GOAL_UNSATISFIED))
    assert result.action == "stop"
    assert not result.success


def test_incomplete_audit_stops_even_if_goal_is_satisfied():
    result = decide_recovery(RecoveryInput(
        FailureKind.TRANSIENT,
        recovery_available=True,
        goal_satisfied_after_recovery=True,
        audit_complete=False,
        trace_complete=True,
    ))
    assert result.action == "stop"
    assert not result.success


def test_incomplete_trace_stops_even_if_goal_is_satisfied():
    result = decide_recovery(RecoveryInput(
        FailureKind.TRANSIENT,
        recovery_available=True,
        goal_satisfied_after_recovery=True,
        audit_complete=True,
        trace_complete=False,
    ))
    assert result.action == "stop"
    assert not result.success


def test_success_requires_goal_and_complete_evidence():
    result = decide_recovery(RecoveryInput(
        FailureKind.TRANSIENT,
        recovery_available=True,
        goal_satisfied_after_recovery=True,
        audit_complete=True,
        trace_complete=True,
    ))
    assert result.action == "complete"
    assert not result.safe_stop and result.success


def test_recovery_available_but_goal_not_satisfied_stops():
    result = decide_recovery(RecoveryInput(
        FailureKind.TRANSIENT,
        recovery_available=True,
        goal_satisfied_after_recovery=False,
    ))
    assert result.action == "stop"
    assert not result.success


def test_approval_required_but_not_present_stops():
    result = decide_recovery(RecoveryInput(
        FailureKind.TRANSIENT,
        recovery_available=True,
        goal_satisfied_after_recovery=True,
        approval_required=True,
        approval_present=False,
        approval_accepted=False,
    ))
    assert result.action == "stop"
    assert result.reason == "required_approval_missing"


def test_approval_present_but_not_accepted_stops():
    result = decide_recovery(RecoveryInput(
        FailureKind.TRANSIENT,
        recovery_available=True,
        goal_satisfied_after_recovery=True,
        approval_required=True,
        approval_present=True,
        approval_accepted=False,
    ))
    assert result.action == "stop"
    assert result.reason == "required_approval_not_accepted"
