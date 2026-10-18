"""Step 41.6: conservative recovery policy and failure classification.

This module is intentionally independent of the runtime adapter. It can be wired
into runtime execution only after its contracts are reviewed and tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureCategory(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    STATE_CONFLICT = "state_conflict"
    VERIFICATION_FAILURE = "verification_failure"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    RECOVER = "recover"
    STOP = "stop"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class FailureContext:
    category: FailureCategory
    attempt: int = 1
    max_attempts: int = 3
    idempotent: bool = False
    state_changed: bool = False
    state_verified: bool = False
    recovery_available: bool = False
    approval_required: bool = False
    approval_granted: bool = False


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    next_attempt: Optional[int] = None


def classify_failure(
    *,
    error_kind: str = "",
    retryable_hint: Optional[bool] = None,
    state_conflict: bool = False,
    verification_failed: bool = False,
) -> FailureCategory:
    """Classify using explicit evidence; ambiguous errors remain UNKNOWN."""
    if verification_failed:
        return FailureCategory.VERIFICATION_FAILURE
    if state_conflict:
        return FailureCategory.STATE_CONFLICT
    normalized = (error_kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"timeout", "temporary_unavailable", "connection_reset", "rate_limited", "transient"}:
        return FailureCategory.TRANSIENT
    if normalized in {"invalid_input", "permission_denied", "unsupported_operation", "permanent"}:
        return FailureCategory.PERMANENT
    if retryable_hint is True:
        return FailureCategory.TRANSIENT
    if retryable_hint is False:
        return FailureCategory.PERMANENT
    return FailureCategory.UNKNOWN


def decide_recovery(context: FailureContext) -> RecoveryDecision:
    """Return a conservative action; never retry uncertain/non-idempotent effects."""
    if context.attempt < 1 or context.max_attempts < 1:
        return RecoveryDecision(RecoveryAction.STOP, "attempt counters must be positive")
    if context.attempt > context.max_attempts:
        return RecoveryDecision(RecoveryAction.STOP, "retry limit already exceeded")
    if context.approval_required and not context.approval_granted:
        return RecoveryDecision(RecoveryAction.STOP, "required approval is missing")
    if context.category == FailureCategory.VERIFICATION_FAILURE:
        if context.state_changed:
            if context.state_verified:
                return RecoveryDecision(RecoveryAction.RECOVER, "state changed; use verified recovery path", None)
            return RecoveryDecision(RecoveryAction.MANUAL_REVIEW, "state changed but final state is unverified")
        return RecoveryDecision(RecoveryAction.RECOVER, "verification failed; reconcile state before further action", None)
    if context.category == FailureCategory.STATE_CONFLICT:
        if context.recovery_available and (not context.state_changed or context.state_verified):
            return RecoveryDecision(RecoveryAction.RECOVER, "state conflict requires explicit recovery plan")
        return RecoveryDecision(RecoveryAction.MANUAL_REVIEW, "state conflict has no safely verified recovery path")
    if context.category == FailureCategory.PERMANENT:
        return RecoveryDecision(RecoveryAction.STOP, "failure is classified as permanent")
    if context.category == FailureCategory.TRANSIENT:
        if not context.idempotent:
            return RecoveryDecision(RecoveryAction.STOP, "retry blocked because action is not known to be idempotent")
        if context.state_changed and not context.state_verified:
            return RecoveryDecision(RecoveryAction.MANUAL_REVIEW, "possible partial side effect must be reconciled")
        if context.attempt >= context.max_attempts:
            return RecoveryDecision(RecoveryAction.STOP, "retry limit reached")
        return RecoveryDecision(RecoveryAction.RETRY, "transient failure; bounded idempotent retry is safe", context.attempt + 1)
    return RecoveryDecision(RecoveryAction.MANUAL_REVIEW, "failure category is unknown; fail closed")
