from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from agent.llm.recovery_policy import FailureCategory, RecoveryAction
from agent.llm.bounded_retry_runtime import BoundedRetryRuntime, RetryRuntimeConfig


def test_transient_idempotent_retries_then_succeeds():
    calls = []
    def operation():
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError("temporary")
        return "ok"
    runtime = BoundedRetryRuntime(RetryRuntimeConfig(max_attempts=3), sleeper=lambda _: None)
    result = runtime.execute(operation, idempotent=True,
        classify_exception=lambda exc: FailureCategory.TRANSIENT)
    assert result.success and result.value == "ok"
    assert result.attempts == 3 and len(calls) == 3
    assert [x.action for x in result.history] == ["retry", "retry", "success"]


def test_non_idempotent_never_retries():
    calls = []
    def operation():
        calls.append(1)
        raise TimeoutError("temporary")
    result = BoundedRetryRuntime(sleeper=lambda _: None).execute(operation, idempotent=False,
        classify_exception=lambda exc: FailureCategory.TRANSIENT)
    assert not result.success and len(calls) == 1
    assert result.final_action == RecoveryAction.STOP.value


def test_permanent_failure_stops_immediately():
    calls = []
    def operation():
        calls.append(1)
        raise ValueError("bad input")
    result = BoundedRetryRuntime(sleeper=lambda _: None).execute(operation, idempotent=True,
        classify_exception=lambda exc: FailureCategory.PERMANENT)
    assert len(calls) == 1 and result.final_action == "stop"


def test_partial_unverified_side_effect_requires_manual_review():
    calls = []
    def operation():
        calls.append(1)
        raise TimeoutError("ambiguous after write")
    result = BoundedRetryRuntime(sleeper=lambda _: None).execute(operation, idempotent=True,
        classify_exception=lambda exc: FailureCategory.TRANSIENT,
        state_changed=lambda: True, state_verified=lambda: False)
    assert len(calls) == 1 and result.final_action == "manual_review"


def test_approval_missing_stops():
    calls = []
    def operation():
        calls.append(1)
        raise RuntimeError("failure")
    result = BoundedRetryRuntime(sleeper=lambda _: None).execute(operation, idempotent=True,
        classify_exception=lambda exc: FailureCategory.TRANSIENT,
        approval_required=True, approval_granted=False)
    assert len(calls) == 1 and result.final_action == "stop"


def test_retry_limit_is_enforced():
    calls = []
    def operation():
        calls.append(1)
        raise TimeoutError("temporary")
    result = BoundedRetryRuntime(RetryRuntimeConfig(max_attempts=2), sleeper=lambda _: None).execute(
        operation, idempotent=True, classify_exception=lambda exc: FailureCategory.TRANSIENT)
    assert len(calls) == 2 and result.attempts == 2
    assert result.final_action == "stop"
    assert result.history[-1].reason == "retry limit reached"


def test_unknown_failure_fails_closed():
    calls = []
    def operation():
        calls.append(1)
        raise RuntimeError("unclassified")
    result = BoundedRetryRuntime(sleeper=lambda _: None).execute(operation, idempotent=True,
        classify_exception=lambda exc: FailureCategory.UNKNOWN)
    assert len(calls) == 1 and result.final_action == "manual_review"


def test_backoff_is_bounded_and_deterministic():
    delays = []
    calls = []
    def operation():
        calls.append(1)
        if len(calls) < 4:
            raise TimeoutError("temporary")
        return True
    runtime = BoundedRetryRuntime(RetryRuntimeConfig(max_attempts=4, base_delay_seconds=0.25,
        max_delay_seconds=0.5), sleeper=delays.append)
    result = runtime.execute(operation, idempotent=True,
        classify_exception=lambda exc: FailureCategory.TRANSIENT)
    assert result.success and delays == [0.25, 0.5, 0.5]


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        RetryRuntimeConfig(max_attempts=0)
