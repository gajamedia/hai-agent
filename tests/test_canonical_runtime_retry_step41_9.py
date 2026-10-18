from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
from agent.llm.bounded_retry_runtime import RetryRuntimeConfig
from agent.llm.canonical_runtime_retry import (
    CanonicalRetryIntegrationError,
    execute_canonical_single_action_with_retry,
)
from agent.llm.recovery_policy import FailureCategory


class Step:
    action = "safe_read"


class Plan:
    steps = (Step(),)


class MultiPlan:
    steps = (Step(), Step())


class Report:
    def __init__(self, success):
        self.success = success


class Env:
    def observe(self):
        return {"verified": True}


class Runtime:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, plan, **kwargs):
        self.calls.append(kwargs["metadata"])
        value = self.results.pop(0)
        if isinstance(value, BaseException):
            raise value
        return Report(value)


def call(runtime, **overrides):
    args = dict(
        runtime=runtime, plan=Plan(), goal=object(), world_state_factory=lambda: {"fresh": True},
        environment=Env(), metadata={"task_id": "task-a"}, execution_id="exec-a",
        idempotent=True,
        classify_report=lambda report: FailureCategory.TRANSIENT,
        is_success_report=lambda report, state: bool(getattr(report, "success", False)) and state["verified"],
        state_changed=lambda: False, state_verified=lambda: True,
        config=RetryRuntimeConfig(max_attempts=3),
    )
    args.update(overrides)
    return execute_canonical_single_action_with_retry(**args)


def test_canonical_runtime_retries_transient_single_action_then_succeeds():
    runtime = Runtime([False, False, True])
    result = call(runtime)
    assert result.success and result.attempts == 3
    assert len(runtime.calls) == 3
    assert [x["retry_attempt"] for x in runtime.calls] == [1, 2, 3]
    assert all(x["execution_id"] == "exec-a" for x in runtime.calls)


def test_canonical_runtime_stops_at_attempt_limit():
    runtime = Runtime([False, False, False])
    result = call(runtime, config=RetryRuntimeConfig(max_attempts=2))
    assert not result.success and result.attempts == 2 and len(runtime.calls) == 2


def test_non_idempotent_action_never_retries():
    runtime = Runtime([False, True])
    result = call(runtime, idempotent=False)
    assert not result.success and len(runtime.calls) == 1


def test_unknown_failure_does_not_retry():
    runtime = Runtime([False, True])
    result = call(runtime, classify_report=lambda report: FailureCategory.UNKNOWN)
    assert not result.success and len(runtime.calls) == 1
    assert result.final_action == "manual_review"


def test_partial_unverified_state_does_not_retry():
    runtime = Runtime([False, True])
    result = call(runtime, state_changed=lambda: True, state_verified=lambda: False)
    assert not result.success and len(runtime.calls) == 1
    assert result.final_action == "manual_review"


def test_permanent_failure_does_not_retry():
    runtime = Runtime([False, True])
    result = call(runtime, classify_report=lambda report: FailureCategory.PERMANENT)
    assert not result.success and len(runtime.calls) == 1


def test_missing_approval_does_not_retry():
    runtime = Runtime([False, True])
    result = call(runtime, approval_required=True, approval_granted=False)
    assert not result.success and len(runtime.calls) == 1


def test_multistep_plan_is_rejected_before_execution():
    runtime = Runtime([True])
    with pytest.raises(CanonicalRetryIntegrationError, match="exactly one action step"):
        call(runtime, plan=MultiPlan())
    assert not runtime.calls


def test_trace_metadata_marks_each_attempt():
    runtime = Runtime([False, True])
    result = call(runtime)
    assert result.success
    assert runtime.calls[0]["retry_integration"] == "step41_9"
    assert runtime.calls[1]["retry_attempt"] == 2
    assert runtime.calls[1]["retry_max_attempts"] == 3


def test_invalid_classifier_return_is_rejected():
    runtime = Runtime([False])
    with pytest.raises(TypeError, match="must return FailureCategory"):
        call(runtime, classify_report=lambda report: "transient")
