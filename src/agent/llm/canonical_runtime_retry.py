"""Step 41.9: opt-in bounded retry wrapper for canonical single-action plans.

This helper invokes the existing LLMPlanRuntimeExecutor for every attempt. It
never retries multi-step plans, and it requires caller-supplied trusted
classification and deterministic success/state-verification callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from agent.llm.bounded_retry_runtime import (
    BoundedRetryRuntime,
    RetryRuntimeConfig,
    RetryRuntimeResult,
)
from agent.llm.recovery_policy import FailureCategory


class CanonicalRetryIntegrationError(ValueError):
    """Raised when retry cannot safely be bound to a canonical runtime call."""


@dataclass(frozen=True)
class CanonicalRetryAttemptContext:
    execution_id: str
    attempt: int
    action_name: str
    metadata: Mapping[str, Any]


class _UnsuccessfulRuntimeReport(RuntimeError):
    def __init__(self, report: Any) -> None:
        self.report = report
        super().__init__("canonical runtime returned an unsuccessful report")


def execute_canonical_single_action_with_retry(
    runtime: Any,
    plan: Any,
    *,
    goal: Any,
    world_state_factory: Callable[[], Any],
    environment: Any,
    metadata: Mapping[str, Any],
    execution_id: str,
    idempotent: bool,
    classify_report: Callable[[Any], FailureCategory],
    is_success_report: Callable[[Any, Any], bool],
    state_changed: Callable[[], bool],
    state_verified: Callable[[], bool],
    config: RetryRuntimeConfig | None = None,
    approval_required: bool = False,
    approval_granted: bool = False,
    retry_runtime: BoundedRetryRuntime | None = None,
) -> RetryRuntimeResult:
    """Execute one canonical plan step with bounded policy-authorized retries.

    The `is_success_report(report, current_state)` callback must verify the
    actual desired outcome, not merely an LLM/runtime success flag. The
    classifier and state callbacks must be backed by trusted application data.
    `metadata` is copied and augmented per attempt; it is trace context only.
    """
    if runtime is None or not callable(getattr(runtime, "execute", None)):
        raise CanonicalRetryIntegrationError("runtime must expose execute().")
    if not callable(world_state_factory) or not callable(classify_report):
        raise CanonicalRetryIntegrationError("world_state_factory and classify_report must be callable.")
    if not callable(is_success_report) or not callable(state_changed) or not callable(state_verified):
        raise CanonicalRetryIntegrationError("success and state evidence callbacks must be callable.")
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise CanonicalRetryIntegrationError("execution_id must be a non-empty string.")
    steps = getattr(plan, "steps", None)
    if steps is None:
        raise CanonicalRetryIntegrationError("plan must expose steps.")
    try:
        steps = tuple(steps)
    except TypeError as exc:
        raise CanonicalRetryIntegrationError("plan.steps must be iterable.") from exc
    if len(steps) != 1:
        raise CanonicalRetryIntegrationError(
            "retry integration accepts exactly one action step; multi-step plans are rejected."
        )
    action_name = getattr(steps[0], "action", None)
    if not isinstance(action_name, str) or not action_name.strip():
        raise CanonicalRetryIntegrationError("single plan step must expose a non-empty action name.")

    base_metadata = dict(metadata)
    latest_report: dict[str, Any] = {"report": None}

    def operation() -> Any:
        # attempt count is set by the retry runtime before calling this operation.
        attempt = len(retry_history)
        attempt_metadata = {
            **base_metadata,
            "execution_id": execution_id,
            "retry_attempt": attempt,
            "retry_max_attempts": effective_config.max_attempts,
            "retry_integration": "step41_9",
            "retry_action_name": action_name,
        }
        report = runtime.execute(
            plan,
            goal=goal,
            world_state=world_state_factory(),
            environment=environment,
            metadata=attempt_metadata,
        )
        latest_report["report"] = report
        current_state = environment.observe() if callable(getattr(environment, "observe", None)) else None
        if bool(is_success_report(report, current_state)):
            return report
        raise _UnsuccessfulRuntimeReport(report)

    def classify(error: BaseException) -> FailureCategory:
        if isinstance(error, _UnsuccessfulRuntimeReport):
            category = classify_report(error.report)
        else:
            # Runtime exceptions are ambiguous unless the caller explicitly
            # classifies them through the same trusted report classifier.
            category = classify_report({"exception": error, "report": latest_report["report"]})
        if not isinstance(category, FailureCategory):
            raise TypeError("classify_report must return FailureCategory")
        return category

    effective_config = config or RetryRuntimeConfig()
    retry_history: list[int] = []
    # Count attempts at operation entry, not on completion, so exceptions also
    # advance the attempt number. This local list is trace context only.
    original_operation = operation

    def counted_operation() -> Any:
        retry_history.append(len(retry_history) + 1)
        return original_operation()

    executor = retry_runtime or BoundedRetryRuntime(effective_config)
    return executor.execute(
        counted_operation,
        idempotent=idempotent,
        classify_exception=classify,
        is_success=lambda report: bool(is_success_report(
            report,
            environment.observe() if callable(getattr(environment, "observe", None)) else None,
        )),
        state_changed=state_changed,
        state_verified=state_verified,
        approval_required=approval_required,
        approval_granted=approval_granted,
    )
