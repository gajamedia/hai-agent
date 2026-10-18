"""Step 41.8: bounded retry execution primitive built on Step 41.6 policy.

This is an explicit, opt-in callable executor. It does not patch or silently
replace the canonical runtime adapter. Callers must provide trusted failure
classification and side-effect verification callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import sleep
from typing import Any, Callable, Optional

from agent.llm.recovery_policy import (
    FailureCategory,
    FailureContext,
    RecoveryAction,
    decide_recovery,
)


@dataclass(frozen=True)
class RetryRuntimeConfig:
    max_attempts: int = 3
    base_delay_seconds: float = 0.0
    max_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise ValueError("base_delay_seconds must be <= max_delay_seconds")


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    success: bool
    action: str
    category: str
    reason: str
    error: Optional[str] = None


@dataclass
class RetryRuntimeResult:
    success: bool
    value: Any = None
    attempts: int = 0
    final_action: str = RecoveryAction.STOP.value
    reason: str = "not_started"
    history: list[AttemptRecord] = field(default_factory=list)


class BoundedRetryRuntime:
    """Run a callable only when policy authorizes a bounded idempotent retry.

    `classify_exception` must map an exception to a trusted FailureCategory.
    `state_changed` and `state_verified` must be based on trusted observations.
    A thrown exception is not automatically considered transient.
    """

    def __init__(self, config: RetryRuntimeConfig | None = None,
                 sleeper: Callable[[float], None] = sleep) -> None:
        self.config = config or RetryRuntimeConfig()
        self._sleep = sleeper

    def execute(
        self,
        operation: Callable[[], Any],
        *,
        idempotent: bool,
        classify_exception: Callable[[BaseException], FailureCategory],
        is_success: Callable[[Any], bool] = lambda value: value is not False,
        state_changed: Callable[[], bool] = lambda: False,
        state_verified: Callable[[], bool] = lambda: False,
        approval_required: bool = False,
        approval_granted: bool = False,
    ) -> RetryRuntimeResult:
        if not callable(operation) or not callable(classify_exception):
            raise TypeError("operation and classify_exception must be callable")
        result = RetryRuntimeResult(success=False)
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                value = operation()
                if is_success(value):
                    result.success = True
                    result.value = value
                    result.attempts = attempt
                    result.final_action = "success"
                    result.reason = "operation_succeeded"
                    result.history.append(AttemptRecord(attempt, True, "success", "none", result.reason))
                    return result
                error = RuntimeError("operation returned an unsuccessful result")
            except BaseException as exc:
                # Do not swallow process-control signals such as KeyboardInterrupt/SystemExit.
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                error = exc

            category = classify_exception(error)
            if not isinstance(category, FailureCategory):
                raise TypeError("classify_exception must return FailureCategory")
            changed = bool(state_changed())
            verified = bool(state_verified())
            decision = decide_recovery(FailureContext(
                category=category,
                attempt=attempt,
                max_attempts=self.config.max_attempts,
                idempotent=idempotent,
                state_changed=changed,
                state_verified=verified,
                recovery_available=False,
                approval_required=approval_required,
                approval_granted=approval_granted,
            ))
            result.attempts = attempt
            result.final_action = decision.action.value
            result.reason = decision.reason
            result.history.append(AttemptRecord(
                attempt=attempt,
                success=False,
                action=decision.action.value,
                category=category.value,
                reason=decision.reason,
                error=f"{type(error).__name__}: {error}",
            ))
            if decision.action != RecoveryAction.RETRY:
                return result
            # Policy returns RETRY only when another attempt is within the configured bound.
            delay = min(self.config.base_delay_seconds * (2 ** (attempt - 1)), self.config.max_delay_seconds)
            if delay:
                self._sleep(delay)
        return result
