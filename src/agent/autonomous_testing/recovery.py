"""Deterministic recovery decisions for autonomous test execution."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .contracts import FaultType, RecoveryPolicy


class RecoveryAction(str, Enum):
    RETRY = "retry"
    STOP = "stop"
    RECONCILE = "reconcile"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    attempt: int
    retries_used: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "attempt": self.attempt,
            "retries_used": self.retries_used,
            "reason": self.reason,
        }


class RecoveryController:
    """Decide whether a known failed attempt may be retried.

    max_attempts is the number of retries allowed after the initial attempt.
    Unknown outcomes always require reconciliation and can never be retried.
    """

    def decide(
        self,
        *,
        policy: RecoveryPolicy,
        fault_type: FaultType,
        attempt: int,
        outcome_known: bool,
        action_executed: Optional[bool],
    ) -> RecoveryDecision:
        if not isinstance(policy, RecoveryPolicy):
            raise TypeError("policy must be RecoveryPolicy")
        try:
            fault_type = FaultType(fault_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported fault_type") from exc
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if not isinstance(outcome_known, bool):
            raise TypeError("outcome_known must be bool")
        if action_executed not in (True, False, None):
            raise TypeError("action_executed must be bool or None")

        retries_used = attempt - 1

        if not outcome_known or action_executed is None:
            return RecoveryDecision(
                RecoveryAction.RECONCILE,
                attempt,
                retries_used,
                "Execution outcome is unknown; stop retries and reconcile state first.",
            )

        if action_executed:
            return RecoveryDecision(
                RecoveryAction.STOP,
                attempt,
                retries_used,
                "Action was executed; do not replay it as a recovery retry.",
            )

        if not policy.enabled:
            return RecoveryDecision(
                RecoveryAction.STOP,
                attempt,
                retries_used,
                "Recovery is disabled by policy.",
            )

        if fault_type not in policy.retryable_fault_types:
            return RecoveryDecision(
                RecoveryAction.STOP,
                attempt,
                retries_used,
                "Fault type is not permitted by the recovery policy.",
            )

        if retries_used >= policy.max_attempts:
            return RecoveryDecision(
                RecoveryAction.STOP,
                attempt,
                retries_used,
                "Maximum retry allowance has been exhausted.",
            )

        return RecoveryDecision(
            RecoveryAction.RETRY,
            attempt,
            retries_used,
            "Known non-execution and retryable fault; one controlled retry is allowed.",
        )


__all__ = ["RecoveryAction", "RecoveryDecision", "RecoveryController"]
