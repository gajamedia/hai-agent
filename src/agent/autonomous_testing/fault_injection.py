"""Test-only fault injection adapter.

Faults are applied around the isolated environment's public run() boundary.
This adapter never registers capabilities or calls action handlers directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .contracts import FaultInjectionPolicy, FaultType


@dataclass(frozen=True)
class InjectedFaultRecord:
    fault_type: FaultType
    target_action: str
    occurrence: int
    outcome_known: bool
    action_executed: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type.value,
            "target_action": self.target_action,
            "occurrence": self.occurrence,
            "outcome_known": self.outcome_known,
            "action_executed": self.action_executed,
            "message": self.message,
        }


class UnknownExecutionOutcome(RuntimeError):
    """Raised when the action may have executed but its observation was lost."""

    def __init__(self, record: InjectedFaultRecord):
        super().__init__(record.message)
        self.record = record
        self.outcome_known = False
        self.action_executed = None


class FaultInjectionAdapter:
    """Wrap an isolated environment and inject deterministic, test-only faults."""

    def __init__(self, environment: Any, policy: Optional[FaultInjectionPolicy] = None):
        if environment is None or not callable(getattr(environment, "run", None)):
            raise TypeError("environment must expose callable run()")
        self.environment = environment
        self.policy = policy or FaultInjectionPolicy(enabled=False)
        if not isinstance(self.policy, FaultInjectionPolicy):
            raise TypeError("policy must be FaultInjectionPolicy")
        if self.policy.enabled and self.policy.trigger_step not in (None, 1):
            raise ValueError(
                "This adapter supports only trigger_step=1; multi-step injection "
                "requires a separately approved canonical dispatch boundary."
            )
        self._occurrences = 0
        self._records: list[InjectedFaultRecord] = []

    @property
    def records(self) -> tuple[InjectedFaultRecord, ...]:
        return tuple(self._records)

    @property
    def occurrence_count(self) -> int:
        return self._occurrences

    def run(self, **kwargs: Any) -> dict[str, Any]:
        actions = kwargs.get("actions")
        if not isinstance(actions, list) or len(actions) != 1:
            raise ValueError(
                "Fault injection currently supports exactly one action per run call."
            )

        action = actions[0]
        policy = self.policy
        should_inject = (
            policy.enabled
            and policy.fault_type is not None
            and (policy.target_action is None or policy.target_action == action)
            and self._occurrences < policy.max_occurrences
        )
        if not should_inject:
            return self.environment.run(**kwargs)

        self._occurrences += 1
        fault_type = policy.fault_type

        if fault_type in (FaultType.ACTION_FAILURE, FaultType.TEMPORARY_UNAVAILABILITY):
            # Inject before dispatch: the adapter can prove no action was sent
            # to the underlying runtime, so this is a known non-execution.
            message = (
                "Injected action failure before dispatch."
                if fault_type == FaultType.ACTION_FAILURE
                else "Injected temporary unavailability before dispatch."
            )
            record = InjectedFaultRecord(
                fault_type=fault_type,
                target_action=action,
                occurrence=self._occurrences,
                outcome_known=True,
                action_executed=False,
                message=message,
            )
            self._records.append(record)
            return {
                "status": "injected_fault",
                "success": False,
                "action_executed": False,
                "outcome_known": True,
                "fault": record.to_dict(),
            }

        if fault_type == FaultType.OBSERVATION_INTERRUPTION:
            # Dispatch once through the real runtime. Losing observation after
            # dispatch means the action may have happened: callers MUST reconcile.
            result = self.environment.run(**kwargs)
            record = InjectedFaultRecord(
                fault_type=fault_type,
                target_action=action,
                occurrence=self._occurrences,
                outcome_known=False,
                action_executed=None,
                message=(
                    "Injected observation interruption after dispatch; "
                    "execution outcome requires reconciliation."
                ),
            )
            self._records.append(record)
            raise UnknownExecutionOutcome(record) from None

        raise ValueError(f"Unsupported fault type: {fault_type!r}")


__all__ = [
    "FaultInjectionAdapter",
    "InjectedFaultRecord",
    "UnknownExecutionOutcome",
]
