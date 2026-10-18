"""Step 41.7 bridge: apply Step 41.6 recovery policy to canonical P0 scenarios.

The current P0CanonicalRuntimeAdapter supports explicit recovery plans but does
not implement retry loops. Therefore RETRY decisions are deliberately NOT
executed by this bridge; they fail closed and are exposed for caller inspection.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, MutableMapping

from agent.llm.recovery_policy import (
    FailureContext,
    RecoveryAction,
    RecoveryDecision,
    decide_recovery,
)


class RuntimeRecoveryPolicyIntegrationError(ValueError):
    """Raised for invalid scenario or context-factory contracts."""


def bind_recovery_policy(
    scenario: Any,
    context_factory: Callable[[Any, Any, Any], FailureContext],
    *,
    decision_sink: MutableMapping[str, Any] | None = None,
) -> Any:
    """Return a dataclass-replaced P0RuntimeScenario governed by recovery policy.

    `context_factory(scenario, report, final_state)` must use trusted runtime
    evidence to produce a FailureContext. Recovery is allowed only when the
    policy chooses RECOVER and an explicit original recovery plan factory exists.
    Existing recoverability checks remain mandatory when present.
    """
    if not callable(context_factory):
        raise RuntimeRecoveryPolicyIntegrationError("context_factory must be callable")
    if getattr(scenario, "recovery_plan_factory", None) is None:
        raise RuntimeRecoveryPolicyIntegrationError(
            "scenario must provide an explicit recovery_plan_factory"
        )

    original_recoverable = getattr(scenario, "is_recoverable_failure", None)
    original_plan_factory = scenario.recovery_plan_factory
    last_decision: dict[str, RecoveryDecision | None] = {"decision": None}

    def policy_recoverable(report: Any, final_state: Any) -> bool:
        if original_recoverable is not None and not bool(original_recoverable(report, final_state)):
            last_decision["decision"] = RecoveryDecision(
                RecoveryAction.STOP,
                "existing scenario recoverability guard rejected recovery",
            )
            if decision_sink is not None:
                decision_sink.update(last_decision)
            return False

        context = context_factory(scenario, report, final_state)
        if not isinstance(context, FailureContext):
            raise RuntimeRecoveryPolicyIntegrationError(
                "context_factory must return FailureContext"
            )
        decision = decide_recovery(context)
        last_decision["decision"] = decision
        if decision_sink is not None:
            decision_sink.update(last_decision)
        # Adapter has no retry-loop contract. Never misrepresent RETRY as RECOVER.
        return decision.action == RecoveryAction.RECOVER

    def policy_plan_factory(current_scenario: Any, report: Any, final_state: Any) -> Any:
        decision = last_decision["decision"]
        if decision is None or decision.action != RecoveryAction.RECOVER:
            raise RuntimeRecoveryPolicyIntegrationError(
                "recovery plan requested without an affirmative RECOVER decision"
            )
        plan = original_plan_factory(current_scenario, report, final_state)
        if plan is None:
            raise RuntimeRecoveryPolicyIntegrationError(
                "original recovery_plan_factory returned None"
            )
        return plan

    metadata = dict(getattr(scenario, "metadata", None) or {})
    metadata["recovery_policy_integration"] = "step41_7"
    metadata["retry_execution_supported_by_adapter"] = False
    return replace(
        scenario,
        is_recoverable_failure=policy_recoverable,
        recovery_plan_factory=policy_plan_factory,
        metadata=metadata,
    )
