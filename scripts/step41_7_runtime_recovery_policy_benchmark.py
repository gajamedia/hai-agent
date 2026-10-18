"""Deterministic integration-contract benchmark for Step 41.7 (no production claims)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from agent.llm.recovery_policy import FailureCategory, FailureContext, RecoveryAction
from agent.llm.runtime_recovery_policy import bind_recovery_policy


@dataclass(frozen=True)
class Scenario:
    is_recoverable_failure: object = lambda report, state: True
    recovery_plan_factory: object = lambda scenario, report, state: {"plan": "explicit"}
    metadata: object = None


def main() -> int:
    cases = [
        ("state_conflict_with_recovery", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True), False, RecoveryAction.RECOVER),
        ("state_conflict_without_recovery", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=False), False, RecoveryAction.MANUAL_REVIEW),
        ("transient_idempotent_retry", FailureContext(FailureCategory.TRANSIENT, idempotent=True), False, RecoveryAction.RETRY),
        ("transient_non_idempotent_stop", FailureContext(FailureCategory.TRANSIENT, idempotent=False), False, RecoveryAction.STOP),
        ("permanent_stop", FailureContext(FailureCategory.PERMANENT), False, RecoveryAction.STOP),
        ("unknown_manual_review", FailureContext(FailureCategory.UNKNOWN), False, RecoveryAction.MANUAL_REVIEW),
        ("approval_missing_stop", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True, approval_required=True), False, RecoveryAction.STOP),
        ("partial_effect_manual_review", FailureContext(FailureCategory.TRANSIENT, idempotent=True, state_changed=True, state_verified=False), False, RecoveryAction.MANUAL_REVIEW),
        ("retry_limit_stop", FailureContext(FailureCategory.TRANSIENT, idempotent=True, attempt=3, max_attempts=3), False, RecoveryAction.STOP),
        ("verified_verification_recovery", FailureContext(FailureCategory.VERIFICATION_FAILURE, state_changed=True, state_verified=True, recovery_available=True), False, RecoveryAction.RECOVER),
        ("existing_guard_rejects", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True), True, RecoveryAction.STOP),
        ("approval_granted_state_recovery", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True, approval_required=True, approval_granted=True), False, RecoveryAction.RECOVER),
    ]
    results = []
    for name, context, reject_guard, expected in cases:
        scenario = Scenario(is_recoverable_failure=(lambda r, s: False) if reject_guard else (lambda r, s: True))
        decisions = {}
        bound = bind_recovery_policy(scenario, lambda sc, report, state, c=context: c, decision_sink=decisions)
        allowed = bound.is_recoverable_failure({"success": False}, {})
        actual = decisions["decision"].action
        expected_allowed = expected == RecoveryAction.RECOVER and not reject_guard
        passed = actual == expected and allowed == expected_allowed
        results.append({"case": name, "expected_action": expected.value, "actual_action": actual.value,
                        "expected_recovery_allowed": expected_allowed, "actual_recovery_allowed": allowed,
                        "passed": passed, "reason": decisions["decision"].reason})
    report = {
        "step": "41.7",
        "kind": "runtime_recovery_policy_integration_contract",
        "scope_note": "Tests policy binding to a scenario contract. RETRY is intentionally not executed because the current canonical adapter has no retry-loop contract; this is not production runtime proof.",
        "total_cases": len(results),
        "passed_cases": sum(r["passed"] for r in results),
        "failed_cases": [r for r in results if not r["passed"]],
        "passed": all(r["passed"] for r in results),
        "cases": results,
    }
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "artifacts" / "step41_7" / f"integration-report-{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
