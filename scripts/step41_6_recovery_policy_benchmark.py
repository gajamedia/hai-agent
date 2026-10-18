"""Deterministic policy contract matrix; not an integrated runtime benchmark."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.llm.recovery_policy import (  # noqa: E402
    FailureCategory, FailureContext, RecoveryAction, classify_failure, decide_recovery,
)

CASES = [
    ("transient_idempotent_retry", FailureContext(FailureCategory.TRANSIENT, idempotent=True), RecoveryAction.RETRY),
    ("transient_non_idempotent_stop", FailureContext(FailureCategory.TRANSIENT), RecoveryAction.STOP),
    ("permanent_stop", FailureContext(FailureCategory.PERMANENT), RecoveryAction.STOP),
    ("unknown_manual_review", FailureContext(FailureCategory.UNKNOWN), RecoveryAction.MANUAL_REVIEW),
    ("state_conflict_manual_review", FailureContext(FailureCategory.STATE_CONFLICT), RecoveryAction.MANUAL_REVIEW),
    ("state_conflict_recovery", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True), RecoveryAction.RECOVER),
    ("verification_partial_side_effect_review", FailureContext(FailureCategory.VERIFICATION_FAILURE, state_changed=True), RecoveryAction.MANUAL_REVIEW),
    ("retry_limit_stop", FailureContext(FailureCategory.TRANSIENT, attempt=3, max_attempts=3, idempotent=True), RecoveryAction.STOP),
    ("approval_missing_stop", FailureContext(FailureCategory.STATE_CONFLICT, recovery_available=True, approval_required=True), RecoveryAction.STOP),
    ("partial_effect_review", FailureContext(FailureCategory.TRANSIENT, idempotent=True, state_changed=True), RecoveryAction.MANUAL_REVIEW),
]


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = ROOT / "artifacts" / "step41_6"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, context, expected in CASES:
        decision = decide_recovery(context)
        rows.append({
            "case": name, "category": context.category.value,
            "expected_action": expected.value, "actual_action": decision.action.value,
            "passed": decision.action == expected, "reason": decision.reason,
            "next_attempt": decision.next_attempt,
        })
    classification_cases = [
        ("timeout", classify_failure(error_kind="timeout"), FailureCategory.TRANSIENT),
        ("invalid_input", classify_failure(error_kind="invalid_input"), FailureCategory.PERMANENT),
        ("state_conflict", classify_failure(state_conflict=True), FailureCategory.STATE_CONFLICT),
        ("verification_failed", classify_failure(verification_failed=True), FailureCategory.VERIFICATION_FAILURE),
        ("ambiguous", classify_failure(error_kind="mystery"), FailureCategory.UNKNOWN),
    ]
    for name, actual, expected in classification_cases:
        rows.append({"case": "classify_" + name, "expected_category": expected.value,
                     "actual_category": actual.value, "passed": actual == expected})
    report = {
        "step": "41.6", "kind": "recovery_policy_contract_matrix",
        "scope_note": "Pure policy contract matrix only; not integrated into canonical runtime or proof of production retry safety.",
        "total_cases": len(rows), "passed_cases": sum(row["passed"] for row in rows),
        "failed_cases": [row for row in rows if not row["passed"]],
        "passed": all(row["passed"] for row in rows), "cases": rows,
    }
    path = outdir / f"policy-report-{stamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {path.relative_to(ROOT)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
