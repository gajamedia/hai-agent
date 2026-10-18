from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent.llm.bounded_retry_runtime import BoundedRetryRuntime, RetryRuntimeConfig
from agent.llm.recovery_policy import FailureCategory


def run_case(name, operation_factory, kwargs, expected_calls, expected_action, expected_success=False):
    calls = []
    operation = operation_factory(calls)
    result = BoundedRetryRuntime(kwargs.pop("config", RetryRuntimeConfig(max_attempts=3)), sleeper=lambda _: None).execute(
        operation, idempotent=kwargs.pop("idempotent", True),
        classify_exception=kwargs.pop("classify_exception", lambda exc: FailureCategory.TRANSIENT), **kwargs)
    passed = len(calls) == expected_calls and result.final_action == expected_action and result.success == expected_success
    return {"case": name, "calls": len(calls), "expected_calls": expected_calls,
            "final_action": result.final_action, "expected_action": expected_action,
            "success": result.success, "passed": passed,
            "history": [r.__dict__ for r in result.history]}


def fail_twice_then_succeed(calls):
    def op():
        calls.append(1)
        if len(calls) <= 2: raise TimeoutError("temporary")
        return "ok"
    return op


def always_timeout(calls):
    def op(): calls.append(1); raise TimeoutError("temporary")
    return op


def permanent(calls):
    def op(): calls.append(1); raise ValueError("invalid")
    return op


def partial(calls):
    def op(): calls.append(1); raise TimeoutError("ambiguous")
    return op


def unknown(calls):
    def op(): calls.append(1); raise RuntimeError("unknown")
    return op


def success(calls):
    def op(): calls.append(1); return "ok"
    return op


def main():
    cases = [
        run_case("transient_retry_then_success", fail_twice_then_succeed, {}, 3, "success", True),
        run_case("non_idempotent_transient_stop", always_timeout, {"idempotent": False}, 1, "stop"),
        run_case("permanent_stop", permanent, {"classify_exception": lambda e: FailureCategory.PERMANENT}, 1, "stop"),
        run_case("unknown_manual_review", unknown, {"classify_exception": lambda e: FailureCategory.UNKNOWN}, 1, "manual_review"),
        run_case("partial_effect_review", partial, {"state_changed": lambda: True, "state_verified": lambda: False}, 1, "manual_review"),
        run_case("approval_missing_stop", always_timeout, {"approval_required": True}, 1, "stop"),
        run_case("two_attempt_limit", always_timeout, {"config": RetryRuntimeConfig(max_attempts=2)}, 2, "stop"),
        run_case("success_first_try", success, {}, 1, "success", True),
    ]
    report = {"step": "41.8", "kind": "bounded_retry_runtime_contract_benchmark",
        "scope_note": "Standalone bounded callable executor; caller supplies trusted classification and state evidence. Not yet wired into the canonical P0 adapter.",
        "total_cases": len(cases), "passed_cases": sum(c["passed"] for c in cases),
        "failed_cases": [c["case"] for c in cases if not c["passed"]],
        "passed": all(c["passed"] for c in cases), "cases": cases}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "artifacts" / "step41_8" / f"bounded-retry-report-{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {out}")
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
