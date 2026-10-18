from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent.llm.bounded_retry_runtime import RetryRuntimeConfig
from agent.llm.canonical_runtime_retry import execute_canonical_single_action_with_retry
from agent.llm.recovery_policy import FailureCategory

class Step: action = "safe_read"
class Plan: steps = (Step(),)
class Report:
    def __init__(self, success): self.success = success
class Env:
    def observe(self): return {"verified": True}
class Runtime:
    def __init__(self, results): self.results, self.calls = list(results), []
    def execute(self, plan, **kwargs):
        self.calls.append(kwargs["metadata"])
        item = self.results.pop(0)
        if isinstance(item, BaseException): raise item
        return Report(item)

def run_case(name, results, expected_calls, expected_success, **options):
    runtime = Runtime(results)
    args = dict(
        runtime=runtime, plan=Plan(), goal=None, world_state_factory=lambda: {}, environment=Env(),
        metadata={"task_id": name}, execution_id=f"step41_9-{name}", idempotent=True,
        classify_report=lambda report: FailureCategory.TRANSIENT,
        is_success_report=lambda report, state: bool(getattr(report, "success", False)) and state["verified"],
        state_changed=lambda: False, state_verified=lambda: True,
        config=RetryRuntimeConfig(max_attempts=3),
    )
    args.update(options)
    result = execute_canonical_single_action_with_retry(**args)
    passed = len(runtime.calls) == expected_calls and result.success is expected_success
    return {"case": name, "actual_calls": len(runtime.calls), "expected_calls": expected_calls,
            "success": result.success, "expected_success": expected_success,
            "final_action": result.final_action, "passed": passed,
            "attempt_metadata": [x.get("retry_attempt") for x in runtime.calls]}

def main():
    cases = [
        run_case("transient_then_success", [False, True], 2, True),
        run_case("retry_until_third_success", [False, False, True], 3, True),
        run_case("attempt_limit", [False, False, False], 3, False),
        run_case("non_idempotent", [False, True], 1, False, idempotent=False),
        run_case("permanent", [False, True], 1, False, classify_report=lambda r: FailureCategory.PERMANENT),
        run_case("unknown", [False, True], 1, False, classify_report=lambda r: FailureCategory.UNKNOWN),
        run_case("partial_effect_unverified", [False, True], 1, False, state_changed=lambda: True, state_verified=lambda: False),
        run_case("approval_missing", [False, True], 1, False, approval_required=True, approval_granted=False),
    ]
    report = {"step":"41.9", "kind":"canonical_single_action_retry_integration_contract",
              "scope_note":"Exercises an opt-in wrapper that calls the canonical runtime execute() per attempt. Uses a fake runtime contract; not a production adapter end-to-end proof.",
              "total_cases":len(cases), "passed_cases":sum(c["passed"] for c in cases),
              "failed_cases":[c["case"] for c in cases if not c["passed"]],
              "passed":all(c["passed"] for c in cases), "cases":cases}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "artifacts" / "step41_9" / f"canonical-retry-report-{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print(f"Report: {out}")
    return 0 if report["passed"] else 1
if __name__ == "__main__": raise SystemExit(main())
