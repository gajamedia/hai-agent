from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SPEC = importlib.util.spec_from_file_location(
    "step41_4_runner", ROOT / "scripts" / "step41_4_cross_domain_holdout.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.p0_order_fulfillment_provider import (
    OrderFulfillmentEnvironment,
    OrderFulfillmentP0ScenarioProvider,
)


def test_domain_is_distinct_from_room_and_observes_order_state():
    env = OrderFulfillmentEnvironment()
    assert env.observe() == {
        "stock_status": "expected",
        "order_status": "created",
        "picked": False,
        "packed": False,
        "dispatched": False,
    }
    assert env.execute("receive_stock")["success"] is True
    assert env.observe()["stock_status"] == "available"


def test_provider_has_30_unique_tasks_and_20_10_split():
    provider = OrderFulfillmentP0ScenarioProvider()
    tasks = tuple(provider.tasks())
    assert len(tasks) == 30
    assert len({task.task_id for task in tasks}) == 30
    assert sum(task.payload["split"] == "development" for task in tasks) == 20
    assert sum(task.payload["split"] == "holdout" for task in tasks) == 10
    assert all(task.payload["domain"] == "order_fulfillment" for task in tasks)


def test_holdout_uses_distinct_initial_state_and_action_composition():
    provider = OrderFulfillmentP0ScenarioProvider()
    tasks = {task.task_id: task for task in provider.tasks()}
    development = tasks["task-11"].payload
    holdout = tasks["task-21"].payload
    assert development["initial_stock_status"] == "expected"
    assert holdout["initial_stock_status"] == "available"
    assert development["route"] != holdout["route"]


def test_provider_builds_fresh_canonical_runtime_scenarios():
    provider = OrderFulfillmentP0ScenarioProvider()
    task = next(task for task in provider.tasks() if task.task_id == "task-21")
    scenario = provider.build_scenario(task, 1)
    assert scenario.environment.observe()["stock_status"] == "available"
    assert scenario.goal.target_state == {"order_status": "dispatched", "dispatched": True}
    assert [step.action for step in scenario.plan.steps] == [
        "pick_order", "pack_order", "dispatch_order"
    ]
    assert scenario.metadata["split"] == "holdout"


def test_split_report_separates_metrics_and_rejects_missing_rows(tmp_path):
    evidence = tmp_path / "evidence.jsonl"
    rows = []
    for number in range(1, 31):
        for run_index in (1, 2, 3):
            rows.append({
                "task_id": f"task-{number:02d}",
                "run_index": run_index,
                "correct_completion": True,
                "deterministic_verification_passed": True,
                "unauthorized_high_risk_actions": 0,
                "duplicate_harmful_side_effects": 0,
                "audit_evidence_complete": True,
                "required_approval_evidence_complete": True,
                "trace_integrity_passed": True,
            })
    evidence.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "split.json"
    report = runner._write_split_report(evidence, output, "contract-run")
    assert report["development"]["tasks"] == 20
    assert report["development"]["runs"] == 60
    assert report["holdout"]["tasks"] == 10
    assert report["holdout"]["runs"] == 30
    assert report["holdout"]["correct_completion_rate"] == 1.0
    assert json.loads(output.read_text(encoding="utf-8"))["step"] == "41.4"

    evidence.write_text("\n".join(json.dumps(row) for row in rows[:-1]), encoding="utf-8")
    try:
        runner._write_split_report(evidence, output, "bad-run")
    except ValueError as exc:
        assert "Expected 90 evidence rows" in str(exc)
    else:
        raise AssertionError("Incomplete evidence must be rejected")


def test_run_id_validation_blocks_path_traversal():
    assert runner._safe_run_id("run-001")
    assert not runner._safe_run_id("../run")
    assert not runner._safe_run_id("run/001")
