from __future__ import annotations

import json
from pathlib import Path

from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.p0_canonical_runtime_adapter import P0CanonicalRuntimeAdapter
from agent.llm.p0_recovery_replay_provider import build_provider


def test_recoverable_failure_runs_explicit_recovery_and_reaches_goal(tmp_path):
    provider = build_provider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "runs")

    evidence = adapter.execute(provider.tasks()[0], 1)

    assert evidence.recoverable_failure is True
    assert evidence.recovery_succeeded is True
    assert evidence.correct_completion is True
    assert evidence.deterministic_verification_passed is True
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.duplicate_harmful_side_effects == 0


def test_recovery_execution_is_present_in_verified_trace(tmp_path):
    provider = build_provider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "runs")
    evidence = adapter.execute(provider.tasks()[0], 1)

    trace_paths = list((tmp_path / "runs").glob("task-01/run-01/attempt-*/execution_trace.jsonl"))
    assert len(trace_paths) == 1
    store = ExecutionTraceStore(trace_paths[0])
    verification = store.verify()
    assert verification["valid"] is True

    execution_id = evidence.notes.split("execution_id=", 1)[1].split(";", 1)[0]
    rows = store.replay(execution_id)
    assert rows
    # Replay is scoped to the one execution and includes records for its recovery
    # generation, rather than treating recovery as an unrelated benchmark run.
    assert any("generation" in json.dumps(row, sort_keys=True) for row in rows)


def test_provider_injects_only_a_pre_mutation_failure():
    provider = build_provider()
    task = provider.tasks()[0]
    scenario = provider.build_scenario(task, 1)

    assert scenario.environment.observe() == {"box": "closed"}
    assert callable(scenario.is_recoverable_failure)
    assert callable(scenario.recovery_plan_factory)
    assert scenario.duplicate_harmful_side_effect_count() == 0
    assert scenario.metadata["injected_failure"] == "one safe transient failure before mutation"


def test_provider_builds_fresh_scenarios_for_each_run():
    provider = build_provider()
    task = provider.tasks()[0]
    first = provider.build_scenario(task, 1)
    second = provider.build_scenario(task, 2)

    assert first.agent is not second.agent
    assert first.environment is not second.environment
    assert first.goal is not second.goal
    assert first.environment.observe() == second.environment.observe() == {"box": "closed"}
