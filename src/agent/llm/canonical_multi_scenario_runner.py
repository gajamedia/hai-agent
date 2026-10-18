"""Step 41.1: execute multi-scenario cases through the canonical runtime chain.

The application supplies a fresh scenario binding for each case/run. This module
never creates domain actions or infers a plan from natural-language text.
Python 3.10+.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.llm.execution_security import ExecutionSecurityPolicy, SQLiteExecutionSecurityAuditSink
from agent.llm.execution_trace_runtime_observer import ExecutionTraceRuntimeObserver
from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI
from agent.llm.multi_scenario_evaluation import ScenarioCase, ScenarioExecution


class CanonicalMultiScenarioRunnerError(RuntimeError):
    """Raised when a scenario cannot be safely bound to canonical execution."""


@dataclass(frozen=True)
class CanonicalScenarioBinding:
    """Fresh application-owned objects required for one scenario execution."""
    agent: Any
    environment: Any
    plan: Any
    goal: Any
    decision_factory: Callable[[Any], Any]
    action_risks: Mapping[str, float]
    approval_required_actions: frozenset[str] = frozenset()
    approval_checker: Callable[[Any], bool] | None = None
    risk_threshold: float = 0.65
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.decision_factory):
            raise TypeError("decision_factory must be callable")
        if not isinstance(self.action_risks, Mapping):
            raise TypeError("action_risks must be a mapping")
        if not 0.0 <= float(self.risk_threshold) <= 1.0:
            raise ValueError("risk_threshold must be between 0 and 1")
        if self.approval_checker is not None and not callable(self.approval_checker):
            raise TypeError("approval_checker must be callable when provided")
        object.__setattr__(self, "action_risks", dict(self.action_risks))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


ScenarioBindingFactory = Callable[[ScenarioCase, int], CanonicalScenarioBinding]


class CanonicalMultiScenarioRunner:
    """Callable runner compatible with ``evaluate_scenarios``.

    The binding factory must return fresh Agent and environment instances per run,
    and must construct a plan and goal appropriate for that case. This runner uses
    the repository's canonical bridge/API/executor path, security policy, SQLite
    audit sink, and hash-chained trace store.
    """
    def __init__(self, binding_factory: ScenarioBindingFactory, *, artifact_dir: str | Path = "artifacts/step41_1") -> None:
        if not callable(binding_factory):
            raise TypeError("binding_factory must be callable")
        self.binding_factory = binding_factory
        self.artifact_dir = Path(artifact_dir)
        self._agents: list[Any] = []
        self._environments: list[Any] = []

    def __call__(self, case: ScenarioCase, run_index: int) -> ScenarioExecution:
        if run_index < 1:
            raise ValueError("run_index must be >= 1")
        binding = self.binding_factory(case, run_index)
        if not isinstance(binding, CanonicalScenarioBinding):
            raise CanonicalMultiScenarioRunnerError("binding_factory must return CanonicalScenarioBinding")
        if binding.agent is None or binding.environment is None or binding.goal is None or binding.plan is None:
            raise CanonicalMultiScenarioRunnerError("agent, environment, goal, and plan are required")
        if not callable(getattr(binding.environment, "observe", None)):
            raise CanonicalMultiScenarioRunnerError("environment must expose observe()")
        if not callable(binding.decision_factory):
            raise CanonicalMultiScenarioRunnerError("decision_factory must be callable")
        if any(binding.agent is old for old in self._agents):
            raise CanonicalMultiScenarioRunnerError("binding_factory reused an Agent across runs")
        if any(binding.environment is old for old in self._environments):
            raise CanonicalMultiScenarioRunnerError("binding_factory reused an environment across runs")
        self._agents.append(binding.agent)
        self._environments.append(binding.environment)

        run_id = f"{case.scenario_id}-run-{run_index:02d}"
        run_dir = self.artifact_dir / case.domain / case.scenario_id / f"run-{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        audit_sink = SQLiteExecutionSecurityAuditSink(run_dir / "security_audit.sqlite3")
        policy = ExecutionSecurityPolicy(
            action_risks=dict(binding.action_risks),
            risk_threshold=binding.risk_threshold,
            approval_required_actions=frozenset(binding.approval_required_actions),
            approval_checker=binding.approval_checker,
            audit_sink=audit_sink,
        )
        binding.agent.execution_security_policy = policy
        api = AgentCanonicalSingleActionAPI(
            binding.agent, decision_factory=binding.decision_factory, security_policy=policy
        )
        bridge = LLMPlanSingleActionBridge.for_api(api)
        trace_store = ExecutionTraceStore(run_dir / "execution_trace.jsonl")
        observer = ExecutionTraceRuntimeObserver(trace_store, raise_on_error=True, include_world_state=False)
        runtime = LLMPlanRuntimeExecutor.for_bridge(bridge, runtime_event_observer=observer)
        if runtime.bridge is not bridge or bridge.api is not api:
            raise CanonicalMultiScenarioRunnerError("canonical runtime chain binding failed")
        if api.executor.binding.agent is not binding.agent:
            raise CanonicalMultiScenarioRunnerError("canonical executor is not bound to scenario Agent")

        before = dict(binding.environment.observe())
        if dict(case.initial_state) != before:
            raise CanonicalMultiScenarioRunnerError(
                f"initial state mismatch for {case.scenario_id}: expected {dict(case.initial_state)!r}, got {before!r}"
            )
        report = runtime.execute(
            binding.plan,
            goal=binding.goal,
            world_state=before,
            environment=binding.environment,
            metadata={
                "stage": "41.1",
                "scenario_id": case.scenario_id,
                "domain": case.domain,
                "run_index": run_index,
                "run_id": run_id,
                "holdout": case.holdout,
                **dict(binding.metadata),
            },
        )
        final_state = dict(binding.environment.observe())
        records = tuple(getattr(report, "records", ()) or ())
        executed_actions = tuple(
            str(getattr(getattr(record, "step", None), "action", "unknown"))
            for record in records
            if not getattr(record, "skipped", False)
        )
        trace_integrity = trace_store.verify()
        audit_events = audit_sink.read_all()
        evidence = {
            "run_id": run_id,
            "runtime_report_completed": bool(getattr(report, "completed", False)),
            "trace_integrity": dict(trace_integrity) if isinstance(trace_integrity, Mapping) else {"valid": False},
            "trace_event_count": len(trace_store.replay(run_id)),
            "trace_observer_errors": int(getattr(observer, "error_count", -1)),
            "security_audit_event_count": len(audit_events),
            "security_audit_path": str(run_dir / "security_audit.sqlite3"),
            "trace_path": str(run_dir / "execution_trace.jsonl"),
        }
        evidence["trace_integrity_passed"] = evidence["trace_integrity"].get("valid") is True and evidence["trace_observer_errors"] == 0
        return ScenarioExecution(
            final_state=final_state,
            runtime_success=bool(getattr(report, "success", False)),
            actions_attempted=len(records),
            executed_actions=executed_actions,
            failure_reason=None if bool(getattr(report, "success", False)) else str(getattr(report, "stop_reason", "runtime_failed")),
            adapter_name="canonical_multi_scenario_runner",
            evidence=evidence,
        )


__all__ = ["CanonicalMultiScenarioRunner", "CanonicalMultiScenarioRunnerError", "CanonicalScenarioBinding", "ScenarioBindingFactory"]
