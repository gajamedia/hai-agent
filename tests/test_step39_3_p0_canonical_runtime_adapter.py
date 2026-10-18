from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.agent_decision import AgentDecision, DecisionMode
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.p0_canonical_runtime_adapter import (
    P0CanonicalRuntimeAdapter,
    P0CanonicalRuntimeAdapterError,
    P0RuntimeScenario,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


class FakeEnvironment:
    def __init__(self):
        self.state = {"box": "closed"}

    def observe(self):
        return dict(self.state)


class FakeUnifiedExecutor:
    # AgentUnifiedExecutorBinding validates that the canonical executor exposes
    # execute(...), even when it prefers the execute_action(...) path.
    def execute(self, **kwargs):
        return self.execute_action(**kwargs)

    def execute_action(self, *, action, environment, **kwargs):
        result = action.execute(environment=environment)
        return result


class FakeAgent:
    def __init__(self, action):
        self.actions = {action.name: action}
        self.unified_executor = FakeUnifiedExecutor()
        self.execution_security_policy = None


class Provider:
    def __init__(self, *, risk=0.1, approval_checker=None, recovery=False):
        self.risk = risk
        self.approval_checker = approval_checker
        self.recovery = recovery
        self.created_agents = []
        self.created_environments = []
        self.duplicates = {"count": 0}

    def tasks(self):
        return [
            P0BenchmarkTask(
                task_id=f"task-{i:02d}",
                description=f"controlled task {i}",
                payload={"target": "open"},
            )
            for i in range(1, 31)
        ]

    def build_scenario(self, task, run_index):
        env = FakeEnvironment()

        def open_box(*, environment):
            environment.state["box"] = "open"
            return {"success": True, "action": "open_box"}

        action = Action(
            name="open_box",
            description="Open box",
            handler=open_box,
            preconditions={"box": "closed"},
            effects={"box": "open"},
        )
        agent = FakeAgent(action)
        self.created_agents.append(agent)
        self.created_environments.append(env)
        goal = Goal(description="open box", target_state={"box": "open"})
        plan = LLMPlan(
            goal="open box",
            rationale="one controlled action",
            steps=(
                LLMPlanStep(
                    index=1,
                    action="open_box",
                    purpose="open the box",
                    expected_observation="box is open",
                    success_criteria="box == open",
                    risk="low",
                ),
            ),
            confidence=0.99,
            provider="test",
            model="test",
        )

        return P0RuntimeScenario(
            agent=agent,
            environment=env,
            plan=plan,
            goal=goal,
            decision_factory=lambda request: AgentDecision(
                mode=DecisionMode.EXPLOIT,
                reason="controlled test",
                uncertainty=0.0,
                expected_information_gain=0.0,
                risk=0.0,
            ),
            action_risks={"open_box": self.risk},
            approval_checker=self.approval_checker,
            duplicate_harmful_side_effect_count=lambda: self.duplicates["count"],
        )


def test_runs_through_canonical_runtime_and_persists_audit_and_trace(tmp_path):
    provider = Provider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "runs")
    task = provider.tasks()[0]

    evidence = adapter.execute(task, 1)

    assert evidence.correct_completion is True
    assert evidence.deterministic_verification_passed is True
    assert evidence.audit_evidence_complete is True
    assert evidence.required_approval_evidence_complete is True
    assert evidence.trace_integrity_passed is True
    assert evidence.unauthorized_high_risk_actions == 0
    assert evidence.duplicate_harmful_side_effects == 0
    assert "trace_events=" in evidence.notes


def test_unknown_or_high_risk_action_without_approval_is_blocked_and_audited(tmp_path):
    provider = Provider(risk=0.95, approval_checker=None)
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "runs")

    evidence = adapter.execute(provider.tasks()[0], 1)

    assert evidence.correct_completion is False
    assert evidence.audit_evidence_complete is True
    assert evidence.trace_integrity_passed is True
    assert evidence.unauthorized_high_risk_actions == 0
    # The denial itself is preserved in the durable audit database.
    from agent.llm.execution_security import SQLiteExecutionSecurityAuditSink
    audit_paths = list((tmp_path / "runs" / "task-01" / "run-01").glob("attempt-*/security_audit.sqlite3"))
    assert len(audit_paths) == 1
    sink = SQLiteExecutionSecurityAuditSink(audit_paths[0])
    events = sink.read_all()
    assert any(event.event_type == "authorization" and not event.allowed for event in events)
    assert not any(event.event_type == "execution_outcome" for event in events)


def test_scenario_requires_deterministic_goal_and_side_effect_ledger():
    provider = Provider()
    scenario = provider.build_scenario(provider.tasks()[0], 1)
    bad_goal = Goal(description="no target")
    invalid = P0RuntimeScenario(
        agent=scenario.agent,
        environment=scenario.environment,
        plan=scenario.plan,
        goal=bad_goal,
        decision_factory=scenario.decision_factory,
        action_risks=scenario.action_risks,
        duplicate_harmful_side_effect_count=scenario.duplicate_harmful_side_effect_count,
    )
    with pytest.raises(P0CanonicalRuntimeAdapterError, match="target_state"):
        P0CanonicalRuntimeAdapter._validate_scenario(invalid)


def test_provider_must_create_fresh_agent_and_environment_each_run(tmp_path):
    class ReusingProvider(Provider):
        def __init__(self):
            super().__init__()
            self.cached = None

        def build_scenario(self, task, run_index):
            if self.cached is None:
                self.cached = super().build_scenario(task, run_index)
            return self.cached

    provider = ReusingProvider()
    adapter = P0CanonicalRuntimeAdapter(provider, artifact_dir=tmp_path / "runs")
    adapter.execute(provider.tasks()[0], 1)
    with pytest.raises(P0CanonicalRuntimeAdapterError, match="reused an Agent"):
        adapter.execute(provider.tasks()[1], 1)
