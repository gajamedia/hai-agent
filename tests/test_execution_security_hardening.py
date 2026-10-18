from __future__ import annotations

import sqlite3

import pytest

from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI
from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.execution_security import (
    ExecutionSecurityAuditError,
    ExecutionSecurityError,
    ExecutionSecurityPolicy,
    SQLiteExecutionSecurityAuditSink,
)


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True}


class FakeAgent:
    def __init__(self, policy=None):
        self.actions = {"open_box": object(), "deploy": object()}
        self.unified_executor = FakeExecutor()
        if policy is not None:
            self.execution_security_policy = policy


def request(action="open_box", metadata=None):
    return AgentExecutionRequest(
        action=action,
        step={"action": action},
        position=1,
        metadata=metadata or {},
    )


def factory(req):
    return {"action": req.action}


def test_binding_direct_call_is_fail_closed_for_unregistered_risk_policy():
    agent = FakeAgent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=factory)
    with pytest.raises(ExecutionSecurityError):
        api.execute_from_boundary_request(request("open_box"))
    assert agent.unified_executor.calls == []


def test_metadata_approval_cannot_bypass_high_risk_gate():
    policy = ExecutionSecurityPolicy(action_risks={"deploy": 0.95})
    agent = FakeAgent(policy)
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=factory)
    with pytest.raises(ExecutionSecurityError):
        api.execute_from_boundary_request(request("deploy", {"approved": True}))
    assert agent.unified_executor.calls == []


def test_trusted_approval_allows_high_risk_action():
    policy = ExecutionSecurityPolicy(
        action_risks={"deploy": 0.95},
        approval_checker=lambda req: req.action == "deploy",
    )
    agent = FakeAgent(policy)
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=factory)
    result = api.execute_from_boundary_request(request("deploy"))
    assert result.success is True
    assert len(agent.unified_executor.calls) == 1


def test_sqlite_audit_persists_allowed_and_denied_decisions(tmp_path):
    sink = SQLiteExecutionSecurityAuditSink(tmp_path / "security-audit.sqlite3")
    policy = ExecutionSecurityPolicy(
        action_risks={"open_box": 0.1, "deploy": 0.95},
        audit_sink=sink,
    )
    policy.authorize(request("open_box"))
    with pytest.raises(ExecutionSecurityError):
        policy.authorize(request("deploy"))

    events = sink.read_all()
    assert len(events) == 2
    assert events[0].action == "open_box" and events[0].allowed is True
    assert events[1].action == "deploy" and events[1].allowed is False


def test_audit_persistence_failure_blocks_execution():
    def broken_sink(_event):
        raise OSError("disk unavailable")

    policy = ExecutionSecurityPolicy(
        action_risks={"open_box": 0.1},
        audit_sink=broken_sink,
    )
    agent = FakeAgent(policy)
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=factory)
    with pytest.raises(ExecutionSecurityAuditError):
        api.execute_from_boundary_request(request("open_box"))
    assert agent.unified_executor.calls == []


def test_sqlite_audit_includes_execution_outcome(tmp_path):
    sink = SQLiteExecutionSecurityAuditSink(tmp_path / "outcome-audit.sqlite3")
    policy = ExecutionSecurityPolicy(
        action_risks={"open_box": 0.1},
        audit_sink=sink,
    )
    agent = FakeAgent(policy)
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=factory)

    result = api.execute_from_boundary_request(request("open_box"))

    assert result.success is True
    events = sink.read_all()
    assert [event.event_type for event in events] == [
        "authorization",
        "execution_outcome",
    ]
    assert events[-1].execution_success is True


def test_unknown_execution_outcome_is_persisted_as_null(tmp_path):
    sink = SQLiteExecutionSecurityAuditSink(
        tmp_path / "unknown-outcome.sqlite3"
    )
    policy = ExecutionSecurityPolicy(
        action_risks={"open_box": 0.1},
        audit_sink=sink,
    )

    policy.record_execution_outcome(
        request("open_box"),
        success=None,
    )

    with sqlite3.connect(str(sink.path)) as connection:
        row = connection.execute(
            """
            SELECT reason, execution_success
            FROM execution_security_audit
            WHERE event_type = 'execution_outcome'
            """
        ).fetchone()

    assert row == ("execution_outcome_unknown", None)
