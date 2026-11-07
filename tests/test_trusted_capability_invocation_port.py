from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.trusted_execution_context import TrustedExecutionContext
from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI
from agent.llm.capability_execution_contracts import CapabilityReadResult
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.trusted_capability_invocation_port import (
    CapabilityInvocationSpec,
    TrustedCapabilityInvocationConfigurationError,
    TrustedCapabilityInvocationPort,
)
from agent.llm.unified_executor_binding import (
    AgentUnifiedExecutorBinding,
    AgentUnifiedExecutorBindingConfigurationError,
)


def principal():
    return AuthenticatedPrincipal(
        actor_id="user-1",
        roles=("analyst",),
        permissions=("business:database:read",),
        issuer="test",
        subject="subject-1",
        is_authenticated=True,
    )


class FakeGateway:
    def __init__(self):
        self.calls = []

    def read(self, principal, request, *, execution_id):
        self.calls.append((principal, request, execution_id))
        return CapabilityReadResult(
            target=request.target,
            capability_id=request.capability_id,
            data=[{"id": 1, "name": "sample"}],
        )


class FakeAction:
    def __init__(self, name):
        self.name = name
        self.executed = 0

    def execute(self, **kwargs):
        self.executed += 1
        raise AssertionError("placeholder Action handler must never execute")


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True}


class FakeAgent:
    def __init__(self):
        self.actions = {"open_box": FakeAction("open_box")}
        self.unified_executor = FakeExecutor()
        self.execution_security_policy = ExecutionSecurityPolicy(
            action_risks={"read_business_data": 0.1, "open_box": 0.1}
        )


def build_port(gateway):
    return TrustedCapabilityInvocationPort(
        gateway,
        {
            "read_business_data": CapabilityInvocationSpec(
                action_name="read_business_data",
                target="business_database",
                capability_id="business_database.select",
            )
        },
    )


def request(action="read_business_data", step=None, metadata=None):
    return AgentExecutionRequest(
        action=action,
        step=step or {
            "action": action,
            "arguments": {"table": "customers", "limit": 10},
            "target": "attacker-target",
            "capability_id": "attacker-capability",
        },
        position=1,
        goal="read customer data",
        metadata=metadata or {"source": "plan"},
    )


def test_port_uses_server_owned_target_and_capability_mapping():
    gateway = FakeGateway()
    port = build_port(gateway)
    context = TrustedExecutionContext(principal())

    result = port.invoke(request(), execution_context=context)

    assert result.data == [{"id": 1, "name": "sample"}]
    seen_principal, seen_request, seen_execution_id = gateway.calls[0]
    assert seen_principal is context.principal
    assert seen_request.target == "business_database"
    assert seen_request.capability_id == "business_database.select"
    assert seen_request.arguments == {"table": "customers", "limit": 10}
    assert seen_execution_id == context.execution_id


def test_port_rejects_missing_or_untrusted_context():
    port = build_port(FakeGateway())
    with pytest.raises(TrustedCapabilityInvocationConfigurationError):
        port.invoke(request(), execution_context=object())


def test_port_rejects_unmapped_action():
    port = build_port(FakeGateway())
    context = TrustedExecutionContext(principal())
    with pytest.raises(Exception, match="not mapped"):
        port.invoke(request(action="other"), execution_context=context)


def test_port_requires_structured_arguments_mapping():
    port = build_port(FakeGateway())
    context = TrustedExecutionContext(principal())
    with pytest.raises(Exception, match="mapping"):
        port.invoke(
            request(step={"action": "read_business_data", "arguments": "bad"}),
            execution_context=context,
        )


def test_binding_routes_capability_without_agent_action_or_simulation_executor():
    gateway = FakeGateway()
    port = build_port(gateway)
    agent = FakeAgent()
    placeholder = FakeAction("read_business_data")
    agent.actions["read_business_data"] = placeholder
    binding = AgentUnifiedExecutorBinding(
        agent,
        decision_factory=lambda req: (_ for _ in ()).throw(
            AssertionError("decision factory must not run for a capability")
        ),
        trusted_capability_invocation_port=port,
    )
    context = TrustedExecutionContext(principal())

    report = binding.execute_request(request(), execution_context=context)

    assert report.result.data[0]["id"] == 1
    assert placeholder.executed == 0
    assert agent.unified_executor.calls == []
    assert len(gateway.calls) == 1
    assert report.decision.capability_id == "business_database.select"


def test_binding_routes_capability_even_when_placeholder_is_absent_from_agent_registry():
    gateway = FakeGateway()
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(
        agent,
        decision_factory=lambda req: "not-used",
        trusted_capability_invocation_port=build_port(gateway),
    )
    report = binding.execute_request(
        request(),
        execution_context=TrustedExecutionContext(principal()),
    )
    assert report.result.data
    assert agent.unified_executor.calls == []


def test_capability_action_without_context_fails_closed_without_simulation_fallback():
    gateway = FakeGateway()
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(
        agent,
        decision_factory=lambda req: "not-used",
        trusted_capability_invocation_port=build_port(gateway),
    )
    with pytest.raises(AgentUnifiedExecutorBindingConfigurationError, match="TrustedExecutionContext"):
        binding.execute_request(request())
    assert gateway.calls == []
    assert agent.unified_executor.calls == []


def test_non_capability_actions_keep_existing_simulation_path():
    gateway = FakeGateway()
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(
        agent,
        decision_factory=lambda req: "decision",
        trusted_capability_invocation_port=build_port(gateway),
    )
    report = binding.execute_request(
        AgentExecutionRequest(
            action="open_box", step={"action": "open_box"}, position=1
        )
    )
    assert report.result["success"] is True
    assert len(agent.unified_executor.calls) == 1
    assert gateway.calls == []


def test_agent_canonical_api_injects_port_and_propagates_context():
    gateway = FakeGateway()
    agent = FakeAgent()
    api = AgentCanonicalSingleActionAPI(
        agent,
        decision_factory=lambda req: "not-used",
        trusted_capability_invocation_port=build_port(gateway),
    )
    context = TrustedExecutionContext(principal())
    from agent.llm.canonical_single_action import CanonicalSingleActionRequest

    result = api.execute(
        CanonicalSingleActionRequest(
            action="read_business_data",
            step={
                "action": "read_business_data",
                "arguments": {"table": "customers"},
            },
            position=1,
        ),
        execution_context=context,
    )
    assert result.success is True
    assert gateway.calls[0][2] == context.execution_id


def test_execution_metadata_cannot_override_capability_identity():
    gateway = FakeGateway()
    port = build_port(gateway)
    context = TrustedExecutionContext(principal())
    req = request(
        metadata={
            "target": "attacker",
            "capability_id": "attacker",
            "execution_id": "caller-controlled",
        }
    )
    port.invoke(req, execution_context=context)
    _, gateway_request, execution_id = gateway.calls[0]
    assert gateway_request.target == "business_database"
    assert gateway_request.capability_id == "business_database.select"
    assert execution_id == context.execution_id
