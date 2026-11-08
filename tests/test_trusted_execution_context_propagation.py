from dataclasses import dataclass

from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI
from agent.llm.canonical_single_action import (
    CanonicalSingleActionExecutor,
    CanonicalSingleActionRequest,
)
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.unified_executor_binding import (
    AgentUnifiedExecutionReport,
    AgentUnifiedExecutorBinding,
)


@dataclass
class FakeAction:
    name: str


class CapturingBinding:
    def __init__(self):
        self.calls = []

    def execute_request(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return AgentUnifiedExecutionReport(
            request=request,
            decision={"action": request.action},
            result={"success": True},
        )


def canonical_request(metadata=None):
    return CanonicalSingleActionRequest(
        action="open_box",
        step={"action": "open_box"},
        position=1,
        metadata=metadata or {"source": "plan"},
    )


def test_canonical_executor_forwards_exact_context_object_out_of_band():
    binding = CapturingBinding()
    executor = CanonicalSingleActionExecutor(binding)
    context = object()
    request = canonical_request()

    executor.execute(request, execution_context=context)

    forwarded = binding.calls[0][1]["execution_context"]
    assert forwarded is context
    assert "execution_context" not in request.metadata
    assert "execution_context" not in binding.calls[0][0].metadata


def test_boundary_request_bridge_forwards_same_context_object():
    binding = CapturingBinding()
    executor = CanonicalSingleActionExecutor(binding)
    context = object()
    request = AgentExecutionRequest(
        action="open_box",
        step={"action": "open_box"},
        position=1,
        metadata={"source": "legacy"},
    )

    executor.execute_from_boundary_request(
        request,
        execution_context=context,
        environment="test-env",
    )

    assert binding.calls[0][1]["execution_context"] is context
    assert binding.calls[0][1]["environment"] == "test-env"
    assert "execution_context" not in request.metadata


def test_agent_api_forwards_context_to_canonical_executor():
    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, request, *, execution_context=None):
            self.calls.append((request, execution_context))
            return "result"

    class Agent:
        actions = {"open_box": FakeAction("open_box")}
        unified_executor = object()

    api = AgentCanonicalSingleActionAPI(
        Agent(), decision_factory=lambda request: {"action": request.action}
    )
    fake_executor = FakeExecutor()
    api.executor = fake_executor
    context = object()
    request = canonical_request()

    assert api.execute(request, execution_context=context) == "result"
    assert fake_executor.calls[0][0] is request
    assert fake_executor.calls[0][1] is context
    assert "execution_context" not in request.metadata


def test_binding_forwards_context_only_to_explicit_trusted_executor_port():
    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            raise AssertionError("generic simulation path must not be called")

        def execute_with_trusted_context(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True}

    class Agent:
        def __init__(self):
            self.actions = {"open_box": FakeAction("open_box")}
            self.unified_executor = FakeExecutor()
            self.execution_security_policy = ExecutionSecurityPolicy(
                action_risks={"open_box": 0.1}
            )

    agent = Agent()
    binding = AgentUnifiedExecutorBinding(
        agent, decision_factory=lambda request: {"action": request.action}
    )
    context = object()
    request = AgentExecutionRequest(
        action="open_box",
        step={"action": "open_box"},
        position=1,
        metadata={"source": "plan"},
    )

    binding.execute_request(request, execution_context=context)

    forwarded = agent.unified_executor.calls[0]
    assert forwarded["execution_context"] is context
    assert "execution_context" not in request.metadata
    assert "execution_context" not in request.metadata


def test_binding_fails_closed_when_context_has_no_trusted_executor_port():
    class FakeExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, **kwargs):
            self.calls.append(kwargs)
            return {"success": True}

    class Agent:
        def __init__(self):
            self.actions = {"open_box": FakeAction("open_box")}
            self.unified_executor = FakeExecutor()
            self.execution_security_policy = ExecutionSecurityPolicy(
                action_risks={"open_box": 0.1}
            )

    agent = Agent()
    binding = AgentUnifiedExecutorBinding(
        agent, decision_factory=lambda request: {"action": request.action}
    )
    request = AgentExecutionRequest(
        action="open_box", step={"action": "open_box"}, position=1
    )

    import pytest
    from agent.llm.unified_executor_binding import AgentUnifiedExecutorBindingConfigurationError

    with pytest.raises(AgentUnifiedExecutorBindingConfigurationError, match="execute_with_trusted_context"):
        binding.execute_request(request, execution_context=object())
    assert agent.unified_executor.calls == []
