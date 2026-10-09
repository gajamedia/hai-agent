from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent.llm.agent_canonical_single_action_api import (
    AgentCanonicalSingleActionAPI,
    AgentCanonicalSingleActionAPIConfigurationError,
)
from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.canonical_single_action import CanonicalSingleActionRequest


@dataclass(frozen=True)
class FakeDecision:
    action: str


@dataclass
class FakeExecutorResult:
    value: Any


class FakeUnifiedExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return FakeExecutorResult({"success": True, "action": kwargs["decision"].action})


class FakeAgent:
    def __init__(self):
        self.actions = {"open_box": object(), "take_key": object()}
        self.unified_executor = FakeUnifiedExecutor()
        self.configure_called = 0

    def configure_planner(self):
        self.configure_called += 1
        if self.unified_executor is None:
            self.unified_executor = FakeUnifiedExecutor()


def build_agent() -> FakeAgent:
    return FakeAgent()


def decision_factory(request):
    return FakeDecision(action=request.action)


def request(**overrides):
    values = {
        "action": "open_box",
        "step": "open_box",
        "position": 1,
        "goal": "open the box",
        "world_state": {"box": "closed"},
        "environment": "env",
        "strategies": ["safe"],
        "information_gain": {"open_box": 0.8},
        "action_risk": {"open_box": 0.2},
        "exploration_risk": 0.1,
        "action_cost": {"open_box": 1.0},
        "experience_id": "exp-1",
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return CanonicalSingleActionRequest(**values)


def test_rejects_none_agent():
    with pytest.raises(AgentCanonicalSingleActionAPIConfigurationError):
        AgentCanonicalSingleActionAPI(None, decision_factory=decision_factory)


def test_rejects_non_callable_decision_factory():
    with pytest.raises(AgentCanonicalSingleActionAPIConfigurationError):
        AgentCanonicalSingleActionAPI(build_agent(), decision_factory="bad")


def test_rejects_agent_without_actions_registry():
    class BadAgent:
        unified_executor = FakeUnifiedExecutor()

    with pytest.raises(AgentCanonicalSingleActionAPIConfigurationError):
        AgentCanonicalSingleActionAPI(BadAgent(), decision_factory=decision_factory)


def test_rejects_agent_without_execution_primitive_or_configurator():
    class BadAgent:
        actions = {"open_box": object()}

    with pytest.raises(AgentCanonicalSingleActionAPIConfigurationError):
        AgentCanonicalSingleActionAPI(BadAgent(), decision_factory=decision_factory)


def test_binds_existing_agent():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    assert api.agent is agent
    assert api.executor.binding.agent is agent


def test_for_agent_factory():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI.for_agent(
        agent,
        decision_factory=decision_factory,
    )
    assert isinstance(api, AgentCanonicalSingleActionAPI)


def test_execute_delegates_to_existing_unified_executor():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)

    result = api.execute(request())

    assert result.success is True
    assert len(agent.unified_executor.calls) == 1
    assert agent.unified_executor.calls[0]["environment"] == "env"


def test_decision_factory_receives_boundary_request():
    agent = build_agent()
    seen = {}

    def factory(boundary_request):
        seen["request"] = boundary_request
        return FakeDecision(action=boundary_request.action)

    api = AgentCanonicalSingleActionAPI(agent, decision_factory=factory)
    canonical = request()
    api.execute(canonical)

    assert isinstance(seen["request"], AgentExecutionRequest)
    assert seen["request"].action == "open_box"
    assert seen["request"].goal == "open the box"


def test_last_result_is_none_before_execution():
    api = AgentCanonicalSingleActionAPI(
        build_agent(),
        decision_factory=decision_factory,
    )
    assert api.last_result is None


def test_last_result_is_updated_after_execution():
    api = AgentCanonicalSingleActionAPI(
        build_agent(),
        decision_factory=decision_factory,
    )
    result = api.execute(request())
    assert api.last_result is result


def test_success_property_is_forwarded():
    api = AgentCanonicalSingleActionAPI(
        build_agent(),
        decision_factory=decision_factory,
    )
    assert api.execute(request()).success is True


def test_execute_passes_information_gain():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request(information_gain={"open_box": 0.91}))
    assert agent.unified_executor.calls[-1]["information_gain"] == {"open_box": 0.91}


def test_execute_passes_action_risk():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request(action_risk={"open_box": 0.33}))
    assert agent.unified_executor.calls[-1]["action_risk"] == {"open_box": 0.33}


def test_execute_passes_action_cost():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request(action_cost={"open_box": 2.5}))
    assert agent.unified_executor.calls[-1]["action_cost"] == {"open_box": 2.5}


def test_execute_passes_exploration_risk():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request(exploration_risk=0.7))
    assert agent.unified_executor.calls[-1]["exploration_risk"] == 0.7


def test_execute_passes_strategies():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request(strategies=["safe", "fast"]))
    assert agent.unified_executor.calls[-1]["strategies"] == ["safe", "fast"]


def test_execute_rejects_unregistered_action():
    api = AgentCanonicalSingleActionAPI(
        build_agent(),
        decision_factory=decision_factory,
    )
    with pytest.raises(RuntimeError, match="not registered"):
        api.execute(request(action="unknown"))



def test_attach_adds_explicit_agent_capabilities():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)

    returned = api.attach()

    assert returned is agent
    assert agent.canonical_single_action_api is api
    assert callable(agent.execute_single_action)


def test_attached_execute_single_action_uses_same_api():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.attach()

    result = agent.execute_single_action(request())

    assert result.success is True
    assert api.last_result is result


def test_attach_rejects_existing_different_api():
    agent = build_agent()
    agent.canonical_single_action_api = object()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)

    with pytest.raises(AgentCanonicalSingleActionAPIConfigurationError, match="different"):
        api.attach()


def test_detach_removes_only_installed_capabilities():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.attach()

    api.detach()

    assert not hasattr(agent, "canonical_single_action_api")
    assert not hasattr(agent, "execute_single_action")


def test_execute_from_boundary_request_bridge():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)

    boundary = AgentExecutionRequest(
        action="open_box",
        step="open_box",
        position=1,
        goal="open the box",
        world_state={"box": "closed"},
        experience_id="exp-boundary",
        metadata={"legacy": True},
    )

    result = api.execute_from_boundary_request(
        boundary,
        environment="env",
    )

    assert result.success is True
    assert result.request.experience_id == "exp-boundary"


def test_bridge_preserves_metadata():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)

    boundary = AgentExecutionRequest(
        action="open_box",
        step="open_box",
        position=1,
        goal="open the box",
        metadata={"legacy": True, "trace_id": "abc"},
    )

    result = api.execute_from_boundary_request(boundary)
    assert result.request.metadata == {"legacy": True, "trace_id": "abc"}


def test_canonical_api_does_not_call_decide_and_act():
    agent = build_agent()
    calls = []

    def decide_and_act(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("decide_and_act must not be called")

    agent.decide_and_act = decide_and_act

    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    result = api.execute(request())

    assert result.success is True
    assert calls == []


def test_canonical_api_reuses_existing_executor():
    agent = build_agent()
    existing = agent.unified_executor
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request())
    assert agent.unified_executor is existing


def test_canonical_api_executes_exactly_one_call_per_request():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request())
    assert len(agent.unified_executor.calls) == 1


def test_two_requests_produce_two_independent_calls():
    agent = build_agent()
    api = AgentCanonicalSingleActionAPI(agent, decision_factory=decision_factory)
    api.execute(request(experience_id="exp-1"))
    api.execute(request(experience_id="exp-2"))
    assert len(agent.unified_executor.calls) == 2
    assert agent.unified_executor.calls[0]["experience_id"] == "exp-1"
    assert agent.unified_executor.calls[1]["experience_id"] == "exp-2"
