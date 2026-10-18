from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.unified_executor_binding import (
    AgentUnifiedExecutionReport,
    AgentUnifiedExecutorBinding,
    AgentUnifiedExecutorBindingConfigurationError,
    AgentUnifiedExecutorBindingError,
)


@dataclass
class FakeAction:
    name: str


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def execute(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True, "action": kwargs["decision"]}


class FakeAgent:
    def __init__(self):
        self.actions = {
            "open_box": FakeAction("open_box"),
            "close_box": FakeAction("close_box"),
        }
        self.execution_security_policy = ExecutionSecurityPolicy(
            action_risks={"open_box": 0.1, "close_box": 0.1}
        )
        self.unified_executor = FakeExecutor()
        self.configure_calls = 0

    def configure_planner(self):
        self.configure_calls += 1
        if self.unified_executor is None:
            self.unified_executor = FakeExecutor()


def request(action="open_box", position=1):
    return AgentExecutionRequest(
        action=action,
        step={"index": position, "action": action},
        position=position,
        goal="open box",
        world_state={"box": "closed"},
        experience_id="ep-1",
        metadata={"source": "llm"},
    )


def test_constructor_requires_agent():
    with pytest.raises(ValueError, match="agent cannot be None"):
        AgentUnifiedExecutorBinding(None, decision_factory=lambda req: object())


def test_constructor_requires_callable_factory():
    with pytest.raises(TypeError, match="decision_factory"):
        AgentUnifiedExecutorBinding(FakeAgent(), decision_factory="nope")


def test_registered_action_is_required():
    binding = AgentUnifiedExecutorBinding(FakeAgent(), decision_factory=lambda req: "decision")
    with pytest.raises(AgentUnifiedExecutorBindingError, match="not registered"):
        binding.execute(request("missing"))


def test_unified_executor_is_required():
    agent = FakeAgent()
    agent.unified_executor = None
    agent.configure_planner = None
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    with pytest.raises(AgentUnifiedExecutorBindingConfigurationError, match="unified_executor"):
        binding.execute(request())


def test_configure_planner_can_restore_executor():
    agent = FakeAgent()
    agent.unified_executor = None
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    result = binding.execute(request())
    assert result["success"] is True
    assert agent.configure_calls == 1


def test_decision_factory_receives_request():
    seen = []
    binding = AgentUnifiedExecutorBinding(
        FakeAgent(),
        decision_factory=lambda req: seen.append(req) or "decision:open_box",
    )
    binding.execute(request())
    assert seen[0].action == "open_box"
    assert seen[0].position == 1


def test_result_is_forwarded_from_existing_executor():
    binding = AgentUnifiedExecutorBinding(FakeAgent(), decision_factory=lambda req: "decision")
    assert binding.execute(request()) == {"success": True, "action": "decision"}


def test_report_contains_request_decision_and_result():
    binding = AgentUnifiedExecutorBinding(FakeAgent(), decision_factory=lambda req: "decision")
    report = binding.execute_request(request())
    assert isinstance(report, AgentUnifiedExecutionReport)
    assert report.request.action == "open_box"
    assert report.decision == "decision"
    assert report.result["success"] is True


def test_last_report_is_updated():
    binding = AgentUnifiedExecutorBinding(FakeAgent(), decision_factory=lambda req: "decision")
    report = binding.execute_request(request())
    assert binding.last_report is report


def test_environment_is_forwarded():
    agent = FakeAgent()
    env = object()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request(), environment=env)
    assert agent.unified_executor.calls[-1]["environment"] is env


def test_goal_is_forwarded():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request())
    assert agent.unified_executor.calls[-1]["goal"] == "open box"


def test_world_state_is_forwarded():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request())
    assert agent.unified_executor.calls[-1]["world_state"] == {"box": "closed"}


def test_experience_id_is_forwarded():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request())
    assert agent.unified_executor.calls[-1]["experience_id"] == "ep-1"


def test_action_registry_is_forwarded_as_action_objects():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request())
    actions = agent.unified_executor.calls[-1]["actions"]
    assert [action.name for action in actions] == ["open_box", "close_box"]


def test_risk_information_gain_and_cost_are_forwarded():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(
        request(),
        information_gain={"open_box": 0.8},
        action_risk={"open_box": 0.2},
        exploration_risk=0.4,
        action_cost={"open_box": 0.1},
    )
    kwargs = agent.unified_executor.calls[-1]
    assert kwargs["information_gain"] == {"open_box": 0.8}
    assert kwargs["action_risk"] == {"open_box": 0.2}
    assert kwargs["exploration_risk"] == 0.4
    assert kwargs["action_cost"] == {"open_box": 0.1}


def test_strategies_are_forwarded():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    strategies = ["strategy-a"]
    binding.execute(request(), strategies=strategies)
    assert agent.unified_executor.calls[-1]["strategies"] is strategies


def test_mapping_actions_are_supported():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request("close_box", 2))
    assert agent.unified_executor.calls[-1]["decision"] == "decision"


def test_non_mapping_actions_are_supported():
    agent = FakeAgent()
    agent.actions = [FakeAction("open_box"), FakeAction("close_box")]
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request())
    assert agent.unified_executor.calls[-1]["actions"] == agent.actions


def test_non_mapping_actions_reject_unknown_action():
    agent = FakeAgent()
    agent.actions = [FakeAction("open_box")]
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    with pytest.raises(AgentUnifiedExecutorBindingError, match="not registered"):
        binding.execute(request("close_box"))


def test_action_registry_absence_is_configuration_error():
    agent = FakeAgent()
    del agent.actions
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    with pytest.raises(AgentUnifiedExecutorBindingConfigurationError, match="actions registry"):
        binding.execute(request())


def test_action_registry_invalid_iterable_is_configuration_error():
    agent = FakeAgent()
    agent.actions = 123
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    with pytest.raises(AgentUnifiedExecutorBindingConfigurationError, match="mapping or iterable"):
        binding.execute(request())


def test_none_decision_is_rejected_before_executor_call():
    agent = FakeAgent()
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: None)
    with pytest.raises(AgentUnifiedExecutorBindingError, match="returned None"):
        binding.execute(request())
    assert agent.unified_executor.calls == []


def test_request_type_is_enforced():
    binding = AgentUnifiedExecutorBinding(FakeAgent(), decision_factory=lambda req: "decision")
    with pytest.raises(TypeError, match="AgentExecutionRequest"):
        binding.execute("bad-request")


def test_convenience_execute_returns_only_executor_output():
    binding = AgentUnifiedExecutorBinding(FakeAgent(), decision_factory=lambda req: "decision")
    output = binding.execute(request())
    assert output == {"success": True, "action": "decision"}


def test_binding_does_not_call_decide_and_act():
    agent = FakeAgent()
    called = []
    agent.decide_and_act = lambda *args, **kwargs: called.append(True)
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    binding.execute(request())
    assert called == []


def test_executor_exception_is_propagated():
    agent = FakeAgent()
    def raise_error(**kwargs):
        raise RuntimeError("executor failed")
    agent.unified_executor.execute = raise_error
    binding = AgentUnifiedExecutorBinding(agent, decision_factory=lambda req: "decision")
    with pytest.raises(RuntimeError, match="executor failed"):
        binding.execute(request())
