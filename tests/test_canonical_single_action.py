from dataclasses import dataclass

import pytest

from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.canonical_single_action import (
    CanonicalSingleActionConfigurationError,
    CanonicalSingleActionExecutor,
    CanonicalSingleActionRequest,
)
from agent.llm.unified_executor_binding import AgentUnifiedExecutionReport


@dataclass
class FakeBinding:
    calls: list[dict]

    def execute_request(self, request, **kwargs):
        self.calls.append({"request": request, "kwargs": kwargs})
        return AgentUnifiedExecutionReport(
            request=request,
            decision={"action": request.action},
            result={"success": True, "action": request.action},
        )


def make_request(**overrides):
    data = {
        "action": "open_box",
        "step": {"action": "open_box"},
        "position": 1,
        "goal": "open the box",
        "world_state": {"box": "closed"},
        "environment": object(),
        "strategies": ["safe"],
        "information_gain": {"open_box": 0.2},
        "action_risk": {"open_box": 0.1},
        "exploration_risk": 0.3,
        "action_cost": {"open_box": 1.0},
        "experience_id": "exp-1",
        "metadata": {"source": "llm"},
    }
    data.update(overrides)
    return CanonicalSingleActionRequest(**data)


def test_request_normalizes_action_and_mappings():
    request = make_request(action="  open_box  ")
    assert request.action == "open_box"
    assert request.metadata == {"source": "llm"}
    assert request.information_gain["open_box"] == 0.2


def test_request_rejects_empty_action():
    with pytest.raises(ValueError, match="action cannot be empty"):
        make_request(action="   ")


def test_request_rejects_invalid_position():
    with pytest.raises(ValueError, match="position must be >= 1"):
        make_request(position=0)


def test_request_rejects_invalid_exploration_risk():
    with pytest.raises(ValueError, match="exploration_risk"):
        make_request(exploration_risk=1.1)


def test_executor_requires_execution_binding():
    with pytest.raises(CanonicalSingleActionConfigurationError):
        CanonicalSingleActionExecutor(object())


def test_execute_forwards_all_runtime_arguments_once():
    binding = FakeBinding([])
    executor = CanonicalSingleActionExecutor(binding)
    request = make_request()

    result = executor.execute(request)

    assert len(binding.calls) == 1
    call = binding.calls[0]
    assert call["request"].action == "open_box"
    assert call["kwargs"]["environment"] is request.environment
    assert call["kwargs"]["strategies"] == request.strategies
    assert call["kwargs"]["information_gain"] == dict(request.information_gain)
    assert call["kwargs"]["action_risk"] == dict(request.action_risk)
    assert call["kwargs"]["exploration_risk"] == request.exploration_risk
    assert call["kwargs"]["action_cost"] == dict(request.action_cost)


def test_execute_returns_canonical_result():
    binding = FakeBinding([])
    executor = CanonicalSingleActionExecutor(binding)

    result = executor.execute(make_request())

    assert result.request.action == "open_box"
    assert result.boundary_request.action == "open_box"
    assert result.decision == {"action": "open_box"}
    assert result.result["success"] is True
    assert result.success is True


def test_execute_updates_last_result():
    executor = CanonicalSingleActionExecutor(FakeBinding([]))
    result = executor.execute(make_request())
    assert executor.last_result is result


def test_execute_rejects_wrong_request_type():
    executor = CanonicalSingleActionExecutor(FakeBinding([]))
    with pytest.raises(TypeError, match="CanonicalSingleActionRequest"):
        executor.execute(object())


def test_normalize_request_builds_agent_boundary_request():
    boundary_request, canonical = executor = CanonicalSingleActionExecutor.normalize_request(
        make_request()
    )
    assert boundary_request.action == canonical.action
    assert boundary_request.position == canonical.position
    assert boundary_request.metadata == canonical.metadata


def test_boundary_request_metadata_isolated_from_original():
    metadata = {"trace": ["a"]}
    request = make_request(metadata=metadata)
    boundary_request, _ = CanonicalSingleActionExecutor.normalize_request(request)
    assert boundary_request.metadata == metadata
    assert boundary_request.metadata is not request.metadata


def test_execute_from_boundary_request_accepts_execution_arguments():
    binding = FakeBinding([])
    executor = CanonicalSingleActionExecutor(binding)
    boundary_request = AgentExecutionRequest(
        action="open_box",
        step={"action": "open_box"},
        position=1,
        goal="open the box",
        world_state={"box": "closed"},
        experience_id="exp-2",
        metadata={"source": "boundary"},
    )

    result = executor.execute_from_boundary_request(
        boundary_request,
        environment="env",
        strategies=["safe"],
        information_gain={"open_box": 0.4},
        action_risk={"open_box": 0.2},
        exploration_risk=0.1,
        action_cost={"open_box": 2.0},
    )

    assert result.request.experience_id == "exp-2"
    assert binding.calls[0]["kwargs"]["environment"] == "env"


def test_execute_from_boundary_request_rejects_unknown_arguments():
    executor = CanonicalSingleActionExecutor(FakeBinding([]))
    boundary_request = AgentExecutionRequest(
        action="open_box",
        step={"action": "open_box"},
        position=1,
    )

    with pytest.raises(TypeError, match="Unsupported execution arguments"):
        executor.execute_from_boundary_request(
            boundary_request,
            unsupported=True,
        )


def test_execute_does_not_call_decide_and_act():
    class AgentLikeBinding(FakeBinding):
        pass

    binding = AgentLikeBinding([])
    executor = CanonicalSingleActionExecutor(binding)
    executor.execute(make_request())
    assert len(binding.calls) == 1


def test_execute_keeps_step_object_identity():
    step = {"action": "open_box", "payload": object()}
    request = make_request(step=step)
    result = CanonicalSingleActionExecutor(FakeBinding([])).execute(request)
    assert result.request.step is step
    assert result.boundary_request.step is step


def test_execute_preserves_goal_and_world_state():
    request = make_request()
    result = CanonicalSingleActionExecutor(FakeBinding([])).execute(request)
    assert result.boundary_request.goal == request.goal
    assert result.boundary_request.world_state == request.world_state


def test_execute_preserves_experience_id():
    request = make_request(experience_id="abc")
    result = CanonicalSingleActionExecutor(FakeBinding([])).execute(request)
    assert result.boundary_request.experience_id == "abc"


def test_success_property_handles_false_dict():
    class FailingBinding(FakeBinding):
        def execute_request(self, request, **kwargs):
            return AgentUnifiedExecutionReport(
                request=request,
                decision={"action": request.action},
                result={"success": False},
            )

    result = CanonicalSingleActionExecutor(FailingBinding([])).execute(make_request())
    assert result.success is False


def test_success_property_handles_truthy_non_mapping():
    class TruthyBinding(FakeBinding):
        def execute_request(self, request, **kwargs):
            return AgentUnifiedExecutionReport(
                request=request,
                decision={"action": request.action},
                result=1,
            )

    result = CanonicalSingleActionExecutor(TruthyBinding([])).execute(make_request())
    assert result.success is True


def test_for_agent_constructs_binding():
    class DummyAgent:
        def __init__(self):
            self.actions = {"open_box": object()}
            self.unified_executor = object()

    def decision_factory(request):
        return {"action": request.action}

    executor = CanonicalSingleActionExecutor.for_agent(
        DummyAgent(),
        decision_factory=decision_factory,
    )
    assert executor.binding is not None


def test_for_agent_rejects_none_agent():
    def decision_factory(request):
        return {"action": request.action}

    with pytest.raises(ValueError, match="agent cannot be None"):
        CanonicalSingleActionExecutor.for_agent(
            None,
            decision_factory=decision_factory,
        )


def test_for_agent_rejects_non_callable_factory():
    class DummyAgent:
        pass

    with pytest.raises(TypeError, match="decision_factory must be callable"):
        CanonicalSingleActionExecutor.for_agent(
            DummyAgent(),
            decision_factory=None,
        )


def test_request_default_collections_are_independent():
    a = CanonicalSingleActionRequest(action="a", step={}, position=1)
    b = CanonicalSingleActionRequest(action="b", step={}, position=1)
    assert a.metadata == {}
    assert b.metadata == {}
    assert a.metadata is not b.metadata
