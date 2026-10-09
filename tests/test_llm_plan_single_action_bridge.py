from __future__ import annotations

from dataclasses import replace

import pytest

from agent.llm.agent_canonical_single_action_api import (
    AgentCanonicalSingleActionAPI,
)
from agent.llm.canonical_single_action import CanonicalSingleActionResult
from agent.llm.llm_plan_single_action_bridge import (
    LLMPlanSingleActionBridge,
    LLMPlanSingleActionBridgeConfigurationError,
    LLMPlanSingleActionBridgeRequestError,
    LLMPlanSingleActionExecutionRequest,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.result = object()

    def execute(self, request):
        self.calls.append(request)
        return self.result


def make_plan() -> LLMPlan:
    return LLMPlan(
        goal="open the box",
        rationale="Use the available action.",
        steps=(
            LLMPlanStep(
                index=1,
                action="open_box",
                purpose="Open the box.",
                preconditions=("box is closed",),
                expected_observation="box is open",
                success_criteria="box state is open",
                risk="low",
            ),
            LLMPlanStep(
                index=2,
                action="take_key",
                purpose="Take the key.",
            ),
        ),
        assumptions=("key exists",),
        constraints=("do not force the lid",),
        stop_conditions=("stop if box is missing",),
        confidence=0.8,
        provider="opencode",
        model="test-model",
    )


def make_bridge():
    api = FakeAPI()
    bridge = LLMPlanSingleActionBridge(api)
    return bridge, api


def test_rejects_none_api():
    with pytest.raises(LLMPlanSingleActionBridgeConfigurationError):
        LLMPlanSingleActionBridge(None)


def test_rejects_api_without_execute():
    with pytest.raises(LLMPlanSingleActionBridgeConfigurationError):
        LLMPlanSingleActionBridge(object())


def test_for_api_builds_bridge():
    api = FakeAPI()
    bridge = LLMPlanSingleActionBridge.for_api(api)
    assert bridge.api is api


def test_build_request_from_step_object():
    bridge, _ = make_bridge()
    plan = make_plan()
    trace = bridge.build_request(plan, plan.steps[0])
    assert isinstance(trace, LLMPlanSingleActionExecutionRequest)
    assert trace.action == "open_box"
    assert trace.position == 1
    assert trace.request.action == "open_box"


def test_build_request_from_step_index():
    bridge, _ = make_bridge()
    trace = bridge.build_request(make_plan(), 2)
    assert trace.step.action == "take_key"
    assert trace.request.position == 2


def test_defaults_goal_to_plan_goal():
    bridge, _ = make_bridge()
    trace = bridge.build_request(make_plan(), 1)
    assert trace.request.goal == "open the box"


def test_explicit_goal_overrides_plan_goal():
    bridge, _ = make_bridge()
    trace = bridge.build_request(make_plan(), 1, goal="different goal")
    assert trace.request.goal == "different goal"


def test_preserves_runtime_inputs():
    bridge, _ = make_bridge()
    world = {"box": "closed"}
    env = object()
    strategies = [object()]
    trace = bridge.build_request(
        make_plan(),
        1,
        world_state=world,
        environment=env,
        strategies=strategies,
        information_gain={"open_box": 0.6},
        action_risk={"open_box": 0.2},
        exploration_risk=0.3,
        action_cost={"open_box": 0.1},
        experience_id="exp-1",
    )
    assert trace.request.world_state == world
    assert trace.request.environment is env
    assert trace.request.strategies == strategies
    assert trace.request.information_gain == {"open_box": 0.6}
    assert trace.request.action_risk == {"open_box": 0.2}
    assert trace.request.exploration_risk == 0.3
    assert trace.request.action_cost == {"open_box": 0.1}
    assert trace.request.experience_id == "exp-1"


def test_adds_llm_provenance_metadata():
    bridge, _ = make_bridge()
    trace = bridge.build_request(make_plan(), 1, metadata={"run": 4})
    metadata = trace.request.metadata
    assert metadata["source"] == "llm_plan_single_action_bridge"
    assert metadata["llm_plan_goal"] == "open the box"
    assert metadata["llm_plan_confidence"] == 0.8
    assert metadata["llm_plan_provider"] == "opencode"
    assert metadata["llm_plan_model"] == "test-model"
    assert metadata["llm_plan_step_index"] == 1
    assert metadata["run"] == 4


def test_step_object_must_match_plan():
    bridge, _ = make_bridge()
    foreign_step = replace(make_plan().steps[0], purpose="different")
    with pytest.raises(LLMPlanSingleActionBridgeRequestError):
        bridge.build_request(make_plan(), foreign_step)


def test_missing_step_index_is_rejected():
    bridge, _ = make_bridge()
    with pytest.raises(LLMPlanSingleActionBridgeRequestError):
        bridge.build_request(make_plan(), 9)


def test_wrong_step_type_is_rejected():
    bridge, _ = make_bridge()
    with pytest.raises(TypeError):
        bridge.build_request(make_plan(), "1")


def test_bool_step_index_is_rejected():
    bridge, _ = make_bridge()
    with pytest.raises(LLMPlanSingleActionBridgeRequestError):
        bridge.build_request(make_plan(), True)


def test_execute_step_calls_api_exactly_once():
    bridge, api = make_bridge()
    result = bridge.execute_step(make_plan(), 1)
    assert len(api.calls) == 1
    assert api.calls[0].action == "open_box"
    assert result is api.result


def test_execute_step_records_trace_and_result():
    bridge, api = make_bridge()
    result = bridge.execute_step(make_plan(), 2)
    assert bridge.last_result is result
    assert bridge.last_execution_request is not None
    assert bridge.last_execution_request.step.action == "take_key"
    assert len(api.calls) == 1


def test_execute_step_index_is_alias():
    bridge, api = make_bridge()
    result = bridge.execute_step_index(make_plan(), 2)
    assert result is api.result
    assert api.calls[0].position == 2


def test_plan_is_not_mutated_by_build():
    bridge, _ = make_bridge()
    plan = make_plan()
    before = plan.to_dict()
    bridge.build_request(plan, 1)
    assert plan.to_dict() == before


def test_plan_is_not_mutated_by_execute():
    bridge, _ = make_bridge()
    plan = make_plan()
    before = plan.to_dict()
    bridge.execute_step(plan, 1)
    assert plan.to_dict() == before


def test_build_keeps_step_object_in_canonical_request():
    bridge, _ = make_bridge()
    plan = make_plan()
    trace = bridge.build_request(plan, 1)
    assert trace.request.step is trace.step


def test_canonical_request_position_comes_from_plan_step():
    bridge, _ = make_bridge()
    trace = bridge.build_request(make_plan(), 2)
    assert trace.request.position == trace.step.index == 2


def test_bridge_does_not_execute_during_build():
    bridge, api = make_bridge()
    bridge.build_request(make_plan(), 1)
    assert api.calls == []


def test_bridge_does_not_need_agent_directly():
    bridge, _ = make_bridge()
    assert not hasattr(bridge, "agent")


def test_execute_result_is_opaque_to_bridge():
    bridge, api = make_bridge()
    opaque = object()
    api.result = opaque
    assert bridge.execute_step(make_plan(), 1) is opaque


def test_metadata_is_copied_without_mutating_caller_mapping():
    bridge, _ = make_bridge()
    metadata = {"custom": "value"}
    trace = bridge.build_request(make_plan(), 1, metadata=metadata)
    metadata["custom"] = "changed"
    assert trace.request.metadata["custom"] == "value"


def test_mapping_inputs_are_copied():
    bridge, _ = make_bridge()
    gains = {"open_box": 0.5}
    trace = bridge.build_request(make_plan(), 1, information_gain=gains)
    gains["open_box"] = 0.9
    assert trace.request.information_gain["open_box"] == 0.5


def test_execute_requires_no_multi_step_loop():
    bridge, api = make_bridge()
    bridge.execute_step(make_plan(), 1)
    assert len(api.calls) == 1
    assert api.calls[0].action == "open_box"
