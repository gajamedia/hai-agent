from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeConfigurationError,
    LLMPlanRuntimeExecutor,
    LLMPlanRuntimeValidationError,
    LLMPlanStepExecutionRecord,
)
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class FakeResult:
    success: bool
    result: object


class FakeAPI:
    def __init__(self, actions=("open_box", "take_key")):
        self.calls = []
        self.agent = type("AgentStub", (), {"actions": {name: object() for name in actions}})()
        self.results = [FakeResult(True, {"observation": {"box": "open"}})]

    def execute(self, request):
        self.calls.append(request)
        if self.results:
            return self.results.pop(0)
        return FakeResult(True, {})


def make_plan(*, risk1="low", risk2=None) -> LLMPlan:
    return LLMPlan(
        goal="open the box",
        rationale="Complete the goal in order.",
        steps=(
            LLMPlanStep(
                index=1,
                action="open_box",
                purpose="Open the box.",
                expected_observation="box is open",
                success_criteria="box is open",
                risk=risk1,
            ),
            LLMPlanStep(
                index=2,
                action="take_key",
                purpose="Take the key.",
                expected_observation="key is held",
                success_criteria="key is held",
                risk=risk2,
            ),
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
        provider="test",
        model="test-model",
    )


def make_runtime(api=None, **kwargs):
    api = api or FakeAPI()
    bridge = LLMPlanSingleActionBridge.for_api(api)
    runtime = LLMPlanRuntimeExecutor.for_bridge(bridge, **kwargs)
    return runtime, api


def test_rejects_missing_bridge():
    with pytest.raises(LLMPlanRuntimeConfigurationError):
        LLMPlanRuntimeExecutor(None)


def test_rejects_invalid_validator():
    bridge = LLMPlanSingleActionBridge(FakeAPI())
    with pytest.raises(LLMPlanRuntimeConfigurationError):
        LLMPlanRuntimeExecutor(bridge, validator=object())


def test_rejects_invalid_stop_evaluator():
    bridge = LLMPlanSingleActionBridge(FakeAPI())
    with pytest.raises(LLMPlanRuntimeConfigurationError):
        LLMPlanRuntimeExecutor(bridge, stop_condition_evaluator=object())


def test_rejects_non_plan():
    runtime, _ = make_runtime()
    with pytest.raises(TypeError):
        runtime.execute(object())


def test_validates_before_first_execution():
    api = FakeAPI(actions=("open_box",))
    runtime, api = make_runtime(api)
    with pytest.raises(LLMPlanRuntimeValidationError) as exc:
        runtime.execute(make_plan())
    assert not api.calls
    assert not exc.value.validation.valid


def test_executes_all_steps_in_order():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime, api = make_runtime(api)

    report = runtime.execute(make_plan())

    assert report.success
    assert report.completed
    assert not report.stopped
    assert report.executed_step_indices == (1, 2)
    assert [call.action for call in api.calls] == ["open_box", "take_key"]


def test_routes_each_step_through_bridge_once():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime, api = make_runtime(api)
    report = runtime.execute(make_plan())

    assert len(api.calls) == len(report.records) == 2
    assert all(isinstance(record, LLMPlanStepExecutionRecord) for record in report.records)


def test_stops_after_failed_step():
    api = FakeAPI()
    api.results = [FakeResult(False, {"error": "failed"}), FakeResult(True, {})]
    runtime, api = make_runtime(api)

    report = runtime.execute(make_plan())

    assert report.stopped
    assert not report.success
    assert report.stop_reason == "step_failed"
    assert report.executed_step_indices == (1,)
    assert len(api.calls) == 1
    assert report.next_step_index == 2


def test_can_continue_after_failure_when_stop_disabled():
    api = FakeAPI()
    api.results = [FakeResult(False, {}), FakeResult(True, {})]
    runtime, api = make_runtime(api, stop_on_step_failure=False)

    report = runtime.execute(make_plan())

    assert report.completed is False
    assert report.stopped is False
    assert report.success is False
    assert len(api.calls) == 2
    assert report.failed_step_indices == (1,)


def test_custom_success_evaluator_can_override_result():
    api = FakeAPI()
    api.results = [FakeResult(False, {"semantic_success": True}), FakeResult(True, {})]
    runtime, _ = make_runtime(
        api,
        step_success_evaluator=lambda step, result, context: (
            bool(result.result.get("semantic_success"))
            if step.index == 1
            else result.success
        ),
    )

    report = runtime.execute(make_plan())
    assert report.success


def test_world_state_is_observed_between_steps():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    observations = []

    class Environment:
        def observe(self):
            value = {"tick": len(observations) + 1}
            observations.append(value)
            return value

    runtime, _ = make_runtime(api)
    report = runtime.execute(make_plan(), environment=Environment())

    assert observations == [{"tick": 1}, {"tick": 2}]
    assert report.final_world_state == {"tick": 2}


def test_result_mapping_can_supply_observation_without_environment():
    api = FakeAPI()
    api.results = [
        FakeResult(True, {"observation": {"box": "open"}}),
        FakeResult(True, {"observation": {"key": "held"}}),
    ]
    runtime, _ = make_runtime(api)

    report = runtime.execute(make_plan())
    assert report.final_world_state == {"key": "held"}


def test_stop_condition_can_stop_before_next_step():
    api = FakeAPI()
    api.results = [FakeResult(True, {})]
    runtime, api = make_runtime(
        api,
        stop_condition_evaluator=lambda condition, context: (
            condition == "stop after first"
            and context["phase"] == "after_step"
            and context["step"].index == 1
        ),
    )
    plan = LLMPlan(
        goal="open the box",
        rationale="stop test",
        steps=make_plan().steps,
        assumptions=(),
        constraints=(),
        stop_conditions=("stop after first",),
        confidence=0.9,
    )

    report = runtime.execute(plan)
    assert report.stopped
    assert report.stop_reason == "stop_condition:stop after first"
    assert report.next_step_index == 2
    assert len(api.calls) == 1
    assert report.records[-1].stop_requested


def test_stop_condition_can_stop_before_first_step():
    runtime, api = make_runtime(
        stop_condition_evaluator=lambda condition, context: condition == "do not run"
    )
    plan = LLMPlan(
        goal="open the box",
        rationale="stop test",
        steps=make_plan().steps,
        assumptions=(),
        constraints=(),
        stop_conditions=("do not run",),
        confidence=0.9,
    )

    report = runtime.execute(plan)
    assert report.stopped
    assert report.stop_reason == "stop_condition:do not run"
    assert report.next_step_index == 1
    assert not api.calls


def test_step_execution_exception_is_recorded_and_stops():
    class RaisingAPI(FakeAPI):
        def execute(self, request):
            self.calls.append(request)
            raise RuntimeError("boom")

    api = RaisingAPI()
    runtime, _ = make_runtime(api)

    report = runtime.execute(make_plan())
    assert report.stopped
    assert report.stop_reason == "step_execution_error"
    assert report.records[0].error == "boom"
    assert report.success is False


def test_step_execution_exception_can_be_reraised():
    class RaisingAPI(FakeAPI):
        def execute(self, request):
            raise RuntimeError("boom")

    runtime, _ = make_runtime(RaisingAPI(), raise_on_step_error=True)
    with pytest.raises(RuntimeError, match="boom"):
        runtime.execute(make_plan())


def test_last_report_is_saved():
    runtime, _ = make_runtime()
    report = runtime.execute(make_plan())
    assert runtime.last_report is report


def test_plan_is_not_mutated():
    runtime, _ = make_runtime()
    plan = make_plan()
    before = plan.to_dict()
    runtime.execute(plan)
    assert plan.to_dict() == before


def test_runtime_metadata_reaches_bridge_request():
    runtime, api = make_runtime()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime.execute(make_plan(), metadata={"run_id": 7})
    first = api.calls[0]
    assert first.metadata["run_id"] == 7
    assert first.metadata["runtime_executor"] == "llm_plan_runtime_execution"


def test_action_risk_uses_numeric_projection_from_textual_step_risk():
    runtime, api = make_runtime()
    api.results = [FakeResult(True, {})]
    report = runtime.execute(
        LLMPlan(
            goal="open the box",
            rationale="single step",
            steps=(make_plan(risk1="high").steps[0],),
            assumptions=(),
            constraints=(),
            stop_conditions=(),
            confidence=0.9,
        )
    )
    assert report.success
    assert api.calls[0].action_risk["open_box"] == 0.75


def test_explicit_action_risk_overrides_projected_risk():
    runtime, api = make_runtime()
    api.results = [FakeResult(True, {})]
    plan = LLMPlan(
        goal="open the box",
        rationale="single step",
        steps=(make_plan(risk1="high").steps[0],),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
    )
    report = runtime.execute(plan, action_risk={"open_box": 0.2})
    assert report.success
    assert api.calls[0].action_risk["open_box"] == 0.2


def test_explicit_available_actions_are_used_for_validation():
    runtime, api = make_runtime(FakeAPI(actions=("open_box", "take_key")))
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    report = runtime.execute(make_plan(), available_actions=["open_box", "take_key"])
    assert report.success


def test_missing_available_action_fails_before_execution():
    api = FakeAPI(actions=("open_box",))
    runtime, api = make_runtime(api)
    with pytest.raises(LLMPlanRuntimeValidationError):
        runtime.execute(make_plan())
    assert not api.calls


def test_goal_mismatch_fails_before_execution():
    runtime, api = make_runtime()
    with pytest.raises(LLMPlanRuntimeValidationError):
        runtime.execute(make_plan(), goal="different goal")
    assert not api.calls


def test_low_confidence_plan_fails_when_validator_requires_more_confidence():
    from agent.llm.plan_validation import LLMPlanValidator

    runtime, api = make_runtime(
        validator=LLMPlanValidator(min_confidence=0.95)
    )
    with pytest.raises(LLMPlanRuntimeValidationError):
        runtime.execute(make_plan())
    assert not api.calls


def test_report_exposes_validation_result():
    runtime, _ = make_runtime()
    report = runtime.execute(make_plan())
    assert report.validation.valid
    assert report.validation.checked_step_count == 2
    assert report.validation.executable_step_indices == (1, 2)


def test_final_world_state_defaults_to_input_when_no_observation_exists():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime, _ = make_runtime(api)
    report = runtime.execute(make_plan(), world_state={"box": "closed"})
    assert report.final_world_state == {"box": "closed"}


def test_experience_id_reaches_each_step():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime, api = make_runtime(api)
    runtime.execute(make_plan(), experience_id="exp-24-6")
    assert [call.experience_id for call in api.calls] == ["exp-24-6", "exp-24-6"]


def test_records_keep_canonical_bridge_request_trace():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime, _ = make_runtime(api)
    report = runtime.execute(make_plan())
    assert report.records[0].request is not None
    assert report.records[0].request.request.action == "open_box"


def test_success_means_all_steps_successful():
    api = FakeAPI()
    api.results = [FakeResult(True, {}), FakeResult(False, {})]
    runtime, _ = make_runtime(api)
    report = runtime.execute(make_plan())
    assert report.success is False
    assert report.completed is False


def test_single_step_plan_completes_cleanly():
    api = FakeAPI()
    api.results = [FakeResult(True, {})]
    runtime, _ = make_runtime(api)
    plan = LLMPlan(
        goal="open the box",
        rationale="single step",
        steps=(make_plan().steps[0],),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
    )
    report = runtime.execute(plan)
    assert report.success
    assert report.next_step_index is None


def test_runtime_does_not_call_agent_decide_and_act():
    api = FakeAPI()
    agent = api.agent
    agent.decide_and_act_called = False
    agent.decide_and_act = lambda *args, **kwargs: setattr(agent, "decide_and_act_called", True)
    runtime, _ = make_runtime(api)
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime.execute(make_plan())
    assert agent.decide_and_act_called is False


def test_runtime_does_not_call_an_llm_service():
    api = FakeAPI()
    api.agent.llm_complete_called = False
    api.agent.llm_complete = lambda *args, **kwargs: setattr(api.agent, "llm_complete_called", True)
    runtime, _ = make_runtime(api)
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    runtime.execute(make_plan())
    assert api.agent.llm_complete_called is False
