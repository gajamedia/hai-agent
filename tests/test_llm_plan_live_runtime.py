from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.llm_plan_live_runtime import (
    LLMPlanLiveRuntimeConfigurationError,
    LLMPlanLiveRuntimeExecutionResult,
    LLMPlanLiveRuntimeExecutor,
    LLMPlanLiveRuntimeStateError,
)
from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeEvent,
    LLMPlanRuntimeExecutor,
)
from agent.llm.llm_plan_runtime_state import (
    LLMPlanRuntimeStateTracker,
    LLMPlanRuntimeStatus,
)
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.plan_validation import LLMPlanValidationResult
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class FakeResult:
    success: bool
    result: object


class FakeAPI:
    def __init__(self, actions=("open_box", "take_key", "finish")):
        self.calls = []
        self.agent = type(
            "AgentStub",
            (),
            {"actions": {name: object() for name in actions}},
        )()
        self.results = []

    def execute(self, request):
        self.calls.append(request)
        if self.results:
            return self.results.pop(0)
        return FakeResult(True, {})


def make_plan(n=3) -> LLMPlan:
    return LLMPlan(
        goal="test goal",
        rationale="execute steps",
        steps=tuple(
            LLMPlanStep(
                index=i,
                action=("open_box", "take_key", "finish")[i - 1],
                purpose=f"purpose {i}",
                risk="low",
            )
            for i in range(1, n + 1)
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
    )


def make_runtime(api=None, **kwargs):
    api = api or FakeAPI()
    bridge = LLMPlanSingleActionBridge.for_api(api)
    runtime = LLMPlanRuntimeExecutor.for_bridge(bridge, **kwargs)
    return runtime, api


def make_live(api=None, **kwargs):
    state_observer = kwargs.pop("state_observer", None)
    runtime, api = make_runtime(api, **kwargs)
    live = LLMPlanLiveRuntimeExecutor(
        runtime,
        state_observer=state_observer,
    )
    return live, runtime, api


def test_runtime_event_is_immutable():
    event = LLMPlanRuntimeEvent("started", make_plan())
    with pytest.raises(Exception):
        event.phase = "x"


def test_runtime_event_observer_is_configurable():
    runtime, _ = make_runtime(runtime_event_observer=lambda event: None)
    assert runtime.runtime_event_observer is not None


def test_runtime_event_observer_must_be_callable():
    api = FakeAPI()
    bridge = LLMPlanSingleActionBridge.for_api(api)
    with pytest.raises(Exception):
        LLMPlanRuntimeExecutor(bridge, runtime_event_observer=object())


def test_add_event_observer_requires_callable():
    runtime, _ = make_runtime()
    with pytest.raises(TypeError):
        runtime.add_event_observer(object())


def test_remove_missing_event_observer_is_safe():
    runtime, _ = make_runtime()
    runtime.remove_event_observer(lambda event: None)


def test_event_observer_receives_started_and_completed():
    events = []
    runtime, api = make_runtime(runtime_event_observer=events.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    runtime.execute(make_plan())
    assert [event.phase for event in events] == [
        "started",
        "step_recorded",
        "step_recorded",
        "step_recorded",
        "completed",
    ]


def test_event_observer_receives_stop_on_step_failure():
    events = []
    runtime, api = make_runtime(runtime_event_observer=events.append)
    api.results = [FakeResult(False, {"error": "x"})]
    report = runtime.execute(make_plan())
    assert report.stopped
    assert [event.phase for event in events] == [
        "started",
        "step_recorded",
        "stopped",
    ]


def test_event_observer_can_be_added_without_replacing_primary():
    primary, extra = [], []
    runtime, api = make_runtime(runtime_event_observer=primary.append)
    runtime.add_event_observer(extra.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    runtime.execute(make_plan())
    assert len(primary) == len(extra) == 5


def test_event_observer_can_be_removed():
    primary, extra = [], []
    runtime, api = make_runtime(runtime_event_observer=primary.append)
    runtime.add_event_observer(extra.append)
    runtime.remove_event_observer(extra.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    runtime.execute(make_plan())
    assert len(primary) == 5
    assert extra == []


def test_live_executor_requires_runtime_executor():
    with pytest.raises(LLMPlanLiveRuntimeConfigurationError):
        LLMPlanLiveRuntimeExecutor(None)


def test_live_executor_requires_event_support():
    with pytest.raises(LLMPlanLiveRuntimeConfigurationError):
        LLMPlanLiveRuntimeExecutor(object())


def test_live_executor_rejects_invalid_tracker():
    runtime, _ = make_runtime()
    with pytest.raises(LLMPlanLiveRuntimeConfigurationError):
        LLMPlanLiveRuntimeExecutor(runtime, state_tracker=object())


def test_live_executor_rejects_invalid_state_observer():
    runtime, _ = make_runtime()
    with pytest.raises(LLMPlanLiveRuntimeConfigurationError):
        LLMPlanLiveRuntimeExecutor(runtime, state_observer=object())


def test_live_execute_returns_live_result():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert isinstance(result, LLMPlanLiveRuntimeExecutionResult)
    assert result.success is True
    assert result.completed is True
    assert result.progress == 1.0


def test_live_state_is_running_after_started_event():
    states = []
    live, _, api = make_live(state_observer=states.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    live.execute(make_plan())
    assert states[0].status is LLMPlanRuntimeStatus.RUNNING
    assert states[0].next_step_index == 1


def test_live_state_updates_after_each_step():
    states = []
    live, _, api = make_live(state_observer=states.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    live.execute(make_plan())
    assert [state.attempted_step_indices for state in states[:4]] == [
        (),
        (1,),
        (1, 2),
        (1, 2, 3),
    ]


def test_live_state_history_contains_terminal_completed_state():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert result.state_history[-1].status is LLMPlanRuntimeStatus.COMPLETED
    assert result.state_history[-1].success is True


def test_live_state_stops_after_failure():
    live, _, api = make_live()
    api.results = [FakeResult(False, {})]
    result = live.execute(make_plan())
    assert result.stopped is True
    assert result.final_state.status is LLMPlanRuntimeStatus.STOPPED
    assert result.final_state.failed_step_indices == (1,)
    assert result.final_state.next_step_index == 2


def test_live_state_stop_reason_matches_report():
    live, _, api = make_live()
    api.results = [FakeResult(False, {})]
    result = live.execute(make_plan())
    assert result.final_state.stop_reason == result.report.stop_reason


def test_live_state_tracks_observation():
    live, _, api = make_live()
    api.results = [
        FakeResult(True, {"observation": {"tick": 1}}),
        FakeResult(True, {"observation": {"tick": 2}}),
        FakeResult(True, {"observation": {"tick": 3}}),
    ]
    result = live.execute(make_plan())
    assert result.final_state.last_observation == {"tick": 3}


def test_live_state_observer_called_for_each_state():
    states = []
    live, _, api = make_live(state_observer=states.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert len(states) == len(result.state_history)


def test_live_state_observer_receives_immutable_snapshots():
    states = []
    live, _, api = make_live(state_observer=states.append)
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    live.execute(make_plan())
    assert states[0] is not states[-1]
    assert states[0].attempted_step_indices == ()


def test_live_executor_resets_tracker_between_runs():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    first = live.execute(make_plan())
    api.results = [FakeResult(True, {}), FakeResult(True, {})]
    second = live.execute(make_plan(2))
    assert first.final_state.planned_step_count == 3
    assert second.final_state.planned_step_count == 2
    assert second.final_state.attempted_step_indices == (1, 2)


def test_live_last_result_is_updated():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert live.last_result is result


def test_live_last_state_matches_result():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert live.last_state == result.final_state


def test_live_report_only_returns_underlying_report():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    report = live.execute_report_only(make_plan())
    assert report.completed
    assert live.last_result is not None
    assert live.last_result.report is report


def test_live_rejects_invalid_plan_before_tracker_start():
    live, _, api = make_live()
    invalid = LLMPlan(
        goal="test goal",
        rationale="invalid",
        steps=(LLMPlanStep(1, "missing", "x"),),
        confidence=0.9,
    )
    with pytest.raises(Exception):
        live.execute(invalid)
    assert live.last_state.status is LLMPlanRuntimeStatus.IDLE
    assert not api.calls


def test_live_does_not_execute_when_validation_fails():
    live, _, api = make_live()
    invalid = LLMPlan(
        goal="test goal",
        rationale="invalid",
        steps=(LLMPlanStep(1, "missing", "x"),),
        confidence=0.9,
    )
    with pytest.raises(Exception):
        live.execute(invalid)
    assert api.calls == []


def test_live_is_single_execution_engine():
    live, runtime, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert result.report is runtime.last_report


def test_live_preserves_underlying_executor_failure_configuration():
    live, _, api = make_live(stop_on_step_failure=False)
    api.results = [FakeResult(False, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert result.report.stopped is False
    assert result.final_state.status is LLMPlanRuntimeStatus.FAILED


def test_live_is_reentrant_sequentially():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    first = live.execute(make_plan())
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    second = live.execute(make_plan())
    assert first.final_state.completed and second.final_state.completed


def test_live_exposes_success_property_from_report():
    live, _, api = make_live()
    api.results = [FakeResult(True, {}), FakeResult(True, {}), FakeResult(True, {})]
    result = live.execute(make_plan())
    assert result.success == result.report.success


def test_live_exposes_next_step_from_tracker():
    live, _, api = make_live()
    api.results = [FakeResult(False, {})]
    result = live.execute(make_plan())
    assert result.next_step_index == 2


def test_state_error_type_is_exported_and_usable():
    assert issubclass(LLMPlanLiveRuntimeStateError, RuntimeError)
