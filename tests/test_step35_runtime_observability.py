from __future__ import annotations

from dataclasses import dataclass

from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeEvent,
    LLMPlanRuntimeExecutor,
)
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class FakeResult:
    success: bool
    result: object


class FakeAPI:
    def __init__(self, outcomes, actions=("open_box", "take_key")):
        self.outcomes = list(outcomes)
        self.calls = []
        self.agent = type(
            "AgentStub", (), {"actions": {name: object() for name in actions}}
        )()

    def execute(self, request):
        self.calls.append(request)
        if self.outcomes:
            return self.outcomes.pop(0)
        return FakeResult(True, {})


def make_plan() -> LLMPlan:
    return LLMPlan(
        goal="open the box",
        rationale="Trace observability test plan.",
        steps=(
            LLMPlanStep(
                index=1,
                action="open_box",
                purpose="Open the box.",
                expected_observation="box is open",
                success_criteria="box is open",
                risk="low",
            ),
            LLMPlanStep(
                index=2,
                action="take_key",
                purpose="Take the key.",
                expected_observation="key is held",
                success_criteria="key is held",
                risk="low",
            ),
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
        provider="test",
        model="test-model",
    )


def make_runtime(outcomes, **kwargs):
    actions = kwargs.pop("actions", ("open_box", "take_key"))
    api = FakeAPI(outcomes, actions=actions)
    bridge = LLMPlanSingleActionBridge.for_api(api)
    events: list[LLMPlanRuntimeEvent] = []
    runtime = LLMPlanRuntimeExecutor.for_bridge(
        bridge, runtime_event_observer=events.append, **kwargs
    )
    return runtime, api, events


def test_success_trace_has_ordered_lifecycle_and_consistent_report():
    runtime, api, events = make_runtime(
        [FakeResult(True, {}), FakeResult(True, {})]
    )

    report = runtime.execute(make_plan(), metadata={"run_id": "trace-35"})

    assert [event.phase for event in events] == [
        "started",
        "step_recorded",
        "step_recorded",
        "completed",
    ]
    assert len(api.calls) == len(report.records) == 2
    assert report.success and report.completed and not report.stopped
    assert report.metadata["run_id"] == "trace-35"
    assert report.metadata["executed_step_count"] == 2
    assert report.metadata["planned_step_count"] == 2
    assert [event.record.step.index for event in events if event.phase == "step_recorded"] == [1, 2]
    assert events[-1].metadata["run_id"] == "trace-35"


def test_failed_step_trace_records_failure_then_stop_without_completed_event():
    runtime, api, events = make_runtime(
        [FakeResult(False, {"error": "action rejected"}), FakeResult(True, {})]
    )

    report = runtime.execute(make_plan())

    assert [event.phase for event in events] == ["started", "step_recorded", "stopped"]
    assert report.stopped and not report.success and not report.completed
    assert report.stop_reason == "step_failed"
    assert report.failed_step_indices == (1,)
    assert report.next_step_index == 2
    assert len(api.calls) == len(report.records) == 1
    assert events[1].record is report.records[0]
    assert events[-1].stop_reason == report.stop_reason


def test_executor_exception_is_traceable_and_does_not_execute_later_steps():
    class RaisingAPI(FakeAPI):
        def execute(self, request):
            self.calls.append(request)
            raise RuntimeError("synthetic executor failure")

    api = RaisingAPI([])
    events: list[LLMPlanRuntimeEvent] = []
    runtime = LLMPlanRuntimeExecutor.for_bridge(
        LLMPlanSingleActionBridge.for_api(api),
        runtime_event_observer=events.append,
    )

    report = runtime.execute(make_plan())

    assert [event.phase for event in events] == ["started", "step_recorded", "stopped"]
    assert report.stop_reason == "step_execution_error"
    assert report.records[0].error == "synthetic executor failure"
    assert report.success is False
    assert len(api.calls) == len(report.records) == 1
    assert events[-1].record is report.records[0]


def test_additional_observer_receives_events_once_and_can_be_removed():
    runtime, _, primary_events = make_runtime(
        [FakeResult(True, {}), FakeResult(True, {})]
    )
    secondary_events: list[LLMPlanRuntimeEvent] = []
    observer = secondary_events.append
    runtime.add_event_observer(observer)
    runtime.add_event_observer(observer)

    runtime.execute(make_plan())

    assert [event.phase for event in secondary_events] == [
        "started",
        "step_recorded",
        "step_recorded",
        "completed",
    ]
    assert len(secondary_events) == len(primary_events)

    runtime.remove_event_observer(observer)
    secondary_events.clear()
    runtime.execute(make_plan())
    assert secondary_events == []


def test_invalid_plan_emits_no_execution_trace_and_makes_no_action_calls():
    runtime, api, events = make_runtime([FakeResult(True, {})], actions=("open_box",))

    try:
        runtime.execute(make_plan())
    except Exception as exc:
        assert "validation" in type(exc).__name__.lower()
    else:
        raise AssertionError("An invalid plan should fail validation before execution.")

    assert api.calls == []
    assert events == []
    assert runtime.last_report is None
