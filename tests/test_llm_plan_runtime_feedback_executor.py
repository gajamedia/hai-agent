from __future__ import annotations

from dataclasses import dataclass

from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_runtime_feedback_consumption import (
    LLMPlanRuntimeFeedbackAwareBridge,
    LLMPlanRuntimeFeedbackConsumer,
)
from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class FakeResult:
    success: bool
    result: dict


class FakeAPI:
    def __init__(self):
        self.calls = []
        self.agent = type(
            "AgentStub",
            (),
            {"actions": {"open_box": object(), "take_key": object()}},
        )()
        self.results = [FakeResult(True, {"observation": {"box": "open"}})]

    def execute(self, request):
        self.calls.append(request)
        if self.results:
            return self.results.pop(0)
        return FakeResult(True, {})


def make_plan():
    return LLMPlan(
        goal="take the key",
        rationale="Execute two ordered actions.",
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


def make_stop_feedback():
    return LLMPlanRuntimeLearningFeedback(
        step_index=1,
        action="open_box",
        success=False,
        progress_delta=0.0,
        evidence_quality=1.0,
        confidence=1.0,
        diagnosis="unsafe_to_continue",
        lesson="Do not continue the current plan.",
        next_step="Replan before taking another action.",
        change_required=True,
        continue_recommended=False,
        exploration_recommended=False,
        risk_level="high",
        risk_score=0.75,
        information_gain=0.0,
    )


def test_feedback_blocked_step_is_skipped_not_recorded_as_executed():
    api = FakeAPI()
    consumer = LLMPlanRuntimeFeedbackConsumer()
    events = []

    def on_event(event):
        events.append(event)
        if event.phase == "step_recorded" and event.step and event.step.index == 1:
            consumer.consume(make_stop_feedback())

    action_bridge = LLMPlanSingleActionBridge.for_api(api)
    feedback_bridge = LLMPlanRuntimeFeedbackAwareBridge(action_bridge, consumer)
    runtime = LLMPlanRuntimeExecutor(
        feedback_bridge,
        runtime_event_observer=on_event,
    )

    report = runtime.execute(make_plan())

    assert [request.action for request in api.calls] == ["open_box"]
    assert report.stopped is True
    assert report.success is False
    assert report.completed is False
    assert report.stop_reason == "runtime_feedback_stop"
    assert report.next_step_index == 2

    assert [record.step.index for record in report.records] == [1, 2]
    blocked_record = report.records[1]
    assert blocked_record.skipped is True
    assert blocked_record.request is None
    assert blocked_record.success is False
    assert blocked_record.error == "runtime_feedback_stop"

    assert report.executed_step_indices == (1,)
    assert report.failed_step_indices == ()
    assert report.metadata["executed_step_count"] == 1

    # Experience integrators consume step_recorded. A blocked step must not
    # emit that event, or it would be learned as if the action had occurred.
    recorded_indices = [
        event.step.index
        for event in events
        if event.phase == "step_recorded" and event.step is not None
    ]
    assert recorded_indices == [1]
    assert any(
        event.phase == "step_blocked"
        and event.step is not None
        and event.step.index == 2
        for event in events
    )
    assert any(
        event.phase == "stopped"
        and event.stop_reason == "runtime_feedback_stop"
        and event.next_step_index == 2
        for event in events
    )
