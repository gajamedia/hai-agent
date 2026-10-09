from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent.llm.llm_plan_adaptive_replanning import LLMPlanAdaptiveReplanner
from agent.llm.llm_plan_adaptive_runtime import (
    LLMPlanAdaptiveRuntime,
    LLMPlanAdaptiveRuntimeConfigurationError,
    LLMPlanAdaptiveRuntimeReplanningError,
    LLMPlanAdaptiveRuntimeSafetyError,
)
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class Result:
    success: bool
    result: dict[str, Any]


class FakeBridge:
    def __init__(self, *, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []
        self.last_execution_request = None

    def execute_step(self, plan, step, **kwargs):
        self.calls.append(step.action)
        success = step.action not in self.failures
        return Result(success=success, result={"world_state": {"last_action": step.action}})


class FeedbackSource:
    def __init__(self) -> None:
        self.by_step: dict[int, LLMPlanRuntimeLearningFeedback] = {}

    @property
    def last_feedback(self):
        if not self.by_step:
            return None
        return self.by_step[max(self.by_step)]

    def get_feedback(self, step_index: int):
        return self.by_step.get(step_index)



def feedback(
    step_index: int,
    *,
    action: str = "action",
    success: bool | None = True,
    progress_delta: float = 0.0,
    change_required: bool = True,
    continue_recommended: bool = True,
    exploration_recommended: bool = True,
    risk_level: str = "medium",
) -> LLMPlanRuntimeLearningFeedback:
    return LLMPlanRuntimeLearningFeedback(
        step_index=step_index,
        action=action,
        success=success,
        progress_delta=progress_delta,
        evidence_quality=0.8,
        confidence=0.8,
        diagnosis="test",
        lesson="adjust",
        next_step="replan",
        change_required=change_required,
        continue_recommended=continue_recommended,
        exploration_recommended=exploration_recommended,
        risk_level=risk_level,
        risk_score=0.5,
        information_gain=max(progress_delta, 0.0),
    )


def plan(goal: str = "reach goal", actions: tuple[str, ...] = ("a", "b", "c")) -> LLMPlan:
    return LLMPlan(
        goal=goal,
        rationale="test",
        steps=tuple(
            LLMPlanStep(
                index=i,
                action=action,
                purpose=f"purpose-{action}",
                expected_observation=f"obs-{action}",
                success_criteria=f"done-{action}",
                risk=0.25,
            )
            for i, action in enumerate(actions, start=1)
        ),
        confidence=0.9,
    )


def replacement_plan(goal: str, actions: tuple[str, ...]) -> LLMPlan:
    return plan(goal, actions)


def make_runtime(bridge: FakeBridge, *, stop_condition_evaluator=None):
    return LLMPlanRuntimeExecutor(
        bridge,
        stop_condition_evaluator=stop_condition_evaluator,
    )


def test_requires_runtime_event_support():
    class NoEvents:
        def execute(self, plan, **kwargs):
            return None

    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    with pytest.raises(LLMPlanAdaptiveRuntimeConfigurationError):
        LLMPlanAdaptiveRuntime(NoEvents(), replanner)


def test_requires_replanner():
    runtime = make_runtime(FakeBridge())
    with pytest.raises(LLMPlanAdaptiveRuntimeConfigurationError):
        LLMPlanAdaptiveRuntime(runtime, object())


def test_invalid_feedback_source_rejected():
    runtime = make_runtime(FakeBridge())
    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    with pytest.raises(LLMPlanAdaptiveRuntimeConfigurationError):
        LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=object())


def test_invalid_max_segments_rejected():
    runtime = make_runtime(FakeBridge())
    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    with pytest.raises(LLMPlanAdaptiveRuntimeConfigurationError):
        LLMPlanAdaptiveRuntime(runtime, replanner, max_segments=0)


def test_no_feedback_runs_single_segment_successfully():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=FeedbackSource())

    result = controller.execute(plan())

    assert result.success is True
    assert result.completed is True
    assert result.replanned is False
    assert result.segment_count == 1
    assert bridge.calls == ["a", "b", "c"]


def test_feedback_after_step_causes_safe_boundary_and_replan():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1, action="a")
    seen = []

    def provider(request):
        seen.append(request)
        return replacement_plan(request.goal, ("x", "y"))

    replanner = LLMPlanAdaptiveReplanner(provider)
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    result = controller.execute(plan())

    assert result.success is True
    assert result.replanned is True
    assert result.segment_count == 2
    assert result.completed_successful_step_indices == (1, 2, 3)
    assert bridge.calls == ["a", "x", "y"]
    assert seen[0].preserved_step_indices == (1,)
    assert seen[0].remaining_step_indices == (2, 3)
    assert result.replans[0].replacement_step_indices == (2, 3)


def test_original_plan_is_not_mutated_by_internal_stop_marker():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    original = plan()
    original_conditions = original.stop_conditions
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    controller.execute(original)

    assert original.stop_conditions == original_conditions == ()


def test_runtime_stop_evaluator_is_restored_after_execution():
    bridge = FakeBridge()
    called = []

    def original(condition, context):
        called.append(condition)
        return False

    runtime = make_runtime(bridge, stop_condition_evaluator=original)
    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(runtime, replanner)

    controller.execute(plan())

    assert runtime.stop_condition_evaluator is original
    assert called == ["__adaptive_runtime_replanning__"] * 6


def test_existing_stop_condition_still_wins_after_step_and_can_coexist_with_replan():
    bridge = FakeBridge()

    def stop_after_step(condition, context):
        return condition == "halt" and context["phase"] == "after_step"

    runtime = make_runtime(bridge, stop_condition_evaluator=stop_after_step)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    replanner = LLMPlanAdaptiveReplanner(
        lambda request: replacement_plan(request.goal, ("x",))
    )
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)
    original = plan()
    original = LLMPlan(
        goal=original.goal,
        rationale=original.rationale,
        steps=original.steps,
        stop_conditions=("halt",),
        confidence=original.confidence,
    )

    result = controller.execute(original)

    assert result.replanned is True
    assert bridge.calls == ["a", "x"]


def test_feedback_source_must_match_recorded_step():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[99] = feedback(99)
    replanner = LLMPlanAdaptiveReplanner(lambda request: plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    result = controller.execute(plan())

    assert result.success is True
    assert result.replanned is False


def test_failed_step_can_trigger_replan_and_failed_step_is_not_preserved():
    bridge = FakeBridge(failures={"b"})
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[2] = feedback(2, action="b", success=False, progress_delta=0.0)
    seen = []

    def provider(request):
        seen.append(request)
        return replacement_plan(request.goal, ("recover", "finish"))

    replanner = LLMPlanAdaptiveReplanner(provider)
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    result = controller.execute(plan())

    assert result.success is True
    assert result.replanned is True
    assert seen[0].preserved_step_indices == (1,)
    assert seen[0].failed_step_index == 2
    assert result.replans[0].replacement_step_indices == (3, 4)
    assert bridge.calls == ["a", "b", "recover", "finish"]


def test_successful_steps_are_aggregated_once_across_generations():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    replanner = LLMPlanAdaptiveReplanner(lambda request: replacement_plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    result = controller.execute(plan())

    assert result.completed_successful_step_indices == (1, 2)
    assert result.generation_count == 2


def test_replan_budget_prevents_second_replan_when_budget_is_zero():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    replanner = LLMPlanAdaptiveReplanner(lambda request: replacement_plan(request.goal, ("x",)), max_replans=0)
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    result = controller.execute(plan())

    assert result.success is True
    assert result.replanned is False
    assert bridge.calls == ["a", "b", "c"]


def test_global_completed_step_repetition_is_blocked_across_generations():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    source.by_step[2] = feedback(2)
    calls = []

    original = plan()
    exact_original_step = original.steps[0]

    def provider(request):
        calls.append(request.generation)
        if request.generation == 2:
            return replacement_plan(request.goal, ("x",))
        return LLMPlan(
            goal=request.goal,
            rationale="bad retry",
            steps=(
                LLMPlanStep(
                    index=1,
                    action=exact_original_step.action,
                    purpose=exact_original_step.purpose,
                    preconditions=exact_original_step.preconditions,
                    expected_observation=exact_original_step.expected_observation,
                    success_criteria=exact_original_step.success_criteria,
                    risk=exact_original_step.risk,
                ),
            ),
            confidence=0.9,
        )

    replanner = LLMPlanAdaptiveReplanner(provider, max_replans=3)
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    with pytest.raises(LLMPlanAdaptiveRuntimeSafetyError):
        controller.execute(original)

    assert calls == [2, 3]


def test_provider_exception_is_wrapped():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)

    def provider(_request):
        raise ValueError("provider exploded")

    replanner = LLMPlanAdaptiveReplanner(provider)
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    with pytest.raises(LLMPlanAdaptiveRuntimeReplanningError, match="provider exploded"):
        controller.execute(plan())


def test_custom_event_observer_receives_replan_lifecycle():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    events = []
    replanner = LLMPlanAdaptiveReplanner(lambda request: replacement_plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(
        runtime,
        replanner,
        feedback_source=source,
        event_observer=events.append,
    )

    result = controller.execute(plan())

    phases = [event.phase for event in events]
    assert phases == ["segment_started", "replan_requested", "segment_finished", "replanned", "segment_started", "segment_finished", "completed"]
    assert result.success is True


def test_reset_clears_last_trace_and_feedback():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    replanner = LLMPlanAdaptiveReplanner(lambda request: replacement_plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(runtime, replanner, feedback_source=source)

    controller.execute(plan())
    assert controller.last_result is not None
    assert controller.last_feedback is not None
    controller.reset()
    assert controller.last_result is None
    assert controller.last_feedback is None
    assert controller.last_replanning is None


def test_nested_live_runtime_shape_is_supported():
    class Wrapper:
        def __init__(self, runtime):
            self.runtime_executor = runtime

        def execute(self, plan, **kwargs):
            return self.runtime_executor.execute(plan, **kwargs)

    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    source = FeedbackSource()
    source.by_step[1] = feedback(1)
    replanner = LLMPlanAdaptiveReplanner(lambda request: replacement_plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(Wrapper(runtime), replanner, feedback_source=source)

    result = controller.execute(plan())

    assert result.success is True
    assert bridge.calls == ["a", "x"]


def test_runtime_kwargs_are_forwarded():
    bridge = FakeBridge()
    runtime = make_runtime(bridge)
    replanner = LLMPlanAdaptiveReplanner(lambda request: replacement_plan(request.goal, ("x",)))
    controller = LLMPlanAdaptiveRuntime(runtime, replanner)

    environment = object()
    result = controller.execute(
        plan(),
        environment=environment,
        strategies={"s": 1},
        information_gain={"a": 0.2},
        action_risk={"a": 0.4},
        exploration_risk=0.1,
        action_cost={"a": 0.3},
        experience_id="exp-1",
    )

    assert result.success is True
