from dataclasses import dataclass
from types import SimpleNamespace

from agent.llm.closed_loop_multi_goal_runtime import ClosedLoopMultiGoalRuntime
from agent.llm.goal_management import GoalLifecycleManager
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class Record:
    step: LLMPlanStep
    success: bool
    skipped: bool = False


def plan(goal="Reach target", indices=(1,)):
    return LLMPlan(
        goal=goal,
        rationale="test",
        steps=tuple(
            LLMPlanStep(index=i, action=f"action_{i}", purpose=f"purpose_{i}")
            for i in indices
        ),
        confidence=0.9,
    )


def failed_result(plan_obj, state):
    failed_step = plan_obj.steps[-1]
    return SimpleNamespace(
        records=tuple(
            Record(step, success=(step.index != failed_step.index))
            for step in plan_obj.steps
        ),
        success=False,
        completed=False,
        stopped=True,
        final_world_state=dict(state),
        next_step_index=failed_step.index + 1,
        stop_reason="step_failed",
    )


def successful_result(plan_obj, state):
    return SimpleNamespace(
        records=tuple(Record(step, success=True) for step in plan_obj.steps),
        success=True,
        completed=True,
        stopped=False,
        final_world_state=dict(state),
        next_step_index=None,
        stop_reason=None,
    )


class FakeRuntime:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def execute(self, plan_obj, **kwargs):
        self.calls.append((plan_obj, kwargs))
        result = self.results.pop(0)
        return result(plan_obj) if callable(result) else result


class FakeReplanner:
    def __init__(self, replacement):
        self.replacement = replacement
        self.calls = []
        self.adaptive_replanner = SimpleNamespace(
            evaluate_trigger=lambda feedback: SimpleNamespace(
                should_replan=bool(
                    feedback.change_required or not feedback.continue_recommended
                )
            )
        )

    def replan(self, plan_obj, goal_obj, **kwargs):
        self.calls.append((plan_obj, goal_obj, kwargs))
        return SimpleNamespace(replacement_plan=self.replacement)


def make_manager():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "Reach target", priority=1.0)
    return manager


def test_failure_without_feedback_synthesizes_recovery_and_replans():
    initial = plan(indices=(1, 2))
    replacement = plan(indices=(10,))
    runtime_engine = FakeRuntime(
        [
            lambda p: failed_result(p, {"position": "blocked", "revision": 2}),
            lambda p: successful_result(p, {"position": "target", "revision": 3}),
        ]
    )
    replanner = FakeReplanner(replacement)
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_engine,
        make_manager(),
        goal_replanner=replanner,
        max_segments=3,
    )

    report = runtime.execute(initial, world_state={"position": "start"})

    assert len(runtime_engine.calls) == 2
    assert len(replanner.calls) == 1
    call_kwargs = replanner.calls[0][2]
    assert call_kwargs["world_state"] == {"position": "blocked", "revision": 2}
    assert call_kwargs["failed_step_index"] == 2
    assert call_kwargs["successful_step_indices"] == (1,)
    feedback = call_kwargs["feedback"]
    assert feedback.success is False
    assert feedback.change_required is True
    assert feedback.continue_recommended is False
    assert feedback.metadata["synthetic"] is True
    assert feedback.metadata["trigger"] == "execution_failure_recovery"
    assert report.replan_count == 1
    assert report.segments[0].metadata["recovery_feedback_used"] is True
    assert report.metadata["recovery_feedback_segment_count"] == 1


def test_recovery_replanning_is_bounded_by_max_segments():
    initial = plan()
    replacement = plan(indices=(10,))
    runtime_engine = FakeRuntime([
        lambda p: failed_result(p, {"revision": 1}),
        lambda p: failed_result(p, {"revision": 2}),
    ])
    replanner = FakeReplanner(replacement)
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_engine,
        make_manager(),
        goal_replanner=replanner,
        max_segments=2,
    )

    report = runtime.execute(initial)

    assert report.stopped is True
    assert report.success is False
    assert report.stop_reason == "max_segments_reached"
    assert report.segment_count == 2
    assert report.replan_count == 1
    assert len(runtime_engine.calls) == 2


def test_failure_does_not_replan_when_recovery_is_disabled():
    initial = plan()
    runtime_engine = FakeRuntime([
        lambda p: failed_result(p, {"revision": 1}),
    ])
    replanner = FakeReplanner(plan(indices=(10,)))
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_engine,
        make_manager(),
        goal_replanner=replanner,
        replan_on_execution_failure=False,
    )

    report = runtime.execute(initial)

    assert len(replanner.calls) == 0
    assert report.stopped is True
    assert report.success is False
    assert report.stop_reason == "execution_failure_without_replan"
