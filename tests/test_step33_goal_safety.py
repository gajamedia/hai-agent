from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from agent.llm.closed_loop_multi_goal_runtime import ClosedLoopMultiGoalRuntime
from agent.llm.goal_management import GoalLifecycleManager, GoalStatus, ManagedGoal
from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.llm.llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback


@dataclass
class Record:
    step: LLMPlanStep
    success: bool
    skipped: bool = False


def goal(gid="g1", desc="Reach target", priority=0.8, progress=0.0, status=GoalStatus.PENDING, sequence=1):
    return ManagedGoal(
        goal_id=gid,
        description=desc,
        priority=priority,
        progress=progress,
        status=status,
        sequence=sequence,
    )


def plan(goal_text="Reach target", indices=(1,)):
    return LLMPlan(
        goal=goal_text,
        rationale="test",
        steps=tuple(
            LLMPlanStep(index=i, action=f"action_{i}", purpose=f"purpose_{i}", risk="low")
            for i in indices
        ),
        confidence=0.9,
    )


def failure_result(plan_obj, *, failed_index=None):
    idx = failed_index or plan_obj.steps[-1].index
    records = tuple(Record(step=step, success=step.index != idx) for step in plan_obj.steps)
    return SimpleNamespace(
        records=records,
        success=False,
        completed=False,
        stopped=True,
        final_world_state={"failed": idx},
        next_step_index=idx + 1,
        stop_reason="step_failed",
    )


def feedback(*, step=1, change_required=False, continue_recommended=True, risk_level="low", progress_delta=0.2):
    return LLMPlanRuntimeLearningFeedback(
        step_index=step,
        action=f"action_{step}",
        success=True,
        progress_delta=progress_delta,
        evidence_quality=1.0,
        confidence=1.0,
        diagnosis="test",
        lesson="test",
        next_step="continue",
        change_required=change_required,
        continue_recommended=continue_recommended,
        exploration_recommended=False,
        risk_level=risk_level,
        risk_score=0.75 if risk_level == "high" else 0.25,
        information_gain=max(0.0, progress_delta),
    )


class FakeRuntime:
    def __init__(self, results=None, *, raise_error=False):
        self.results = list(results or [])
        self.calls = []
        self.raise_error = raise_error

    def execute(self, plan, **kwargs):
        self.calls.append((plan, kwargs))
        if self.raise_error:
            raise RuntimeError("boom")
        if self.results:
            item = self.results.pop(0)
            return item(plan) if callable(item) else item
        return SimpleNamespace(
            records=tuple(Record(step=s, success=True) for s in plan.steps),
            success=True,
            completed=True,
            stopped=False,
            final_world_state={},
            next_step_index=max((s.index for s in plan.steps), default=0) + 1,
            stop_reason=None,
        )


class FakeGoalReplanner:
    def __init__(self, replacement=None):
        self.replacement = replacement
        self.calls = []
        self.adaptive_replanner = SimpleNamespace(
            evaluate_trigger=lambda fb: SimpleNamespace(
                should_replan=bool(fb.change_required or not fb.continue_recommended)
            )
        )

    def replan(self, plan_obj, goal_obj, **kwargs):
        self.calls.append((plan_obj, goal_obj, kwargs))
        return SimpleNamespace(replacement_plan=self.replacement or plan(goal_obj.description, (10,)))


@dataclass
class ReplanningResultWithValidation:
    replacement_plan: LLMPlan | None
    validation: object


def _manager_two_goals():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "Reach target", priority=0.9)
    manager.add_goal("g2", "Open gate", priority=0.8)
    manager.set_current_goal("g1")
    return manager


def test_rejects_replacement_plan_that_targets_wrong_goal_before_execution():
    manager = _manager_two_goals()
    runtime_executor = FakeRuntime([failure_result(plan("Reach target"))])
    replanner = FakeGoalReplanner(plan("Open gate", (10,)))
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_executor,
        manager,
        goal_replanner=replanner,
        max_segments=3,
    )

    report = runtime.execute(
        plan("Reach target"),
        initial_goal_id="g1",
        feedback_provider=lambda *_: feedback(change_required=True, continue_recommended=False),
    )

    assert len(runtime_executor.calls) == 1
    assert report.stopped is True
    assert report.success is False
    assert report.stop_reason == "unsafe_replacement_plan:replacement_plan_goal_mismatch"
    assert report.metadata["replacement_plan_safety_rejection_count"] == 1
    assert report.metadata["replacement_plan_safety_rejections"][0]["goal_id"] == "g1"


def test_rejects_replacement_plan_when_planner_validation_failed():
    manager = _manager_two_goals()
    runtime_executor = FakeRuntime([failure_result(plan("Reach target"))])
    replanner = FakeGoalReplanner()
    replanner.replan = lambda *_args, **_kwargs: ReplanningResultWithValidation(
        replacement_plan=plan("Reach target", (10,)),
        validation=SimpleNamespace(valid=False),
    )
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_executor,
        manager,
        goal_replanner=replanner,
        max_segments=3,
    )

    report = runtime.execute(
        plan("Reach target"),
        initial_goal_id="g1",
        feedback_provider=lambda *_: feedback(change_required=True, continue_recommended=False),
    )

    assert len(runtime_executor.calls) == 1
    assert report.success is False
    assert report.stop_reason == "unsafe_replacement_plan:replacement_plan_validation_failed"
    assert report.metadata["replacement_plan_safety_rejection_count"] == 1


def test_rejects_empty_replacement_plan_and_preserves_goal_selection():
    manager = _manager_two_goals()
    runtime_executor = FakeRuntime([failure_result(plan("Reach target"))])
    empty_plan = LLMPlan(
        goal="Reach target",
        rationale="empty candidate",
        steps=(),
        confidence=0.9,
    )
    replanner = FakeGoalReplanner(empty_plan)
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_executor,
        manager,
        goal_replanner=replanner,
        max_segments=3,
    )

    report = runtime.execute(
        plan("Reach target"),
        initial_goal_id="g1",
        feedback_provider=lambda *_: feedback(change_required=True, continue_recommended=False),
    )

    assert len(runtime_executor.calls) == 1
    assert manager.current_goal_id == "g1"
    assert report.success is False
    assert report.stop_reason == "unsafe_replacement_plan:replacement_plan_empty"


def test_safety_gate_allows_valid_replacement_plan_for_active_goal():
    manager = _manager_two_goals()
    runtime_executor = FakeRuntime([
        failure_result(plan("Reach target")),
        lambda p: SimpleNamespace(
            records=tuple(SimpleNamespace(step=s, success=True, skipped=False) for s in p.steps),
            success=True,
            completed=True,
            stopped=False,
            final_world_state={"target": True},
            next_step_index=99,
            stop_reason=None,
        ),
    ])
    replanner = FakeGoalReplanner(plan("Reach target", (10,)))
    runtime = ClosedLoopMultiGoalRuntime(
        runtime_executor,
        manager,
        goal_replanner=replanner,
        max_segments=3,
    )

    feedback_calls = {"count": 0}
    def feedback_once(*_args):
        feedback_calls["count"] += 1
        if feedback_calls["count"] == 1:
            return feedback(change_required=True, continue_recommended=False)
        return None

    report = runtime.execute(
        plan("Reach target"),
        initial_goal_id="g1",
        feedback_provider=feedback_once,
        evidence_provider=lambda *_: [],
    )

    assert report.metadata["goal_safety_gate_enabled"] is True
    assert report.metadata["replacement_plan_safety_rejection_count"] == 0
    assert len(runtime_executor.calls) == 2
