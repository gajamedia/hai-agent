from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.llm.closed_loop_multi_goal_runtime import (
    ClosedLoopMultiGoalRuntime,
    ClosedLoopMultiGoalRuntimeConfigurationError,
    ClosedLoopMultiGoalRuntimeCoordinationError,
    ClosedLoopMultiGoalRuntimeEvent,
    ClosedLoopMultiGoalRuntimeExecutionError,
    ClosedLoopMultiGoalRuntimeExecutionReport,
    ClosedLoopMultiGoalRuntimeInputError,
    ClosedLoopMultiGoalRuntimeReplanningError,
    ClosedLoopMultiGoalRuntimeSegment,
)
from agent.llm.goal_conflict_resolution import GoalConflictRelation
from agent.llm.goal_management import GoalLifecycleManager, GoalStatus, ManagedGoal
from agent.llm.goal_progress_evaluation import GoalAchievementEvidence
from agent.llm.llm_plan_runtime_learning_feedback import LLMPlanRuntimeLearningFeedback
from agent.llm.planning import LLMPlan, LLMPlanStep


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
            LLMPlanStep(
                index=i,
                action=f"action_{i}",
                purpose=f"purpose_{i}",
                risk="low",
            )
            for i in indices
        ),
        confidence=0.9,
    )


def success_result(plan_obj, *, final_world_state=None, completed=True, success=True, next_step_index=None):
    records = tuple(Record(step=step, success=True) for step in plan_obj.steps)
    return SimpleNamespace(
        records=records,
        success=success,
        completed=completed,
        stopped=not completed,
        final_world_state=final_world_state,
        next_step_index=next_step_index or (max((s.index for s in plan_obj.steps), default=0) + 1),
        stop_reason=None if success else "failed",
    )


def failure_result(plan_obj, *, failed_index=None):
    idx = failed_index or plan_obj.steps[-1].index
    records = tuple(
        Record(step=step, success=step.index != idx)
        for step in plan_obj.steps
    )
    return SimpleNamespace(
        records=records,
        success=False,
        completed=False,
        stopped=True,
        final_world_state={"failed": idx},
        next_step_index=idx + 1,
        stop_reason="step_failed",
    )


def evidence_achieved(*, progress=1.0):
    return [
        GoalAchievementEvidence(
            source="test",
            success=True,
            progress=progress,
            confidence=1.0,
            quality=1.0,
            achieved=True,
        )
    ]


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
        return success_result(plan)


@dataclass
class FakeReplanningResult:
    replacement_plan: LLMPlan | None


class FakeGoalReplanner:
    def __init__(self, replacement=None):
        self.replacement = replacement
        self.calls = []
        self.adaptive_replanner = SimpleNamespace(
            evaluate_trigger=lambda fb: SimpleNamespace(should_replan=bool(fb.change_required or not fb.continue_recommended))
        )

    def replan(self, plan_obj, goal_obj, **kwargs):
        self.calls.append((plan_obj, goal_obj, kwargs))
        return FakeReplanningResult(self.replacement or plan(goal_obj.description, (10,)))


# 1-10: construction and immutable contracts

def test_constructor_accepts_minimal_runtime_and_manager():
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager())
    assert runtime.max_segments == 20


def test_constructor_requires_runtime_execute():
    with pytest.raises(ClosedLoopMultiGoalRuntimeConfigurationError):
        ClosedLoopMultiGoalRuntime(object(), GoalLifecycleManager())


def test_constructor_requires_goal_manager():
    with pytest.raises(ClosedLoopMultiGoalRuntimeConfigurationError):
        ClosedLoopMultiGoalRuntime(FakeRuntime(), object())


def test_constructor_rejects_invalid_replanner():
    with pytest.raises(ClosedLoopMultiGoalRuntimeConfigurationError):
        ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager(), goal_replanner=object())


def test_constructor_rejects_invalid_segment_limit():
    with pytest.raises(ClosedLoopMultiGoalRuntimeConfigurationError):
        ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager(), max_segments=0)


def test_constructor_rejects_negative_switch_limit():
    with pytest.raises(ClosedLoopMultiGoalRuntimeConfigurationError):
        ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager(), max_goal_switches=-1)


def test_event_requires_nonempty_phase():
    with pytest.raises(Exception):
        ClosedLoopMultiGoalRuntimeEvent(phase="", generation=1)


def test_event_requires_positive_generation():
    with pytest.raises(Exception):
        ClosedLoopMultiGoalRuntimeEvent(phase="started", generation=0)


def test_event_copies_metadata():
    data = {"x": 1}
    event = ClosedLoopMultiGoalRuntimeEvent(phase="started", generation=1, metadata=data)
    data["x"] = 2
    assert event.metadata["x"] == 1


def test_reset_clears_runtime_trace():
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager())
    runtime.last_evaluations["g1"] = object()
    runtime.last_arbitration = object()
    runtime.last_conflict_resolution = object()
    runtime.reset()
    assert runtime.last_evaluations == {}
    assert runtime.last_arbitration is None
    assert runtime.last_conflict_resolution is None


# 11-20: static helpers

def test_successful_step_indices_extracts_successes():
    p = plan(indices=(1, 2))
    result = SimpleNamespace(records=(Record(p.steps[0], True), Record(p.steps[1], False)))
    assert ClosedLoopMultiGoalRuntime._successful_step_indices(result) == (1,)


def test_successful_step_indices_ignores_skipped():
    p = plan(indices=(1,))
    result = SimpleNamespace(records=(Record(p.steps[0], True, skipped=True),))
    assert ClosedLoopMultiGoalRuntime._successful_step_indices(result) == ()


def test_failed_step_index_returns_latest_failed():
    p = plan(indices=(1, 2))
    result = SimpleNamespace(records=(Record(p.steps[0], False), Record(p.steps[1], False)))
    assert ClosedLoopMultiGoalRuntime._failed_step_index(result) == 2


def test_failed_step_index_returns_none_when_all_successful():
    p = plan(indices=(1,))
    result = SimpleNamespace(records=(Record(p.steps[0], True),))
    assert ClosedLoopMultiGoalRuntime._failed_step_index(result) is None


def test_next_step_index_prefers_report():
    result = SimpleNamespace(records=(), next_step_index=7)
    assert ClosedLoopMultiGoalRuntime._next_step_index(result) == 7


def test_next_step_index_falls_back_to_records():
    p = plan(indices=(5,))
    result = SimpleNamespace(records=(Record(p.steps[0], True),), next_step_index=None)
    assert ClosedLoopMultiGoalRuntime._next_step_index(result) == 6


def test_is_success_reads_report():
    assert ClosedLoopMultiGoalRuntime._is_success(SimpleNamespace(success=True)) is True


def test_final_world_state_prefers_report_value():
    assert ClosedLoopMultiGoalRuntime._final_world_state(SimpleNamespace(final_world_state={"x": 1}), {}) == {"x": 1}


def test_final_world_state_keeps_current_when_report_has_none():
    current = {"x": 1}
    assert ClosedLoopMultiGoalRuntime._final_world_state(SimpleNamespace(final_world_state=None), current) is current


def test_remaining_plan_indices_filters_completed():
    p = plan(indices=(1, 2, 3))
    assert ClosedLoopMultiGoalRuntime._remaining_plan_indices(p, (1, 3)) == (2,)


# 21-32: event observer and coordination

def test_add_observer_requires_callable():
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager())
    with pytest.raises(ClosedLoopMultiGoalRuntimeConfigurationError):
        runtime.add_event_observer(object())


def test_add_observer_deduplicates():
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager())
    observer = lambda event: None
    runtime.add_event_observer(observer)
    runtime.add_event_observer(observer)
    assert runtime._observers == [observer]


def test_remove_observer_is_idempotent():
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager())
    observer = lambda event: None
    runtime.remove_event_observer(observer)
    assert runtime._observers == []


def test_coordination_selects_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=0.9)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    arbitration, conflict = runtime._build_goal_context(
        urgency_provider=None,
        relations_provider=None,
        blocked_goal_ids_provider=None,
        metadata={},
    )
    assert arbitration.selected_goal_id == "g1"
    assert conflict.conflict_count == 0


def test_coordination_respects_blocked_provider():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=0.9)
    manager.add_goal("g2", "G2", priority=0.8)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    arbitration, _ = runtime._build_goal_context(
        urgency_provider=None,
        relations_provider=None,
        blocked_goal_ids_provider=lambda goals, meta: ("g1",),
        metadata={},
    )
    assert arbitration.selected_goal_id == "g2"


def test_coordination_respects_urgency_provider():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=0.4)
    manager.add_goal("g2", "G2", priority=0.9)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    arbitration, _ = runtime._build_goal_context(
        urgency_provider=lambda goals, meta: {"g1": 1.0, "g2": 1.0},
        relations_provider=None,
        blocked_goal_ids_provider=None,
        metadata={},
    )
    assert arbitration.selected_goal_id == "g2"


def test_coordination_detects_conflict():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=0.9)
    manager.add_goal("g2", "G2", priority=0.8)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    relation = GoalConflictRelation("g1", "g2", reason="shared resource", severity=0.8)
    _, conflict = runtime._build_goal_context(
        urgency_provider=None,
        relations_provider=lambda goals, meta: (relation,),
        blocked_goal_ids_provider=None,
        metadata={},
    )
    assert conflict.conflict_count == 1
    assert conflict.resolved


def test_apply_arbitration_sets_current_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    arb, _ = runtime._build_goal_context(urgency_provider=None, relations_provider=None, blocked_goal_ids_provider=None, metadata={})
    selected = runtime._apply_arbitration(arb)
    assert selected.goal_id == "g1"
    assert manager.current_goal_id == "g1"


def test_all_goals_terminal_helper():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    manager.set_current_goal("g1")
    manager.complete("g1")
    assert ClosedLoopMultiGoalRuntime._all_goals_terminal(manager) is True


def test_all_goals_completed_helper_false_when_failed():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    manager.set_current_goal("g1")
    manager.fail("g1")
    assert ClosedLoopMultiGoalRuntime._all_goals_completed(manager) is False


def test_normalize_evidence_accepts_single_mapping():
    value = {"source": "x"}
    assert ClosedLoopMultiGoalRuntime._normalize_evidence(value) == (value,)


def test_normalize_evidence_accepts_iterable():
    values = [{"source": "a"}, {"source": "b"}]
    assert ClosedLoopMultiGoalRuntime._normalize_evidence(values) == tuple(values)


# 33-44: single goal lifecycle

def test_execute_requires_plan():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    with pytest.raises(ClosedLoopMultiGoalRuntimeInputError):
        runtime.execute(object())


def test_execute_requires_goals():
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), GoalLifecycleManager())
    with pytest.raises(ClosedLoopMultiGoalRuntimeCoordinationError):
        runtime.execute(plan("G1"))


def test_execute_rejects_initial_goal_mismatch():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    with pytest.raises(ClosedLoopMultiGoalRuntimeInputError):
        runtime.execute(plan("Wrong"))


def test_execute_single_goal_can_complete_with_evidence():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=1.0)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    report = runtime.execute(
        plan("G1"),
        evidence_provider=lambda goal_obj, result, world, meta: evidence_achieved(),
        world_state={"start": True},
    )
    assert report.success is True
    assert report.completed is True
    assert report.completed_goal_ids == ("g1",)
    assert manager.get_goal("g1").status == GoalStatus.COMPLETED


def test_execute_single_goal_updates_progress_without_completing():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    manager.update_progress("g1", 0.2)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    report = runtime.execute(
        plan("G1"),
        evidence_provider=lambda goal_obj, result, world, meta: [
            GoalAchievementEvidence(source="progress", progress=0.6, confidence=1.0, quality=1.0)
        ],
    )
    assert report.success is False
    assert manager.get_goal("g1").progress == 0.6
    assert report.stop_reason == "segment_succeeded_without_goal_completion"


def test_execute_records_segment():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    report = runtime.execute(plan("G1"), evidence_provider=lambda *args: evidence_achieved())
    assert isinstance(report.segments[0], ClosedLoopMultiGoalRuntimeSegment)
    assert report.segment_count == 1


def test_execute_emits_finished_event():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    events = []
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    runtime.add_event_observer(events.append)
    runtime.execute(plan("G1"), evidence_provider=lambda *args: evidence_achieved())
    assert events[-1].phase == "completed"


def test_execute_wraps_runtime_error():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(raise_error=True), manager)
    with pytest.raises(ClosedLoopMultiGoalRuntimeExecutionError):
        runtime.execute(plan("G1"))


def test_execute_stops_after_failure_without_feedback():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(results=[lambda p: failure_result(p)]), manager)
    report = runtime.execute(plan("G1"))
    assert report.stopped is True
    assert report.success is False
    assert report.stop_reason == "execution_failure_without_replan"


def test_execute_requires_replanner_when_goal_switch_needs_new_plan():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=1.0)
    manager.add_goal("g2", "G2", priority=1.0)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager, max_goal_switches=0)
    # Same priority and no evaluation evidence leaves g1 retained; a direct current-goal
    # selection to g2 is therefore handled explicitly below.
    manager.set_current_goal("g2")
    with pytest.raises(ClosedLoopMultiGoalRuntimeInputError):
        runtime.execute(plan("G1"))


def test_execute_honors_max_segments():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager, max_segments=1)
    report = runtime.execute(plan("G1"), evidence_provider=lambda *args: [
        GoalAchievementEvidence(source="p", progress=0.5, confidence=1.0, quality=1.0)
    ])
    assert report.segment_count == 1
    assert report.stopped is True


def test_execute_stops_on_unresolved_conflict():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=0.8)
    manager.add_goal("g2", "G2", priority=0.7)
    relation = GoalConflictRelation("g1", "g2", reason="hard conflict", severity=1.0)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    report = runtime.execute(
        plan("G1"),
        relations_provider=lambda goals, meta: (relation,),
        blocked_goal_ids_provider=lambda goals, meta: ("g1", "g2"),
    )
    assert report.stop_reason == "unresolved_goal_conflict"
    assert report.stopped is True


# 45-52: feedback and replanning

def test_execute_replans_on_runtime_feedback():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    replacement = plan("G1", (10,))
    fake_replanner = FakeGoalReplanner(replacement)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(results=[lambda p: failure_result(p)]), manager, goal_replanner=fake_replanner)
    report = runtime.execute(
        plan("G1", (1,)),
        feedback_provider=lambda *args: feedback(change_required=True, continue_recommended=False, progress_delta=0.0),
    )
    assert fake_replanner.calls
    assert report.replan_count == 1
    assert report.final_plan.goal == "G1"


def test_feedback_replan_wraps_replanner_error():
    class BadReplanner:
        adaptive_replanner = SimpleNamespace(evaluate_trigger=lambda fb: SimpleNamespace(should_replan=True))
        def replan(self, *args, **kwargs):
            raise RuntimeError("replan failed")

    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(results=[lambda p: failure_result(p)]), manager, goal_replanner=BadReplanner())
    with pytest.raises(ClosedLoopMultiGoalRuntimeReplanningError):
        runtime.execute(
            plan("G1"),
            feedback_provider=lambda *args: feedback(change_required=True, continue_recommended=False, progress_delta=0.0),
        )


def test_replanner_is_not_called_for_positive_segment_without_trigger():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    fake_replanner = FakeGoalReplanner(plan("G1", (10,)))
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager, goal_replanner=fake_replanner)
    runtime.execute(plan("G1"), feedback_provider=lambda *args: feedback(progress_delta=0.5))
    assert fake_replanner.calls == []


def test_execute_multi_goal_switches_and_replans():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=1.0)
    manager.add_goal("g2", "G2", priority=1.0)
    fake_replanner = FakeGoalReplanner(plan("G2", (20,)))
    runtime = ClosedLoopMultiGoalRuntime(
        FakeRuntime(),
        manager,
        goal_replanner=fake_replanner,
        max_segments=3,
    )
    report = runtime.execute(
        plan("G1", (1,)),
        evidence_provider=lambda goal_obj, result, world, meta: evidence_achieved(),
    )
    assert fake_replanner.calls
    assert report.goal_switch_count == 1
    assert report.replan_count == 1
    assert manager.current_goal_id is None


def test_multi_goal_completion_requires_all_goals_completed():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=1.0)
    manager.add_goal("g2", "G2", priority=1.0)
    fake_replanner = FakeGoalReplanner(plan("G2", (20,)))
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager, goal_replanner=fake_replanner, max_segments=3)
    calls = {"g1": 0, "g2": 0}
    def evidence(goal_obj, result, world, meta):
        calls[goal_obj.goal_id] += 1
        return evidence_achieved()
    report = runtime.execute(plan("G1"), evidence_provider=evidence)
    assert report.completed_goal_ids == ("g1", "g2")
    assert report.success is True
    assert manager.get_goal("g1").status == GoalStatus.COMPLETED
    assert manager.get_goal("g2").status == GoalStatus.COMPLETED


# 53-60: integration/report surface

def test_report_exposes_goal_ids():
    report = ClosedLoopMultiGoalRuntimeExecutionReport(
        initial_goal_id="g1",
        final_goal_id="g2",
        initial_plan=plan("G1"),
        final_plan=plan("G2"),
        segments=(),
        evaluations={},
        completed_goal_ids=("g1",),
        failed_goal_ids=(),
        final_world_state=None,
        completed=False,
        stopped=True,
        success=False,
        stop_reason="x",
        goal_switch_count=1,
        replan_count=1,
    )
    assert report.goal_ids == ()


def test_report_all_goals_completed_property():
    report = ClosedLoopMultiGoalRuntimeExecutionReport(
        initial_goal_id="g1",
        final_goal_id="g1",
        initial_plan=plan("G1"),
        final_plan=plan("G1"),
        segments=(),
        evaluations={},
        completed_goal_ids=("g1",),
        failed_goal_ids=(),
        final_world_state=None,
        completed=True,
        stopped=False,
        success=True,
        stop_reason=None,
        goal_switch_count=0,
        replan_count=0,
    )
    assert report.all_goals_completed is True


def test_report_all_goals_completed_false_with_failures():
    report = ClosedLoopMultiGoalRuntimeExecutionReport(
        initial_goal_id="g1",
        final_goal_id="g1",
        initial_plan=plan("G1"),
        final_plan=plan("G1"),
        segments=(),
        evaluations={},
        completed_goal_ids=("g1",),
        failed_goal_ids=("g2",),
        final_world_state=None,
        completed=True,
        stopped=False,
        success=False,
        stop_reason="failed_goal",
        goal_switch_count=0,
        replan_count=0,
    )
    assert report.all_goals_completed is False


def test_event_observer_receives_coordination_and_segment_events():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    events = []
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    runtime.add_event_observer(events.append)
    runtime.execute(plan("G1"))
    phases = [event.phase for event in events]
    assert "coordinated" in phases
    assert "segment_started" in phases
    assert "segment_finished" in phases


def test_runtime_stores_last_result():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    report = runtime.execute(plan("G1"))
    assert runtime.last_result is report


def test_runtime_stores_last_arbitration_and_conflict():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    runtime.execute(plan("G1"))
    assert runtime.last_arbitration is not None
    assert runtime.last_conflict_resolution is not None


def test_runtime_metadata_marks_stage():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    report = runtime.execute(plan("G1"), metadata={"request_id": "abc"})
    assert report.metadata["stage"] == "25.8"
    assert report.metadata["request_id"] == "abc"


def test_segment_is_immutable():
    manager = GoalLifecycleManager()
    segment = ClosedLoopMultiGoalRuntimeSegment(
        generation=1,
        goal_id="g1",
        plan=plan("G1"),
        runtime_result=object(),
        evaluation=None,
        arbitration=None,
        conflict_resolution=None,
        replanning=None,
        switched_from_goal_id=None,
        success=True,
        world_state=None,
        next_goal_id=None,
    )
    with pytest.raises(Exception):
        segment.success = False


# 61-64: safety boundaries and utility behavior

def test_no_goal_switch_without_replanner_is_safe_stop():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1", priority=0.5)
    manager.add_goal("g2", "G2", priority=1.0)
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    # Initial plan is for g2 after arbitration; use current goal g1 so it is selected initially.
    manager.set_current_goal("g1")
    report = runtime.execute(plan("G1"), evidence_provider=lambda *args: evidence_achieved())
    # g1 completes, then g2 is selected; without a replanner the switch cannot be planned safely.
    assert report.success is False


def test_non_callable_providers_are_detected_when_invoked():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    with pytest.raises(ClosedLoopMultiGoalRuntimeInputError):
        runtime.execute(plan("G1"), urgency_provider=object())


def test_report_final_plan_tracks_replacement_plan():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    replanner = FakeGoalReplanner(plan("G1", (9,)))
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(results=[lambda p: failure_result(p)]), manager, goal_replanner=replanner)
    report = runtime.execute(
        plan("G1", (1,)),
        feedback_provider=lambda *args: feedback(change_required=True, continue_recommended=False, progress_delta=0.0),
    )
    assert report.final_plan.steps[0].index == 9


def test_failure_feedback_can_trigger_replan_without_goal_switch():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    replanner = FakeGoalReplanner(plan("G1", (7,)))
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(results=[lambda p: failure_result(p)]), manager, goal_replanner=replanner)
    runtime.execute(
        plan("G1", (1,)),
        feedback_provider=lambda *args: feedback(change_required=True, continue_recommended=False, progress_delta=0.0),
    )
    assert len(replanner.calls) == 1



def test_execute_rejects_noncallable_evidence_provider():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    with pytest.raises(ClosedLoopMultiGoalRuntimeInputError):
        runtime.execute(plan("G1"), evidence_provider=object())


def test_execute_rejects_invalid_feedback_provider_result():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "G1")
    runtime = ClosedLoopMultiGoalRuntime(FakeRuntime(), manager)
    with pytest.raises(ClosedLoopMultiGoalRuntimeInputError):
        runtime.execute(plan("G1"), feedback_provider=lambda *args: "invalid")
