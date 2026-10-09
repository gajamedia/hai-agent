from dataclasses import replace

import pytest

from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeExecutionReport,
    LLMPlanStepExecutionRecord,
)
from agent.llm.llm_plan_runtime_state import (
    LLMPlanRuntimeStateTransitionError,
    LLMPlanRuntimeStateTracker,
    LLMPlanRuntimeStatus,
)
from agent.llm.plan_validation import LLMPlanValidationResult
from agent.llm.planning import LLMPlan, LLMPlanStep


def plan(n=3):
    return LLMPlan(
        goal="test goal",
        rationale="test",
        steps=tuple(
            LLMPlanStep(
                index=i,
                action=f"action_{i}",
                purpose=f"purpose {i}",
            )
            for i in range(1, n + 1)
        ),
        confidence=0.9,
    )


def record(index, success=True, skipped=False, observation=None):
    p = plan(3)
    step = p.steps[index - 1]
    return LLMPlanStepExecutionRecord(
        step=step,
        request=None,
        result={"success": success},
        success=success,
        observation=observation if observation is not None else {"step": index},
        skipped=skipped,
    )


def test_initial_state_is_idle():
    s = LLMPlanRuntimeStateTracker().snapshot()
    assert s.status is LLMPlanRuntimeStatus.IDLE
    assert s.progress == 0.0
    assert s.planned_step_count == 0


def test_start_requires_plan():
    with pytest.raises(TypeError):
        LLMPlanRuntimeStateTracker().start(None)


def test_start_rejects_empty_plan():
    empty = LLMPlan(goal="g", rationale="r", steps=())
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        LLMPlanRuntimeStateTracker().start(empty)


def test_start_sets_first_next_step():
    s = LLMPlanRuntimeStateTracker().start(plan())
    assert s.status is LLMPlanRuntimeStatus.RUNNING
    assert s.next_step_index == 1


def test_record_step_updates_attempted_and_progress():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    s = t.record_step(record(1, True))
    assert s.current_step_index == 1
    assert s.next_step_index == 2
    assert s.attempted_step_indices == (1,)
    assert s.succeeded_step_indices == (1,)
    assert s.progress == pytest.approx(1 / 3)
    assert s.success_rate == 1.0


def test_failed_step_updates_failure_metrics():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    s = t.record_step(record(1, False))
    assert s.failed_step_indices == (1,)
    assert s.success_rate == 0.0


def test_duplicate_step_record_rejected():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.record_step(record(1))
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.record_step(record(1))


def test_unknown_step_rejected():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    foreign = LLMPlanStep(9, "x", "x")
    rec = LLMPlanStepExecutionRecord(foreign, None, success=True)
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.record_step(rec)


def test_remaining_steps_are_exposed():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.record_step(record(1))
    assert t.snapshot().remaining_step_indices == (2, 3)


def test_can_continue_while_running():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    assert t.snapshot().can_continue


def test_mark_stopped_sets_stop_reason_and_next_step():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.record_step(record(1))
    s = t.mark_stopped(reason="step_failed", next_step_index=2)
    assert s.status is LLMPlanRuntimeStatus.STOPPED
    assert s.stopped is True
    assert s.stop_reason == "step_failed"
    assert s.next_step_index == 2
    assert not s.can_continue


def test_mark_stopped_requires_reason():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    with pytest.raises(ValueError):
        t.mark_stopped(reason="")


def test_mark_stopped_rejects_unknown_next_index():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.mark_stopped(reason="stop", next_step_index=99)


def test_mark_completed_success():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.record_step(record(1))
    t.record_step(record(2))
    t.record_step(record(3))
    s = t.mark_completed(success=True, final_world_state={"done": True})
    assert s.status is LLMPlanRuntimeStatus.COMPLETED
    assert s.completed is True
    assert s.success is True
    assert s.next_step_index is None
    assert s.last_observation == {"done": True}


def test_mark_completed_failure_sets_failed_status():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    s = t.mark_completed(success=False)
    assert s.status is LLMPlanRuntimeStatus.FAILED
    assert s.completed is True
    assert s.success is False


def test_cannot_record_after_stopped():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.mark_stopped(reason="manual")
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.record_step(record(1))


def test_cannot_record_after_completed():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.mark_completed(success=False)
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.record_step(record(1))


def test_mark_skipped_tracks_skipped_step():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    s = t.mark_skipped(1, reason="condition")
    assert s.skipped_step_indices == (1,)
    assert s.next_step_index == 2


def test_mark_skipped_unknown_rejected():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.mark_skipped(99)


def test_start_metadata_is_preserved():
    s = LLMPlanRuntimeStateTracker().start(plan(), metadata={"run_id": "r1"})
    assert s.metadata["run_id"] == "r1"
    assert s.metadata["planned_step_count"] == 3


def test_snapshot_is_immutable_dataclass():
    s = LLMPlanRuntimeStateTracker().start(plan())
    with pytest.raises(Exception):
        s.status = LLMPlanRuntimeStatus.COMPLETED


def test_apply_report_completed():
    p = plan(2)
    validation = LLMPlanValidationResult(valid=True, issues=(), checked_step_count=len(p.steps), executable_step_indices=tuple(step.index for step in p.steps))
    report = LLMPlanRuntimeExecutionReport(
        plan=p,
        validation=validation,
        records=(record(1), record(2)),
        completed=True,
        stopped=False,
        success=True,
        stop_reason=None,
        final_world_state={"done": True},
        next_step_index=None,
        metadata={"run_id": "r1"},
    )
    t = LLMPlanRuntimeStateTracker()
    s = t.apply_report(report)
    assert s.status is LLMPlanRuntimeStatus.COMPLETED
    assert s.attempted_step_indices == (1, 2)
    assert s.progress == 1.0
    assert s.success_rate == 1.0
    assert s.last_observation == {"done": True}


def test_apply_report_stopped():
    p = plan(3)
    validation = LLMPlanValidationResult(valid=True, issues=(), checked_step_count=len(p.steps), executable_step_indices=tuple(step.index for step in p.steps))
    report = LLMPlanRuntimeExecutionReport(
        plan=p,
        validation=validation,
        records=(record(1, False),),
        completed=False,
        stopped=True,
        success=False,
        stop_reason="step_failed",
        final_world_state={"step": 1},
        next_step_index=2,
    )
    s = LLMPlanRuntimeStateTracker().apply_report(report)
    assert s.status is LLMPlanRuntimeStatus.STOPPED
    assert s.next_step_index == 2
    assert s.stop_reason == "step_failed"
    assert s.progress == pytest.approx(1 / 3)


def test_apply_report_noncompleted_nonstopped_remains_running():
    p = plan(3)
    validation = LLMPlanValidationResult(valid=True, issues=(), checked_step_count=len(p.steps), executable_step_indices=tuple(step.index for step in p.steps))
    report = LLMPlanRuntimeExecutionReport(
        plan=p,
        validation=validation,
        records=(record(1),),
        completed=False,
        stopped=False,
        success=False,
        stop_reason=None,
        final_world_state={"step": 1},
        next_step_index=2,
    )
    s = LLMPlanRuntimeStateTracker().apply_report(report)
    assert s.status is LLMPlanRuntimeStatus.RUNNING
    assert s.can_continue
    assert s.next_step_index == 2


def test_progress_counts_attempts_not_successes():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.record_step(record(1, False))
    t.record_step(record(2, True))
    assert t.snapshot().progress == pytest.approx(2 / 3)
    assert t.snapshot().success_rate == 0.5


def test_reset_returns_idle():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.reset()
    s = t.snapshot()
    assert s.status is LLMPlanRuntimeStatus.IDLE
    assert s.plan_goal is None


def test_record_before_start_rejected():
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        LLMPlanRuntimeStateTracker().record_step(record(1))


def test_mark_complete_before_start_rejected():
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        LLMPlanRuntimeStateTracker().mark_completed(success=True)


def test_mark_stop_before_start_rejected():
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        LLMPlanRuntimeStateTracker().mark_stopped(reason="x")


def test_apply_report_type_validation():
    with pytest.raises(TypeError):
        LLMPlanRuntimeStateTracker().apply_report(None)


def test_metadata_updates_on_record():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    s = t.record_step(record(1))
    assert s.metadata["last_recorded_step_index"] == 1
    assert s.metadata["attempted_step_count"] == 1


def test_successful_steps_are_ordered():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.record_step(record(1))
    t.record_step(record(2))
    assert t.snapshot().succeeded_step_indices == (1, 2)


def test_remaining_steps_empty_after_all_attempts():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan(2))
    t.record_step(record(1))
    t.record_step(record(2))
    assert t.snapshot().remaining_step_indices == ()


def test_remaining_steps_excludes_skipped_steps():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    t.mark_skipped(1, reason="condition")
    assert t.snapshot().remaining_step_indices == (2, 3)


def test_out_of_order_step_record_is_rejected():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan())
    with pytest.raises(LLMPlanRuntimeStateTransitionError):
        t.record_step(record(2))


def test_skipping_all_steps_leaves_no_next_step():
    t = LLMPlanRuntimeStateTracker()
    t.start(plan(2))
    t.mark_skipped(1)
    s = t.mark_skipped(2)
    assert s.next_step_index is None
    assert s.remaining_step_indices == ()
