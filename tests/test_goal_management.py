from __future__ import annotations

import pytest

from agent.llm.goal_management import (
    GoalHierarchyError,
    GoalLifecycleError,
    GoalLifecycleManager,
    GoalManagementError,
    GoalNotFoundError,
    GoalSelectionPolicy,
    GoalStatus,
    ManagedGoal,
)


def test_managed_goal_defaults_and_bounds():
    goal = ManagedGoal("g1", "Do work")
    assert goal.status is GoalStatus.PENDING
    assert goal.priority == 0.5
    assert goal.progress == 0.0
    assert goal.sequence == 0


def test_empty_goal_id_rejected():
    with pytest.raises(ValueError):
        ManagedGoal("", "work")


def test_empty_description_rejected():
    with pytest.raises(ValueError):
        ManagedGoal("g1", "  ")


@pytest.mark.parametrize("field", ["priority", "progress"])
def test_bounded_fields_reject_invalid(field):
    kwargs = {field: 1.1}
    with pytest.raises(ValueError):
        ManagedGoal("g1", "work", **kwargs)


def test_add_goal_assigns_monotonic_sequence():
    manager = GoalLifecycleManager()
    first = manager.add_goal("g1", "first")
    second = manager.add_goal("g2", "second")
    assert first.sequence == 1
    assert second.sequence == 2


def test_duplicate_goal_rejected():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "first")
    with pytest.raises(GoalManagementError):
        manager.add_goal("g1", "duplicate")


def test_parent_must_exist():
    manager = GoalLifecycleManager()
    with pytest.raises(GoalHierarchyError):
        manager.add_goal("child", "child", parent_id="missing")


def test_cancelled_parent_cannot_receive_child():
    manager = GoalLifecycleManager()
    manager.add_goal("parent", "parent")
    manager.cancel("parent")
    with pytest.raises(GoalHierarchyError):
        manager.add_goal("child", "child", parent_id="parent")


def test_get_unknown_goal_rejected():
    manager = GoalLifecycleManager()
    with pytest.raises(GoalNotFoundError):
        manager.get_goal("missing")


def test_activate_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    goal = manager.activate("g1")
    assert goal.status is GoalStatus.ACTIVE


def test_pending_to_paused_is_invalid():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    with pytest.raises(GoalLifecycleError):
        manager.pause("g1")


def test_active_can_pause():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    goal = manager.pause("g1")
    assert goal.status is GoalStatus.PAUSED
    assert manager.current_goal_id is None


def test_paused_can_resume():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    manager.pause("g1")
    goal = manager.resume("g1")
    assert goal.status is GoalStatus.ACTIVE


def test_active_can_complete_and_progress_becomes_one():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    goal = manager.complete("g1")
    assert goal.status is GoalStatus.COMPLETED
    assert goal.progress == 1.0


def test_active_can_fail():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    goal = manager.fail("g1")
    assert goal.status is GoalStatus.FAILED


def test_active_can_cancel():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    goal = manager.cancel("g1")
    assert goal.status is GoalStatus.CANCELLED


@pytest.mark.parametrize("terminal_transition", ["activate", "pause", "resume", "complete", "fail", "cancel"])
def test_terminal_goal_cannot_transition(terminal_transition):
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    manager.complete("g1")
    with pytest.raises(GoalLifecycleError):
        getattr(manager, terminal_transition)("g1")


def test_terminal_goal_cannot_update_progress():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.activate("g1")
    manager.complete("g1")
    with pytest.raises(GoalLifecycleError):
        manager.update_progress("g1", 0.5)


def test_progress_update_is_bounded():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    with pytest.raises(ValueError):
        manager.update_progress("g1", -0.1)
    with pytest.raises(ValueError):
        manager.update_progress("g1", 1.1)


def test_children_of_returns_only_direct_children():
    manager = GoalLifecycleManager()
    manager.add_goal("root", "root")
    manager.add_goal("child", "child", parent_id="root")
    manager.add_goal("grand", "grand", parent_id="child")
    assert [goal.goal_id for goal in manager.children_of("root")] == ["child"]


def test_selection_chooses_highest_priority():
    manager = GoalLifecycleManager()
    manager.add_goal("low", "low", priority=0.2)
    manager.add_goal("high", "high", priority=0.9)
    selection = manager.select_next_goal()
    assert selection.selected_goal_id == "high"
    assert manager.current_goal_id == "high"
    assert selection.reason == "highest_priority_goal"


def test_selection_activates_pending_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work", priority=0.8)
    manager.select_next_goal()
    assert manager.get_goal("g1").status is GoalStatus.ACTIVE


def test_selection_continues_current_goal_on_tie():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "first", priority=0.8)
    manager.add_goal("g2", "second", priority=0.8)
    manager.set_current_goal("g1")
    selection = manager.select_next_goal()
    assert selection.selected_goal_id == "g1"
    assert not selection.preempted
    assert selection.reason == "continue_current_goal"


def test_higher_priority_goal_preempts_current():
    manager = GoalLifecycleManager()
    manager.add_goal("current", "current", priority=0.5)
    manager.add_goal("urgent", "urgent", priority=0.9)
    manager.set_current_goal("current")
    selection = manager.select_next_goal()
    assert selection.selected_goal_id == "urgent"
    assert selection.preempted


def test_preemption_margin_prevents_small_priority_change():
    policy = GoalSelectionPolicy(preempt_priority_margin=0.2)
    manager = GoalLifecycleManager(selection_policy=policy)
    manager.add_goal("current", "current", priority=0.5)
    manager.add_goal("other", "other", priority=0.6)
    manager.set_current_goal("current")
    selection = manager.select_next_goal()
    assert selection.selected_goal_id == "current"
    assert not selection.preempted


def test_current_goal_not_eligible_switches_to_candidate():
    manager = GoalLifecycleManager()
    manager.add_goal("current", "current", priority=0.5)
    manager.add_goal("other", "other", priority=0.4)
    manager.set_current_goal("current")
    manager.complete("current")
    selection = manager.select_next_goal()
    assert selection.selected_goal_id == "other"
    assert selection.reason == "highest_priority_goal"


def test_no_eligible_goals_returns_none():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.cancel("g1")
    selection = manager.select_next_goal()
    assert selection.selected_goal_id is None
    assert selection.reason == "no_eligible_goals"


def test_set_current_goal_activates_pending_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    goal = manager.set_current_goal("g1")
    assert goal.status is GoalStatus.ACTIVE
    assert manager.current_goal_id == "g1"


def test_set_current_goal_cannot_select_terminal_goal():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.cancel("g1")
    with pytest.raises(GoalLifecycleError):
        manager.set_current_goal("g1")


def test_set_current_goal_none_clears_selection():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "work")
    manager.set_current_goal("g1")
    assert manager.set_current_goal(None) is None
    assert manager.current_goal_id is None


def test_snapshot_is_deterministic():
    manager = GoalLifecycleManager()
    manager.add_goal("g1", "one")
    manager.add_goal("g2", "two")
    manager.set_current_goal("g1")
    snapshot = manager.snapshot()
    assert snapshot.sequence == 2
    assert snapshot.current_goal_id == "g1"
    assert [goal.goal_id for goal in snapshot.goals] == ["g1", "g2"]


def test_list_goals_can_exclude_terminal():
    manager = GoalLifecycleManager()
    manager.add_goal("active", "active")
    manager.add_goal("done", "done")
    manager.activate("done")
    manager.complete("done")
    assert [goal.goal_id for goal in manager.list_goals(include_terminal=False)] == ["active"]


def test_metadata_and_payload_are_preserved():
    manager = GoalLifecycleManager()
    payload = {"target": "x"}
    metadata = {"source": "test"}
    goal = manager.add_goal(
        "g1",
        "work",
        payload=payload,
        metadata=metadata,
    )
    assert goal.payload == payload
    assert dict(goal.metadata) == metadata
