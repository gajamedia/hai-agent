from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.agent_goal_management_integration import (
    AgentGoalManagementIntegrationConfigurationError,
    AgentGoalManagementIntegrator,
    AgentGoalManagementSnapshot,
    AgentGoalProjectionError,
)
from agent.llm.goal_management import GoalLifecycleManager, GoalSelectionPolicy, GoalStatus, GoalLifecycleError


class DummyAgent:
    pass


def test_for_agent_creates_manager_and_binds_it():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    assert isinstance(integration.manager, GoalLifecycleManager)
    assert agent.goal_manager is integration.manager


def test_existing_manager_can_be_reused():
    agent = DummyAgent()
    manager = GoalLifecycleManager()
    integration = AgentGoalManagementIntegrator.for_agent(agent, manager=manager)
    assert integration.manager is manager


def test_manager_and_policy_cannot_both_be_supplied():
    with pytest.raises(AgentGoalManagementIntegrationConfigurationError):
        AgentGoalManagementIntegrator.for_agent(
            DummyAgent(),
            manager=GoalLifecycleManager(),
            selection_policy=GoalSelectionPolicy(),
        )


def test_none_agent_rejected():
    with pytest.raises(AgentGoalManagementIntegrationConfigurationError):
        AgentGoalManagementIntegrator.for_agent(None)


def test_invalid_manager_rejected():
    with pytest.raises(AgentGoalManagementIntegrationConfigurationError):
        AgentGoalManagementIntegrator(DummyAgent(), object())


def test_add_goal_projects_current_state_without_selecting():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work", priority=0.8)
    assert agent.current_goal_id is None
    assert agent.current_goal is None
    assert agent.goal_status is None
    assert agent.goal_progress is None


def test_set_current_goal_projects_goal():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    goal = integration.set_current_goal("g1")
    assert goal.status is GoalStatus.ACTIVE
    assert agent.current_goal_id == "g1"
    assert agent.current_goal == goal
    assert agent.goal_status is GoalStatus.ACTIVE
    assert agent.goal_progress == 0.0


def test_select_next_goal_projects_selected_goal():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("low", "low", priority=0.2)
    integration.add_goal("high", "high", priority=0.9)
    selection = integration.select_next_goal()
    assert selection.selected_goal_id == "high"
    assert agent.current_goal_id == "high"


def test_projection_can_be_disabled():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(
        agent,
        project_to_agent=False,
    )
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    assert not hasattr(agent, "current_goal_id")
    assert not hasattr(agent, "current_goal")
    assert agent.goal_manager is integration.manager


def test_snapshot_is_immutable_view():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work", priority=0.7)
    integration.set_current_goal("g1")
    snapshot = integration.snapshot()
    assert isinstance(snapshot, AgentGoalManagementSnapshot)
    assert snapshot.current_goal_id == "g1"
    assert snapshot.goal_status is GoalStatus.ACTIVE
    assert snapshot.goal_progress == 0.0
    assert snapshot.manager_sequence == 1


def test_complete_updates_projection_and_clears_current_goal():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    goal = integration.complete("g1")
    assert goal.status is GoalStatus.COMPLETED
    assert agent.current_goal_id is None
    assert agent.current_goal is None
    assert agent.goal_status is None
    assert agent.goal_progress is None


def test_pause_updates_projection_and_clears_current_goal():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    goal = integration.pause("g1")
    assert goal.status is GoalStatus.PAUSED
    assert agent.current_goal_id is None


def test_resume_updates_projection():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.pause("g1")
    goal = integration.resume("g1")
    assert goal.status is GoalStatus.ACTIVE
    assert agent.current_goal_id is None  # resume changes lifecycle, not current selection


def test_resume_can_then_be_selected():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.pause("g1")
    integration.resume("g1")
    integration.set_current_goal("g1")
    assert agent.current_goal_id == "g1"


def test_update_progress_projects_value():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    goal = integration.update_progress("g1", 0.4)
    assert goal.progress == 0.4
    assert agent.goal_progress == 0.4


def test_fail_clears_current_projection():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.fail("g1")
    assert agent.current_goal_id is None


def test_cancel_clears_current_projection():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.cancel("g1")
    assert agent.current_goal_id is None


def test_activate_does_not_implicitly_make_goal_current():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    goal = integration.activate("g1")
    assert goal.status is GoalStatus.ACTIVE
    assert integration.current_goal is None


def test_select_next_goal_preempts_as_manager_defines():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("current", "current", priority=0.5)
    integration.add_goal("urgent", "urgent", priority=0.9)
    integration.set_current_goal("current")
    selection = integration.select_next_goal()
    assert selection.preempted is True
    assert selection.selected_goal_id == "urgent"
    assert agent.current_goal_id == "urgent"


def test_restore_projection_matches_manager_state():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    agent.current_goal_id = "corrupt"
    snapshot = integration.restore_agent_projection()
    assert snapshot.current_goal_id == "g1"
    assert agent.current_goal_id == "g1"


def test_goal_management_snapshot_is_projected():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    assert agent.goal_management_snapshot == integration.snapshot()


def test_detach_removes_integration_projection():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.detach()
    assert not hasattr(agent, "goal_manager")
    assert not hasattr(agent, "current_goal_id")
    assert not hasattr(agent, "goal_management_snapshot")


def test_detach_restores_preexisting_goal_manager():
    agent = DummyAgent()
    original = object()
    agent.goal_manager = original
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    assert agent.goal_manager is integration.manager
    integration.detach()
    assert agent.goal_manager is original


def test_detach_can_be_called_twice():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.detach()
    integration.detach()
    assert not hasattr(agent, "goal_manager")


def test_projection_error_is_raised_for_read_only_agent():
    class ReadOnly:
        __slots__ = ()

    with pytest.raises(AgentGoalManagementIntegrationConfigurationError):
        AgentGoalManagementIntegrator.for_agent(ReadOnly())


def test_projection_error_is_raised_for_read_only_current_field():
    class PartiallyReadOnly:
        @property
        def current_goal_id(self):
            return None

    with pytest.raises(AgentGoalProjectionError):
        AgentGoalManagementIntegrator.for_agent(PartiallyReadOnly())


def test_goal_operations_delegate_to_single_manager():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.update_progress("g1", 0.25)
    assert integration.manager.get_goal("g1").progress == 0.25
    assert integration.current_goal is integration.manager.get_goal("g1")


def test_no_execution_api_is_required():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    assert not hasattr(agent, "execute")
    assert not hasattr(agent, "llm_complete")


def test_no_llm_call_is_performed_during_goal_selection():
    class LLMSpy:
        calls = 0
        def complete(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("LLM must not be called")

    agent = DummyAgent()
    agent.llm = LLMSpy()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work", priority=0.8)
    integration.select_next_goal()
    assert agent.llm.calls == 0


def test_set_current_goal_none_clears_projection():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    integration.set_current_goal(None)
    assert agent.current_goal_id is None
    assert agent.current_goal is None


def test_pending_goal_can_become_current_via_selection():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work", priority=0.6)
    integration.select_next_goal()
    assert agent.goal_status is GoalStatus.ACTIVE


def test_manager_remains_source_of_truth_after_agent_projection_change():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.set_current_goal("g1")
    agent.current_goal_id = "fake"
    assert integration.manager.current_goal_id == "g1"


def test_snapshot_has_no_mutation_side_effects():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    sequence_before = integration.manager.snapshot().sequence
    integration.snapshot()
    assert integration.manager.snapshot().sequence == sequence_before


def test_metadata_and_payload_round_trip_through_integration():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    payload = {"x": 1}
    metadata = {"source": "agent"}
    goal = integration.add_goal("g1", "work", payload=payload, metadata=metadata)
    assert goal.payload == payload
    assert dict(goal.metadata) == metadata


def test_hierarchical_goal_operations_delegate():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("root", "root")
    integration.add_goal("child", "child", parent_id="root")
    assert [g.goal_id for g in integration.manager.children_of("root")] == ["child"]


def test_terminal_goal_operations_continue_to_use_manager_errors():
    agent = DummyAgent()
    integration = AgentGoalManagementIntegrator.for_agent(agent)
    integration.add_goal("g1", "work")
    integration.complete("g1") if False else None
    integration.set_current_goal("g1")
    integration.complete("g1")
    with pytest.raises(GoalLifecycleError):
        integration.pause("g1")
