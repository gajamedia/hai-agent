from __future__ import annotations

import pytest

from agent.core.action import Action
from agent.core.agent import Agent
from agent.llm.capability_registry import (
    CapabilityRegistrationError,
    CapabilityStatus,
    TrustedCapabilityRegistry,
)
from agent.llm.capability_lifecycle_manager import (
    CapabilityLifecycleManager,
    CapabilityLifecycleSynchronizationError,
)
from agent.llm.dynamic_workspace_runtime_adapter import (
    DynamicWorkspaceRuntimeAdapter,
)


CAPABILITY_ID = "application.integration_probe"
ACTION_NAME = "integration_probe"


def make_action(name: str = ACTION_NAME) -> Action:
    return Action(
        name=name,
        description=f"Integration test action: {name}",
        handler=lambda **kwargs: {"success": True, "action": name},
    )


def make_integrated_stack(*, register_through_adapter: bool = True):
    """Build the real Agent/registry/adapter/manager stack used by the app."""
    agent = Agent(name="capability-lifecycle-integration-test")
    action = make_action()

    registry = TrustedCapabilityRegistry()
    registry.register_trusted(
        capability_id=CAPABILITY_ID,
        action=action,
        risk_level="low",
    )

    adapter = DynamicWorkspaceRuntimeAdapter(
        registry=registry,
        agent=agent,
    )

    if register_through_adapter:
        report = adapter.synchronize()
        assert ACTION_NAME in report.synchronized_action_names
    else:
        agent.register_action(action)

    manager = CapabilityLifecycleManager(
        registry=registry,
        adapter=adapter,
    )
    return agent, action, registry, adapter, manager


def planner_names(agent: Agent) -> set[str]:
    return {action.name for action in agent.planner.actions}


def assert_planner_graph_is_shared(agent: Agent) -> None:
    assert agent.adaptive_planner.planner is agent.planner
    assert agent.hierarchical_planner.planner is agent.planner
    assert (
        agent.risk_aware_hierarchical_planner.hierarchical_planner
        is agent.hierarchical_planner
    )
    assert agent.unified_executor is not None


def test_real_agent_adapter_and_manager_disable_then_reactivate():
    agent, action, registry, adapter, manager = make_integrated_stack()

    assert ACTION_NAME in agent.actions
    assert agent.actions[ACTION_NAME] is action
    assert ACTION_NAME in planner_names(agent)
    assert_planner_graph_is_shared(agent)

    disabled = manager.disable(CAPABILITY_ID)

    assert disabled.status == CapabilityStatus.DISABLED
    assert ACTION_NAME not in planner_names(agent)
    assert agent.actions[ACTION_NAME] is action
    assert manager.inspect().is_consistent
    assert_planner_graph_is_shared(agent)

    activated = manager.activate(CAPABILITY_ID)

    assert activated.status == CapabilityStatus.ACTIVE
    assert ACTION_NAME in planner_names(agent)
    assert agent.actions[ACTION_NAME] is action
    assert manager.inspect().is_consistent
    assert_planner_graph_is_shared(agent)


def test_real_agent_preserves_unrelated_actions_during_lifecycle_changes():
    agent, capability_action, registry, adapter, manager = make_integrated_stack()
    unrelated_action = make_action("local_diagnostic")
    agent.register_action(unrelated_action)

    assert {ACTION_NAME, "local_diagnostic"}.issubset(planner_names(agent))

    manager.disable(CAPABILITY_ID)

    assert ACTION_NAME not in planner_names(agent)
    assert "local_diagnostic" in planner_names(agent)
    assert agent.actions[ACTION_NAME] is capability_action
    assert agent.actions["local_diagnostic"] is unrelated_action
    assert manager.inspect().is_consistent

    # A normal planner rebuild must continue to enforce the disabled state.
    agent.configure_planner()
    assert ACTION_NAME not in planner_names(agent)
    assert "local_diagnostic" in planner_names(agent)
    assert manager.inspect().is_consistent


def test_real_agent_revoke_is_terminal_and_survives_repeated_rebuilds():
    agent, action, registry, adapter, manager = make_integrated_stack()

    manager.revoke(CAPABILITY_ID)

    assert manager.get_status(CAPABILITY_ID) == CapabilityStatus.REVOKED
    assert ACTION_NAME not in planner_names(agent)
    assert agent.actions[ACTION_NAME] is action

    with pytest.raises(CapabilityRegistrationError):
        manager.activate(CAPABILITY_ID)

    for _ in range(3):
        agent.configure_planner()
        assert ACTION_NAME not in planner_names(agent)
        assert agent.actions[ACTION_NAME] is action

    assert manager.inspect().is_consistent


def test_real_agent_adapter_synchronization_adds_trusted_action_without_manual_registration():
    agent, action, registry, adapter, manager = make_integrated_stack()

    assert agent.actions[ACTION_NAME] is action
    assert ACTION_NAME in planner_names(agent)
    assert manager.inspect().is_consistent

    report = adapter.inspect()
    assert not report.conflicts
    assert ACTION_NAME in report.already_registered_action_names


def test_real_agent_configure_planner_rolls_back_entire_graph_when_hook_fails():
    agent = Agent(name="capability-lifecycle-hook-rollback-test")
    action = make_action("rollback_probe")
    agent.register_action(action)

    before = (
        agent.planner,
        agent.adaptive_planner,
        agent.hierarchical_planner,
        agent.risk_aware_hierarchical_planner,
        agent.unified_executor,
    )

    def fail_sync():
        raise RuntimeError("injected planner synchronization failure")

    agent.set_planner_sync_hook(fail_sync)

    with pytest.raises(RuntimeError, match="injected planner synchronization failure"):
        agent.configure_planner()

    after = (
        agent.planner,
        agent.adaptive_planner,
        agent.hierarchical_planner,
        agent.risk_aware_hierarchical_planner,
        agent.unified_executor,
    )
    assert all(current is previous for current, previous in zip(after, before))
    assert_planner_graph_is_shared(agent)


def test_real_agent_lifecycle_transition_rolls_back_registry_and_planner_on_hook_failure():
    agent = Agent(name="capability-lifecycle-transition-rollback-test")
    action = make_action()
    registry = TrustedCapabilityRegistry()
    registry.register_trusted(
        capability_id=CAPABILITY_ID,
        action=action,
        risk_level="low",
    )
    agent.register_action(action)

    adapter = DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)
    manager = CapabilityLifecycleManager(registry=registry, adapter=adapter)

    before_record = registry.get(CAPABILITY_ID)
    before_planner = agent.planner

    # Replace the callback implementation while retaining the exact installed
    # callback identity expected by the manager and Agent.
    original_sync = adapter._planner_sync_callback

    def fail_sync():
        raise RuntimeError("injected lifecycle synchronization failure")

    adapter._planner_sync_callback = fail_sync
    agent._planner_sync_hook = fail_sync
    try:
        with pytest.raises(CapabilityLifecycleSynchronizationError):
            manager.disable(CAPABILITY_ID)
    finally:
        adapter._planner_sync_callback = original_sync
        agent._planner_sync_hook = original_sync

    assert registry.get(CAPABILITY_ID) is before_record
    assert registry.get(CAPABILITY_ID).status == CapabilityStatus.ACTIVE
    assert agent.planner is before_planner
    assert ACTION_NAME in planner_names(agent)
    assert manager.inspect().is_consistent


def test_adapter_detach_stops_future_lifecycle_planner_synchronization():
    agent, action, registry, adapter, manager = make_integrated_stack()
    assert adapter.detach() is True

    # Manager construction is intentionally bound to an installed adapter hook;
    # after detach, a new manager must reject the detached adapter.
    from agent.llm.capability_lifecycle_manager import CapabilityLifecycleValidationError

    with pytest.raises(CapabilityLifecycleValidationError):
        CapabilityLifecycleManager(registry=registry, adapter=adapter)

    # The existing manager still references the same adapter, but the hook has
    # been detached. It must not be used to mutate lifecycle state silently.
    # This call is expected to fail closed because configure_planner no longer
    # invokes the adapter's planner filter.
    with pytest.raises(CapabilityLifecycleSynchronizationError):
        manager.disable(CAPABILITY_ID)

    assert registry.get(CAPABILITY_ID).status == CapabilityStatus.ACTIVE
    assert agent.actions[ACTION_NAME] is action



def test_inspection_marks_unsynchronized_active_capability_inconsistent():
    agent = Agent(name="capability-lifecycle-unsynchronized-inspection-test")
    action = make_action()
    registry = TrustedCapabilityRegistry()
    registry.register_trusted(
        capability_id=CAPABILITY_ID,
        action=action,
        risk_level="low",
    )
    adapter = DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)
    manager = CapabilityLifecycleManager(registry=registry, adapter=adapter)

    inspection = manager.inspect()

    assert ACTION_NAME in inspection.adapter_report.ready_action_names
    assert inspection.planner_sync_hook_attached is True
    assert inspection.is_consistent is False
    assert ACTION_NAME not in agent.actions
    assert ACTION_NAME not in planner_names(agent)
