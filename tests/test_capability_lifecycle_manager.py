from types import SimpleNamespace

import pytest

from agent.core.action import Action
from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityRegistrationError,
    CapabilityStatus,
    TrustedCapabilityRegistry,
)
from agent.llm.capability_lifecycle_manager import (
    CapabilityLifecycleManager,
    CapabilityLifecycleSynchronizationError,
    CapabilityLifecycleValidationError,
)
from agent.llm.dynamic_workspace_runtime_adapter import (
    DynamicWorkspaceRuntimeAdapter,
)


def make_action(name="inspect_room"):
    return Action(
        name=name,
        description=f"Test action: {name}",
        handler=lambda **kwargs: {"success": True},
    )


class FakePlanner:
    def __init__(self, actions=None):
        self.actions = list(actions or [])


class FakeAgent:
    def __init__(self, actions=None):
        self.actions = dict(actions or {})
        self._planner_sync_hook = None
        self.fail_configuration = False
        self.configure_calls = 0
        self._set_graph()

    def _set_graph(self):
        self.planner = FakePlanner(self.actions.values())
        self.adaptive_planner = SimpleNamespace(planner=self.planner)
        self.hierarchical_planner = SimpleNamespace(planner=self.planner)
        self.risk_aware_hierarchical_planner = SimpleNamespace(
            hierarchical_planner=self.hierarchical_planner
        )
        self.unified_executor = object()

    def set_planner_sync_hook(self, hook):
        if self._planner_sync_hook is not None and self._planner_sync_hook is not hook:
            raise RuntimeError("hook already installed")
        self._planner_sync_hook = hook

    def clear_planner_sync_hook(self, hook=None):
        if self._planner_sync_hook is None:
            return False
        if hook is not None and self._planner_sync_hook is not hook:
            return False
        self._planner_sync_hook = None
        return True

    def configure_planner(self):
        self.configure_calls += 1
        previous = (
            self.planner,
            self.adaptive_planner,
            self.hierarchical_planner,
            self.risk_aware_hierarchical_planner,
            self.unified_executor,
        )
        try:
            self._set_graph()
            if self._planner_sync_hook is not None:
                self._planner_sync_hook()
            if self.fail_configuration:
                raise RuntimeError("simulated configure failure")
        except BaseException:
            (
                self.planner,
                self.adaptive_planner,
                self.hierarchical_planner,
                self.risk_aware_hierarchical_planner,
                self.unified_executor,
            ) = previous
            raise


def make_manager(*, status=CapabilityStatus.ACTIVE, fail_configuration=False):
    action = make_action()
    registry = TrustedCapabilityRegistry()
    registry.register_trusted(
        capability_id="application.inspect_room",
        action=action,
        risk_level="low",
    )
    if status == CapabilityStatus.DISABLED:
        registry.disable("application.inspect_room")
    elif status == CapabilityStatus.REVOKED:
        registry.revoke("application.inspect_room")

    agent = FakeAgent({"inspect_room": action})
    agent.fail_configuration = fail_configuration
    adapter = DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)
    manager = CapabilityLifecycleManager(registry=registry, adapter=adapter)
    return manager, registry, agent, action


def test_disable_updates_registry_and_removes_action_from_planner():
    manager, registry, agent, action = make_manager()

    result = manager.disable("application.inspect_room")

    assert result.previous_status == CapabilityStatus.ACTIVE
    assert result.status == CapabilityStatus.DISABLED
    assert result.changed is True
    assert registry.get("application.inspect_room", include_inactive=True).status == CapabilityStatus.DISABLED
    assert agent.planner.actions == []
    assert agent.actions["inspect_room"] is action


def test_revoke_updates_registry_and_removes_action_from_planner():
    manager, registry, agent, action = make_manager()

    result = manager.revoke("application.inspect_room")

    assert result.status == CapabilityStatus.REVOKED
    assert agent.planner.actions == []
    assert agent.actions["inspect_room"] is action


def test_activate_restores_disabled_capability_to_planner():
    manager, registry, agent, action = make_manager(status=CapabilityStatus.DISABLED)

    result = manager.activate("application.inspect_room")

    assert result.previous_status == CapabilityStatus.DISABLED
    assert result.status == CapabilityStatus.ACTIVE
    assert agent.planner.actions == [action]


def test_revoke_is_terminal():
    manager, registry, agent, _ = make_manager()

    manager.revoke("application.inspect_room")

    with pytest.raises(CapabilityRegistrationError):
        manager.activate("application.inspect_room")

    assert manager.get_status("application.inspect_room") == CapabilityStatus.REVOKED
    assert agent.planner.actions == []


def test_disable_is_idempotent_and_does_not_increment_record_version():
    manager, registry, agent, _ = make_manager()

    first = manager.disable("application.inspect_room")
    second = manager.disable("application.inspect_room")

    assert first.changed is True
    assert second.changed is False
    assert first.record_version == second.record_version
    assert agent.planner.actions == []


def test_activation_of_already_active_capability_is_idempotent():
    manager, registry, agent, action = make_manager()

    first = manager.activate("application.inspect_room")
    second = manager.activate("application.inspect_room")

    assert first.changed is False
    assert second.changed is False
    assert agent.planner.actions == [action]


def test_planner_failure_rolls_back_disable_status_and_record():
    manager, registry, agent, action = make_manager()
    before_record = registry.get("application.inspect_room")
    before_registry_version = registry.snapshot().version
    before_planner = agent.planner

    agent.fail_configuration = True
    with pytest.raises(CapabilityLifecycleSynchronizationError):
        manager.disable("application.inspect_room")

    restored = registry.get("application.inspect_room")
    assert restored is before_record
    assert restored.status == CapabilityStatus.ACTIVE
    assert registry.snapshot().version > before_registry_version
    assert agent.planner is before_planner
    assert agent.planner.actions == [action]


def test_planner_failure_rolls_back_revoke_instead_of_leaving_terminal_state():
    manager, registry, agent, action = make_manager()
    before_record = registry.get("application.inspect_room")
    before_planner = agent.planner

    agent.fail_configuration = True
    with pytest.raises(CapabilityLifecycleSynchronizationError):
        manager.revoke("application.inspect_room")

    assert registry.get("application.inspect_room") is before_record
    assert registry.get("application.inspect_room").status == CapabilityStatus.ACTIVE
    assert agent.planner is before_planner
    assert agent.planner.actions == [action]


def test_planner_failure_rolls_back_activation():
    manager, registry, agent, action = make_manager(status=CapabilityStatus.DISABLED)
    before_record = registry.get("application.inspect_room", include_inactive=True)
    before_planner = agent.planner

    agent.fail_configuration = True
    with pytest.raises(CapabilityLifecycleSynchronizationError):
        manager.activate("application.inspect_room")

    assert registry.get("application.inspect_room", include_inactive=True) is before_record
    assert manager.get_status("application.inspect_room") == CapabilityStatus.DISABLED
    assert agent.planner is before_planner
    assert agent.planner.actions == [action]


def test_builtin_capability_cannot_be_disabled():
    action = make_action("open_box")
    registry = TrustedCapabilityRegistry(builtin_actions=[action])
    agent = FakeAgent({"open_box": action})
    adapter = DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)
    manager = CapabilityLifecycleManager(registry=registry, adapter=adapter)

    with pytest.raises(CapabilityLifecycleValidationError):
        manager.disable("builtin.open_box")

    assert registry.get("builtin.open_box").status == CapabilityStatus.ACTIVE
    assert agent.planner.actions == [action]


def test_unknown_capability_raises_not_found():
    manager, _, _, _ = make_manager()

    with pytest.raises(CapabilityNotFoundError):
        manager.disable("application.unknown")


def test_inspection_reports_consistent_active_capability():
    manager, _, _, _ = make_manager()

    report = manager.inspect()

    assert report.is_consistent is True
    assert report.missing_active_action_names == ()
    assert report.inactive_planner_action_names == ()


def test_inspection_detects_stale_planner_state():
    manager, registry, agent, action = make_manager()

    registry.disable("application.inspect_room")
    report = manager.inspect()

    assert report.is_consistent is False
    assert report.inactive_planner_action_names == ("inspect_room",)


def test_manager_rejects_adapter_for_different_registry():
    registry = TrustedCapabilityRegistry()
    other_registry = TrustedCapabilityRegistry()
    action = make_action()
    registry.register_trusted(
        capability_id="application.inspect_room",
        action=action,
        risk_level="low",
    )
    agent = FakeAgent({"inspect_room": action})
    adapter = DynamicWorkspaceRuntimeAdapter(registry=other_registry, agent=agent)

    with pytest.raises(CapabilityLifecycleValidationError):
        CapabilityLifecycleManager(registry=registry, adapter=adapter)


def test_manager_rejects_detached_adapter():
    manager, _, _, _ = make_manager()
    manager.adapter.detach()

    with pytest.raises(CapabilityLifecycleValidationError):
        CapabilityLifecycleManager(
            registry=manager.registry,
            adapter=manager.adapter,
        )


def test_status_type_is_validated():
    manager, _, _, _ = make_manager()

    with pytest.raises(CapabilityLifecycleValidationError):
        manager.transition("application.inspect_room", "disabled")


def test_inspection_reports_active_action_missing_from_agent_mapping():
    manager, _, agent, _ = make_manager()
    agent.actions.pop("inspect_room")

    report = manager.inspect()

    assert report.active_registry_action_names == ("inspect_room",)
    assert report.missing_active_action_names == ("inspect_room",)
    assert report.is_consistent is False
    assert report.adapter_report.ready_action_names == ("inspect_room",)


def test_inspection_reports_active_action_missing_from_planner():
    manager, _, agent, action = make_manager()
    agent.planner.actions.clear()

    report = manager.inspect()

    assert report.active_registry_action_names == ("inspect_room",)
    assert report.missing_active_action_names == ("inspect_room",)
    assert report.is_consistent is False
    assert report.adapter_report.already_registered_action_names == ("inspect_room",)
    assert action not in agent.planner.actions


def test_inspection_reports_active_action_mapping_identity_conflict():
    manager, _, agent, _ = make_manager()
    replacement = make_action("inspect_room")
    agent.actions["inspect_room"] = replacement

    report = manager.inspect()

    assert report.active_registry_action_names == ("inspect_room",)
    assert report.missing_active_action_names == ("inspect_room",)
    assert report.is_consistent is False
    assert report.adapter_report.conflicts == ("agent_action_conflict:inspect_room",)


def test_inspection_reports_detached_hook_as_inconsistent():
    manager, _, _, _ = make_manager()
    assert manager.adapter.detach() is True

    report = manager.inspect()

    assert report.planner_sync_hook_attached is False
    assert report.is_consistent is False

