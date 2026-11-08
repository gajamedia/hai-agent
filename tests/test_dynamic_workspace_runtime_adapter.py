from types import SimpleNamespace

import pytest

from agent.core.action import Action
from agent.llm.capability_registry import (
    CapabilityStatus,
    TrustedCapabilityRegistry,
)
from agent.llm.dynamic_workspace_runtime_adapter import (
    DynamicWorkspaceAdapterValidationError,
    DynamicWorkspaceRuntimeAdapter,
    DynamicWorkspaceSynchronizationError,
)


PLANNER_ATTRIBUTES = (
    "planner",
    "adaptive_planner",
    "hierarchical_planner",
    "risk_aware_hierarchical_planner",
    "unified_executor",
)


def make_action(name: str) -> Action:
    return Action(
        name=name,
        description=f"Test action: {name}",
        handler=lambda **kwargs: {"success": True, "message": "test"},
    )


class FakePlanner:
    def __init__(self, actions=None):
        self.actions = list(actions or [])


class FakeAgent:
    def __init__(self, actions=None):
        self.actions = dict(actions or {})
        self.configure_calls = 0
        self.fail_configuration = False
        self._planner_sync_hook = None
        self._set_planner_graph(list(self.actions.values()))

    def set_planner_sync_hook(self, hook):
        if not callable(hook):
            raise TypeError("hook must be callable")
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

    def _set_planner_graph(self, actions):
        self.planner = FakePlanner(actions)
        self.adaptive_planner = SimpleNamespace(planner=self.planner)
        self.hierarchical_planner = SimpleNamespace(planner=self.planner)
        self.risk_aware_hierarchical_planner = SimpleNamespace(
            hierarchical_planner=self.hierarchical_planner
        )
        self.unified_executor = object()

    def configure_planner(self):
        self.configure_calls += 1
        previous = {
            name: getattr(self, name)
            for name in PLANNER_ATTRIBUTES
        }
        try:
            self._set_planner_graph(list(self.actions.values()))
            if self._planner_sync_hook is not None:
                self._planner_sync_hook()
            if self.fail_configuration:
                raise RuntimeError("simulated planner failure")
        except BaseException:
            for name, value in previous.items():
                setattr(self, name, value)
            raise


def make_registry(*actions):
    registry = TrustedCapabilityRegistry()
    for action in actions:
        registry.register_trusted(
            capability_id=f"application.{action.name}",
            action=action,
            risk_level="low",
        )
    return registry


def make_adapter(registry, agent):
    return DynamicWorkspaceRuntimeAdapter(registry=registry, agent=agent)


# Inspection
def test_inspect_reports_active_capability_as_ready():
    action = make_action("inspect_room")
    report = make_adapter(make_registry(action), FakeAgent()).inspect()
    assert report.ready_action_names == ("inspect_room",)
    assert report.can_synchronize is True


def test_inspect_detects_same_action_object_already_registered():
    action = make_action("inspect_room")
    report = make_adapter(
        make_registry(action), FakeAgent({"inspect_room": action})
    ).inspect()
    assert report.already_registered_action_names == ("inspect_room",)
    assert report.conflicts == ()


def test_inspect_detects_action_name_collision():
    registry_action = make_action("inspect_room")
    agent_action = make_action("inspect_room")
    report = make_adapter(
        make_registry(registry_action),
        FakeAgent({"inspect_room": agent_action}),
    ).inspect()
    assert report.conflicts == ("agent_action_conflict:inspect_room",)
    assert report.can_synchronize is False


def test_inspect_reports_inactive_action_without_calling_it_a_conflict():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.disable("application.inspect_room")
    report = make_adapter(
        registry, FakeAgent({"inspect_room": action})
    ).inspect()
    assert report.inactive_registered_action_names == ("inspect_room",)
    assert report.can_synchronize is True


def test_inspect_does_not_include_disabled_capability_as_ready():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.disable("application.inspect_room")
    report = make_adapter(registry, FakeAgent()).inspect()
    assert report.ready == ()
    assert report.inactive_registered == ()


def test_inspect_does_not_include_revoked_capability_as_ready():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.revoke("application.inspect_room")
    report = make_adapter(registry, FakeAgent()).inspect()
    assert report.ready == ()
    assert report.inactive_registered == ()


def test_inspect_preserves_agent_action_registry():
    existing = make_action("open_box")
    action = make_action("inspect_room")
    agent = FakeAgent({"open_box": existing})
    before = dict(agent.actions)
    make_adapter(make_registry(action), agent).inspect()
    assert agent.actions == before
    assert agent.actions["open_box"] is existing


def test_inspect_does_not_register_ready_capability():
    action = make_action("inspect_room")
    agent = FakeAgent()
    make_adapter(make_registry(action), agent).inspect()
    assert "inspect_room" not in agent.actions


def test_report_summary_is_read_only():
    action = make_action("inspect_room")
    report = make_adapter(make_registry(action), FakeAgent()).inspect()
    assert report.summary["ready"] == 1
    with pytest.raises(TypeError):
        report.summary["ready"] = 100


def test_adapter_rejects_invalid_registry():
    with pytest.raises(DynamicWorkspaceAdapterValidationError):
        DynamicWorkspaceRuntimeAdapter(registry=None, agent=FakeAgent())


def test_adapter_rejects_agent_without_mutable_actions():
    with pytest.raises(DynamicWorkspaceAdapterValidationError):
        DynamicWorkspaceRuntimeAdapter(
            registry=TrustedCapabilityRegistry(),
            agent=SimpleNamespace(actions=None),
        )


# Agent registration
def test_synchronize_registers_active_capability():
    action = make_action("inspect_room")
    agent = FakeAgent()
    result = make_adapter(make_registry(action), agent).synchronize()
    assert agent.actions["inspect_room"] is action
    assert result.synchronized_action_names == ("inspect_room",)
    assert result.changed is True
    assert agent.configure_calls == 1


def test_synchronize_registers_multiple_actions_in_one_batch():
    first = make_action("inspect_room")
    second = make_action("read_terminal")
    agent = FakeAgent()
    result = make_adapter(make_registry(first, second), agent).synchronize()
    assert agent.actions["inspect_room"] is first
    assert agent.actions["read_terminal"] is second
    assert set(result.synchronized_action_names) == {"inspect_room", "read_terminal"}
    assert agent.configure_calls == 1


def test_synchronize_preserves_existing_agent_actions():
    existing = make_action("open_box")
    new_action = make_action("inspect_room")
    agent = FakeAgent({"open_box": existing})
    make_adapter(make_registry(new_action), agent).synchronize()
    assert agent.actions["open_box"] is existing
    assert agent.actions["inspect_room"] is new_action


def test_synchronize_is_idempotent():
    action = make_action("inspect_room")
    agent = FakeAgent()
    adapter = make_adapter(make_registry(action), agent)
    first = adapter.synchronize()
    second = adapter.synchronize()
    assert first.changed is True
    assert second.changed is False
    assert second.already_registered_action_names == ("inspect_room",)
    assert agent.configure_calls == 1


def test_synchronize_does_not_rebuild_planner_when_nothing_is_ready():
    action = make_action("inspect_room")
    agent = FakeAgent({"inspect_room": action})
    result = make_adapter(make_registry(action), agent).synchronize()
    assert result.changed is False
    assert agent.configure_calls == 0


def test_synchronize_rejects_action_name_collision():
    registry_action = make_action("inspect_room")
    agent_action = make_action("inspect_room")
    agent = FakeAgent({"inspect_room": agent_action})
    with pytest.raises(DynamicWorkspaceSynchronizationError):
        make_adapter(make_registry(registry_action), agent).synchronize()
    assert agent.actions["inspect_room"] is agent_action
    assert agent.configure_calls == 0


def test_synchronize_retains_disabled_action_in_agent_mapping():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.disable("application.inspect_room")
    agent = FakeAgent({"inspect_room": action})
    result = make_adapter(registry, agent).synchronize()
    assert result.changed is False
    assert agent.actions["inspect_room"] is action


def test_synchronize_retains_revoked_action_in_agent_mapping():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.revoke("application.inspect_room")
    agent = FakeAgent({"inspect_room": action})
    make_adapter(registry, agent).synchronize()
    assert agent.actions["inspect_room"] is action


def test_synchronize_rolls_back_actions_when_planner_fails():
    action = make_action("inspect_room")
    existing = make_action("open_box")
    agent = FakeAgent({"open_box": existing})
    agent.fail_configuration = True
    before = dict(agent.actions)
    with pytest.raises(DynamicWorkspaceSynchronizationError):
        make_adapter(make_registry(action), agent).synchronize()
    assert agent.actions == before
    assert agent.actions["open_box"] is existing
    assert "inspect_room" not in agent.actions


def test_synchronize_rolls_back_all_planner_references():
    action = make_action("inspect_room")
    agent = FakeAgent()
    before = {name: getattr(agent, name) for name in PLANNER_ATTRIBUTES}
    agent.fail_configuration = True
    with pytest.raises(DynamicWorkspaceSynchronizationError):
        make_adapter(make_registry(action), agent).synchronize()
    for name in PLANNER_ATTRIBUTES:
        assert getattr(agent, name) is before[name]


def test_synchronize_preserves_original_action_mapping_object():
    action = make_action("inspect_room")
    agent = FakeAgent()
    original_mapping = agent.actions
    make_adapter(make_registry(action), agent).synchronize()
    assert agent.actions is original_mapping


# Planner filtering
def test_synchronize_planner_excludes_disabled_action_but_keeps_agent_action():
    active = make_action("inspect_room")
    disabled = make_action("read_terminal")
    registry = make_registry(active, disabled)
    registry.disable("application.read_terminal")
    agent = FakeAgent({"inspect_room": active, "read_terminal": disabled})
    result = make_adapter(registry, agent).synchronize_planner()
    assert result.available_action_names == ("inspect_room",)
    assert result.excluded_action_names == ("read_terminal",)
    assert result.changed is True
    assert agent.planner.actions == [active]
    assert agent.actions["read_terminal"] is disabled


def test_synchronize_planner_excludes_revoked_action():
    revoked = make_action("read_terminal")
    registry = make_registry(revoked)
    registry.revoke("application.read_terminal")
    agent = FakeAgent({"read_terminal": revoked})
    result = make_adapter(registry, agent).synchronize_planner()
    assert result.excluded_action_names == ("read_terminal",)
    assert agent.planner.actions == []


def test_synchronize_planner_preserves_non_registry_actions():
    builtin = make_action("open_box")
    active = make_action("inspect_room")
    disabled = make_action("read_terminal")
    registry = make_registry(active, disabled)
    registry.disable("application.read_terminal")
    agent = FakeAgent({
        "open_box": builtin,
        "inspect_room": active,
        "read_terminal": disabled,
    })
    make_adapter(registry, agent).synchronize_planner()
    assert agent.planner.actions == [builtin, active]


def test_synchronize_planner_is_idempotent():
    action = make_action("inspect_room")
    registry = make_registry(action)
    agent = FakeAgent({"inspect_room": action})
    adapter = make_adapter(registry, agent)
    first = adapter.synchronize_planner()
    second = adapter.synchronize_planner()
    assert first.changed is False
    assert second.changed is False
    assert agent.planner.actions == [action]


def test_synchronize_planner_requires_shared_planner_references():
    action = make_action("inspect_room")
    agent = FakeAgent({"inspect_room": action})
    agent.adaptive_planner = SimpleNamespace(planner=FakePlanner([action]))
    with pytest.raises(DynamicWorkspaceSynchronizationError):
        make_adapter(make_registry(action), agent).synchronize_planner()


def test_synchronize_planner_fails_closed_on_identity_mismatch():
    registered = make_action("inspect_room")
    substituted = make_action("inspect_room")
    registry = make_registry(registered)
    agent = FakeAgent({"inspect_room": registered})
    agent.planner.actions[:] = [substituted]
    before = list(agent.planner.actions)
    with pytest.raises(DynamicWorkspaceSynchronizationError):
        make_adapter(registry, agent).synchronize_planner()
    assert agent.planner.actions == before


def test_synchronize_planner_preserves_list_identity():
    action = make_action("inspect_room")
    disabled = make_action("read_terminal")
    registry = make_registry(action, disabled)
    registry.disable("application.read_terminal")
    agent = FakeAgent({"inspect_room": action, "read_terminal": disabled})
    original = agent.planner.actions
    make_adapter(registry, agent).synchronize_planner()
    assert agent.planner.actions is original


def test_synchronize_planner_detects_active_action_missing_from_planner():
    action = make_action("inspect_room")
    registry = make_registry(action)
    agent = FakeAgent({"inspect_room": action})
    agent.planner.actions.clear()
    with pytest.raises(DynamicWorkspaceSynchronizationError):
        make_adapter(registry, agent).synchronize_planner()



# Automatic planner lifecycle synchronization
def test_adapter_installs_planner_sync_hook():
    action = make_action("inspect_room")
    agent = FakeAgent({"inspect_room": action})
    adapter = make_adapter(make_registry(action), agent)
    assert agent._planner_sync_hook is adapter._planner_sync_callback


def test_configure_planner_automatically_excludes_disabled_capability():
    action = make_action("inspect_room")
    registry = make_registry(action)
    agent = FakeAgent({"inspect_room": action})
    make_adapter(registry, agent)

    registry.disable("application.inspect_room")
    agent.configure_planner()

    assert agent.actions["inspect_room"] is action
    assert agent.planner.actions == []


def test_configure_planner_automatically_excludes_revoked_capability():
    action = make_action("inspect_room")
    registry = make_registry(action)
    agent = FakeAgent({"inspect_room": action})
    make_adapter(registry, agent)

    registry.revoke("application.inspect_room")
    agent.configure_planner()

    assert agent.actions["inspect_room"] is action
    assert agent.planner.actions == []


def test_configure_planner_reintroduces_reactivated_capability():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.disable("application.inspect_room")
    agent = FakeAgent({"inspect_room": action})
    adapter = make_adapter(registry, agent)

    # The first explicit rebuild applies the disabled state.
    agent.configure_planner()
    assert agent.planner.actions == []

    registry.activate("application.inspect_room")
    agent.configure_planner()

    assert agent.planner.actions == [action]



def test_reactivation_is_applied_by_next_planner_rebuild():
    action = make_action("inspect_room")
    registry = make_registry(action)
    registry.disable("application.inspect_room")
    agent = FakeAgent({"inspect_room": action})
    adapter = make_adapter(registry, agent)

    # First rebuild applies the disabled state through the installed hook.
    agent.configure_planner()
    assert agent.planner.actions == []
    assert agent.actions["inspect_room"] is action

    # Reactivation is reflected by the next full planner rebuild. The planner
    # is rebuilt from Agent.actions, then the hook validates/filters that list.
    registry.activate("application.inspect_room")
    agent.configure_planner()

    assert agent.planner.actions == [action]
    assert agent.actions["inspect_room"] is action
    assert adapter.synchronize_planner().changed is False


def test_detach_removes_only_this_adapters_hook():
    action = make_action("inspect_room")
    agent = FakeAgent({"inspect_room": action})
    adapter = make_adapter(make_registry(action), agent)

    assert adapter.detach() is True
    assert agent._planner_sync_hook is None
    assert adapter.detach() is False


def test_agent_rejects_a_second_different_planner_hook():
    agent = FakeAgent()
    first = make_adapter(TrustedCapabilityRegistry(), agent)

    with pytest.raises(DynamicWorkspaceAdapterValidationError):
        make_adapter(TrustedCapabilityRegistry(), agent)

    assert agent._planner_sync_hook is first._planner_sync_callback
