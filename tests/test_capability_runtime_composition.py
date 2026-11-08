from __future__ import annotations

from agent.core.action import Action
from types import SimpleNamespace
from agent.llm.cognitive_capability_runtime import (
    compose_cognitive_capability_runtime,
)


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.actions = {}
        self.planner = SimpleNamespace(actions=[])
        self._planner_sync_hook = None

    def register_action(self, action):
        self.actions[action.name] = action
        self.configure_planner()

    def set_planner_sync_hook(self, callback):
        self._planner_sync_hook = callback

    def clear_planner_sync_hook(self, callback):
        if self._planner_sync_hook is callback:
            self._planner_sync_hook = None
            return True
        return False

    def configure_planner(self):
        self.planner.actions = list(self.actions.values())
        if self._planner_sync_hook is not None:
            self._planner_sync_hook()


def test_composition_shares_one_registry_between_adapter_and_lifecycle_manager():
    agent = FakeAgent("p2-composition-test")

    runtime = compose_cognitive_capability_runtime(agent)

    assert runtime.agent is agent
    assert runtime.adapter is not None
    assert runtime.lifecycle_manager is not None
    assert runtime.adapter.registry is runtime.registry
    assert runtime.lifecycle_manager.registry is runtime.registry
    assert runtime.lifecycle_manager.adapter is runtime.adapter
    assert runtime.lifecycle_manager.agent is agent
    assert runtime.inspect()["composition_ready"] is True
    assert runtime.inspect()["execution_enabled"] is False

    runtime.adapter.detach()


def test_composition_does_not_merge_or_mutate_preexisting_agent_actions():
    agent = FakeAgent("p2-composition-preservation-test")
    existing = Action(
        name="existing_local_action",
        description="Existing local action",
        handler=lambda **kwargs: {"success": True},
    )
    agent.register_action(existing)
    before = dict(agent.actions)

    runtime = compose_cognitive_capability_runtime(agent)

    assert agent.actions == before
    assert agent.actions["existing_local_action"] is existing
    assert runtime.registry.snapshot().records == {}
    assert "environment.observe" in runtime.action_builder.capabilities
    assert "environment.observe" not in agent.actions

    runtime.adapter.detach()


def test_lifecycle_manager_controls_only_the_composed_registry():
    agent = FakeAgent("p2-composition-lifecycle-test")
    runtime = compose_cognitive_capability_runtime(agent)

    assert runtime.lifecycle_manager.inspect().is_consistent
    assert runtime.registry.list_all() == ()

    runtime.adapter.detach()
