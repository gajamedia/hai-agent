from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.core.action import Action
from agent.llm.cognitive_capability_runtime import (
    CognitiveCapabilityRuntimeError,
    create_cognitive_capability_runtime,
)
from agent.llm.cognitive_environment import (
    CognitiveEnvironment,
    CognitiveEnvironmentValidationError,
)


class FakeAgent:
    def __init__(self):
        self.actions = {
            "existing_action": Action(
                name="existing_action",
                description="Existing action",
                handler=lambda **kwargs: {"success": True},
            )
        }
        self.register_action_calls = 0

    def register_action(self, action):
        self.register_action_calls += 1
        self.actions[action.name] = action


def test_environment_starts_with_empty_observation():
    environment = CognitiveEnvironment()

    snapshot = environment.observe()

    assert snapshot["version"] == 0
    assert dict(snapshot["observation"]) == {}


def test_environment_publish_returns_detached_immutable_snapshot():
    source = {"world": {"items": ["key", "box"]}}
    environment = CognitiveEnvironment()

    assert environment.publish(source) == 1
    source["world"]["items"].append("door")

    snapshot = environment.observe()
    assert snapshot["version"] == 1
    assert snapshot["observation"]["world"]["items"] == ("key", "box")

    with pytest.raises(TypeError):
        snapshot["observation"]["new"] = "value"

    with pytest.raises(TypeError):
        snapshot["observation"]["world"]["new"] = "value"


def test_environment_rejects_non_mapping_observation():
    environment = CognitiveEnvironment()

    with pytest.raises(CognitiveEnvironmentValidationError):
        environment.publish(["not", "a", "mapping"])


def test_environment_rejects_invalid_keys():
    environment = CognitiveEnvironment()

    with pytest.raises(CognitiveEnvironmentValidationError):
        environment.publish({"": "invalid"})


def test_capability_runtime_is_empty_and_does_not_mutate_agent():
    agent = FakeAgent()
    before = dict(agent.actions)

    runtime = create_cognitive_capability_runtime(agent)

    assert runtime.agent is agent
    assert runtime.registry.snapshot().records == {}
    assert runtime.active_capability_count == 0
    assert runtime.adapter is None
    assert runtime.lifecycle_manager is None
    assert runtime.inspect()["execution_enabled"] is False
    assert agent.actions == before
    assert agent.register_action_calls == 0


def test_capability_runtime_has_only_application_owned_observe_builder_entry():
    agent = FakeAgent()
    runtime = create_cognitive_capability_runtime(agent)

    assert set(runtime.action_builder.capabilities) == {"environment.observe"}
    assert runtime.registry.snapshot().records == {}
    assert "environment.observe" not in agent.actions


def test_factory_requires_agent():
    with pytest.raises(CognitiveCapabilityRuntimeError):
        create_cognitive_capability_runtime(None)
