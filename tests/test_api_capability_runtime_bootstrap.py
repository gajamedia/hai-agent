from __future__ import annotations

from types import SimpleNamespace

from api.main import _create_composed_cognitive_capability_runtime


def test_capability_composition_is_skipped_for_execution_only_runtime_stub():
    cognitive_runtime = SimpleNamespace(agent=object())

    runtime = _create_composed_cognitive_capability_runtime(
        app=None,
        cognitive_runtime=cognitive_runtime,
    )

    assert runtime is None


def test_capability_composition_is_skipped_when_agent_lacks_planner_sync_hook():
    cognitive_runtime = SimpleNamespace(agent=SimpleNamespace(actions={}))

    runtime = _create_composed_cognitive_capability_runtime(
        app=None,
        cognitive_runtime=cognitive_runtime,
    )

    assert runtime is None
