
import pytest
from unittest.mock import patch

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.memory.persistence.database import SQLiteDatabase
from agent.memory.persistence.procedural_repository import (
    ProceduralMemoryRepository,
)
from agent.memory.procedural_store import ProceduralMemoryStore


def create_persistent_store(database_path):
    database = SQLiteDatabase(database_path)
    repository = ProceduralMemoryRepository(database)
    return ProceduralMemoryStore(repository=repository)


def create_agent():
    agent = Agent(name="procedural-persistence-agent")

    agent.register_actions([
        Action(
            name="take_key",
            description="Take the key",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock the door",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ])

    return agent


def create_memory_with_outcomes(store):
    memory = store.get_or_create(
        "open_the_door",
        ["take_key", "unlock_door"],
    )

    memory.record_success(["episode-success"])
    memory.record_failure(["episode-failure-1"])
    memory.record_failure(["episode-failure-2"])

    store.save_all()

    return memory


def test_procedural_memory_outcomes_survive_store_reload(
    tmp_path,
):
    database_path = tmp_path / "procedural.sqlite3"

    store = create_persistent_store(database_path)
    original = create_memory_with_outcomes(store)

    assert original.success_count == 1
    assert original.failure_count == 2
    assert original.confidence == pytest.approx(0.4)

    # Simulate a restart by constructing a new store
    # and repository using the same database file.
    restored_store = create_persistent_store(database_path)
    restored = restored_store.latest()

    assert restored is not None
    assert restored.memory_id == original.memory_id
    assert restored.skill == "open_the_door"
    assert restored.steps == [
        "take_key",
        "unlock_door",
    ]
    assert restored.success_count == 1
    assert restored.failure_count == 2
    assert restored.confidence == pytest.approx(0.4)

    assert set(restored.source_episodes) == {
        "episode-success",
        "episode-failure-1",
        "episode-failure-2",
    }


def test_restored_unreliable_procedure_uses_symbolic_fallback(
    tmp_path,
):
    database_path = tmp_path / "procedural.sqlite3"

    store = create_persistent_store(database_path)
    create_memory_with_outcomes(store)

    # Load the learned procedure into a fresh store.
    restored_store = create_persistent_store(database_path)

    agent = create_agent()

    planner = HierarchicalPlanner(
        planner=agent.planner,
        decomposer=GoalDecomposer(),
        procedural_store=restored_store,
        causal_store=None,
    )

    goal = Goal(
        description="open the door",
        target_state={
            "door": "unlocked",
        },
    )

    world_state = {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
    }

    with patch.object(
        agent.planner,
        "plan",
        wraps=agent.planner.plan,
    ) as symbolic_plan:
        result = planner.plan(
            goal=goal,
            world_state=world_state,
        )

    # The restored procedure has more failures than successes.
    # Therefore, classical symbolic planning must be used.
    symbolic_plan.assert_called_once()

    assert [
        action.name for action in result
    ] == [
        "take_key",
        "unlock_door",
    ]
