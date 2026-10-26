import importlib.util
from pathlib import Path

from agent.core.goal import Goal
from agent.memory import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore


_HELPER_PATH = (
    Path(__file__).parent
    / "test_episodic_persistence_planner_ranking.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_ranking_helpers",
    _HELPER_PATH,
)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

GOAL_DESCRIPTION = "open the door"


def add_outcomes(store, action, success, count, world_state):
    for index in range(count):
        store.add(
            EpisodicMemory(
                memory_id=(
                    f"{action}-{success}-{index}"
                ),
                content=(
                    f"Action {action} "
                    f"{'succeeded' if success else 'failed'}"
                ),
                importance=1.0 if success else 0.8,
                confidence=1.0,
                source="robustness-test",
                tags=[
                    "experience",
                    action,
                    "success" if success else "failure",
                ],
                goal=GOAL_DESCRIPTION,
                action=action,
                observation_before=dict(world_state),
                observation_after=None,
                result={"success": success},
                cycle=index + 1,
            )
        )


def test_strategy_ranking_adapts_as_evidence_accumulates():
    world_state = _HELPERS.make_world_state()
    store = EpisodicMemoryStore()

    add_outcomes(
        store, "take_key", False, 8, world_state
    )
    add_outcomes(
        store, "find_spare_key", True, 8, world_state
    )

    planner = _HELPERS.make_planner()
    strategies = _HELPERS.make_strategies()
    goal = Goal(GOAL_DESCRIPTION)

    initial_ranking = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=goal,
        world_state=world_state,
        episodic_memory=store,
    )

    assert initial_ranking[0][0].name == "spare-key"

    # Add substantial new positive evidence for take_key.
    add_outcomes(
        store, "take_key", True, 40, world_state
    )

    updated_ranking = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=goal,
        world_state=world_state,
        episodic_memory=store,
    )

    assert updated_ranking[0][0].name == "take-key"
    assert updated_ranking[0][1] > updated_ranking[1][1]
