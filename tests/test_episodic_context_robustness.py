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
    "_context_ranking_helpers",
    _HELPER_PATH,
)
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

GOAL_DESCRIPTION = "open the door"


def add_episode(
    store,
    memory_id,
    goal,
    action,
    success,
    world_state,
):
    store.add(
        EpisodicMemory(
            memory_id=memory_id,
            content=f"{action}: {'success' if success else 'failure'}",
            importance=1.0,
            confidence=1.0,
            source="context-test",
            tags=["experience", action],
            goal=goal,
            action=action,
            observation_before=dict(world_state),
            observation_after=None,
            result={"success": success},
            cycle=1,
        )
    )


def test_unrelated_context_does_not_dominate_strategy_ranking():
    world_state = _HELPERS.make_world_state()
    store = EpisodicMemoryStore()

    # Relevant evidence: take_key failed in the matching goal/context.
    for index in range(8):
        add_episode(
            store,
            f"relevant-failure-{index}",
            GOAL_DESCRIPTION,
            "take_key",
            False,
            world_state,
        )

    # Relevant evidence: spare-key strategy succeeded.
    for index in range(8):
        add_episode(
            store,
            f"relevant-success-{index}",
            GOAL_DESCRIPTION,
            "find_spare_key",
            True,
            world_state,
        )

    # Many successes, but from another goal and a different context.
    unrelated_state = {
        "door": "open",
        "box": "closed",
        "key": "missing",
        "spare_key": "unavailable",
    }
    for index in range(30):
        add_episode(
            store,
            f"unrelated-success-{index}",
            "prepare the kitchen",
            "take_key",
            True,
            unrelated_state,
        )

    planner = _HELPERS.make_planner()
    ranked = planner._rank_with_episodic_memory(
        strategies=_HELPERS.make_strategies(),
        goal=Goal(GOAL_DESCRIPTION),
        world_state=world_state,
        episodic_memory=store,
    )

    assert len(ranked) == 2
    assert ranked[0][0].name == "spare-key"
    assert ranked[0][1] > ranked[1][1]
