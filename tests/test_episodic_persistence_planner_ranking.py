from agent.core.goal import Goal
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.cognition.planner import Planner
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector
from agent.core.action import Action
from agent.learning.strategy import Strategy
from agent.memory import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore
from agent.memory.persistence import (
    EpisodicMemoryRepository,
    SQLiteDatabase,
)

GOAL_DESCRIPTION = "open the door"


def make_world_state():
    return {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
        "spare_key": "available",
    }


def make_strategies():
    return [
        Strategy(
            name="take-key",
            goal=GOAL_DESCRIPTION,
            steps=["take_key", "unlock_door"],
        ),
        Strategy(
            name="spare-key",
            goal=GOAL_DESCRIPTION,
            steps=["find_spare_key", "unlock_door"],
        ),
    ]


def make_planner():
    actions = [
        Action(
            name="take_key",
            description="Take key from the box",
            handler=lambda **kwargs: {"success": True},
            preconditions={"box": "open", "key": "inside_box"},
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find the spare key",
            handler=lambda **kwargs: {"success": True},
            preconditions={"spare_key": "available"},
            effects={"spare_key": "found", "key": "agent"},
        ),
        Action(
            name="unlock_door",
            description="Unlock the door",
            handler=lambda **kwargs: {"success": True},
            preconditions={"key": "agent", "door": "locked"},
            effects={"door": "unlocked"},
        ),
    ]

    base_planner = Planner(actions)
    decomposer = GoalDecomposer()

    def decompose_open_door(goal):
        obtain_key = Goal("obtain the key")
        take_key = Goal(
            "take the key",
            target_state={"key": "agent"},
        )
        obtain_key.add_sub_goal(take_key)

        unlock_door = Goal(
            "unlock the door",
            target_state={"door": "unlocked"},
        )
        unlock_door.add_dependency(obtain_key)

        return [obtain_key, unlock_door]

    decomposer.register_rule(
        GOAL_DESCRIPTION,
        decompose_open_door,
    )

    hierarchical = HierarchicalPlanner(
        planner=base_planner,
        decomposer=decomposer,
    )

    return RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical,
        risk_strategy_selector=RiskAwareStrategySelector(),
    )


def add_episodes(store, action, success, count=12):
    for index in range(count):
        outcome = "success" if success else "failure"

        store.add(
            EpisodicMemory(
                memory_id=f"{action}-{success}-{index}",
                content=f"Past action {action}: {outcome}",
                importance=1.0 if success else 0.8,
                confidence=1.0,
                source="test",
                tags=["experience", action, outcome],
                goal=GOAL_DESCRIPTION,
                action=action,
                observation_before=make_world_state(),
                observation_after=None,
                result={"success": success},
                cycle=index + 1,
            )
        )


def test_persisted_episodes_change_planner_ranking_after_restart(
    tmp_path,
):
    database_path = tmp_path / "episodic_memory.db"

    database_before = SQLiteDatabase(database_path)
    repository_before = EpisodicMemoryRepository(database_before)
    store_before = EpisodicMemoryStore(repository=repository_before)

    add_episodes(store_before, "take_key", success=False)
    add_episodes(store_before, "find_spare_key", success=True)

    assert store_before.count() == 24
    assert repository_before.count() == 24

    # Recreate the persistence stack using the same SQLite database.
    del store_before
    del repository_before
    del database_before

    database_after = SQLiteDatabase(database_path)
    repository_after = EpisodicMemoryRepository(database_after)
    restored_store = EpisodicMemoryStore(repository=repository_after)

    assert restored_store.count() == 24

    # Retrieve all persisted episodes, then validate each action's outcome.
    restored_episodes = restored_store.retrieve(
        goal=GOAL_DESCRIPTION,
        context=make_world_state(),
        limit=100,
        min_score=0.0,
    )

    failures = [
        episode
        for episode in restored_episodes
        if episode.action == "take_key"
    ]
    successes = [
        episode
        for episode in restored_episodes
        if episode.action == "find_spare_key"
    ]

    assert len(restored_episodes) == 24
    assert len(failures) == 12
    assert all(
        episode.result["success"] is False
        for episode in failures
    )
    assert len(successes) == 12
    assert all(
        episode.result["success"] is True
        for episode in successes
    )

    planner = make_planner()
    ranked = planner._rank_with_episodic_memory(
        strategies=make_strategies(),
        goal=Goal(GOAL_DESCRIPTION),
        world_state=make_world_state(),
        episodic_memory=restored_store,
    )

    assert len(ranked) == 2
    assert ranked[0][0].name == "spare-key"
    assert ranked[0][1] > ranked[1][1]
