
from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.cognition.planner import Planner
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector
from agent.learning.strategy import Strategy
from agent.memory import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore


GOAL_DESCRIPTION = "open the door"


def make_actions():
    return [
        Action(
            name="take_key",
            description="Take the key from the open box",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find the available spare key",
            handler=lambda **kwargs: {"success": True},
            preconditions={"spare_key": "available"},
            effects={
                "spare_key": "found",
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock the locked door",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={"door": "unlocked"},
        ),
    ]


def make_strategy(name, steps):
    return Strategy(
        name=name,
        goal=GOAL_DESCRIPTION,
        steps=steps,
    )


def make_strategies():
    return [
        make_strategy("take-key", ["take_key", "unlock_door"]),
        make_strategy(
            "spare-key",
            ["find_spare_key", "unlock_door"],
        ),
    ]


def make_goal():
    return Goal(GOAL_DESCRIPTION)


def make_world_state():
    return {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
        "spare_key": "available",
    }


def make_planner():
    actions = make_actions()
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
    """Insert real EpisodicMemory instances into the actual store."""
    for index in range(count):
        store.add(
            EpisodicMemory(
                memory_id=f"{action}-{success}-{index}",
                content=(
                    f"Past action {action}: "
                    f"{'success' if success else 'failure'}"
                ),
                goal=GOAL_DESCRIPTION,
                action=action,
                observation_before=make_world_state(),
                result={"success": success},
                cycle=index + 1,
            )
        )


def test_real_episodic_store_changes_strategy_ranking():
    planner = make_planner()
    strategies = make_strategies()
    store = EpisodicMemoryStore()

    add_episodes(store, "take_key", success=False)
    add_episodes(store, "find_spare_key", success=True)

    ranked = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=store,
    )

    assert store.count() == 24
    assert len(ranked) == 2
    assert ranked[0][0].name == "spare-key"
    assert ranked[0][1] > ranked[1][1]


def test_real_episodic_store_retrieves_matching_action_and_goal():
    store = EpisodicMemoryStore()

    expected = EpisodicMemory(
        memory_id="matching-episode",
        content="Successfully took the key.",
        goal=GOAL_DESCRIPTION,
        action="take_key",
        observation_before=make_world_state(),
        result={"success": True},
        cycle=1,
    )

    unrelated_action = EpisodicMemory(
        memory_id="unrelated-action",
        content="Opened the box.",
        goal=GOAL_DESCRIPTION,
        action="open_box",
        observation_before=make_world_state(),
        result={"success": False},
        cycle=2,
    )

    unrelated_goal = EpisodicMemory(
        memory_id="unrelated-goal",
        content="Failed to take a key for another goal.",
        goal="find a treasure",
        action="take_key",
        observation_before=make_world_state(),
        result={"success": False},
        cycle=3,
    )

    store.add(expected)
    store.add(unrelated_action)
    store.add(unrelated_goal)

    retrieved = store.retrieve(
        goal=GOAL_DESCRIPTION,
        action="take_key",
        context=make_world_state(),
        limit=20,
        min_score=0.0,
    )

    assert expected in retrieved


def test_empty_real_episodic_store_preserves_base_ranking():
    planner = make_planner()
    strategies = make_strategies()
    store = EpisodicMemoryStore()

    baseline = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=None,
    )

    actual = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=store,
    )

    assert store.count() == 0
    assert actual == baseline
