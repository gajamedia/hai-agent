from agent.core.goal import Goal
from agent.learning.strategy import Strategy
from agent.memory import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore
from agent.cognition.planner import Planner
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector


WORLD_STATE = {
    "door": "locked",
    "box": "open",
    "key": "inside_box",
    "spare_key": "available",
}
GOAL_DESCRIPTION = "open the door"


def make_planner():
    base_planner = Planner([])
    hierarchical = HierarchicalPlanner(
        planner=base_planner,
        decomposer=GoalDecomposer(),
    )
    return RiskAwareHierarchicalPlanner(
        hierarchical_planner=hierarchical,
        risk_strategy_selector=RiskAwareStrategySelector(),
    )


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


def add_episode(store, memory_id, action, success):
    store.add(
        EpisodicMemory(
            memory_id=memory_id,
            content=f"{action} outcome",
            importance=1.0,
            confidence=1.0,
            source="evidence-robustness-test",
            tags=["experience", action],
            goal=GOAL_DESCRIPTION,
            action=action,
            observation_before=dict(WORLD_STATE),
            observation_after=None,
            result={"success": success},
            cycle=1,
        )
    )


def rank(store):
    return make_planner()._rank_with_episodic_memory(
        strategies=make_strategies(),
        goal=Goal(GOAL_DESCRIPTION),
        world_state=WORLD_STATE,
        episodic_memory=store,
    )


def test_single_success_does_not_overpower_repeated_failures():
    store = EpisodicMemoryStore()

    for index in range(8):
        add_episode(
            store, f"take-fail-{index}", "take_key", False
        )

    add_episode(store, "take-single-success", "take_key", True)

    for index in range(5):
        add_episode(
            store,
            f"spare-success-{index}",
            "find_spare_key",
            True,
        )

    ranked = rank(store)

    assert ranked[0][0].name == "spare-key"


def test_balanced_evidence_does_not_create_extreme_bias():
    store = EpisodicMemoryStore()

    for index in range(6):
        add_episode(
            store,
            f"take-balanced-{index}",
            "take_key",
            index % 2 == 0,
        )

    for index in range(6):
        add_episode(
            store,
            f"spare-balanced-{index}",
            "find_spare_key",
            index % 2 == 0,
        )

    ranked = rank(store)

    assert len(ranked) == 2
    assert abs(ranked[0][1] - ranked[1][1]) < 0.05
