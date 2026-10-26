from types import SimpleNamespace

import pytest

from agent.core.goal import Goal
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.learning.strategy import Strategy


class FakeEpisodicMemory:
    """Minimal episodic store test double keyed by action name."""

    def __init__(self, episodes_by_action=None, error=None):
        self.episodes_by_action = episodes_by_action or {}
        self.error = error
        self.calls = []

    def retrieve(
        self,
        *,
        goal=None,
        action=None,
        context=None,
        limit=5,
        min_score=0.0,
    ):
        self.calls.append(
            {
                "goal": goal,
                "action": action,
                "context": context,
                "limit": limit,
                "min_score": min_score,
            }
        )
        if self.error is not None:
            raise self.error
        return list(self.episodes_by_action.get(action, []))


def make_planner():
    # The episodic ranking helper only needs the selector. The real planner
    # is intentionally None here so the test isolates ranking behavior.
    planner = RiskAwareHierarchicalPlanner.__new__(
        RiskAwareHierarchicalPlanner
    )
    planner.risk_strategy_selector = RiskAwareStrategySelector()
    return planner


def make_strategy(name, step):
    return Strategy(
        name=name,
        goal="open the door",
        steps=[step],
    )


def episode(action, success, goal="open the door"):
    return SimpleNamespace(
        action=action,
        goal=goal,
        result={"success": success},
    )


def rank(planner, strategies, memory):
    return planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=Goal("open the door"),
        world_state={"door": "locked"},
        episodic_memory=memory,
    )


def test_relevant_successful_episodes_improve_strategy_ranking():
    planner = make_planner()
    successful = make_strategy("successful", "take_key")
    unsuccessful = make_strategy("unsuccessful", "find_spare_key")
    memory = FakeEpisodicMemory(
        {
            "take_key": [episode("take_key", True) for _ in range(8)],
            "find_spare_key": [episode("find_spare_key", False) for _ in range(8)],
        }
    )

    ranked = rank(planner, [unsuccessful, successful], memory)

    assert [strategy.name for strategy, _ in ranked] == [
        "successful",
        "unsuccessful",
    ]
    assert ranked[0][1] > ranked[1][1]


def test_relevant_failed_episodes_lower_strategy_ranking():
    planner = make_planner()
    mostly_successful = make_strategy("mostly-successful", "take_key")
    mostly_failed = make_strategy("mostly-failed", "find_spare_key")
    memory = FakeEpisodicMemory(
        {
            "take_key": [episode("take_key", True) for _ in range(8)],
            "find_spare_key": [episode("find_spare_key", False) for _ in range(8)],
        }
    )

    ranked = rank(planner, [mostly_failed, mostly_successful], memory)

    assert ranked[0][0] is mostly_successful
    assert ranked[1][0] is mostly_failed


def test_unrelated_goal_episodes_do_not_change_base_ranking():
    planner = make_planner()
    first = make_strategy("first", "take_key")
    second = make_strategy("second", "find_spare_key")
    strategies = [first, second]
    memory = FakeEpisodicMemory(
        {
            "take_key": [
                episode("take_key", True, goal="different goal")
                for _ in range(10)
            ],
            "find_spare_key": [
                episode("find_spare_key", False, goal="different goal")
                for _ in range(10)
            ],
        }
    )

    base = planner.risk_strategy_selector.rank(strategies)
    ranked = rank(planner, strategies, memory)

    assert [(item.name, score) for item, score in ranked] == [
        (item.name, score) for item, score in base
    ]


def test_episode_for_nonmatching_action_is_ignored():
    planner = make_planner()
    strategy = make_strategy("candidate", "take_key")
    memory = FakeEpisodicMemory(
        {
            "take_key": [
                episode("some_other_action", False)
                for _ in range(10)
            ]
        }
    )

    base = planner.risk_strategy_selector.rank([strategy])
    ranked = rank(planner, [strategy], memory)

    assert ranked == base


def test_missing_memory_preserves_original_selector_ranking():
    planner = make_planner()
    strategies = [
        make_strategy("first", "take_key"),
        make_strategy("second", "find_spare_key"),
    ]

    base = planner.risk_strategy_selector.rank(strategies)
    ranked = rank(planner, strategies, None)

    assert ranked == base


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("episodic backend unavailable"),
        ValueError("invalid retrieval request"),
    ],
)
def test_retrieval_error_preserves_original_selector_ranking(error):
    planner = make_planner()
    strategies = [
        make_strategy("first", "take_key"),
        make_strategy("second", "find_spare_key"),
    ]
    memory = FakeEpisodicMemory(error=error)

    base = planner.risk_strategy_selector.rank(strategies)
    ranked = rank(planner, strategies, memory)

    assert ranked == base


def test_episode_without_boolean_success_is_ignored():
    planner = make_planner()
    strategy = make_strategy("candidate", "take_key")
    malformed_episode = SimpleNamespace(
        action="take_key",
        goal="open the door",
        result={"success": "yes"},
    )
    memory = FakeEpisodicMemory(
        {"take_key": [malformed_episode]}
    )

    base = planner.risk_strategy_selector.rank([strategy])
    ranked = rank(planner, [strategy], memory)

    assert ranked == base
