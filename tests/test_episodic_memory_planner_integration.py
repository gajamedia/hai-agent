
from types import SimpleNamespace

from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.agent_decision import AgentDecision, DecisionMode
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.cognition.planner import Planner
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector
from agent.cognition.unified_executor import UnifiedDecisionExecutor
from agent.learning.strategy import Strategy


GOAL_DESCRIPTION = "open the door"


class FakeEpisodicMemory:
    """Test double that returns episodes indexed by action."""

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


def make_episode(action, success, goal=GOAL_DESCRIPTION):
    return SimpleNamespace(
        action=action,
        goal=goal,
        result={"success": success},
    )


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


def make_decision():
    return AgentDecision(
        mode=DecisionMode.EXPLOIT,
        reason="Use known experience to achieve the goal.",
        uncertainty=0.1,
        expected_information_gain=0.1,
        risk=0.1,
    )


def test_executor_forwards_agent_episodic_memory_to_planner():
    class PlannerSpy:
        def __init__(self):
            self.kwargs = None

        def plan(self, **kwargs):
            self.kwargs = kwargs
            return []

    memory = FakeEpisodicMemory()
    agent = SimpleNamespace(episodic_memory=memory)
    planner_spy = PlannerSpy()
    executor = UnifiedDecisionExecutor(
        hierarchical_planner=planner_spy,
    )

    result = executor.execute(
        make_decision(),
        agent=agent,
        goal=make_goal(),
        world_state=make_world_state(),
        strategies=[],
    )

    assert result.plan == []
    assert planner_spy.kwargs is not None
    assert planner_spy.kwargs["episodic_memory"] is memory


def test_executor_preserves_legacy_call_without_agent_memory():
    class PlannerSpy:
        def __init__(self):
            self.kwargs = None

        def plan(self, **kwargs):
            self.kwargs = kwargs
            return []

    planner_spy = PlannerSpy()
    executor = UnifiedDecisionExecutor(
        hierarchical_planner=planner_spy,
    )

    executor.execute(
        make_decision(),
        goal=make_goal(),
        world_state=make_world_state(),
        strategies=[],
    )

    assert "episodic_memory" not in planner_spy.kwargs


def test_real_planner_uses_episode_outcomes_to_adjust_strategy_ranking():
    planner = make_planner()
    strategies = make_strategies()

    memory = FakeEpisodicMemory(
        {
            "take_key": [
                make_episode("take_key", False)
                for _ in range(20)
            ],
            "find_spare_key": [
                make_episode("find_spare_key", True)
                for _ in range(20)
            ],
        }
    )

    ranked = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=memory,
    )

    assert len(ranked) == 2
    assert ranked[0][0].name == "spare-key"
    assert ranked[0][1] > ranked[1][1]
    assert memory.calls
    assert all(call["limit"] == 20 for call in memory.calls)


def test_real_planner_falls_back_when_memory_retrieval_fails():
    planner = make_planner()
    strategies = make_strategies()

    baseline = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=None,
    )

    memory = FakeEpisodicMemory(
        error=RuntimeError("episodic backend unavailable"),
    )

    actual = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=memory,
    )

    assert actual == baseline
    assert memory.calls


def test_real_planner_ignores_episodes_from_another_goal():
    planner = make_planner()
    strategies = make_strategies()

    baseline = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=None,
    )

    memory = FakeEpisodicMemory(
        {
            "take_key": [
                make_episode(
                    "take_key",
                    False,
                    goal="different goal",
                )
                for _ in range(10)
            ],
            "find_spare_key": [
                make_episode(
                    "find_spare_key",
                    True,
                    goal="different goal",
                )
                for _ in range(10)
            ],
        }
    )

    actual = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=make_goal(),
        world_state=make_world_state(),
        episodic_memory=memory,
    )

    assert actual == baseline


def test_strategy_validation_still_rejects_unexecutable_actions():
    planner = make_planner()

    # The action is registered, but its preconditions are not satisfied.
    invalid_strategy = make_strategy(
        "invalid-high-confidence",
        ["take_key", "unlock_door"],
    )

    memory = FakeEpisodicMemory(
        {
            "take_key": [
                make_episode("take_key", True)
                for _ in range(20)
            ],
            "unlock_door": [
                make_episode("unlock_door", True)
                for _ in range(20)
            ],
        }
    )

    plan = planner.plan(
        goal=make_goal(),
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "available",
        },
        strategies=[invalid_strategy],
        episodic_memory=memory,
    )

    # The regular hierarchical fallback should still be returned.
    # It should not return the invalid strategy's unexecutable plan.
    assert [action.name for action in plan] != [
        "take_key",
        "unlock_door",
    ]
