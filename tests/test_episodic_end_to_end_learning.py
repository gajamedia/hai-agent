from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.exploration_executor import ExplorationExecutor
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.cognition.planner import Planner
from agent.cognition.risk_aware_hierarchical_planner import (
    RiskAwareHierarchicalPlanner,
)
from agent.cognition.risk_strategy_selector import RiskAwareStrategySelector
from agent.learning.strategy import Strategy
from agent.world.room import RoomEnvironment


GOAL_DESCRIPTION = "open the door"


def make_actions():
    return [
        Action(
            name="open_box",
            description="Open the box",
            handler=lambda **kwargs: {"success": True},
            preconditions={"box": "closed"},
            effects={"box": "open"},
        ),
        Action(
            name="take_key",
            description="Take the key from the box",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find the spare key",
            handler=lambda **kwargs: {"success": True},
            preconditions={"spare_key": "available"},
            effects={
                "spare_key": "found",
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
            effects={"door": "unlocked"},
        ),
    ]


def make_agent():
    agent = Agent(name="episodic-e2e-agent")
    agent.register_actions(make_actions())
    return agent


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


def record_action(agent, environment, action_name, success):
    """
    Execute an action through the real ExplorationExecutor.

    Failed attempts are created by arranging the environment state so
    the selected RoomEnvironment action cannot succeed.
    """
    action = agent.actions[action_name]

    if action_name == "take_key":
        environment.state["box"] = "closed"
        environment.state["key"] = "inside_box"
    elif action_name == "find_spare_key":
        environment.state["spare_key"] = (
            "available" if success else "unavailable"
        )
    else:
        raise AssertionError(
            f"Unsupported test action: {action_name}"
        )

    experience_id = (
        f"e2e-{action_name}-{'success' if success else 'failure'}-"
        f"{agent.episodic_memory.count()}"
    )
    agent.start_experience(
        experience_id=experience_id,
        goal=GOAL_DESCRIPTION,
    )

    try:
        result = ExplorationExecutor().execute(
            agent=agent,
            environment=environment,
            action=action,
        )
        assert result.success is success

        if success:
            agent.complete_experience()
        else:
            agent.fail_experience()
    finally:
        if agent.current_experience is not None:
            agent.close_experience()

    return agent.episodic_memory.latest()


def test_real_execution_memory_changes_next_strategy_ranking():
    agent = make_agent()
    environment = RoomEnvironment()
    planner = make_planner()
    strategies = make_strategies()
    goal = Goal(GOAL_DESCRIPTION)

    initial_count = agent.episodic_memory.count()

    # Generate real failures for the standard-key strategy.
    for _ in range(8):
        episode = record_action(
            agent,
            environment,
            action_name="take_key",
            success=False,
        )
        assert episode.action == "take_key"
        assert episode.goal == GOAL_DESCRIPTION
        assert episode.result["success"] is False

    # Generate real successes for the spare-key strategy.
    for _ in range(8):
        episode = record_action(
            agent,
            environment,
            action_name="find_spare_key",
            success=True,
        )
        assert episode.action == "find_spare_key"
        assert episode.goal == GOAL_DESCRIPTION
        assert episode.result["success"] is True

    store = agent.episodic_memory

    assert store.count() == initial_count + 16

    # Verify that the actual store retrieves the recorded evidence.
    failed_episodes = store.retrieve(
        goal=GOAL_DESCRIPTION,
        action="take_key",
        context=environment.observe(),
        limit=20,
        min_score=0.0,
    )
    successful_episodes = store.retrieve(
        goal=GOAL_DESCRIPTION,
        action="find_spare_key",
        context=environment.observe(),
        limit=20,
        min_score=0.0,
    )

    assert any(
        episode.result.get("success") is False
        for episode in failed_episodes
    )
    assert any(
        episode.result.get("success") is True
        for episode in successful_episodes
    )

    # Compare the ranking without memory to the ranking using the
    # same real memory store populated by executed actions.
    baseline = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=goal,
        world_state=environment.observe(),
        episodic_memory=None,
    )
    learned = planner._rank_with_episodic_memory(
        strategies=strategies,
        goal=goal,
        world_state=environment.observe(),
        episodic_memory=store,
    )

    assert len(baseline) == 2
    assert len(learned) == 2
    assert learned[0][0].name == "spare-key"
    assert learned[0][1] > learned[1][1]
