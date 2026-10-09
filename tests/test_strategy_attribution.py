from agent.core.agent import Agent
from agent.core.experience import Experience
from agent.core.goal import Goal
from agent.core.action import Action
from agent.learning.strategy import Strategy
from agent.learning.strategy_attribution import (
    StrategyAttributor,
)


def create_episode(
    action,
    success=True,
    cycle=0,
):
    class Episode:
        pass

    episode = Episode()

    episode.action = action

    episode.result = {
        "success": success,
    }

    episode.observation_before = {}

    episode.observation_after = {}

    episode.cycle = cycle

    return episode

def create_experience(
    goal,
    actions,
):
    experience = Experience(
        experience_id="experience-1",
        goal=goal,
    )

    for cycle, action in enumerate(
        actions
    ):
        experience.add_episode(
            create_episode(
                action=action,
                cycle=cycle,
            )
        )

    return experience

def test_strategy_matches_goal_and_actions():

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    experience = create_experience(
        goal="open the box",
        actions=[
            "open_box",
        ],
    )

    attributor = StrategyAttributor()

    result = attributor.attribute(
        experience=experience,
        strategies=[
            strategy,
        ],
    )

    assert result.matched is True

    assert (
        result.strategy
        is strategy
    )


def test_strategy_does_not_match_different_goal():

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    experience = create_experience(
        goal="unlock the door",
        actions=[
            "open_box",
        ],
    )

    attributor = StrategyAttributor()

    result = attributor.attribute(
        experience=experience,
        strategies=[
            strategy,
        ],
    )

    assert result.matched is False

    assert (
        result.strategy
        is None
    )


def test_strategy_does_not_match_different_action_sequence():

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    experience = create_experience(
        goal="open the box",
        actions=[
            "inspect_box",
        ],
    )

    attributor = StrategyAttributor()

    result = attributor.attribute(
        experience=experience,
        strategies=[
            strategy,
        ],
    )

    assert result.matched is False


def test_agent_can_attribute_registered_strategy():

    agent = Agent(
        "attribution-agent"
    )

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    agent.register_strategy(
        strategy
    )

    experience = create_experience(
        goal="open the box",
        actions=[
            "open_box",
        ],
    )

    result = agent.attribute_strategy(
        experience
    )

    assert result.matched is True

    assert (
        result.strategy
        is strategy
    )

    assert (
        agent.active_strategy
        is strategy
    )


def test_agent_does_not_attribute_unmatched_experience():

    agent = Agent(
        "attribution-agent"
    )

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    agent.register_strategy(
        strategy
    )

    experience = create_experience(
        goal="unlock the door",
        actions=[
            "unlock_door",
        ],
    )

    result = agent.attribute_strategy(
        experience
    )

    assert result.matched is False

    assert (
        agent.active_strategy
        is None
    )


def test_completed_experience_updates_attributed_strategy():

    agent = Agent(
        "attribution-agent"
    )

    action = Action(
        name="open_box",
        description="Open the box",
        handler=lambda **kwargs: {
            "success": True,
        },
    )

    agent.register_action(
        action
    )

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    agent.register_strategy(
        strategy
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    agent.set_goal(
        goal
    )

    agent.start_experience(
        experience_id="attribution-success",
        goal="open the box",
    )

    agent.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "open",
        },
        result={
            "success": True,
        },
    )

    agent.complete_experience()

    assert (
        strategy.success_count
        == 1
    )

    assert (
        strategy.progress_count
        == 1
    )

    assert (
        strategy.average_progress
        == 1.0
    )


def test_failed_experience_updates_attributed_strategy():

    agent = Agent(
        "attribution-agent"
    )

    action = Action(
        name="open_box",
        description="Open the box",
        handler=lambda **kwargs: {
            "success": False,
        },
    )

    agent.register_action(
        action
    )

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    agent.register_strategy(
        strategy
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    agent.set_goal(
        goal
    )

    agent.start_experience(
        experience_id="attribution-failure",
        goal="open the box",
    )

    agent.record_experience(
        action="open_box",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": False,
        },
    )

    agent.fail_experience()

    assert (
        strategy.failure_count
        == 1
    )

    assert (
        strategy.progress_count
        == 1
    )

    assert (
        strategy.average_progress
        == 0.0
    )
