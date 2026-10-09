from types import SimpleNamespace

from agent.core.action import Action
from agent.core.experience import Experience
from agent.core.goal import Goal
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def make_episode(
    action: str,
    success: bool,
    cycle: int,
    observation_before: dict,
    observation_after: dict,
):
    return SimpleNamespace(
        action=action,
        result={
            "success": success,
        },
        cycle=cycle,
        observation_before=observation_before,
        observation_after=observation_after,
    )


def make_actions():
    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {
                "success": True,
            },
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
            description="Unlock door",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ]


def make_source_strategy():
    return Strategy(
        name="open_door_original",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )


def make_failed_experience():
    experience = Experience(
        experience_id="exp-evolution-1",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="take_key",
            success=False,
            cycle=1,
            observation_before={
                "door": "locked",
                "box": "closed",
                "key": "agent",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.fail()

    return experience


def test_agent_evolves_and_registers_selected_candidate():
    agent = Agent(
        name="test-agent"
    )

    agent.register_actions(
        make_actions()
    )

    source = make_source_strategy()

    agent.register_strategy(
        source
    )

    agent.set_active_strategy(
        source
    )

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    agent.set_goal(
        goal
    )

    experience = make_failed_experience()

    result = (
        agent.evolve_strategy_from_experience(
            experience=experience,
            goal=goal,
        )
    )

    assert result is not None

    assert result.selection.selected is True

    candidate = (
        result.selection.strategy
    )

    assert candidate is not None

    assert candidate.steps == [
        "open_box",
        "unlock_door",
    ]

    assert candidate.goal == "Open door"

    assert candidate in (
        agent.get_registered_strategies()
    )

    # The original active strategy is not
    # automatically replaced.
    assert agent.get_active_strategy() is source


def test_agent_does_not_evolve_successful_experience():
    agent = Agent(
        name="test-agent"
    )

    source = make_source_strategy()

    agent.register_strategy(
        source
    )

    agent.set_active_strategy(
        source
    )

    experience = Experience(
        experience_id="exp-success",
        goal="Open door",
    )

    experience.add_episode(
        make_episode(
            action="open_box",
            success=True,
            cycle=1,
            observation_before={
                "door": "locked",
                "box": "closed",
                "key": "agent",
            },
            observation_after={
                "door": "locked",
                "box": "open",
                "key": "agent",
            },
        )
    )

    experience.complete()

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    result = (
        agent.evolve_strategy_from_experience(
            experience=experience,
            goal=goal,
        )
    )

    assert result is None

    assert agent.get_registered_strategies() == [
        source
    ]


def test_agent_does_not_evolve_retired_strategy():
    agent = Agent(
        name="test-agent"
    )

    source = make_source_strategy()

    source.retire()

    agent.register_strategy(
        source
    )

    agent.set_active_strategy(
        source
    )

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    experience = make_failed_experience()

    result = (
        agent.evolve_strategy_from_experience(
            experience=experience,
            goal=goal,
        )
    )

    assert result is None

    assert agent.get_registered_strategies() == [
        source
    ]


def test_agent_does_not_register_duplicate_candidate():
    agent = Agent(
        name="test-agent"
    )

    agent.register_actions(
        make_actions()
    )

    source = make_source_strategy()

    candidate = Strategy(
        name="existing_candidate",
        goal="Open door",
        steps=[
            "open_box",
            "unlock_door",
        ],
    )

    agent.register_strategy(
        source
    )

    agent.register_strategy(
        candidate
    )

    agent.set_active_strategy(
        source
    )

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    experience = make_failed_experience()

    result = (
        agent.evolve_strategy_from_experience(
            experience=experience,
            goal=goal,
        )
    )

    assert result is not None

    # Existing candidate is rejected by the validator.
    assert result.selection.selected is False

    assert (
        result.selection.strategy
        is None
    )

    assert len(
        agent.get_registered_strategies()
    ) == 2
