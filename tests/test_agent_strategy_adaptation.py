from agent.core.action import Action
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def create_action():
    return Action(
        name="take_key",
        description="Take key",
        handler=lambda: {
            "success": False,
            "message": "The box is closed.",
        },
        preconditions={
            "box": "open",
        },
        effects={
            "key": "agent",
        },
    )


def create_strategy():
    return Strategy(
        name="strategy_a",
        goal="open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        total_actions=6,
    )


def test_agent_applies_reflection_penalty_to_active_strategy():

    agent = Agent("hAI")

    agent.register_action(
        create_action()
    )

    strategy = create_strategy()

    agent.set_active_strategy(
        strategy
    )

    initial_penalty = (
        strategy.reflection_penalty
    )

    agent.record_experience(
        action="take_key",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": False,
            "message": "The box is closed.",
        },
    )

    assert (
        strategy.reflection_penalty
        > initial_penalty
    )


def test_reflection_is_marked_as_applied():

    agent = Agent("hAI")

    agent.register_action(
        create_action()
    )

    strategy = create_strategy()

    agent.set_active_strategy(
        strategy
    )

    agent.record_experience(
        action="take_key",
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

    reflection = (
        agent.reflection_memory.latest()
    )

    assert reflection is not None

    assert reflection.applied is True


def test_applied_reflection_survives_restart(
    tmp_path,
):

    database_path = (
        tmp_path / "agent.db"
    )

    # -----------------------------------------
    # SESSION 1
    # -----------------------------------------

    agent_1 = Agent(
        "hAI",
        memory_path=database_path,
    )

    agent_1.register_action(
        create_action()
    )

    strategy = create_strategy()

    agent_1.set_active_strategy(
        strategy
    )

    agent_1.record_experience(
        action="take_key",
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

    reflection = (
        agent_1.reflection_memory.latest()
    )

    assert reflection is not None
    assert reflection.applied is True

    del agent_1

    # -----------------------------------------
    # SESSION 2
    # -----------------------------------------

    agent_2 = Agent(
        "hAI",
        memory_path=database_path,
    )

    restored = (
        agent_2.reflection_memory.latest()
    )

    assert restored is not None

    assert restored.action == (
        "take_key"
    )

    assert restored.applied is True
