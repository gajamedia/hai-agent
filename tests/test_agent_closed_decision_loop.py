from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import (
    DecisionMode,
)
from agent.world.room import RoomEnvironment


def create_open_box_action():
    return Action(
        name="open_box",
        description="Open the box",
        handler=lambda **kwargs: {
            "success": True,
        },
        preconditions={
            "box": "closed",
        },
        effects={
            "box": "open",
        },
    )


def create_agent():
    agent = Agent(
        "closed-loop-agent"
    )

    agent.register_action(
        create_open_box_action()
    )

    return agent


def create_goal():
    return Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )


def test_no_knowledge_is_uncertain():

    agent = create_agent()

    assert (
        agent._knowledge_uncertainty()
        == 1.0
    )


def test_first_exploration_creates_procedural_knowledge():

    agent = create_agent()

    result = agent.decide_and_act(
        goal=create_goal(),
        environment=RoomEnvironment(),
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="closed-loop-1",
    )

    assert (
        result.mode
        == DecisionMode.EXPLORE
    )

    procedural = (
        agent.procedural_memory.latest()
    )

    assert procedural is not None

    assert (
        procedural.success_count
        == 1
    )

    assert (
        procedural.steps
        == ["open_box"]
    )


def test_second_exploration_reduces_uncertainty():

    agent = create_agent()

    for index in range(2):

        result = agent.decide_and_act(
            goal=create_goal(),
            environment=RoomEnvironment(),
            information_gain={
                "open_box": 1.0,
            },
            action_risk={
                "open_box": 0.1,
            },
            risk=0.2,
            experience_id=(
                f"closed-loop-{index}"
            ),
        )

        assert (
            result.mode
            == DecisionMode.EXPLORE
        )

    procedural = (
        agent.procedural_memory.latest()
    )

    assert procedural is not None

    assert (
        procedural.success_count
        == 2
    )

    assert (
        agent._knowledge_uncertainty()
        < 0.6
    )


def test_closed_loop_switches_from_explore_to_exploit():

    agent = create_agent()

    # ---------------------------------------------
    # Experience 1
    # ---------------------------------------------

    first = agent.decide_and_act(
        goal=create_goal(),
        environment=RoomEnvironment(),
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="closed-loop-1",
    )

    assert (
        first.mode
        == DecisionMode.EXPLORE
    )

    # ---------------------------------------------
    # Experience 2
    # ---------------------------------------------

    second = agent.decide_and_act(
        goal=create_goal(),
        environment=RoomEnvironment(),
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="closed-loop-2",
    )

    assert (
        second.mode
        == DecisionMode.EXPLORE
    )

    # ---------------------------------------------
    # Third decision should exploit learned
    # procedural knowledge.
    # ---------------------------------------------

    third = agent.decide_and_act(
        goal=create_goal(),
        world_state={
            "box": "closed",
        },
        strategies=[],
        risk=0.1,
    )

    assert (
        third.mode
        == DecisionMode.EXPLOIT
    )

    assert (
        third.exploration_result
        is None
    )
