from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import DecisionMode
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
        "goal-aware-agent"
    )

    agent.register_action(
        create_open_box_action()
    )

    return agent


def test_unrelated_procedure_does_not_reduce_goal_uncertainty():

    agent = create_agent()

    memory = (
        agent.procedural_memory.get_or_create(
            "open_the_box",
            ["open_box"],
        )
    )

    for index in range(20):
        memory.record_success(
            [f"experience-{index}"]
        )

    unrelated_goal = Goal(
        "unlock the door"
    )

    uncertainty = (
        agent._knowledge_uncertainty(
            goal=unrelated_goal
        )
    )

    assert uncertainty == 1.0


def test_matching_procedure_reduces_goal_uncertainty():

    agent = create_agent()

    memory = (
        agent.procedural_memory.get_or_create(
            "open_the_box",
            ["open_box"],
        )
    )

    for index in range(20):
        memory.record_success(
            [f"experience-{index}"]
        )

    goal = Goal(
        "open the box"
    )

    uncertainty = (
        agent._knowledge_uncertainty(
            goal=goal
        )
    )

    assert uncertainty < 0.6


def test_unknown_goal_remains_explorable():

    agent = create_agent()

    memory = (
        agent.procedural_memory.get_or_create(
            "open_the_box",
            ["open_box"],
        )
    )

    for index in range(20):
        memory.record_success(
            [f"experience-{index}"]
        )

    environment = RoomEnvironment()

    result = agent.decide_and_act(
        goal=Goal(
            "unlock the door"
        ),
        environment=environment,
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="goal-specific-1",
    )

    assert (
        result.mode
        == DecisionMode.EXPLORE
    )


def test_known_goal_can_switch_to_exploit():

    agent = create_agent()

    memory = (
        agent.procedural_memory.get_or_create(
            "open_the_box",
            ["open_box"],
        )
    )

    for index in range(20):
        memory.record_success(
            [f"experience-{index}"]
        )

    result = agent.decide_and_act(
        goal=Goal(
            "open the box",
            target_state={
                "box": "open",
            },
        ),
        world_state={
            "box": "closed",
        },
        strategies=[],
        risk=0.1,
    )

    assert (
        result.mode
        == DecisionMode.EXPLOIT
    )
