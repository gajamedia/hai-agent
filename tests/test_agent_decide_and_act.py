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
        "test-agent"
    )

    agent.register_action(
        create_open_box_action()
    )

    return agent


def test_agent_has_unified_decision_components():

    agent = create_agent()

    assert agent.decision_maker is not None
    assert agent.active_explorer is not None
    assert agent.unified_executor is not None

    assert (
        agent.hierarchical_planner
        is not None
    )

    assert (
        agent.risk_aware_hierarchical_planner
        is not None
    )


def test_decide_and_act_requires_goal():

    agent = Agent(
        "test-agent"
    )

    try:
        agent.decide_and_act()

        assert False

    except ValueError as exc:

        assert str(exc) == (
            "A goal is required "
            "for decide_and_act()."
        )


def test_decide_and_act_explores_when_knowledge_is_unknown():

    agent = create_agent()

    environment = RoomEnvironment()

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    result = agent.decide_and_act(
        goal=goal,
        environment=environment,
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="agent-exploration-1",
    )

    assert (
        result.mode
        == DecisionMode.EXPLORE
    )

    assert result.exploration_result is not None

    assert (
        result.exploration_result.action
        is not None
    )

    assert (
        result.exploration_result.action.name
        == "open_box"
    )

    assert (
        environment.state["box"]
        == "open"
    )


def test_decide_and_act_learns_after_exploration():

    agent = create_agent()

    environment = RoomEnvironment()

    goal = Goal(
        "open the box"
    )

    result = agent.decide_and_act(
        goal=goal,
        environment=environment,
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="agent-learning-1",
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
        procedural.steps
        == ["open_box"]
    )


def test_decide_and_act_switches_to_exploit_after_known_procedure():

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
        "open the box",
        target_state={
            "box": "open",
        },
    )

    result = agent.decide_and_act(
        goal=goal,
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


def test_decide_and_act_uses_current_goal():

    agent = create_agent()

    goal = Goal(
        "open the box"
    )

    agent.set_goal(
        goal
    )

    assert (
        agent.current_goal
        is goal
    )

    environment = RoomEnvironment()

    result = agent.decide_and_act(
        environment=environment,
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="agent-current-goal",
    )

    assert result is not None

    assert (
        agent.current_goal
        is goal
    )


def test_decide_and_act_uses_environment_observation():

    agent = create_agent()

    environment = RoomEnvironment()

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    result = agent.decide_and_act(
        goal=goal,
        environment=environment,
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
    )

    assert result is not None

    assert (
        result.mode
        == DecisionMode.EXPLORE
    )
