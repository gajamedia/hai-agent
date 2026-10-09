from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.active_explorer import (
    ActiveExplorer,
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
        name="active-agent"
    )

    agent.register_action(
        create_open_box_action()
    )

    return agent


def test_active_explorer_explores_when_knowledge_is_unknown():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    result = explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
            target_state={
                "box": "open",
            },
        ),
        experience_id="exploration-1",
        exploration_risk=0.2,
    )

    assert result.decision.explore is True

    assert result.action is not None

    assert (
        result.action.name
        == "open_box"
    )

    assert result.execution is not None

    assert (
        result.execution.result["success"]
        is True
    )

    assert (
        environment.observe()["box"]
        == "open"
    )


def test_active_exploration_creates_experience():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    result = explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
        ),
        experience_id="exploration-2",
        exploration_risk=0.2,
    )

    assert result.execution is not None

    assert (
        agent.episodic_memory.count()
        == 1
    )

    assert (
        agent.experience_store.count()
        == 1
    )


def test_active_exploration_learns_procedure():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
        ),
        experience_id="exploration-3",
        exploration_risk=0.2,
    )

    memory = (
        agent.procedural_memory.latest()
    )

    assert memory is not None

    assert (
        memory.skill
        == "open_the_box"
    )

    assert memory.steps == [
        "open_box",
    ]

    assert (
        memory.success_count
        == 1
    )


def test_active_explorer_switches_to_exploit_after_learning():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
        ),
        experience_id="exploration-4",
        exploration_risk=0.2,
    )

    memory = (
        agent.procedural_memory.latest()
    )

    assert memory is not None

    for index in range(20):
        memory.record_success(
            [f"reinforcement-{index}"]
        )

    decision = explorer.decide(
        procedural_memory=memory,
        risk=0.1,
    )

    assert decision.explore is False


def test_exploration_does_not_execute_when_exploit_is_selected():

    agent = create_agent()

    environment = RoomEnvironment()

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

    explorer = ActiveExplorer()

    result = explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 0.1,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
        ),
        experience_id="exploration-5",
        exploration_risk=0.1,
    )

    assert result.decision.explore is False

    assert result.action is None

    assert result.execution is None

    assert (
        agent.episodic_memory.count()
        == 0
    )


def test_exploration_respects_action_preconditions():

    agent = create_agent()

    environment = RoomEnvironment()

    environment.state["box"] = "open"

    explorer = ActiveExplorer()

    result = explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
        ),
        experience_id="exploration-6",
        exploration_risk=0.2,
    )

    assert result.decision.explore is True

    assert result.action is None

    assert result.execution is None

    assert (
        agent.episodic_memory.count()
        == 0
    )


def test_high_exploration_risk_prevents_execution():

    agent = create_agent()

    environment = RoomEnvironment()

    explorer = ActiveExplorer()

    result = explorer.explore_once(
        agent=agent,
        environment=environment,
        actions=[
            create_open_box_action()
        ],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box",
        ),
        experience_id="exploration-7",
        exploration_risk=0.95,
    )

    assert result.decision.explore is False

    assert result.action is None

    assert result.execution is None

    assert (
        agent.episodic_memory.count()
        == 0
    )
