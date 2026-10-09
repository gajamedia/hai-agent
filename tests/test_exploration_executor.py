from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.exploration_executor import (
    ExplorationExecutor,
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
        name="exploration-agent"
    )

    agent.register_action(
        create_open_box_action()
    )

    agent.set_goal(
        Goal(
            description="open the box",
            target_state={
                "box": "open",
            },
        )
    )

    return agent


def test_exploration_executes_action():

    agent = create_agent()

    environment = RoomEnvironment()

    agent.start_experience(
        experience_id="exploration-1",
        goal="open the box",
    )

    executor = ExplorationExecutor()

    result = executor.execute(
        agent=agent,
        environment=environment,
        action=create_open_box_action(),
    )

    assert result.action == "open_box"

    assert (
        result.result["success"]
        is True
    )

    assert (
        result.observation_before["box"]
        == "closed"
    )

    assert (
        result.observation_after["box"]
        == "open"
    )

    assert (
        result.episode is not None
    )


def test_exploration_records_episode():

    agent = create_agent()

    environment = RoomEnvironment()

    agent.start_experience(
        experience_id="exploration-2",
        goal="open the box",
    )

    executor = ExplorationExecutor()

    result = executor.execute(
        agent=agent,
        environment=environment,
        action=create_open_box_action(),
    )

    assert (
        agent.episodic_memory.count()
        == 1
    )

    assert (
        len(
            agent.current_experience.episodes
        )
        == 1
    )

    episode = (
        agent.current_experience.episodes[0]
    )

    assert episode.action == "open_box"

    assert (
        episode.observation_before["box"]
        == "closed"
    )

    assert (
        episode.observation_after["box"]
        == "open"
    )

    assert (
        episode.result["success"]
        is True
    )


def test_successful_exploration_can_become_procedure():

    agent = create_agent()

    environment = RoomEnvironment()

    agent.start_experience(
        experience_id="exploration-3",
        goal="open the box",
    )

    executor = ExplorationExecutor()

    executor.execute(
        agent=agent,
        environment=environment,
        action=create_open_box_action(),
    )

    agent.complete_experience()

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


def test_failed_exploration_is_recorded():

    agent = create_agent()

    environment = RoomEnvironment()

    agent.start_experience(
        experience_id="exploration-4",
        goal="open the box",
    )

    # Deliberately make the action fail by starting
    # from a state where the box is already open.
    environment.state["box"] = "open"

    executor = ExplorationExecutor()

    result = executor.execute(
        agent=agent,
        environment=environment,
        action=create_open_box_action(),
    )

    assert (
        result.result["success"]
        is False
    )

    assert (
        result.observation_before["box"]
        == "open"
    )

    assert (
        result.observation_after["box"]
        == "open"
    )

    assert (
        agent.episodic_memory.count()
        == 1
    )

    assert (
        agent.current_experience
        .episodes[0]
        .result["success"]
        is False
    )


def test_executor_requires_active_experience():

    agent = create_agent()

    environment = RoomEnvironment()

    executor = ExplorationExecutor()

    try:
        executor.execute(
            agent=agent,
            environment=environment,
            action=create_open_box_action(),
        )
    except RuntimeError as exc:
        assert str(exc) == (
            "No active experience."
        )
    else:
        raise AssertionError(
            "Expected RuntimeError."
        )


def test_executor_rejects_missing_dependencies():

    agent = create_agent()

    environment = RoomEnvironment()

    executor = ExplorationExecutor()

    try:
        executor.execute(
            agent=None,
            environment=environment,
            action=create_open_box_action(),
        )
    except ValueError as exc:
        assert str(exc) == (
            "agent cannot be None."
        )
    else:
        raise AssertionError(
            "Expected ValueError."
        )
