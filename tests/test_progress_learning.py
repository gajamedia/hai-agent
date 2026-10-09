from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.core.action import Action
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


def test_progress_learning_detects_positive_progress():

    agent = Agent(
        "progress-agent"
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    agent.register_action(
        create_open_box_action()
    )

    result = agent.decide_and_act(
        goal=goal,
        environment=RoomEnvironment(),
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="progress-1",
    )

    assert result is not None

    progress = (
        agent.last_progress_learning
    )

    assert progress is not None

    assert (
        progress.progress_before
        == 0.0
    )

    assert (
        progress.progress_after
        == 1.0
    )

    assert (
        progress.progress_delta
        == 1.0
    )

    assert (
        progress.completed
        is True
    )


def test_progress_learning_detects_no_progress():

    agent = Agent(
        "progress-agent"
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    agent.start_experience(
        experience_id="progress-2",
        goal=goal.description,
    )

    # No episodes means no learning signal.
    result = agent.progress_learning.learn_from_experience(
        goal=goal,
        episodes=[],
    )

    assert result is None


def test_progress_learning_detects_partial_progress():

    agent = Agent(
        "progress-agent"
    )

    goal = Goal(
        "prepare room",
        target_state={
            "box": "open",
            "door": "unlocked",
        },
    )

    class Episode:
        observation_before = {
            "box": "closed",
            "door": "locked",
        }

        observation_after = {
            "box": "open",
            "door": "locked",
        }

    result = (
        agent.progress_learning.learn_from_experience(
            goal=goal,
            episodes=[
                Episode()
            ],
        )
    )

    assert result is not None

    assert (
        result.progress_before
        == 0.0
    )

    assert (
        result.progress_after
        == 0.5
    )

    assert (
        result.progress_delta
        == 0.5
    )

    assert (
        result.completed
        is False
    )


def test_progress_learning_detects_regression():

    agent = Agent(
        "progress-agent"
    )

    goal = Goal(
        "prepare room",
        target_state={
            "box": "open",
            "door": "unlocked",
        },
    )

    class Episode:
        observation_before = {
            "box": "open",
            "door": "unlocked",
        }

        observation_after = {
            "box": "closed",
            "door": "locked",
        }

    result = (
        agent.progress_learning.learn_from_experience(
            goal=goal,
            episodes=[
                Episode()
            ],
        )
    )

    assert result is not None

    assert (
        result.progress_before
        == 1.0
    )

    assert (
        result.progress_after
        == 0.0
    )

    assert (
        result.progress_delta
        == -1.0
    )

    assert (
        result.completed
        is False
    )


def test_progress_learning_result_contains_goal():

    agent = Agent(
        "progress-agent"
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    class Episode:
        observation_before = {
            "box": "closed",
        }

        observation_after = {
            "box": "open",
        }

    result = (
        agent.progress_learning.learn_from_experience(
            goal=goal,
            episodes=[
                Episode()
            ],
        )
    )

    assert result is not None

    assert (
        result.goal
        == "open the box"
    )
