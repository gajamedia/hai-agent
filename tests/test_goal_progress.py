from agent.core.agent import Agent
from agent.core.goal import Goal


def create_goal():
    return Goal(
        "open the box",
        target_state={
            "box": "open",
            "door": "unlocked",
        },
    )


def test_goal_is_not_satisfied_when_target_state_is_missing():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    assert (
        agent.is_goal_satisfied(
            world_state={
                "box": "closed",
                "door": "locked",
            },
            goal=goal,
        )
        is False
    )


def test_goal_progress_is_zero_when_nothing_matches():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    progress = agent.goal_progress(
        world_state={
            "box": "closed",
            "door": "locked",
        },
        goal=goal,
    )

    assert progress == 0.0


def test_goal_progress_is_half_when_one_target_matches():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    progress = agent.goal_progress(
        world_state={
            "box": "open",
            "door": "locked",
        },
        goal=goal,
    )

    assert progress == 0.5


def test_goal_progress_is_one_when_all_targets_match():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    progress = agent.goal_progress(
        world_state={
            "box": "open",
            "door": "unlocked",
        },
        goal=goal,
    )

    assert progress == 1.0


def test_satisfied_goal_is_detected():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    assert (
        agent.is_goal_satisfied(
            world_state={
                "box": "open",
                "door": "unlocked",
            },
            goal=goal,
        )
        is True
    )


def test_update_goal_status_marks_goal_completed():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    assert goal.completed is False

    completed = agent.update_goal_status(
        world_state={
            "box": "open",
            "door": "unlocked",
        },
        goal=goal,
    )

    assert completed is True

    assert goal.completed is True


def test_update_goal_status_does_not_complete_unsatisfied_goal():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    completed = agent.update_goal_status(
        world_state={
            "box": "open",
            "door": "locked",
        },
        goal=goal,
    )

    assert completed is False

    assert goal.completed is False


def test_goal_methods_use_current_goal_when_omitted():

    agent = Agent(
        "goal-progress-agent"
    )

    goal = create_goal()

    agent.set_goal(
        goal
    )

    agent.observe(
        {
            "box": "open",
            "door": "unlocked",
        }
    )

    assert (
        agent.is_goal_satisfied()
        is True
    )

    assert (
        agent.goal_progress()
        == 1.0
    )

def test_decide_and_act_marks_goal_completed_after_exploration():

    from agent.core.action import Action
    from agent.world.room import RoomEnvironment

    agent = Agent(
        "goal-completion-agent"
    )

    action = Action(
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

    agent.register_action(
        action
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    environment = RoomEnvironment()

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
        experience_id="goal-completion-1",
    )

    assert result is not None

    assert (
        environment.state["box"]
        == "open"
    )

    assert goal.completed is True

    assert (
        agent.goal_progress(
            environment.observe(),
            goal,
        )
        == 1.0
    )
