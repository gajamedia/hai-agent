from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.active_explorer import (
    ActiveExplorer,
)
from agent.cognition.exploration_loop import (
    ActiveExplorationLoop,
)
from agent.world.room import RoomEnvironment


def create_actions():

    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
    ]


def create_agent():

    agent = Agent(
        name="loop-agent"
    )

    agent.register_actions(
        create_actions()
    )

    return agent


def test_loop_executes_multiple_experiments():

    agent = create_agent()

    environment = RoomEnvironment()

    loop = ActiveExplorationLoop(
        max_steps=3
    )

    result = loop.run(
        agent=agent,
        environment=environment,
        actions=create_actions(),
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box"
        ),
        exploration_risk=0.2,
    )

    assert result.steps == 1

    assert result.successful_steps == 1

    assert result.failed_steps == 0


def test_loop_does_not_repeat_same_action():

    agent = create_agent()

    environment = RoomEnvironment()

    loop = ActiveExplorationLoop(
        max_steps=10
    )

    result = loop.run(
        agent=agent,
        environment=environment,
        actions=create_actions(),
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box"
        ),
        exploration_risk=0.2,
    )

    action_names = [
        item.action.name
        for item in result.results
        if item.action is not None
    ]

    assert len(action_names) == len(
        set(action_names)
    )


def test_loop_stops_when_no_action_is_executable():

    agent = create_agent()

    environment = RoomEnvironment()

    environment.state["box"] = "open"

    loop = ActiveExplorationLoop(
        max_steps=3
    )

    result = loop.run(
        agent=agent,
        environment=environment,
        actions=create_actions(),
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box"
        ),
        exploration_risk=0.2,
    )

    assert result.steps == 0


def test_loop_stops_when_exploitation_is_preferred():

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

    loop = ActiveExplorationLoop(
        max_steps=3
    )

    result = loop.run(
        agent=agent,
        environment=environment,
        actions=create_actions(),
        information_gain={
            "open_box": 0.1,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box"
        ),
        exploration_risk=0.1,
    )

    assert result.steps == 0


def test_loop_respects_max_steps():

    actions = [
        Action(
            name="action_one",
            description="Action one",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={},
            effects={
                "one": True,
            },
        ),
        Action(
            name="action_two",
            description="Action two",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={},
            effects={
                "two": True,
            },
        ),
        Action(
            name="action_three",
            description="Action three",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={},
            effects={
                "three": True,
            },
        ),
    ]

    agent = Agent(
        name="max-step-agent"
    )

    agent.register_actions(
        actions
    )

    environment = RoomEnvironment()

    loop = ActiveExplorationLoop(
        max_steps=2
    )

    result = loop.run(
        agent=agent,
        environment=environment,
        actions=actions,
        information_gain={
            "action_one": 1.0,
            "action_two": 1.0,
            "action_three": 1.0,
        },
        risk={
            "action_one": 0.1,
            "action_two": 0.1,
            "action_three": 0.1,
        },
        goal=Goal(
            "explore"
        ),
        exploration_risk=0.2,
    )

    assert result.steps == 2


def test_failed_experiment_is_recorded():

    class FailingEnvironment:

        def __init__(self):
            self.state = {
                "box": "closed",
            }

        def observe(self):
            return self.state.copy()

        def execute(self, action: str):
            return {
                "success": False,
                "message": (
                    f"Action failed: {action}"
                ),
            }

    action = Action(
        name="open_box",
        description="Open box",
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

    agent = Agent(
        name="failed-loop-agent"
    )

    agent.register_action(
        action
    )

    environment = FailingEnvironment()

    loop = ActiveExplorationLoop(
        max_steps=1
    )

    result = loop.run(
        agent=agent,
        environment=environment,
        actions=[action],
        information_gain={
            "open_box": 1.0,
        },
        risk={
            "open_box": 0.1,
        },
        goal=Goal(
            "open the box"
        ),
        exploration_risk=0.2,
    )

    assert result.steps == 1

    assert result.failed_steps == 1

    assert result.successful_steps == 0

    assert (
        result.results[0]
        .execution
        .result["success"]
        is False
    )

    assert (
        agent.episodic_memory.count()
        == 1
    )

    assert (
        agent.current_experience
        is None
    )
