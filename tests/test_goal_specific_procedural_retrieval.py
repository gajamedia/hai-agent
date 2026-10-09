from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal


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


def create_unlock_door_action():
    return Action(
        name="unlock_door",
        description="Unlock the door",
        handler=lambda **kwargs: {
            "success": True,
        },
        preconditions={
            "key": "agent",
            "door": "locked",
        },
        effects={
            "door": "unlocked",
        },
    )


def create_agent():
    agent = Agent(
        "goal-procedural-agent"
    )

    agent.register_actions(
        [
            create_open_box_action(),
            create_unlock_door_action(),
        ]
    )

    return agent


def learn_open_box(agent):
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

    return memory


def test_matching_goal_has_low_uncertainty():

    agent = create_agent()

    learn_open_box(agent)

    uncertainty = (
        agent._knowledge_uncertainty(
            Goal("open the box")
        )
    )

    assert uncertainty < 0.6


def test_unrelated_goal_does_not_reuse_procedure():

    agent = create_agent()

    learn_open_box(agent)

    uncertainty = (
        agent._knowledge_uncertainty(
            Goal("unlock the door")
        )
    )

    assert uncertainty == 1.0


def test_procedure_is_valid_for_matching_world_state():

    agent = create_agent()

    learn_open_box(agent)

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    plan = agent.hierarchical_planner.plan(
        world_state={
            "box": "closed",
        },
        goal=goal,
    )

    assert plan is not None

    assert [
        action.name
        for action in plan
    ] == [
        "open_box"
    ]


def test_invalid_procedure_falls_back_when_precondition_is_not_satisfied():

    agent = create_agent()

    learn_open_box(agent)

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    plan = agent.hierarchical_planner.plan(
        world_state={
            "box": "already_open",
        },
        goal=goal,
    )

    assert plan is not None

    assert [
        action.name
        for action in plan
    ] != [
        "open_box"
    ]


def test_different_goal_does_not_retrieve_open_box_procedure():

    agent = create_agent()

    learn_open_box(agent)

    goal = Goal(
        "unlock the door",
        target_state={
            "door": "unlocked",
        },
    )

    plan = agent.hierarchical_planner.plan(
        world_state={
            "door": "locked",
            "key": "missing",
        },
        goal=goal,
    )

    assert plan is not None

    assert all(
        action.name != "open_box"
        for action in plan
    )
