from agent.core.action import Action
from agent.core.goal import Goal
from agent.core.agent import Agent
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import (
    HierarchicalPlanner,
)


def create_actions():
    return [
        Action(
            name="take_key",
            description="Take the key",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
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
        ),
    ]


def create_agent():
    agent = Agent(
        name="learning-agent"
    )

    agent.register_actions(
        create_actions()
    )

    return agent


def record_successful_door_experience(
    agent,
):
    goal = Goal(
        description="open the door"
    )

    agent.set_goal(
        goal
    )

    agent.start_experience(
        experience_id="experience-1",
        goal="open the door",
    )

    agent.record_experience(
        action="take_key",
        observation_before={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        observation_after={
            "door": "locked",
            "box": "open",
            "key": "agent",
        },
        result={
            "success": True,
        },
    )

    agent.step()

    agent.record_experience(
        action="unlock_door",
        observation_before={
            "door": "locked",
            "box": "open",
            "key": "agent",
        },
        observation_after={
            "door": "unlocked",
            "box": "open",
            "key": "agent",
        },
        result={
            "success": True,
        },
    )

    agent.complete_experience()


def test_successful_experience_creates_procedural_memory():

    agent = create_agent()

    record_successful_door_experience(
        agent
    )

    assert (
        agent.procedural_memory.count()
        == 1
    )

    memory = (
        agent.procedural_memory.latest()
    )

    assert memory is not None

    assert memory.skill == (
        "open_the_door"
    )

    assert memory.steps == [
        "take_key",
        "unlock_door",
    ]

    assert memory.success_count == 1


def test_successful_experience_creates_causal_memory():

    agent = create_agent()

    record_successful_door_experience(
        agent
    )

    assert (
        agent.causal_memory.count()
        >= 1
    )

    causal_memories = (
        agent.causal_memory.all()
    )

    assert any(
        memory.cause == "take_key"
        and memory.effect == "unlock_door"
        and memory.relation == "enables"
        for memory in causal_memories
    )


def test_learned_procedure_is_reused_by_new_planner():

    agent = create_agent()

    record_successful_door_experience(
        agent
    )

    planner = HierarchicalPlanner(
        planner=agent.planner,
        decomposer=GoalDecomposer(),
        procedural_store=(
            agent.procedural_memory
        ),
        causal_store=(
            agent.causal_memory
        ),
    )

    goal = Goal(
        description="open the door",
        target_state={
            "door": "unlocked",
        },
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_learning_does_not_create_procedure_from_failed_experience():

    agent = create_agent()

    goal = Goal(
        description="open the door"
    )

    agent.set_goal(
        goal
    )

    agent.start_experience(
        experience_id="failed-experience",
        goal="open the door",
    )

    agent.record_experience(
        action="take_key",
        observation_before={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        observation_after={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        result={
            "success": False,
        },
    )

    agent.fail_experience()

    assert (
        agent.procedural_memory.count()
        == 0
    )


def test_failed_procedure_is_not_reused():

    agent = create_agent()

    # Learn a valid procedure first.
    record_successful_door_experience(
        agent
    )

    memory = (
        agent.procedural_memory.latest()
    )

    assert memory is not None

    # Introduce two failures so failure count
    # becomes greater than success count.
    memory.record_failure(
        ["failed-1"]
    )

    memory.record_failure(
        ["failed-2"]
    )

    planner = HierarchicalPlanner(
        planner=agent.planner,
        decomposer=GoalDecomposer(),
        procedural_store=(
            agent.procedural_memory
        ),
        causal_store=(
            agent.causal_memory
        ),
    )

    goal = Goal(
        description="open the door",
        target_state={
            "door": "unlocked",
        },
    )

    plan = planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
    )

    # Classical planner should still be able to
    # produce the correct plan.
    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]
