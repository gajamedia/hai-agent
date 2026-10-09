from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.world.room import RoomEnvironment


def create_actions(environment):

    return [
        Action(
            name="open_box",
            description="Open the box",
            handler=lambda: (
                environment.execute(
                    "open_box"
                )
            ),
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: (
                environment.execute(
                    "take_key"
                )
            ),
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
    ]


def test_agent_automatically_learns_causal_relationship():

    environment = RoomEnvironment()

    agent = Agent("hAI")

    actions = create_actions(
        environment
    )

    agent.register_actions(
        actions
    )

    agent.set_goal(
        Goal("open the door")
    )

    before = environment.observe()

    environment.execute(
        "open_box"
    )

    after = environment.observe()

    agent.record_experience(
        action="open_box",
        observation_before=before,
        observation_after=after,
        result={
            "success": True,
        },
    )

    before = environment.observe()

    environment.execute(
        "take_key"
    )

    after = environment.observe()

    agent.record_experience(
        action="take_key",
        observation_before=before,
        observation_after=after,
        result={
            "success": True,
        },
    )

    causal_memories = (
        agent.causal_memory.all()
    )

    assert len(
        causal_memories
    ) == 1

    causal = causal_memories[0]

    assert causal.cause == (
        "open_box"
    )

    assert causal.relation == (
        "enables"
    )

    assert causal.effect == (
        "take_key"
    )
