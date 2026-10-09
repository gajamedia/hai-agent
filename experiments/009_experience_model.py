from agent.core.agent import Agent
from agent.core.experience_recorder import (
    ExperienceRecorder,
)
from agent.core.experience_store import (
    ExperienceStore,
)
from agent.core.goal import Goal

from agent.cognition.planner import Planner
from agent.tools.action_system import ActionSystem
from agent.tools.room_actions import (
    create_room_actions,
)

from agent.world.model import WorldModel
from agent.world.room import RoomEnvironment


def run_experience(
    agent,
    environment,
    planner,
    action_system,
    experience_recorder,
    experience_number,
):

    environment.reset()

    world_model = WorldModel()

    goal = Goal(
        description="Open the door",
        priority=10,
    )

    agent.set_goal(goal)

    experience_recorder.start(
        experience_id=(
            f"experience-{experience_number}"
        ),
        goal=goal.description,
    )

    observation = environment.observe()

    agent.observe(
        observation
    )

    world_model.update(
        observation
    )

    world_state = {
        "door": world_model.get("door"),
        "box": world_model.get("box"),
        "key": world_model.get("key"),
    }

    plan = planner.plan(
        world_state,
        {
            "door": "unlocked",
        },
    )

    agent.set_plan(
        [
            action.name
            for action in plan
        ]
    )

    for action in plan:

        observation_before = (
            environment.observe()
        )

        agent.act(
            action.name
        )

        result = action_system.execute(
            action.name
        )

        observation_after = (
            environment.observe()
        )

        agent.observe(
            observation_after
        )

        world_model.update(
            observation_after
        )

        episode = agent.record_experience(
            action=action.name,
            observation_before=(
                observation_before
            ),
            observation_after=(
                observation_after
            ),
            result=result,
        )

        experience_recorder.add_episode(
            episode
        )

        agent.step()

    final_world = environment.observe()

    if (
        final_world.get("door")
        == "unlocked"
    ):

        goal.complete()

        experience_recorder.complete()

    else:

        goal.fail()

        experience_recorder.fail()

    experience_recorder.close()

    return goal


def main():

    agent = Agent(
        "Agent-001"
    )

    environment = RoomEnvironment()

    action_system = ActionSystem()

    actions = create_room_actions(
        environment
    )

    for action in actions:
        action_system.register(
            action
        )

    planner = Planner(
        actions
    )

    experience_store = (
        ExperienceStore()
    )

    experience_recorder = (
        ExperienceRecorder(
            experience_store
        )
    )

    print("================================")
    print("EXPERIENCE MODEL")
    print("================================")
    print()

    for number in range(1, 4):

        goal = run_experience(
            agent=agent,
            environment=environment,
            planner=planner,
            action_system=action_system,
            experience_recorder=(
                experience_recorder
            ),
            experience_number=number,
        )

        experience = (
            experience_store.latest()
        )

        print(
            f"EXPERIENCE #{number}"
        )

        print(
            "  Goal:",
            goal.description,
        )

        print(
            "  Completed:",
            experience.completed,
        )

        print(
            "  Episodes:",
            experience.action_count,
        )

        print(
            "  Actions:",
            experience.actions,
        )

        print(
            "  Cycle range:",
            experience.cycle_start,
            "->",
            experience.cycle_end,
        )

        print()

    print("================================")
    print("SUMMARY")
    print("================================")

    print(
        "Total experiences:",
        experience_store.count(),
    )

    print(
        "Successful experiences:",
        len(
            experience_store.successful()
        ),
    )

    print(
        "Failed experiences:",
        len(
            experience_store.failed()
        ),
    )

    print(
        "Total episodes:",
        agent.episodic_memory.count(),
    )


if __name__ == "__main__":
    main()
