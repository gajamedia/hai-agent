from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.learning.procedural_learning import (
    ProceduralLearning,
)
from agent.memory.procedural_store import (
    ProceduralMemoryStore,
)
from agent.tools.action_system import ActionSystem
from agent.tools.room_actions import (
    create_room_actions,
)
from agent.world.model import WorldModel
from agent.world.room import RoomEnvironment


def run_episode(
    agent,
    environment,
    planner,
    action_system,
):

    environment.reset()

    world_model = WorldModel()

    goal = Goal(
        description="Open the door",
        priority=10,
    )

    agent.set_goal(goal)

    observation = environment.observe()

    agent.observe(observation)

    world_model.update(observation)

    world_state = {
        "door": world_model.get("door"),
        "box": world_model.get("box"),
        "key": world_model.get("key"),
    }

    goal_state = {
        "door": "unlocked",
    }

    plan = planner.plan(
        world_state,
        goal_state,
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

        agent.act(action.name)

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

        agent.record_experience(
            action=action.name,
            observation_before=observation_before,
            observation_after=observation_after,
            result=result,
        )

        agent.step()

    final_world = environment.observe()

    if final_world.get("door") == "unlocked":
        goal.complete()
    else:
        goal.fail()

    return goal


def main():

    agent = Agent("Agent-001")

    environment = RoomEnvironment()

    action_system = ActionSystem()

    actions = create_room_actions(
        environment
    )

    for action in actions:
        action_system.register(action)

    planner = Planner(actions)

    procedural_store = (
        ProceduralMemoryStore()
    )

    procedural_learning = (
        ProceduralLearning(
            procedural_store
        )
    )

    print("================================")
    print("PROCEDURAL LEARNING")
    print("================================")
    print()

    for experience_number in range(1, 4):

        print(
            f"EXPERIENCE #{experience_number}"
        )
        print("----------------------------")

        goal = run_episode(
            agent=agent,
            environment=environment,
            planner=planner,
            action_system=action_system,
        )

        print(
            "Goal completed:",
            goal.completed,
        )

        print(
            "Episodes:",
            agent.episodic_memory.count(),
        )

        episodes = (
            agent.episodic_memory.all()
        )

        procedural_learning.learn_from_episode_sequence(
            goal=goal.description,
            episodes=episodes[
                -3:
            ],
        )

        print(
            "Procedures:",
            procedural_store.count(),
        )

        print()

    print("================================")
    print("LEARNED PROCEDURE")
    print("================================")
    print()

    memories = procedural_store.all()

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        print(
            f"Procedure #{index}"
        )

        print(
            "  Skill:",
            memory.skill,
        )

        print(
            "  Steps:",
            memory.steps,
        )

        print(
            "  Success:",
            memory.success_count,
        )

        print(
            "  Failure:",
            memory.failure_count,
        )

        print(
            "  Success rate:",
            memory.success_rate,
        )

        print(
            "  Confidence:",
            memory.confidence,
        )

        print(
            "  Source episodes:",
            len(memory.source_episodes),
        )

        print()

    print("================================")
    print("FINAL SUMMARY")
    print("================================")

    print(
        "Total episodes:",
        agent.episodic_memory.count(),
    )

    print(
        "Total procedures:",
        procedural_store.count(),
    )


if __name__ == "__main__":
    main()
