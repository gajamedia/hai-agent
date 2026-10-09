from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.tools.action_system import ActionSystem
from agent.tools.room_actions import create_room_actions
from agent.world.model import WorldModel
from agent.world.room import RoomEnvironment


def main():

    # --------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------

    agent = Agent("Agent-001")

    # --------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------

    environment = RoomEnvironment()

    # --------------------------------------------------
    # WORLD MODEL
    # --------------------------------------------------

    world_model = WorldModel()

    # --------------------------------------------------
    # ACTION SYSTEM
    # --------------------------------------------------

    action_system = ActionSystem()

    actions = create_room_actions(environment)

    for action in actions:
        action_system.register(action)

    # --------------------------------------------------
    # PLANNER
    # --------------------------------------------------

    planner = Planner(actions)

    # --------------------------------------------------
    # GOAL
    # --------------------------------------------------

    goal = Goal(
        description="Open the door",
        priority=10,
    )

    agent.set_goal(goal)

    goal_state = {
        "door": "unlocked"
    }

    # --------------------------------------------------
    # INITIAL OBSERVATION
    # --------------------------------------------------

    observation = environment.observe()

    agent.observe(observation)

    world_model.update(observation)

    # --------------------------------------------------
    # GENERATE PLAN
    # --------------------------------------------------

    world_state = {
        "door": world_model.get("door"),
        "box": world_model.get("box"),
        "key": world_model.get("key"),
    }

    plan = planner.plan(
        world_state,
        goal_state,
    )

    agent.set_plan(
        [action.name for action in plan]
    )

    print("================================")
    print("SEMANTIC MEMORY EXPERIMENT")
    print("================================")
    print()

    print("Generated Plan:")

    for index, action in enumerate(
        plan,
        start=1,
    ):
        print(
            f"{index}. {action.name}"
        )

    print()

    # --------------------------------------------------
    # EXECUTE
    # --------------------------------------------------

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

        episode = agent.record_experience(
            action=action.name,
            observation_before=observation_before,
            observation_after=observation_after,
            result=result,
        )

        agent.step()

        print(
            f"Action: {action.name}"
        )

        print(
            f"Result: {result['success']}"
        )

        print(
            f"Episode: {episode.memory_id}"
        )

        print()

    # --------------------------------------------------
    # CHECK FINAL GOAL
    # --------------------------------------------------

    final_world = environment.observe()

    if final_world.get("door") == "unlocked":
        goal.complete()
    else:
        goal.fail()


    # --------------------------------------------------
    # DISPLAY SEMANTIC MEMORY
    # --------------------------------------------------

    print("================================")
    print("SEMANTIC MEMORY")
    print("================================")
    print()

    memories = agent.semantic_memory.all()

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        print(
            f"Knowledge {index}"
        )

        print(
            "  Subject:",
            memory.subject,
        )

        print(
            "  Predicate:",
            memory.predicate,
        )

        print(
            "  Object:",
            memory.object,
        )

        print(
            "  Confidence:",
            memory.confidence,
        )

        print(
            "  Evidence:",
            memory.evidence_count,
        )

        print(
            "  Source Episodes:",
            memory.source_episodes,
        )

        print()

    # --------------------------------------------------
    # FINAL
    # --------------------------------------------------

    print("================================")
    print("FINAL")
    print("================================")

    print(
        "Episodes:",
        agent.episodic_memory.count(),
    )

    print(
        "Semantic Memories:",
        agent.semantic_memory.count(),
    )

    print(
        "Goal completed:",
        goal.completed,
    )


if __name__ == "__main__":
    main()
