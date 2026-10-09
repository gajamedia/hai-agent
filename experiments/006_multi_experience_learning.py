from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.tools.action_system import ActionSystem
from agent.tools.room_actions import create_room_actions
from agent.world.model import WorldModel
from agent.world.room import RoomEnvironment


def run_episode(agent, environment, planner, action_system):
    """
    Run one complete experience.
    """

    # Reset environment
    environment.reset()

    # Create world model for this experience
    world_model = WorldModel()

    # Goal
    goal = Goal(
        description="Open the door",
        priority=10,
    )

    agent.set_goal(goal)

    goal_state = {
        "door": "unlocked"
    }

    # Initial observation
    observation = environment.observe()

    agent.observe(observation)
    world_model.update(observation)

    # Current world
    world_state = {
        "door": world_model.get("door"),
        "box": world_model.get("box"),
        "key": world_model.get("key"),
    }

    # Plan
    plan = planner.plan(
        world_state,
        goal_state,
    )

    agent.set_plan(
        [action.name for action in plan]
    )

    # Execute
    for action in plan:

        observation_before = environment.observe()

        agent.act(action.name)

        result = action_system.execute(
            action.name
        )

        observation_after = environment.observe()

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

    # Evaluate goal
    final_world = environment.observe()

    if final_world.get("door") == "unlocked":
        goal.complete()
    else:
        goal.fail()

    return goal


def main():

    # --------------------------------------------------
    # CREATE AGENT
    # --------------------------------------------------

    agent = Agent("Agent-001")

    # --------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------

    environment = RoomEnvironment()

    # --------------------------------------------------
    # ACTION SYSTEM
    # --------------------------------------------------

    action_system = ActionSystem()

    actions = create_room_actions(
        environment
    )

    for action in actions:
        action_system.register(action)

    # --------------------------------------------------
    # PLANNER
    # --------------------------------------------------

    planner = Planner(actions)

    # --------------------------------------------------
    # RUN EXPERIENCES
    # --------------------------------------------------

    print("================================")
    print("MULTI EXPERIENCE LEARNING")
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

        print(
            "Semantic memories:",
            agent.semantic_memory.count(),
        )

        print()

    # --------------------------------------------------
    # DISPLAY LEARNED KNOWLEDGE
    # --------------------------------------------------

    print("================================")
    print("LEARNED KNOWLEDGE")
    print("================================")
    print()

    memories = agent.semantic_memory.all()

    for index, memory in enumerate(
        memories,
        start=1,
    ):

        print(
            f"Knowledge #{index}"
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
            "  Evidence:",
            memory.evidence_count,
        )

        print(
            "  Contradictions:",
            memory.contradiction_count,
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

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("================================")
    print("FINAL SUMMARY")
    print("================================")

    print(
        "Total episodes:",
        agent.episodic_memory.count(),
    )

    print(
        "Total semantic memories:",
        agent.semantic_memory.count(),
    )


if __name__ == "__main__":
    main()
