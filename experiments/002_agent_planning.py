from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.tools.action_system import ActionSystem
from agent.tools.room_actions import create_room_actions
from agent.world.model import WorldModel
from agent.world.room import RoomEnvironment


def main():
    # ============================================================
    # 1. CREATE AGENT
    # ============================================================

    agent = Agent("Agent-001")

    # ============================================================
    # 2. CREATE ENVIRONMENT
    # ============================================================

    environment = RoomEnvironment()

    # ============================================================
    # 3. CREATE WORLD MODEL
    # ============================================================

    world_model = WorldModel()

    # ============================================================
    # 4. CREATE ACTION SYSTEM
    # ============================================================

    action_system = ActionSystem()

    actions = create_room_actions(environment)

    for action in actions:
        action_system.register(action)

    # ============================================================
    # 5. CREATE PLANNER
    # ============================================================

    planner = Planner(actions)

    # ============================================================
    # 6. SET GOAL
    # ============================================================

    goal = Goal(
        description="Open the door",
        priority=10,
    )

    agent.set_goal(goal)

    goal_state = {
        "door": "unlocked"
    }

    # ============================================================
    # 7. INITIAL OBSERVATION
    # ============================================================

    observation = environment.observe()

    agent.observe(observation)
    world_model.update(observation)

    # ============================================================
    # 8. DISPLAY INITIAL STATE
    # ============================================================

    print("Agent:", agent.name)
    print("Goal:", goal.description)
    print("Goal state:", goal_state)
    print()

    world_state = {
        "door": world_model.get("door"),
        "box": world_model.get("box"),
        "key": world_model.get("key"),
    }

    print("Initial World:")
    print(world_state)
    print()

    # ============================================================
    # 9. GENERATE PLAN
    # ============================================================

    plan = planner.plan(
        world_state,
        goal_state,
    )

    # Store plan in Working Memory
    agent.set_plan(
        [action.name for action in plan]
    )

    print("Generated Plan:")
    print("--------------------------------")

    for index, action in enumerate(plan, start=1):
        print(f"{index}. {action.name}")

    print("--------------------------------")
    print()

    # ============================================================
    # 10. CHECK PLAN
    # ============================================================

    if not plan:
        print("Planner could not create a plan.")

        goal.fail()

        return

    # ============================================================
    # 11. EXECUTE PLAN
    # ============================================================

    for index, action in enumerate(plan, start=1):

        print(f"Executing step {index}: {action.name}")

        # --------------------------------------------------------
        # Capture BEFORE observation
        # --------------------------------------------------------

        observation_before = environment.observe()

        # --------------------------------------------------------
        # Update Working Memory
        # --------------------------------------------------------

        agent.act(action.name)

        print("Working Memory:")
        print(
            "  Current Action:",
            agent.working_memory.current_action
        )

        # --------------------------------------------------------
        # Execute action
        # --------------------------------------------------------

        result = action_system.execute(
            action.name
        )

        print("Result:", result)

        # --------------------------------------------------------
        # Check action result
        # --------------------------------------------------------

        if not result["success"]:

            print("Action failed.")

            goal.fail()

            break

        # --------------------------------------------------------
        # Capture AFTER observation
        # --------------------------------------------------------

        observation_after = environment.observe()

        # --------------------------------------------------------
        # Update Agent observation
        # --------------------------------------------------------

        agent.observe(
            observation_after
        )

        # --------------------------------------------------------
        # Update World Model
        # --------------------------------------------------------

        world_model.update(
            observation_after
        )

        # --------------------------------------------------------
        # Record Episodic Memory
        # --------------------------------------------------------

        episode = agent.record_experience(
            action=action.name,
            observation_before=observation_before,
            observation_after=observation_after,
            result=result,
        )

        print(
            "Episode recorded:",
            episode.memory_id
        )

        # --------------------------------------------------------
        # Advance Agent Cycle
        # --------------------------------------------------------

        agent.step()

        # --------------------------------------------------------
        # Display current world
        # --------------------------------------------------------

        print(
            "World:",
            observation_after
        )

        print()

    # ============================================================
    # 12. CHECK FINAL GOAL
    # ============================================================

    final_world = environment.observe()

    if final_world.get("door") == "unlocked":
        goal.complete()
    else:
        goal.fail()

    # ============================================================
    # 13. DISPLAY EPISODIC MEMORY
    # ============================================================

    print()
    print("Episodic Memory:")
    print("----------------")

    memories = agent.episodic_memory.all()

    for index, memory in enumerate(
        memories,
        start=1,
    ):
        print(f"Episode {index}")

        print("  Goal:")
        print("   ", memory.goal)

        print("  Action:")
        print("   ", memory.action)

        print("  Before:")
        print("   ", memory.observation_before)

        print("  After:")
        print("   ", memory.observation_after)

        print("  Result:")
        print("   ", memory.result)

        print("  Cycle:")
        print("   ", memory.cycle)

        print("  Tags:")
        print("   ", memory.tags)

        print()

    # ============================================================
    # 14. FINAL STATUS
    # ============================================================

    print("================================")
    print("Finished")

    print("Goal completed:", goal.completed)
    print("Goal failed:", goal.failed)

    print("Cycles:", agent.state.cycle)

    print("Episodes:", agent.episodic_memory.count())

    print("Final world:", final_world)


if __name__ == "__main__":
    main()
