from agent.memory import WorkingMemory


def main():
    memory = WorkingMemory()

    print("=== WORKING MEMORY ===")
    print()

    memory.set_goal("Open the door")

    memory.set_plan([
        "open_box",
        "take_key",
        "unlock_door",
    ])

    memory.set_current_action("open_box")

    memory.set_observation({
        "door": "locked",
        "box": "closed",
        "key": "inside_box",
    })

    memory.update_context({
        "agent_location": "room",
        "priority": 10,
    })

    memory.set_reasoning(
        "The door is locked and the key is inside the box. "
        "The box must be opened first."
    )

    print("Goal:")
    print(memory.goal)
    print()

    print("Plan:")
    for index, action in enumerate(memory.plan, start=1):
        print(f"{index}. {action}")
    print()

    print("Current Action:")
    print(memory.current_action)
    print()

    print("Recent Observation:")
    print(memory.recent_observation)
    print()

    print("Context:")
    print(memory.context)
    print()

    print("Reasoning:")
    print(memory.reasoning)
    print()

    print("Has Active Plan:")
    print(memory.has_active_plan)
    print()

    print("=== UPDATE AFTER ACTION ===")
    print()

    memory.set_current_action("take_key")

    memory.set_observation({
        "door": "locked",
        "box": "open",
        "key": "inside_box",
    })

    memory.set_reasoning(
        "The box is open. The key can now be taken."
    )

    print("Current Action:")
    print(memory.current_action)
    print()

    print("Recent Observation:")
    print(memory.recent_observation)
    print()

    print("Reasoning:")
    print(memory.reasoning)
    print()


if __name__ == "__main__":
    main()
