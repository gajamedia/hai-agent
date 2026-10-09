from agent.memory import (
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
)


def main():
    print("=== MEMORY DATA MODEL ===")
    print()

    episodic = EpisodicMemory(
        content="Agent successfully opened the door.",
        goal="Open the door",
        action="unlock_door",
        observation={
            "door": "locked",
            "key": "agent",
        },
        result={
            "success": True,
            "message": "The door is now unlocked.",
        },
        cycle=3,
        importance=0.9,
        confidence=1.0,
        source="room_environment",
        tags=["door", "key", "success"],
    )

    print("EPISODIC MEMORY")
    print("----------------")
    print("ID:", episodic.memory_id)
    print("Goal:", episodic.goal)
    print("Action:", episodic.action)
    print("Observation:", episodic.observation)
    print("Result:", episodic.result)
    print("Cycle:", episodic.cycle)
    print("Importance:", episodic.importance)
    print()

    semantic = SemanticMemory(
        content="A locked door requires a key.",
        subject="door",
        predicate="requires",
        object="key",
        importance=0.8,
        confidence=0.95,
        source="learned_from_experience",
        tags=["door", "key", "requirement"],
    )

    print("SEMANTIC MEMORY")
    print("----------------")
    print("ID:", semantic.memory_id)
    print("Subject:", semantic.subject)
    print("Predicate:", semantic.predicate)
    print("Object:", semantic.object)
    print("Confidence:", semantic.confidence)
    print()

    procedural = ProceduralMemory(
        content="Procedure for opening a locked door.",
        skill_name="open_locked_door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=8,
        failure_count=2,
        importance=1.0,
        confidence=0.8,
        source="learned_from_experience",
        tags=["door", "procedure", "navigation"],
    )

    print("PROCEDURAL MEMORY")
    print("-----------------")
    print("ID:", procedural.memory_id)
    print("Skill:", procedural.skill_name)
    print("Steps:", procedural.steps)
    print("Success:", procedural.success_count)
    print("Failure:", procedural.failure_count)
    print("Success rate:", procedural.success_rate)
    print()


if __name__ == "__main__":
    main()
