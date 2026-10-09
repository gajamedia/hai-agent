from agent.core.experience import Experience
from agent.memory.episodic import EpisodicMemory


def create_episode(
    action: str,
    cycle: int,
):
    return EpisodicMemory(
        content=f"Executed {action}.",
        action=action,
        cycle=cycle,
        result={
            "success": True,
        },
    )


def test_experience_contains_multiple_episodes():

    experience = Experience(
        experience_id="experience-1",
        goal="Open the door",
    )

    experience.add_episode(
        create_episode(
            "open_box",
            1,
        )
    )

    experience.add_episode(
        create_episode(
            "take_key",
            2,
        )
    )

    experience.add_episode(
        create_episode(
            "unlock_door",
            3,
        )
    )

    assert experience.action_count == 3

    assert experience.actions == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert experience.cycle_start == 1
    assert experience.cycle_end == 3


def test_experience_completion():

    experience = Experience(
        experience_id="experience-1",
        goal="Open the door",
    )

    experience.complete()

    assert experience.completed is True
    assert experience.failed is False


def test_experience_failure():

    experience = Experience(
        experience_id="experience-1",
        goal="Open the door",
    )

    experience.fail()

    assert experience.failed is True
    assert experience.completed is False
