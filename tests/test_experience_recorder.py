from agent.core.experience_recorder import (
    ExperienceRecorder,
)
from agent.core.experience_store import (
    ExperienceStore,
)
from agent.memory.episodic import EpisodicMemory


def test_experience_recorder():

    store = ExperienceStore()

    recorder = ExperienceRecorder(
        store
    )

    experience = recorder.start(
        experience_id="experience-1",
        goal="Open the door",
    )

    recorder.add_episode(
        EpisodicMemory(
            content="Opened box.",
            action="open_box",
            cycle=1,
        )
    )

    recorder.add_episode(
        EpisodicMemory(
            content="Took key.",
            action="take_key",
            cycle=2,
        )
    )

    recorder.add_episode(
        EpisodicMemory(
            content="Unlocked door.",
            action="unlock_door",
            cycle=3,
        )
    )

    recorder.complete()

    assert experience.completed is True
    assert experience.action_count == 3
    assert store.count() == 1

    recorder.close()

    assert recorder.current_experience is None


def test_experience_recorder_requires_active_experience():

    store = ExperienceStore()

    recorder = ExperienceRecorder(
        store
    )

    try:
        recorder.add_episode(
            EpisodicMemory(
                content="Invalid.",
                action="open_box",
            )
        )
        assert False
    except RuntimeError as error:
        assert str(error) == (
            "No active experience."
        )
