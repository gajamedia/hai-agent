from agent.memory.episodic_store import EpisodicMemoryStore
from agent.memory.episode_recorder import EpisodeRecorder


def test_episode_recorder():

    store = EpisodicMemoryStore()
    recorder = EpisodeRecorder(store)

    memory = recorder.record(
        goal="Open the door",

        action="open_box",

        observation_before={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },

        observation_after={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },

        result={
            "success": True,
            "message": "The box is now open.",
        },

        cycle=1,
    )

    assert store.count() == 1

    assert memory.goal == "Open the door"
    assert memory.action == "open_box"

    assert memory.observation_before["box"] == "closed"
    assert memory.observation_after["box"] == "open"

    assert memory.result["success"] is True
    assert memory.cycle == 1

    assert "success" in memory.tags
