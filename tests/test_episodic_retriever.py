
from agent.memory.episodic import EpisodicMemory
from agent.memory.episodic_store import EpisodicMemoryStore
from agent.memory.persistence import (
    EpisodicMemoryRepository,
    SQLiteDatabase,
)


def make_episode(
    goal,
    action,
    before=None,
    after=None,
    success=True,
    confidence=1.0,
    importance=0.5,
    memory_id=None,
):
    kwargs = {}

    if memory_id is not None:
        kwargs["memory_id"] = memory_id

    # return EpisodicMemory(
    #     content=f"Goal: {goal}; action: {action}",
    #     goal=goal,
    #     action=action,
    #     observation_before=before,
    #     observation_after=after,
    #     result={"success": success},
    #     confidence=confidence,
    #     importance=importance,
    #     **kwargs,
    # )
    return EpisodicMemory(
        content=f"Goal: {goal}; action: {action}",
        source="test",
        goal=goal,
        action=action,
        observation_before=before,
        observation_after=after,
        result={"success": success},
        confidence=confidence,
        importance=importance,
        **kwargs,
    )

def test_retriever_prefers_matching_goal():
    store = EpisodicMemoryStore()

    unrelated = make_episode(
        goal="prepare the meal",
        action="cook_food",
    )
    relevant = make_episode(
        goal="open the door",
        action="unlock_door",
    )

    store.add(unrelated)
    store.add(relevant)

    results = store.retrieve(goal="open the door")

    assert results
    assert results[0] is relevant


def test_retriever_ranks_matching_action_higher():
    store = EpisodicMemoryStore()

    different_action = make_episode(
        goal="perform task",
        action="close_box",
    )
    matching_action = make_episode(
        goal="perform task",
        action="open_box",
    )

    store.add(different_action)
    store.add(matching_action)

    results = store.retrieve_ranked(
        goal="perform task",
        action="open_box",
    )

    assert results[0].memory is matching_action
    assert results[0].score > results[1].score


def test_retriever_uses_observation_context():
    store = EpisodicMemoryStore()

    closed_box = make_episode(
        goal="open the box",
        action="open_box",
        before={"box": "closed", "key": "available"},
    )
    open_box = make_episode(
        goal="open the box",
        action="open_box",
        before={"box": "open", "key": "missing"},
    )

    store.add(closed_box)
    store.add(open_box)

    results = store.retrieve_ranked(
        goal="open the box",
        action="open_box",
        context={"box": "closed", "key": "available"},
    )

    assert results[0].memory is closed_box


def test_failed_episode_remains_retrievable():
    store = EpisodicMemoryStore()

    failed = make_episode(
        goal="open the door",
        action="unlock_door",
        success=False,
    )
    successful = make_episode(
        goal="open the door",
        action="unlock_door",
        success=True,
    )

    store.add(failed)
    store.add(successful)

    results = store.retrieve_ranked(
        goal="open the door",
        action="unlock_door",
    )

    assert len(results) == 2
    assert {item.memory.result["success"] for item in results} == {
        True,
        False,
    }


def test_retrieval_returns_scores_and_respects_limit():
    store = EpisodicMemoryStore()

    for _ in range(3):
        store.add(
            make_episode(
                goal="open the door",
                action="unlock_door",
            )
        )

    results = store.retrieve_ranked(
        goal="open the door",
        limit=2,
    )

    assert len(results) == 2
    assert all(result.score > 0.0 for result in results)
    assert results[0].score >= results[1].score


def test_retrieval_returns_empty_for_unrelated_query():
    store = EpisodicMemoryStore()

    store.add(
        make_episode(
            goal="open the door",
            action="unlock_door",
        )
    )

    assert store.retrieve(goal="bake a cake") == []


def test_retrieval_handles_invalid_limits():
    store = EpisodicMemoryStore()

    assert store.retrieve(limit=0) == []
    assert store.retrieve(limit=-1) == []


def test_retrieval_works_after_sqlite_reload(tmp_path):
    database = SQLiteDatabase(tmp_path / "episodic.db")
    repository = EpisodicMemoryRepository(database)

    original = EpisodicMemoryStore(repository=repository)
    episode = make_episode(
        goal="open the door",
        action="unlock_door",
        before={"door": "locked"},
        success=True,
        memory_id="persisted-episode",
    )
    original.add(episode)

    restored = EpisodicMemoryStore(repository=repository)

    results = restored.retrieve_ranked(
        goal="open the door",
        action="unlock_door",
        context={"door": "locked"},
    )

    assert len(results) == 1
    assert results[0].memory.memory_id == "persisted-episode"
    assert results[0].memory.action == "unlock_door"
