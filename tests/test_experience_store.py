from agent.core.experience import Experience
from agent.core.experience_store import ExperienceStore


def test_experience_store():

    store = ExperienceStore()

    first = Experience(
        experience_id="experience-1",
        goal="Open the door",
    )

    second = Experience(
        experience_id="experience-2",
        goal="Open the door",
    )

    first.complete()
    second.fail()

    store.add(first)
    store.add(second)

    assert store.count() == 2
    assert store.latest() is second

    assert len(
        store.successful()
    ) == 1

    assert len(
        store.failed()
    ) == 1


def test_experience_store_clear():

    store = ExperienceStore()

    store.add(
        Experience(
            experience_id="experience-1",
            goal="Open the door",
        )
    )

    assert store.count() == 1

    store.clear()

    assert store.count() == 0
    assert store.latest() is None
