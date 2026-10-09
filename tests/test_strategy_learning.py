from agent.core.experience import Experience
from agent.memory.episodic import EpisodicMemory
from agent.learning.strategy_learning import (
    StrategyLearning,
)
from agent.learning.strategy_store import (
    StrategyStore,
)


def create_experience(
    experience_id: str,
    completed: bool,
):

    experience = Experience(
        experience_id=experience_id,
        goal="Open the door",
    )

    experience.add_episode(
        EpisodicMemory(
            content="Opened box.",
            action="open_box",
        )
    )

    experience.add_episode(
        EpisodicMemory(
            content="Took key.",
            action="take_key",
        )
    )

    experience.add_episode(
        EpisodicMemory(
            content="Unlocked door.",
            action="unlock_door",
        )
    )

    if completed:
        experience.complete()
    else:
        experience.fail()

    return experience


def test_strategy_learning():

    store = StrategyStore()

    learning = StrategyLearning(
        strategy_store=store,
    )

    experience = create_experience(
        "experience-1",
        True,
    )

    strategy = (
        learning.learn_from_experience(
            experience
        )
    )

    assert strategy is not None

    assert strategy.goal == (
        "Open the door"
    )

    assert strategy.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert strategy.attempts == 1
    assert strategy.success_count == 1
    assert strategy.failure_count == 0
    assert strategy.success_rate == 1.0
    assert strategy.average_actions == 3.0


def test_strategy_learning_records_failure():

    store = StrategyStore()

    learning = StrategyLearning(
        strategy_store=store,
    )

    experience = create_experience(
        "experience-1",
        False,
    )

    strategy = (
        learning.learn_from_experience(
            experience
        )
    )

    assert strategy is not None
    assert strategy.attempts == 1
    assert strategy.success_count == 0
    assert strategy.failure_count == 1
    assert strategy.success_rate == 0.0


def test_strategy_learning_reuses_same_strategy():

    store = StrategyStore()

    learning = StrategyLearning(
        strategy_store=store,
    )

    first = create_experience(
        "experience-1",
        True,
    )

    second = create_experience(
        "experience-2",
        True,
    )

    first_strategy = (
        learning.learn_from_experience(
            first
        )
    )

    second_strategy = (
        learning.learn_from_experience(
            second
        )
    )

    assert first_strategy is second_strategy
    assert store.count() == 1

    assert second_strategy.attempts == 2
    assert second_strategy.success_count == 2
    assert second_strategy.success_rate == 1.0
