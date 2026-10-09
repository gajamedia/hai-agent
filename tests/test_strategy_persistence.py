from agent.learning.strategy import Strategy
from agent.memory.strategy_repository import (
    StrategyRepository,
)
from agent.memory.strategy_store import (
    StrategyMemoryStore,
)


def test_strategy_persists_and_reloads(tmp_path):

    database_path = (
        tmp_path
        / "agent.db"
    )

    repository = StrategyRepository(
        database_path
    )

    store = StrategyMemoryStore(
        repository=repository
    )

    strategy = Strategy(
        name="open_box",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=7,
        failure_count=2,
        total_actions=9,
        reflection_penalty=0.2,
        total_progress=5.5,
        progress_count=9,
        positive_progress_count=7,
        negative_progress_count=1,
        status="active",
    )

    strategy.source_experiences = [
        "experience-1",
        "experience-2",
    ]

    store.add(
        strategy
    )

    second_repository = (
        StrategyRepository(
            database_path
        )
    )

    second_store = (
        StrategyMemoryStore(
            repository=second_repository
        )
    )

    loaded = (
        second_store.all()
    )

    assert len(
        loaded
    ) == 1

    restored = loaded[0]

    assert (
        restored.name
        == "open_box"
    )

    assert (
        restored.goal
        == "open the box"
    )

    assert (
        restored.steps
        == ["open_box"]
    )

    assert (
        restored.success_count
        == 7
    )

    assert (
        restored.failure_count
        == 2
    )

    assert (
        restored.total_actions
        == 9
    )

    assert (
        restored.reflection_penalty
        == 0.2
    )

    assert (
        restored.total_progress
        == 5.5
    )

    assert (
        restored.progress_count
        == 9
    )

    assert (
        restored.positive_progress_count
        == 7
    )

    assert (
        restored.negative_progress_count
        == 1
    )

    assert (
        restored.source_experiences
        == [
            "experience-1",
            "experience-2",
        ]
    )

    assert (
        restored.status
        == "active"
    )


def test_strategy_update_is_persistent(tmp_path):

    database_path = (
        tmp_path
        / "agent.db"
    )

    repository = StrategyRepository(
        database_path
    )

    store = StrategyMemoryStore(
        repository=repository
    )

    strategy = Strategy(
        name="test",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    store.add(
        strategy
    )

    strategy.record_success(
        experience_id="experience-1",
        action_count=1,
        progress_delta=1.0,
    )

    store.save(
        strategy
    )

    reloaded = (
        StrategyMemoryStore(
            repository=(
                StrategyRepository(
                    database_path
                )
            )
        )
    )

    restored = (
        reloaded.all()[0]
    )

    assert (
        restored.success_count
        == 1
    )

    assert (
        restored.average_progress
        == 1.0
    )


def test_strategy_lifecycle_is_persistent(tmp_path):

    database_path = (
        tmp_path
        / "agent.db"
    )

    repository = StrategyRepository(
        database_path
    )

    store = StrategyMemoryStore(
        repository=repository
    )

    strategy = Strategy(
        name="bad",
        goal="open the box",
        steps=[
            "open_box",
        ],
        success_count=0,
        failure_count=10,
        total_progress=-3.0,
        progress_count=10,
        status="retired",
    )

    store.add(
        strategy
    )

    store.save(
        strategy
    )

    reloaded = (
        StrategyMemoryStore(
            repository=(
                StrategyRepository(
                    database_path
                )
            )
        )
    )

    restored = (
        reloaded.all()[0]
    )

    assert (
        restored.status
        == "retired"
    )
