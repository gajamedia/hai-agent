from agent.learning.strategy import Strategy
from agent.memory.persistence import (
    SelfModelRepository,
    SQLiteDatabase,
)
from agent.self.assessment import SelfAssessment
from agent.self.capability import Capability
from agent.self.model import SelfModel
from agent.self.store import SelfModelStore


def create_model():

    model = SelfModel(
        limitations=[
            "cannot open jammed box",
        ],
        known_successes=[
            "Achieved goal: open the door",
        ],
        known_failures=[
            "Failed goal: retrieve key",
        ],
        confidence=0.8,
        current_condition={
            "box": "closed",
            "door": "locked",
        },
    )

    capability = Capability(
        name="open_box",
        confidence=0.7,
        success_count=2,
        failure_count=1,
        evidence=[
            "episode-001",
            "episode-002",
            "episode-003",
        ],
    )

    model.register_capability(
        "open_box",
        capability,
    )

    strategy = Strategy(
        name="door_strategy",
        goal="open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=1,
        total_actions=10,
        reflection_penalty=0.2,
        source_experiences=[
            "experience-001",
        ],
    )

    model.register_strategy(
        strategy
    )

    assessment = SelfAssessment(
        confidence=0.7,
        uncertainty=0.3,
        risk=0.25,
    )

    model.register_assessment(
        "capability:open_box",
        assessment,
    )

    return model


def test_self_model_can_be_persisted(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = SelfModelRepository(
        database
    )

    model = create_model()

    repository.save(model)

    assert repository.exists() is True

    loaded = repository.get()

    assert loaded is not None

    assert loaded.confidence == 0.8

    assert loaded.limitations == [
        "cannot open jammed box",
    ]

    capability = loaded.get_capability(
        "open_box"
    )

    assert capability is not None
    assert capability.confidence == 0.7
    assert capability.success_count == 2
    assert capability.failure_count == 1

    strategy = (
        loaded.learned_strategies[
            "door_strategy"
        ]
    )

    assert strategy.steps == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

    assert strategy.reflection_penalty == 0.2

    assessment = (
        loaded.get_assessment(
            "capability:open_box"
        )
    )

    assert assessment is not None
    assert assessment.confidence == 0.7
    assert assessment.uncertainty == 0.3
    assert assessment.risk == 0.25


def test_self_model_survives_restart(
    tmp_path,
):

    database_path = (
        tmp_path / "memory.db"
    )

    database_1 = SQLiteDatabase(
        database_path
    )

    repository_1 = SelfModelRepository(
        database_1
    )

    repository_1.save(
        create_model()
    )

    del repository_1
    del database_1

    database_2 = SQLiteDatabase(
        database_path
    )

    repository_2 = SelfModelRepository(
        database_2
    )

    loaded = repository_2.get()

    assert loaded is not None

    assert loaded.get_capability(
        "open_box"
    ) is not None

    assert (
        loaded.learned_strategies[
            "door_strategy"
        ].success_count
        == 3
    )


def test_self_model_store_loads_persistent_model(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = SelfModelRepository(
        database
    )

    store_1 = SelfModelStore(
        repository=repository
    )

    store_1.model = create_model()

    store_1.save()

    del store_1

    store_2 = SelfModelStore(
        repository=repository
    )

    loaded = store_2.get()

    assert loaded.confidence == 0.8

    assert loaded.get_capability(
        "open_box"
    ) is not None


def test_self_model_store_persists_updates(
    tmp_path,
):

    database = SQLiteDatabase(
        tmp_path / "memory.db"
    )

    repository = SelfModelRepository(
        database
    )

    store = SelfModelStore(
        repository=repository
    )

    model = store.get()

    model.confidence = 0.9

    model.register_capability(
        "take_key",
        Capability(
            name="take_key",
            confidence=0.6,
            success_count=1,
        ),
    )

    store.save()

    loaded = repository.get()

    assert loaded is not None
    assert loaded.confidence == 0.9

    capability = loaded.get_capability(
        "take_key"
    )

    assert capability is not None
    assert capability.success_count == 1
