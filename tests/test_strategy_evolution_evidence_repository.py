
from agent.learning.strategy_evolution_evidence import (
    StrategyEvolutionEvidence,
    StrategyEvolutionEvidenceStore,
)
from agent.memory.strategy_evolution_evidence_repository import (
    StrategyEvolutionEvidenceRepository,
)


def test_evidence_survives_store_recreation(tmp_path):
    database_path = tmp_path / "memory.db"

    first_store = StrategyEvolutionEvidenceStore(
        repository=StrategyEvolutionEvidenceRepository(
            database_path
        )
    )

    evidence = StrategyEvolutionEvidence(
        strategy_name="open_door_v2",
        source_strategy_name="open_door",
        source_experience_id="exp-001",
        generation=2,
    )
    first_store.add(evidence)

    evidence.record_trial(
        success=True,
        progress_delta=1.5,
        experience_id="exp-002",
    )
    first_store.save(evidence)

    second_store = StrategyEvolutionEvidenceStore(
        repository=StrategyEvolutionEvidenceRepository(
            database_path
        )
    )

    restored = second_store.get("open_door_v2")

    assert restored is not None
    assert restored.source_strategy_name == "open_door"
    assert restored.generation == 2
    assert restored.trial_count == 1
    assert restored.success_count == 1
    assert restored.failure_count == 0
    assert restored.total_progress == 1.5
    assert restored.source_experience_ids == ["exp-002"]


def test_evidence_store_clear_removes_persisted_data(tmp_path):
    database_path = tmp_path / "memory.db"

    store = StrategyEvolutionEvidenceStore(
        repository=StrategyEvolutionEvidenceRepository(
            database_path
        )
    )
    store.add(
        StrategyEvolutionEvidence(
            strategy_name="strategy_a",
            source_strategy_name="strategy_original",
            source_experience_id="exp-001",
        )
    )

    store.clear()

    restored = StrategyEvolutionEvidenceStore(
        repository=StrategyEvolutionEvidenceRepository(
            database_path
        )
    )

    assert restored.count() == 0


def test_in_memory_evidence_store_remains_supported():
    store = StrategyEvolutionEvidenceStore()
    evidence = store.get_or_create(
        strategy_name="strategy_a",
        source_strategy_name="strategy_original",
        source_experience_id="exp-001",
    )

    evidence.record_trial(success=False, experience_id="exp-002")

    assert store.count() == 1
    assert store.get("strategy_a").failure_count == 1
