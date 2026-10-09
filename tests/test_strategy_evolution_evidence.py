from agent.learning.strategy_evolution_evidence import (
    StrategyEvolutionEvidence,
    StrategyEvolutionEvidenceStore,
)


def make_evidence():
    return StrategyEvolutionEvidence(
        strategy_name="open_door_evolved",
        source_strategy_name="open_door",
        source_experience_id="exp-1",
    )


def test_new_evidence_is_experimental():
    evidence = make_evidence()

    assert evidence.status == "experimental"
    assert evidence.is_experimental is True
    assert evidence.trial_count == 0
    assert evidence.success_count == 0
    assert evidence.failure_count == 0


def test_record_successful_trial():
    evidence = make_evidence()

    evidence.record_trial(
        success=True,
        progress_delta=0.5,
        experience_id="exp-2",
    )

    assert evidence.trial_count == 1
    assert evidence.success_count == 1
    assert evidence.failure_count == 0

    assert evidence.success_rate == 1.0
    assert evidence.failure_rate == 0.0
    assert evidence.average_progress == 0.5

    assert evidence.source_experience_ids == [
        "exp-2"
    ]


def test_record_failed_trial():
    evidence = make_evidence()

    evidence.record_trial(
        success=False,
        progress_delta=-0.5,
        experience_id="exp-2",
    )

    assert evidence.trial_count == 1
    assert evidence.success_count == 0
    assert evidence.failure_count == 1

    assert evidence.success_rate == 0.0
    assert evidence.failure_rate == 1.0
    assert evidence.average_progress == -0.5


def test_multiple_trials_are_aggregated():
    evidence = make_evidence()

    evidence.record_trial(
        success=True,
        progress_delta=0.5,
        experience_id="exp-2",
    )

    evidence.record_trial(
        success=False,
        progress_delta=-0.2,
        experience_id="exp-3",
    )

    evidence.record_trial(
        success=True,
        progress_delta=0.3,
        experience_id="exp-4",
    )

    assert evidence.trial_count == 3
    assert evidence.success_count == 2
    assert evidence.failure_count == 1

    assert evidence.success_rate == round(
        2 / 3,
        6,
    )

    assert evidence.failure_rate == round(
        1 / 3,
        6,
    )

    assert evidence.average_progress == round(
        0.6 / 3,
        6,
    )


def test_duplicate_experience_id_is_not_added_twice():
    evidence = make_evidence()

    evidence.record_trial(
        success=True,
        progress_delta=0.5,
        experience_id="exp-2",
    )

    evidence.record_trial(
        success=True,
        progress_delta=0.5,
        experience_id="exp-2",
    )

    assert evidence.trial_count == 2

    assert evidence.source_experience_ids == [
        "exp-2"
    ]


def test_promote_evidence():
    evidence = make_evidence()

    evidence.promote()

    assert evidence.status == "promoted"
    assert evidence.is_promoted is True
    assert evidence.is_experimental is False


def test_reject_evidence():
    evidence = make_evidence()

    evidence.reject()

    assert evidence.status == "rejected"
    assert evidence.is_rejected is True


def test_rejected_evidence_cannot_be_promoted():
    evidence = make_evidence()

    evidence.reject()
    evidence.promote()

    assert evidence.status == "rejected"
    assert evidence.is_rejected is True


def test_reset_experimental_status():
    evidence = make_evidence()

    evidence.promote()

    assert evidence.is_promoted is True

    evidence.reset_experimental()

    assert evidence.status == "experimental"
    assert evidence.is_experimental is True


def test_rejected_evidence_cannot_be_reset():
    evidence = make_evidence()

    evidence.reject()
    evidence.reset_experimental()

    assert evidence.status == "rejected"


def test_store_add_and_get():
    store = StrategyEvolutionEvidenceStore()

    evidence = make_evidence()

    stored = store.add(
        evidence
    )

    assert stored is evidence
    assert store.count() == 1
    assert store.get(
        "open_door_evolved"
    ) is evidence


def test_store_does_not_duplicate_strategy_evidence():
    store = StrategyEvolutionEvidenceStore()

    first = make_evidence()

    second = StrategyEvolutionEvidence(
        strategy_name="open_door_evolved",
        source_strategy_name="other_source",
        source_experience_id="exp-other",
    )

    result_first = store.add(
        first
    )

    result_second = store.add(
        second
    )

    assert result_first is first
    assert result_second is first

    assert store.count() == 1


def test_store_get_or_create_reuses_existing():
    store = StrategyEvolutionEvidenceStore()

    first = store.get_or_create(
        strategy_name="open_door_evolved",
        source_strategy_name="open_door",
        source_experience_id="exp-1",
    )

    second = store.get_or_create(
        strategy_name="open_door_evolved",
        source_strategy_name="another",
        source_experience_id="exp-2",
    )

    assert first is second
    assert store.count() == 1


def test_store_remove():
    store = StrategyEvolutionEvidenceStore()

    evidence = make_evidence()

    store.add(
        evidence
    )

    removed = store.remove(
        "open_door_evolved"
    )

    assert removed is evidence
    assert store.count() == 0
    assert store.get(
        "open_door_evolved"
    ) is None


def test_store_clear():
    store = StrategyEvolutionEvidenceStore()

    store.add(
        make_evidence()
    )

    assert store.count() == 1

    store.clear()

    assert store.count() == 0


def test_none_evidence_is_rejected():
    store = StrategyEvolutionEvidenceStore()

    try:
        store.add(None)
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "evidence cannot be None."
        )
