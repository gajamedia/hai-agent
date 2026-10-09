from pathlib import Path

import pytest

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SelfModelDecisionInfluenceIntegrator,
)

from agent.learning.self_uncertainty import (
    SelfUncertaintyResult,
)

from agent.learning.self_model_persistence import (
    PersistentSelfModelStore,
    SelfModelContinuitySnapshot,
    build_persistent_self_model_store,
)


# ============================================================
# Helpers
# ============================================================


def make_uncertainty(
    *,
    uncertainty=0.80,
    confidence=0.20,
    evidence_strength=0.80,
) -> SelfUncertaintyResult:

    return SelfUncertaintyResult(
        uncertainty=uncertainty,
        confidence=confidence,
        evidence_strength=evidence_strength,
        average_capability_uncertainty=(
            uncertainty
        ),
        average_capability_confidence=(
            confidence
        ),
        confidence_dispersion=0.0,
        capability_count=1,
        observed_capability_count=1,
        high_uncertainty_count=(
            1
            if uncertainty >= 0.60
            else 0
        ),
        low_confidence_count=(
            1
            if confidence < 0.40
            else 0
        ),
        strongest_capability="planning",
        weakest_capability="planning",
    )


def make_influence(
    uncertainty_result: SelfUncertaintyResult,
):
    return (
        SelfModelDecisionInfluenceIntegrator()
        .assess(
            uncertainty_result
        )
    )

# ============================================================
# Initialization
# ============================================================


def test_store_creates_database(tmp_path: Path):
    database_path = (
        tmp_path
        / "self_model.sqlite3"
    )

    store = PersistentSelfModelStore(
        database_path
    )

    assert database_path.exists()
    assert store.count() == 0


def test_builder_creates_store(tmp_path: Path):
    database_path = (
        tmp_path
        / "self_model.sqlite3"
    )

    store = (
        build_persistent_self_model_store(
            database_path
        )
    )

    assert isinstance(
        store,
        PersistentSelfModelStore,
    )


# ============================================================
# Save / Load
# ============================================================


def test_save_returns_snapshot(tmp_path):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    uncertainty = make_uncertainty()

    influence = make_influence(
        uncertainty
    )

    snapshot = store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    assert isinstance(
        snapshot,
        SelfModelContinuitySnapshot,
    )

    assert snapshot.agent_name == "agent-a"
    assert snapshot.version == 1
    assert snapshot.decision_influence == influence


def test_load_returns_saved_state(tmp_path):
    database_path = (
        tmp_path
        / "state.sqlite3"
    )

    store = PersistentSelfModelStore(
        database_path
    )

    uncertainty = make_uncertainty()

    influence = make_influence(
        uncertainty
    )

    store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    loaded = store.load(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.self_uncertainty
        == uncertainty
    )
    assert (
        loaded.decision_influence
        == influence
    )


def test_persistence_survives_new_store_instance(
    tmp_path,
):
    database_path = (
        tmp_path
        / "state.sqlite3"
    )

    uncertainty = make_uncertainty()

    influence = make_influence(
        uncertainty
    )

    first_store = (
        PersistentSelfModelStore(
            database_path
        )
    )

    first_store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    second_store = (
        PersistentSelfModelStore(
            database_path
        )
    )

    loaded = second_store.load(
        "agent-a"
    )

    assert loaded is not None

    assert (
        loaded.self_uncertainty
        == uncertainty
    )

    assert (
        loaded.decision_influence
        == influence
    )


# ============================================================
# Versioning
# ============================================================


def test_first_save_has_version_one(tmp_path):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    snapshot = store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(),
    )

    assert snapshot.version == 1


def test_second_save_increments_version(tmp_path):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    first = store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(
            uncertainty=0.80,
            confidence=0.20,
        ),
    )

    second = store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(
            uncertainty=0.30,
            confidence=0.70,
        ),
    )

    assert first.version == 1
    assert second.version == 2


def test_version_is_persisted(
    tmp_path,
):
    database_path = (
        tmp_path
        / "state.sqlite3"
    )

    first_store = (
        PersistentSelfModelStore(
            database_path
        )
    )

    first_store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(),
    )

    first_store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(
            uncertainty=0.40,
            confidence=0.60,
        ),
    )

    second_store = (
        PersistentSelfModelStore(
            database_path
        )
    )

    loaded = second_store.load(
        "agent-a"
    )

    assert loaded is not None
    assert loaded.version == 2


# ============================================================
# Influence Continuity
# ============================================================


def test_exploration_influence_is_preserved(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    uncertainty = make_uncertainty(
        uncertainty=0.90,
        confidence=0.10,
    )

    influence = make_influence(
        uncertainty
    )

    store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    loaded = store.load(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.decision_influence
        is not None
    )

    assert (
        loaded.decision_influence.preferred_mode
        == SELF_MODEL_MODE_EXPLORE
    )


def test_exploitation_influence_is_preserved(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    uncertainty = make_uncertainty(
        uncertainty=0.10,
        confidence=0.90,
        evidence_strength=0.90,
    )

    influence = make_influence(
        uncertainty
    )

    store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    loaded = store.load(
        "agent-a"
    )

    assert loaded is not None

    assert (
        loaded.decision_influence
        is not None
    )

    assert (
        loaded.decision_influence.preferred_mode
        == SELF_MODEL_MODE_EXPLOIT
    )


def test_state_can_be_saved_without_decision_influence(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    uncertainty = make_uncertainty()

    store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
    )

    loaded = store.load(
        "agent-a"
    )

    assert loaded is not None
    assert (
        loaded.decision_influence
        is None
    )


# ============================================================
# Agent Isolation
# ============================================================


def test_multiple_agents_are_isolated(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    agent_a = make_uncertainty(
        uncertainty=0.90,
        confidence=0.10,
    )

    agent_b = make_uncertainty(
        uncertainty=0.10,
        confidence=0.90,
        evidence_strength=0.90,
    )

    store.save(
        agent_name="agent-a",
        self_uncertainty=agent_a,
    )

    store.save(
        agent_name="agent-b",
        self_uncertainty=agent_b,
    )

    loaded_a = store.load(
        "agent-a"
    )

    loaded_b = store.load(
        "agent-b"
    )

    assert loaded_a is not None
    assert loaded_b is not None

    assert (
        loaded_a.self_uncertainty
        == agent_a
    )

    assert (
        loaded_b.self_uncertainty
        == agent_b
    )


def test_list_agent_names(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    store.save(
        agent_name="zeta",
        self_uncertainty=make_uncertainty(),
    )

    store.save(
        agent_name="alpha",
        self_uncertainty=make_uncertainty(),
    )

    store.save(
        agent_name="beta",
        self_uncertainty=make_uncertainty(),
    )

    assert store.list_agent_names() == (
        "alpha",
        "beta",
        "zeta",
    )


def test_count_tracks_agents(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    assert store.count() == 0

    store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(),
    )

    assert store.count() == 1

    store.save(
        agent_name="agent-b",
        self_uncertainty=make_uncertainty(),
    )

    assert store.count() == 2


# ============================================================
# Missing State
# ============================================================


def test_load_missing_agent_returns_none(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    assert store.load(
        "missing-agent"
    ) is None


def test_exists_false_for_missing_agent(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    assert not store.exists(
        "missing-agent"
    )


def test_exists_true_for_saved_agent(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(),
    )

    assert store.exists(
        "agent-a"
    )


# ============================================================
# Delete
# ============================================================


def test_delete_removes_state(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    store.save(
        agent_name="agent-a",
        self_uncertainty=make_uncertainty(),
    )

    assert store.delete(
        "agent-a"
    )

    assert store.load(
        "agent-a"
    ) is None


def test_delete_missing_agent_returns_false(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    assert not store.delete(
        "missing-agent"
    )


def test_clear_removes_all_agents(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    for name in (
        "agent-a",
        "agent-b",
        "agent-c",
    ):
        store.save(
            agent_name=name,
            self_uncertainty=make_uncertainty(),
        )

    removed = store.clear()

    assert removed == 3
    assert store.count() == 0


# ============================================================
# Validation
# ============================================================


def test_none_database_path_is_rejected():
    with pytest.raises(ValueError):
        PersistentSelfModelStore(
            None
        )


def test_empty_agent_name_is_rejected(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    with pytest.raises(ValueError):
        store.save(
            agent_name="",
            self_uncertainty=make_uncertainty(),
        )


def test_whitespace_agent_name_is_rejected(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    with pytest.raises(ValueError):
        store.save(
            agent_name="   ",
            self_uncertainty=make_uncertainty(),
        )


def test_non_string_agent_name_is_rejected(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    with pytest.raises(ValueError):
        store.save(
            agent_name=123,
            self_uncertainty=make_uncertainty(),
        )


def test_none_uncertainty_is_rejected(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    with pytest.raises(ValueError):
        store.save(
            agent_name="agent-a",
            self_uncertainty=None,
        )


def test_wrong_uncertainty_type_is_rejected(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    with pytest.raises(ValueError):
        store.save(
            agent_name="agent-a",
            self_uncertainty=object(),
        )


def test_wrong_influence_type_is_rejected(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    with pytest.raises(ValueError):
        store.save(
            agent_name="agent-a",
            self_uncertainty=make_uncertainty(),
            decision_influence=object(),
        )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot_contains_expected_fields(
    tmp_path,
):
    store = PersistentSelfModelStore(
        tmp_path / "state.sqlite3"
    )

    uncertainty = make_uncertainty()
    influence = make_influence(
        uncertainty
    )

    snapshot = store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    data = snapshot.snapshot()

    assert set(data.keys()) == {
        "agent_name",
        "self_uncertainty",
        "decision_influence",
        "version",
        "updated_at",
    }


def test_snapshot_round_trip_is_stable(
    tmp_path,
):
    database_path = (
        tmp_path
        / "state.sqlite3"
    )

    store = PersistentSelfModelStore(
        database_path
    )

    uncertainty = make_uncertainty(
        uncertainty=0.72,
        confidence=0.28,
        evidence_strength=0.64,
    )

    influence = make_influence(
        uncertainty
    )

    saved = store.save(
        agent_name="agent-a",
        self_uncertainty=uncertainty,
        decision_influence=influence,
    )

    loaded = store.load(
        "agent-a"
    )

    assert loaded is not None

    assert (
        loaded.agent_name
        == saved.agent_name
    )

    assert (
        loaded.self_uncertainty
        == saved.self_uncertainty
    )

    assert (
        loaded.decision_influence
        == saved.decision_influence
    )

    assert (
        loaded.version
        == saved.version
    )

    assert (
        loaded.updated_at
        == saved.updated_at
    )
