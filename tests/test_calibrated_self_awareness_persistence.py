from pathlib import Path

from agent.learning.calibrated_self_awareness import (
    CalibratedSelfAwareInfluenceApplier,
)
from agent.learning.calibrated_self_awareness_persistence import (
    CalibratedSelfAwarenessPersistenceSnapshot,
    PersistentCalibratedSelfAwarenessStore,
)
from agent.learning.self_awareness_calibration import (
    SelfAwarenessCalibrator,
)
from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SelfModelDecisionInfluence,
)


def make_influence():
    return SelfModelDecisionInfluence(
        preferred_mode=SELF_MODEL_MODE_EXPLOIT,
        explore_bonus=0.05,
        exploit_bonus=0.20,
        confidence=0.80,
        uncertainty=0.20,
        strength=1.0,
        applied=True,
        reason="persistent test",
    )


def make_calibration():
    calibration = SelfAwarenessCalibrator().calibrate(
        raw_confidence=0.80,
        evidence={
            "execution": {
                "attempts": 10,
                "successes": 8,
                "failures": 2,
            }
        },
    )

    return (
        CalibratedSelfAwareInfluenceApplier()
        .apply(
            make_influence(),
            calibration,
        )
    )


def build_store(
    tmp_path: Path,
):
    return PersistentCalibratedSelfAwarenessStore(
        tmp_path
        / "calibrated.sqlite3"
    )


def test_store_creates_database(tmp_path):
    store = build_store(tmp_path)

    assert store.path.exists()


def test_initial_store_is_empty(tmp_path):
    store = build_store(tmp_path)

    assert store.count() == 0
    assert store.list_agent_names() == []


def test_save_and_load(tmp_path):
    store = build_store(tmp_path)

    influence = make_influence()
    result = make_calibration()

    saved = store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    assert isinstance(
        saved,
        CalibratedSelfAwarenessPersistenceSnapshot,
    )

    loaded = store.load(
        "test-agent"
    )

    assert loaded is not None

    assert (
        loaded.agent_name
        == "test-agent"
    )

    assert (
        loaded.influence
        == result.influence
    )

    assert (
        loaded.calibration_result
        == result
    )


def test_first_save_has_version_one(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    saved = store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    assert saved.version == 1


def test_second_save_increments_version(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    first = store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    second = store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    assert first.version == 1
    assert second.version == 2


def test_exists(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    assert not store.exists(
        "test-agent"
    )

    store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    assert store.exists(
        "test-agent"
    )


def test_delete(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    assert store.delete(
        "test-agent"
    )

    assert not store.exists(
        "test-agent"
    )


def test_delete_unknown_returns_false(tmp_path):
    store = build_store(tmp_path)

    assert not store.delete(
        "unknown"
    )


def test_multiple_agents(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    store.save(
        agent_name="agent-a",
        influence=result.influence,
        calibration_result=result,
    )

    store.save(
        agent_name="agent-b",
        influence=result.influence,
        calibration_result=result,
    )

    assert store.count() == 2

    assert (
        store.list_agent_names()
        == [
            "agent-a",
            "agent-b",
        ]
    )


def test_clear(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    store.save(
        agent_name="agent-a",
        influence=result.influence,
        calibration_result=result,
    )

    store.save(
        agent_name="agent-b",
        influence=result.influence,
        calibration_result=result,
    )

    store.clear()

    assert store.count() == 0


def test_load_unknown_returns_none(tmp_path):
    store = build_store(tmp_path)

    assert (
        store.load("unknown")
        is None
    )


def test_invalid_agent_name_is_rejected(tmp_path):
    store = build_store(tmp_path)

    result = make_calibration()

    try:
        store.save(
            agent_name="",
            influence=result.influence,
            calibration_result=result,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_persistence_snapshot_is_serializable(
    tmp_path,
):
    store = build_store(tmp_path)

    result = make_calibration()

    saved = store.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    snapshot = saved.snapshot

    assert isinstance(
        snapshot,
        dict,
    )

    assert (
        snapshot["agent_name"]
        == "test-agent"
    )

    assert (
        "influence"
        in snapshot
    )

    assert (
        "calibration_result"
        in snapshot
    )


def test_persistence_survives_new_store_instance(
    tmp_path,
):
    path = (
        tmp_path
        / "calibrated.sqlite3"
    )

    store1 = (
        PersistentCalibratedSelfAwarenessStore(
            path
        )
    )

    result = make_calibration()

    store1.save(
        agent_name="test-agent",
        influence=result.influence,
        calibration_result=result,
    )

    store2 = (
        PersistentCalibratedSelfAwarenessStore(
            path
        )
    )

    loaded = store2.load(
        "test-agent"
    )

    assert loaded is not None

    assert (
        loaded.influence
        == result.influence
    )

    assert (
        loaded.calibration_result
        == result
    )
