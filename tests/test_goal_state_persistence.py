from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent.llm.goal_management import (
    GoalLifecycleManager,
    GoalSelectionPolicy,
    GoalStatus,
)
from agent.llm.goal_state_persistence import (
    PersistentGoalStateConfigurationError,
    PersistentGoalStateError,
    PersistentGoalStateLoadResult,
    PersistentGoalStateNotFoundError,
    PersistentGoalStateSerializationError,
    PersistentGoalStateSnapshot,
    PersistentGoalStateStore,
    PersistentGoalStateVersionError,
)


def make_manager() -> GoalLifecycleManager:
    manager = GoalLifecycleManager(selection_policy=GoalSelectionPolicy(0.15))
    manager.add_goal("g1", "primary", priority=0.8, payload={"kind": "main"}, metadata={"tag": "a"})
    manager.add_goal("g2", "secondary", priority=0.6, parent_id="g1", progress=0.0) if False else None
    return manager


def test_error_hierarchy() -> None:
    assert issubclass(PersistentGoalStateConfigurationError, PersistentGoalStateError)
    assert issubclass(PersistentGoalStateSerializationError, PersistentGoalStateError)
    assert issubclass(PersistentGoalStateNotFoundError, PersistentGoalStateError)
    assert issubclass(PersistentGoalStateVersionError, PersistentGoalStateError)


def test_requires_path(tmp_path: Path) -> None:
    with pytest.raises(PersistentGoalStateConfigurationError):
        PersistentGoalStateStore(None)  # type: ignore[arg-type]


def test_directory_path_gets_default_filename(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path)
    assert store.path == tmp_path / "goal_state.sqlite3"


def test_file_path_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    store = PersistentGoalStateStore(path)
    assert store.path == path


def test_parent_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deep" / "state.sqlite3"
    store = PersistentGoalStateStore(path)
    assert store.path.exists()


def test_new_store_is_empty(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    assert not store.exists()
    assert store.count() == 0


def test_load_without_state_raises(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    with pytest.raises(PersistentGoalStateNotFoundError):
        store.load()


def test_delete_without_state_returns_false(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    assert store.delete() is False


def test_clear_without_state_returns_false(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    assert store.clear() is False


def test_save_empty_manager(tmp_path: Path) -> None:
    manager = GoalLifecycleManager(selection_policy=GoalSelectionPolicy(0.25))
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    saved = store.save_manager(manager)
    assert saved.manager_snapshot == manager.snapshot()
    assert saved.preempt_priority_margin == 0.25
    assert saved.schema_version == 1
    assert store.exists()
    assert store.count() == 1


def test_save_returns_snapshot_envelope(tmp_path: Path) -> None:
    manager = make_manager()
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    saved = store.save_manager(manager)
    assert isinstance(saved, PersistentGoalStateSnapshot)
    assert saved.manager_snapshot.goals[0].goal_id == "g1"


def test_save_preserves_goal_fields(tmp_path: Path) -> None:
    manager = make_manager()
    child = manager.add_goal(
        "g2", "secondary", priority=0.6, parent_id="g1", payload=[1, 2], metadata={"x": True}
    )
    manager.update_progress("g1", 0.4)
    manager.set_current_goal("g1")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    loaded = store.load()
    goals = {goal.goal_id: goal for goal in loaded.manager_snapshot.goals}
    assert goals["g1"].priority == 0.8
    assert goals["g1"].progress == 0.4
    assert goals["g1"].metadata == {"tag": "a"}
    assert goals["g1"].payload == {"kind": "main"}
    assert goals["g2"].parent_id == "g1"
    assert child.sequence == goals["g2"].sequence


def test_save_preserves_current_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.set_current_goal("g1")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    assert store.load().manager_snapshot.current_goal_id == "g1"


def test_save_preserves_sequence(tmp_path: Path) -> None:
    manager = make_manager()
    manager.add_goal("g2", "secondary")
    manager.add_goal("g3", "third")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    assert store.load().manager_snapshot.sequence == 3
    assert [g.sequence for g in store.load().manager_snapshot.goals] == [1, 2, 3]


def test_save_preserves_policy_margin(tmp_path: Path) -> None:
    manager = make_manager()
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    result = store.restore_manager()
    assert result.manager.selection_policy.preempt_priority_margin == 0.15


def test_load_roundtrip_manager_snapshot(tmp_path: Path) -> None:
    manager = make_manager()
    manager.add_goal("g2", "secondary", priority=0.2)
    manager.activate("g1")
    manager.update_progress("g1", 0.7)
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    loaded = store.load()
    assert loaded.manager_snapshot == manager.snapshot()


def test_restore_manager_returns_load_result(tmp_path: Path) -> None:
    manager = make_manager()
    manager.activate("g1")
    manager.update_progress("g1", 0.5)
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    result = store.restore_manager()
    assert isinstance(result, PersistentGoalStateLoadResult)
    assert result.manager.snapshot() == manager.snapshot()


def test_restore_pending_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.add_goal("g2", "secondary")
    manager.update_progress("g2", 0.2)
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    restored = store.restore_manager().manager
    assert restored.get_goal("g2").status is GoalStatus.PENDING
    assert restored.get_goal("g2").progress == 0.2


def test_restore_active_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.set_current_goal("g1")
    manager.update_progress("g1", 0.3)
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    restored = store.restore_manager().manager
    assert restored.get_goal("g1").status is GoalStatus.ACTIVE
    assert restored.current_goal_id == "g1"


def test_restore_paused_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.activate("g1")
    manager.pause("g1")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    restored = store.restore_manager().manager
    assert restored.get_goal("g1").status is GoalStatus.PAUSED
    assert restored.current_goal_id is None


def test_restore_completed_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.activate("g1")
    manager.complete("g1")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    restored = store.restore_manager().manager
    goal = restored.get_goal("g1")
    assert goal.status is GoalStatus.COMPLETED
    assert goal.progress == 1.0


def test_restore_failed_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.activate("g1")
    manager.fail("g1")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    assert store.restore_manager().manager.get_goal("g1").status is GoalStatus.FAILED


def test_restore_cancelled_goal(tmp_path: Path) -> None:
    manager = make_manager()
    manager.cancel("g1")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    assert store.restore_manager().manager.get_goal("g1").status is GoalStatus.CANCELLED


def test_delete_existing_state(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(make_manager())
    assert store.delete() is True
    assert not store.exists()


def test_clear_existing_state(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(make_manager())
    assert store.clear() is True
    assert store.count() == 0


def test_second_save_replaces_previous_snapshot(tmp_path: Path) -> None:
    manager = make_manager()
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    manager.add_goal("g2", "new")
    store.save_manager(manager)
    loaded = store.load()
    assert [g.goal_id for g in loaded.manager_snapshot.goals] == ["g1", "g2"]


def test_save_manager_after_status_change_persists_change(tmp_path: Path) -> None:
    manager = make_manager()
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    manager.set_current_goal("g1")
    manager.update_progress("g1", 0.9)
    store.save_manager(manager)
    loaded_goal = store.load().manager_snapshot.goals[0]
    assert loaded_goal.status is GoalStatus.ACTIVE
    assert loaded_goal.progress == 0.9


def test_json_serializable_payload_is_supported(tmp_path: Path) -> None:
    manager = GoalLifecycleManager()
    manager.add_goal("g", "json", payload={"nested": [1, "two", True], "none": None})
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    assert store.restore_manager().manager.get_goal("g").payload == {
        "nested": [1, "two", True], "none": None
    }


def test_non_serializable_payload_raises(tmp_path: Path) -> None:
    manager = GoalLifecycleManager()
    manager.add_goal("g", "bad", payload={"value": object()})
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    with pytest.raises(PersistentGoalStateSerializationError):
        store.save_manager(manager)


def test_non_serializable_metadata_raises(tmp_path: Path) -> None:
    manager = GoalLifecycleManager()
    manager.add_goal("g", "bad", metadata={"value": object()})
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    with pytest.raises(PersistentGoalStateSerializationError):
        store.save_manager(manager)


def test_invalid_persisted_json_raises(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO goal_state(singleton_id, schema_version, payload) VALUES (1, 1, ?)",
            ("{not-json",),
        )
    with pytest.raises(PersistentGoalStateSerializationError):
        store.load()


def test_invalid_schema_version_column_raises(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO goal_state(singleton_id, schema_version, payload) VALUES (1, 99, ?)",
            (json.dumps({}),),
        )
    with pytest.raises(PersistentGoalStateVersionError):
        store.load()


def test_invalid_schema_version_payload_raises(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    payload = {"schema_version": 99, "preempt_priority_margin": 0.0, "goals": [], "sequence": 0}
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO goal_state(singleton_id, schema_version, payload) VALUES (1, 1, ?)",
            (json.dumps(payload),),
        )
    with pytest.raises(PersistentGoalStateVersionError):
        store.load()


def test_invalid_current_goal_reference_raises(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    payload = {"schema_version": 1, "preempt_priority_margin": 0.0, "goals": [], "current_goal_id": "missing", "sequence": 0}
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO goal_state(singleton_id, schema_version, payload) VALUES (1, 1, ?)",
            (json.dumps(payload),),
        )
    with pytest.raises(PersistentGoalStateSerializationError):
        store.load()


def test_invalid_sequence_consistency_raises(tmp_path: Path) -> None:
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    goal = {
        "goal_id": "g",
        "description": "goal",
        "priority": 0.5,
        "parent_id": None,
        "status": "pending",
        "progress": 0.0,
        "payload": None,
        "metadata": {},
        "sequence": 2,
    }
    payload = {"schema_version": 1, "preempt_priority_margin": 0.0, "goals": [goal], "current_goal_id": None, "sequence": 1}
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO goal_state(singleton_id, schema_version, payload) VALUES (1, 1, ?)",
            (json.dumps(payload),),
        )
    with pytest.raises(PersistentGoalStateSerializationError):
        store.load()


def test_restore_preserves_hierarchy(tmp_path: Path) -> None:
    manager = GoalLifecycleManager()
    manager.add_goal("root", "root")
    manager.add_goal("child", "child", parent_id="root")
    manager.add_goal("grand", "grand", parent_id="child")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    restored = store.restore_manager().manager
    assert [g.goal_id for g in restored.children_of("root")] == ["child"]
    assert [g.goal_id for g in restored.children_of("child")] == ["grand"]


def test_restore_preserves_insertion_order(tmp_path: Path) -> None:
    manager = make_manager()
    manager.add_goal("g2", "second")
    manager.add_goal("g3", "third")
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    restored = store.restore_manager().manager
    assert [g.goal_id for g in restored.list_goals()] == ["g1", "g2", "g3"]


def test_restore_empty_manager(tmp_path: Path) -> None:
    manager = GoalLifecycleManager(selection_policy=GoalSelectionPolicy(0.7))
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    result = store.restore_manager()
    assert result.manager.snapshot().goals == ()
    assert result.manager.current_goal_id is None
    assert result.manager.selection_policy.preempt_priority_margin == 0.7


def test_persistence_is_separate_database_file(tmp_path: Path) -> None:
    path = tmp_path / "goal.sqlite3"
    store = PersistentGoalStateStore(path)
    assert path.exists()
    assert path.suffix == ".sqlite3"


def test_save_is_atomic_from_singleton_perspective(tmp_path: Path) -> None:
    manager = make_manager()
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    with sqlite3.connect(store.path) as connection:
        rows = connection.execute("SELECT singleton_id FROM goal_state").fetchall()
    assert rows == [(1,)]


def test_loaded_payload_is_not_shared_with_store_snapshot(tmp_path: Path) -> None:
    manager = GoalLifecycleManager()
    payload = {"items": [1, 2]}
    manager.add_goal("g", "goal", payload=payload)
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    loaded = store.load()
    assert loaded.manager_snapshot.goals[0].payload == payload
    payload["items"].append(3)
    assert loaded.manager_snapshot.goals[0].payload == {"items": [1, 2]}


def test_schema_version_constant_is_one() -> None:
    assert PersistentGoalStateStore.SCHEMA_VERSION == 1


def test_saved_json_is_utf8_and_compact(tmp_path: Path) -> None:
    manager = GoalLifecycleManager()
    manager.add_goal("g", "tujuan ä", metadata={"label": "日本語"})
    store = PersistentGoalStateStore(tmp_path / "state.sqlite3")
    store.save_manager(manager)
    with sqlite3.connect(store.path) as connection:
        payload = connection.execute("SELECT payload FROM goal_state WHERE singleton_id=1").fetchone()[0]
    assert "日本語" in payload
    assert "\n" not in payload
