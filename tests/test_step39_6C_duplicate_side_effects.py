from __future__ import annotations

import sqlite3

import pytest

from agent.llm.p0_idempotent_side_effects import (
    IdempotencyConflictError,
    SQLiteIdempotentSideEffectStore,
)


def _setup_effect_table(database_path):
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE IF NOT EXISTS world_state (name TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    connection.execute(
        "INSERT OR IGNORE INTO world_state (name, value) VALUES ('door_unlock_count', 0)"
    )
    connection.commit()
    connection.close()


def test_duplicate_key_returns_original_result_without_repeating_effect(tmp_path):
    db = tmp_path / "effects.sqlite3"
    _setup_effect_table(db)
    store = SQLiteIdempotentSideEffectStore(db)
    calls = {"count": 0}

    def effect(connection):
        calls["count"] += 1
        connection.execute("UPDATE world_state SET value = value + 1 WHERE name = 'door_unlock_count'")
        return {"success": True, "state": "unlocked"}

    first = store.execute_once(idempotency_key="exec-1:unlock", action_name="unlock_door", effect=effect)
    second = store.execute_once(idempotency_key="exec-1:unlock", action_name="unlock_door", effect=effect)

    assert first.replayed is False
    assert second.replayed is True
    assert first.result == second.result == {"success": True, "state": "unlocked"}
    assert calls["count"] == 1
    with sqlite3.connect(db) as connection:
        count = connection.execute("SELECT value FROM world_state WHERE name='door_unlock_count'").fetchone()[0]
    assert count == 1


def test_same_key_cannot_be_reused_for_different_action(tmp_path):
    store = SQLiteIdempotentSideEffectStore(tmp_path / "effects.sqlite3")
    store.execute_once(
        idempotency_key="execution-7:step-2",
        action_name="unlock_door",
        effect=lambda connection: {"success": True},
    )

    with pytest.raises(IdempotencyConflictError):
        store.execute_once(
            idempotency_key="execution-7:step-2",
            action_name="delete_file",
            effect=lambda connection: {"success": True},
        )


def test_distinct_keys_are_distinct_intended_effects(tmp_path):
    db = tmp_path / "effects.sqlite3"
    _setup_effect_table(db)
    store = SQLiteIdempotentSideEffectStore(db)

    for key in ("execution-1:unlock", "execution-2:unlock"):
        store.execute_once(
            idempotency_key=key,
            action_name="unlock_door",
            effect=lambda connection: (
                connection.execute("UPDATE world_state SET value=value+1 WHERE name='door_unlock_count'"),
                {"success": True},
            )[1],
        )

    with sqlite3.connect(db) as connection:
        count = connection.execute("SELECT value FROM world_state WHERE name='door_unlock_count'").fetchone()[0]
    assert count == 2


def test_failed_effect_rolls_back_both_state_change_and_idempotency_record(tmp_path):
    db = tmp_path / "effects.sqlite3"
    _setup_effect_table(db)
    store = SQLiteIdempotentSideEffectStore(db)

    def failing_effect(connection):
        connection.execute("UPDATE world_state SET value=value+1 WHERE name='door_unlock_count'")
        raise RuntimeError("simulated failure before commit")

    with pytest.raises(RuntimeError, match="simulated failure"):
        store.execute_once(idempotency_key="retry-me", action_name="unlock_door", effect=failing_effect)

    assert store.get_record("retry-me") is None
    with sqlite3.connect(db) as connection:
        count = connection.execute("SELECT value FROM world_state WHERE name='door_unlock_count'").fetchone()[0]
    assert count == 0

    result = store.execute_once(
        idempotency_key="retry-me",
        action_name="unlock_door",
        effect=lambda connection: (
            connection.execute("UPDATE world_state SET value=value+1 WHERE name='door_unlock_count'"),
            {"success": True},
        )[1],
    )
    assert result.replayed is False
    with sqlite3.connect(db) as connection:
        count = connection.execute("SELECT value FROM world_state WHERE name='door_unlock_count'").fetchone()[0]
    assert count == 1


def test_idempotency_record_survives_store_recreation(tmp_path):
    db = tmp_path / "effects.sqlite3"
    first_store = SQLiteIdempotentSideEffectStore(db)
    first_store.execute_once(
        idempotency_key="durable-key",
        action_name="send_notification",
        effect=lambda connection: {"message_id": "msg-123"},
    )

    second_store = SQLiteIdempotentSideEffectStore(db)
    result = second_store.execute_once(
        idempotency_key="durable-key",
        action_name="send_notification",
        effect=lambda connection: pytest.fail("effect must not execute on replay"),
    )
    assert result.replayed is True
    assert result.result == {"message_id": "msg-123"}


def test_rejects_empty_key_and_non_json_result(tmp_path):
    store = SQLiteIdempotentSideEffectStore(tmp_path / "effects.sqlite3")
    with pytest.raises(ValueError, match="idempotency_key"):
        store.execute_once(idempotency_key="  ", action_name="x", effect=lambda connection: {})
    with pytest.raises(TypeError, match="mapping"):
        store.execute_once(idempotency_key="bad-result", action_name="x", effect=lambda connection: "not-json")
    assert store.get_record("bad-result") is None
