from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from agent.world.transactional_room import (
    IdempotencyConflictError,
    SQLiteRoomEnvironment,
)


def make_room(tmp_path, *, spare_key_available=True):
    return SQLiteRoomEnvironment(
        tmp_path / "room.sqlite3",
        spare_key_available=spare_key_available,
    )


def test_duplicate_key_replays_without_repeating_effect(tmp_path):
    room = make_room(tmp_path)
    first = room.execute_idempotent(idempotency_key="req-1", action="open_box")
    second = room.execute_idempotent(idempotency_key="req-1", action="open_box")
    assert first["success"] is True and first["replayed"] is False
    assert second["success"] is True and second["replayed"] is True
    assert room.observe()["box"] == "open"


def test_same_key_different_action_conflicts(tmp_path):
    room = make_room(tmp_path)
    room.execute_idempotent(idempotency_key="req-1", action="open_box")
    with pytest.raises(IdempotencyConflictError):
        room.execute_idempotent(idempotency_key="req-1", action="take_key")


def test_same_key_different_fingerprint_conflicts(tmp_path):
    room = make_room(tmp_path)
    room.execute_idempotent(
        idempotency_key="req-1", action="open_box", request_fingerprint="payload-a"
    )
    with pytest.raises(IdempotencyConflictError):
        room.execute_idempotent(
            idempotency_key="req-1", action="open_box", request_fingerprint="payload-b"
        )


def test_distinct_keys_are_distinct_requests(tmp_path):
    room = make_room(tmp_path)
    first = room.execute_idempotent(idempotency_key="req-1", action="open_box")
    second = room.execute_idempotent(idempotency_key="req-2", action="open_box")
    assert first["replayed"] is False
    assert second["replayed"] is False
    assert second["success"] is False  # box was already open


def test_state_and_ledger_roll_back_together(tmp_path):
    room = make_room(tmp_path)
    with sqlite3.connect(room.database_path) as con:
        con.execute("""
            CREATE TRIGGER fail_ledger_insert
            BEFORE INSERT ON room_idempotency
            BEGIN SELECT RAISE(ABORT, 'injected ledger failure'); END
        """)
    with pytest.raises(sqlite3.IntegrityError):
        room.execute_idempotent(idempotency_key="req-1", action="open_box")
    assert room.observe()["box"] == "closed"
    with sqlite3.connect(room.database_path) as con:
        con.execute("DROP TRIGGER fail_ledger_insert")
    result = room.execute_idempotent(idempotency_key="req-1", action="open_box")
    assert result["success"] is True
    assert result["replayed"] is False


def test_replay_survives_environment_recreation(tmp_path):
    path = tmp_path / "room.sqlite3"
    room1 = SQLiteRoomEnvironment(path)
    room1.execute_idempotent(idempotency_key="req-1", action="open_box")
    room2 = SQLiteRoomEnvironment(path)
    result = room2.execute_idempotent(idempotency_key="req-1", action="open_box")
    assert result["replayed"] is True
    assert room2.observe()["box"] == "open"


def test_concurrent_duplicate_requests_apply_once(tmp_path):
    room = make_room(tmp_path)
    def run_once(_):
        return room.execute_idempotent(idempotency_key="same-key", action="open_box")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_once, range(16)))
    assert sum(not result["replayed"] for result in results) == 1
    assert sum(result["replayed"] for result in results) == 15
    assert room.observe()["box"] == "open"


def test_failed_action_result_is_replayed_within_epoch(tmp_path):
    room = make_room(tmp_path)
    first = room.execute_idempotent(idempotency_key="req-1", action="take_key")
    second = room.execute_idempotent(idempotency_key="req-1", action="take_key")
    assert first["success"] is False
    assert first["replayed"] is False
    assert second["replayed"] is True


def test_reset_advances_epoch_and_old_key_does_not_replay(tmp_path):
    room = make_room(tmp_path)
    first = room.execute_idempotent(idempotency_key="req-1", action="open_box")
    old_epoch = first["epoch"]
    room.reset()
    assert room.epoch == old_epoch + 1
    assert room.observe()["box"] == "closed"
    second = room.execute_idempotent(idempotency_key="req-1", action="open_box")
    assert second["success"] is True
    assert second["replayed"] is False
    assert second["epoch"] == old_epoch + 1
    assert room.observe()["box"] == "open"


def test_empty_fingerprint_is_rejected(tmp_path):
    room = make_room(tmp_path)
    with pytest.raises(ValueError, match="request_fingerprint"):
        room.execute_idempotent(
            idempotency_key="req-1", action="open_box", request_fingerprint=""
        )


def test_initial_configuration_does_not_overwrite_existing_database(tmp_path):
    path = tmp_path / "room.sqlite3"
    first = SQLiteRoomEnvironment(path, spare_key_available=False)
    assert first.observe()["spare_key"] == "unavailable"
    second = SQLiteRoomEnvironment(path, spare_key_available=True)
    assert second.observe()["spare_key"] == "unavailable"
