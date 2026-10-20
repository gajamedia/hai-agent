from __future__ import annotations

import pytest

from api.execution_runtime import ExecutionRuntime
from agent.world.transactional_room import (
    IdempotencyConflictError,
    SQLiteRoomEnvironment,
)
from agent.world.room import RoomEnvironment


def make_runtime(tmp_path, *, database_name="room.sqlite3"):
    room = SQLiteRoomEnvironment(tmp_path / database_name)
    runtime = ExecutionRuntime(
        memory_path=tmp_path / "agent-memory.sqlite3",
        environment=room,
    )
    return runtime, room


def run_open_box(runtime, *, key="request-001", target_state=None):
    return runtime.run(
        actions=["open_box"],
        goal_description="Open the box",
        target_state=target_state or {"box": "open"},
        idempotency_key=key,
    )


def test_runtime_replays_same_key_without_reapplying_action(tmp_path):
    runtime, room = make_runtime(tmp_path)

    first = run_open_box(runtime)
    second = run_open_box(runtime)

    assert first["success"] is True
    assert first["replayed"] is False
    assert second["replayed"] is True
    assert room.observe()["box"] == "open"


def test_runtime_rejects_same_key_with_different_request(tmp_path):
    runtime, room = make_runtime(tmp_path)

    first = run_open_box(runtime)
    assert first["success"] is True

    with pytest.raises(IdempotencyConflictError):
        run_open_box(
            runtime,
            target_state={"door": "unlocked"},
        )

    assert room.observe()["box"] == "open"
    assert room.observe()["door"] == "locked"


def test_runtime_replay_survives_runtime_recreation(tmp_path):
    runtime1, room1 = make_runtime(tmp_path)
    first = run_open_box(runtime1)

    assert first["success"] is True
    assert room1.observe()["box"] == "open"

    runtime2, room2 = make_runtime(tmp_path)
    replay = run_open_box(runtime2)

    assert replay["replayed"] is True
    assert room2.observe()["box"] == "open"


def test_same_key_can_be_used_again_after_room_reset(tmp_path):
    runtime, room = make_runtime(tmp_path)

    first = run_open_box(runtime)
    assert first["success"] is True
    assert first["replayed"] is False

    runtime.reset()
    assert room.observe()["box"] == "closed"

    second = run_open_box(runtime)

    assert second["success"] is True
    assert second["replayed"] is False
    assert room.observe()["box"] == "open"


def test_runtime_rejects_idempotency_key_for_multi_action_request(tmp_path):
    runtime, room = make_runtime(tmp_path)

    with pytest.raises(ValueError, match="single-action"):
        runtime.run(
            actions=["open_box", "take_key"],
            goal_description="Take the key",
            target_state={"key": "agent"},
            idempotency_key="multi-step-001",
        )

    assert room.observe()["box"] == "closed"
    assert room.observe()["key"] == "inside_box"


def test_default_runtime_still_uses_in_memory_room(tmp_path):
    runtime = ExecutionRuntime(memory_path=tmp_path / "agent-memory.sqlite3")

    assert isinstance(runtime.environment, RoomEnvironment)

