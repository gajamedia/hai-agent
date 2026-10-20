from agent.world.room import RoomEnvironment
from agent.world.transactional_room import SQLiteRoomEnvironment
from api.execution_runtime import ExecutionRuntime


def make_runtime(tmp_path, database_name="room.sqlite3"):
    room = SQLiteRoomEnvironment(tmp_path / database_name)
    runtime = ExecutionRuntime(
        memory_path=tmp_path / "agent-memory.sqlite3",
        environment=room,
    )
    return runtime, room


def test_runtime_executes_action_using_transactional_room(tmp_path):
    runtime, room = make_runtime(tmp_path)

    result = runtime.run(
        actions=["open_box"],
        goal_description="Open the box",
        target_state={"box": "open"},
    )

    assert result["success"] is True
    assert result["goal_satisfied"] is True
    assert result["state_after"]["box"] == "open"
    assert room.observe()["box"] == "open"


def test_runtime_reuses_persisted_room_state_after_recreation(tmp_path):
    runtime1, room1 = make_runtime(tmp_path)

    first = runtime1.run(
        actions=["open_box"],
        goal_description="Open the box",
        target_state={"box": "open"},
    )

    assert first["success"] is True
    assert room1.observe()["box"] == "open"

    room2 = SQLiteRoomEnvironment(tmp_path / "room.sqlite3")
    runtime2 = ExecutionRuntime(
        memory_path=tmp_path / "agent-memory-restarted.sqlite3",
        environment=room2,
    )

    assert runtime2.get_state()["box"] == "open"


def test_runtime_reset_advances_room_epoch_and_resets_state(tmp_path):
    runtime, room = make_runtime(tmp_path)

    runtime.run(
        actions=["open_box"],
        goal_description="Open the box",
        target_state={"box": "open"},
    )

    previous_epoch = room.epoch
    state = runtime.reset()

    assert room.epoch == previous_epoch + 1
    assert state["box"] == "closed"
    assert runtime.get_state()["box"] == "closed"


def test_default_runtime_still_uses_in_memory_room(tmp_path):
    runtime = ExecutionRuntime(
        memory_path=tmp_path / "default-agent-memory.sqlite3",
    )

    assert isinstance(runtime.environment, RoomEnvironment)
