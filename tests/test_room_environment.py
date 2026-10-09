from agent.world.room import RoomEnvironment


def test_find_spare_key():

    environment = RoomEnvironment()

    result = environment.execute(
        "find_spare_key"
    )

    assert result["success"] is True

    state = environment.observe()

    assert state["spare_key"] == "found"
    assert state["key"] == "agent"


def test_find_spare_key_cannot_be_used_twice():

    environment = RoomEnvironment()

    first = environment.execute(
        "find_spare_key"
    )

    second = environment.execute(
        "find_spare_key"
    )

    assert first["success"] is True
    assert second["success"] is False


def test_spare_key_can_unlock_door():

    environment = RoomEnvironment()

    environment.execute(
        "find_spare_key"
    )

    result = environment.execute(
        "unlock_door"
    )

    assert result["success"] is True

    state = environment.observe()

    assert state["door"] == "unlocked"
