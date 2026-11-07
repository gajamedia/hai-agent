from pathlib import Path

import pytest

from agent.autonomous_testing.contracts import ROOM_STATE_SCHEMA
from agent.autonomous_testing.environment import (
    IsolatedEnvironmentError,
    IsolatedTestEnvironment,
)

INITIAL_A = {
    "door": "locked",
    "box": "closed",
    "key": "inside_box",
    "spare_key": "available",
    "agent_location": "room",
}
INITIAL_B = {
    "door": "locked",
    "box": "open",
    "key": "agent",
    "spare_key": "unavailable",
    "agent_location": "room",
}


class FakeRuntime:
    def __init__(self, memory_path):
        self.memory_path = Path(memory_path)
        self.state = None
        self.closed = False

    def set_state(self, state):
        self.state = dict(state)
        return dict(self.state)

    def get_state(self):
        return dict(self.state)

    def run(self, **kwargs):
        # Test that the environment delegates instead of dispatching actions.
        return {"delegated": True, "kwargs": kwargs}

    def close(self):
        self.closed = True


def test_creates_private_runtime_and_memory_path_per_environment(tmp_path):
    runtimes = []

    def factory(*, memory_path):
        runtime = FakeRuntime(memory_path)
        runtimes.append(runtime)
        return runtime

    a = IsolatedTestEnvironment(
        initial_state=INITIAL_A, runtime_factory=factory, temp_root=tmp_path
    )
    b = IsolatedTestEnvironment(
        initial_state=INITIAL_B, runtime_factory=factory, temp_root=tmp_path
    )
    a_dir = a.storage_directory
    b_dir = b.storage_directory
    try:
        assert a.runtime is not b.runtime
        assert a.memory_path != b.memory_path
        assert a.storage_directory != b.storage_directory
        assert a.observe() == INITIAL_A
        assert b.observe() == INITIAL_B
        assert a.memory_path.parent.exists()
        assert b.memory_path.parent.exists()
    finally:
        a.close()
        b.close()

    assert runtimes[0].closed is True
    assert runtimes[1].closed is True
    assert not a_dir.exists()
    assert not b_dir.exists()


def test_run_delegates_to_runtime_run_without_direct_action_dispatch(tmp_path):
    env = IsolatedTestEnvironment(
        initial_state=INITIAL_A,
        runtime_factory=lambda *, memory_path: FakeRuntime(memory_path),
        temp_root=tmp_path,
    )
    try:
        result = env.run(
            actions=["open_box"],
            goal_description="Open box",
            target_state={"box": "open"},
        )
        assert result["delegated"] is True
        assert result["kwargs"]["actions"] == ["open_box"]
    finally:
        env.close()


@pytest.mark.parametrize("bad_state", [
    {},
    {**INITIAL_A, "unknown": "value"},
    {**INITIAL_A, "door": "ajar"},
    {**INITIAL_A, "box": 1},
])
def test_rejects_invalid_initial_state_before_factory_call(tmp_path, bad_state):
    called = False

    def factory(*, memory_path):
        nonlocal called
        called = True
        return FakeRuntime(memory_path)

    with pytest.raises(ValueError):
        IsolatedTestEnvironment(
            initial_state=bad_state, runtime_factory=factory, temp_root=tmp_path
        )
    assert called is False


def test_factory_failure_cleans_temporary_directory(tmp_path):
    captured = {}

    def broken_factory(*, memory_path):
        captured["directory"] = Path(memory_path).parent
        raise RuntimeError("factory failure")

    with pytest.raises(IsolatedEnvironmentError):
        IsolatedTestEnvironment(
            initial_state=INITIAL_A, runtime_factory=broken_factory, temp_root=tmp_path
        )
    assert not captured["directory"].exists()


def test_state_initialization_mismatch_fails_closed_and_cleans(tmp_path):
    captured = {}

    class BadRuntime(FakeRuntime):
        def set_state(self, state):
            captured["directory"] = self.memory_path.parent
            self.state = dict(state)
            self.state["door"] = "unlocked"
            return dict(self.state)

    with pytest.raises(IsolatedEnvironmentError):
        IsolatedTestEnvironment(
            initial_state=INITIAL_A,
            runtime_factory=lambda *, memory_path: BadRuntime(memory_path),
            temp_root=tmp_path,
        )
    assert not captured["directory"].exists()


def test_sequential_only_is_required(tmp_path):
    from agent.autonomous_testing.contracts import TestEnvironmentConfig

    config = TestEnvironmentConfig(sequential_only=False)
    with pytest.raises(IsolatedEnvironmentError):
        IsolatedTestEnvironment(
            initial_state=INITIAL_A,
            config=config,
            runtime_factory=lambda *, memory_path: FakeRuntime(memory_path),
            temp_root=tmp_path,
        )


def test_close_is_idempotent_and_runtime_cannot_be_reused(tmp_path):
    env = IsolatedTestEnvironment(
        initial_state=INITIAL_A,
        runtime_factory=lambda *, memory_path: FakeRuntime(memory_path),
        temp_root=tmp_path,
    )
    directory = env.storage_directory
    env.close()
    env.close()
    assert env.is_closed
    assert not directory.exists()
    with pytest.raises(IsolatedEnvironmentError):
        env.observe()


def test_schema_matches_contract():
    assert set(INITIAL_A) == set(ROOM_STATE_SCHEMA)
