from __future__ import annotations

import pytest

from agent.autonomous_testing import environment as environment_module
from agent.autonomous_testing.environment import (
    IsolatedEnvironmentError,
    IsolatedTestEnvironment,
)
from agent.autonomous_testing.contracts import ROOM_STATE_SCHEMA


def _initial_state():
    return {key: next(iter(values)) for key, values in ROOM_STATE_SCHEMA.items()}


class FakeRuntime:
    def __init__(self, initial_state, *, close_error=None, set_state_error=None):
        self.initial_state = dict(initial_state)
        self.close_error = close_error
        self.set_state_error = set_state_error
        self.close_calls = 0

    def set_state(self, state):
        if self.set_state_error is not None:
            raise self.set_state_error
        self.initial_state = dict(state)
        return dict(state)

    def get_state(self):
        return dict(self.initial_state)

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _make_environment(tmp_path, runtime):
    return IsolatedTestEnvironment(
        initial_state=_initial_state(),
        runtime_factory=lambda **_: runtime,
        temp_root=tmp_path,
    )


def test_runtime_close_failure_does_not_skip_private_directory_cleanup(tmp_path):
    runtime = FakeRuntime(_initial_state(), close_error=OSError("runtime close failed"))
    environment = _make_environment(tmp_path, runtime)
    private_dir = environment.storage_directory

    with pytest.raises(IsolatedEnvironmentError, match="runtime close failed"):
        environment.close()

    assert runtime.close_calls == 1
    assert not private_dir.exists()
    assert environment.is_closed is True
    environment.close()
    assert runtime.close_calls == 1
    assert list(tmp_path.iterdir()) == []


def test_temp_directory_cleanup_failure_is_retryable_without_reclosing_runtime(
    tmp_path, monkeypatch
):
    runtime = FakeRuntime(_initial_state())
    environment = _make_environment(tmp_path, runtime)
    private_dir = environment.storage_directory
    real_rmtree = environment_module.shutil.rmtree
    calls = 0

    def fail_once(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("temporary directory is busy")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(environment_module.shutil, "rmtree", fail_once)

    with pytest.raises(IsolatedEnvironmentError, match="temporary directory is busy"):
        environment.close()

    assert runtime.close_calls == 1
    assert private_dir.exists()
    assert environment.is_closed is False

    environment.close()

    assert runtime.close_calls == 1
    assert not private_dir.exists()
    assert environment.is_closed is True
    assert calls == 2


def test_runtime_and_directory_cleanup_failures_are_both_reported_and_retryable(
    tmp_path, monkeypatch
):
    runtime = FakeRuntime(_initial_state(), close_error=OSError("runtime close failed"))
    environment = _make_environment(tmp_path, runtime)
    private_dir = environment.storage_directory
    real_rmtree = environment_module.shutil.rmtree
    calls = 0

    def fail_once(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("directory cleanup failed")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(environment_module.shutil, "rmtree", fail_once)

    with pytest.raises(IsolatedEnvironmentError) as captured:
        environment.close()

    message = str(captured.value)
    assert "runtime close failed" in message
    assert "directory cleanup failed" in message
    assert runtime.close_calls == 1
    assert private_dir.exists()
    assert environment.is_closed is False

    environment.close()

    assert runtime.close_calls == 1
    assert not private_dir.exists()
    assert environment.is_closed is True


def test_partial_construction_attempts_directory_cleanup_after_runtime_close_failure(
    tmp_path, monkeypatch
):
    runtime = FakeRuntime(
        _initial_state(),
        close_error=OSError("partial runtime close failed"),
        set_state_error=ValueError("initial state rejected"),
    )
    real_rmtree = environment_module.shutil.rmtree
    removed = []

    def record_rmtree(path, *args, **kwargs):
        removed.append(path)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(environment_module.shutil, "rmtree", record_rmtree)

    with pytest.raises(IsolatedEnvironmentError) as captured:
        IsolatedTestEnvironment(
            initial_state=_initial_state(),
            runtime_factory=lambda **_: runtime,
            temp_root=tmp_path,
        )

    assert "initial state rejected" in str(captured.value)
    assert "partial runtime close failed" in str(captured.value)
    assert runtime.close_calls == 1
    assert len(removed) == 1
    assert list(tmp_path.iterdir()) == []


def test_successful_close_is_idempotent_and_preserves_temp_root(tmp_path):
    runtime = FakeRuntime(_initial_state())
    environment = _make_environment(tmp_path, runtime)
    private_dir = environment.storage_directory

    environment.close()
    environment.close()

    assert runtime.close_calls == 1
    assert not private_dir.exists()
    assert tmp_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_agent_close_fallback_is_used_when_runtime_has_no_close(tmp_path):
    events = []

    class Agent:
        def close(self):
            events.append("agent-close")

    class RuntimeWithoutClose:
        def __init__(self):
            self.agent = Agent()

        def set_state(self, state):
            return dict(state)

    runtime = RuntimeWithoutClose()
    environment = _make_environment(tmp_path, runtime)
    environment.close()

    assert events == ["agent-close"]
    assert environment.is_closed is True
