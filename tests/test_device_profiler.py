"""Tests for the dependency-free device profiler."""

import json
from pathlib import Path

from agent.device import DeviceProfile, DeviceProfiler


def test_profile_returns_expected_contract(tmp_path):
    profile = DeviceProfiler().profile(target_path=tmp_path)

    assert isinstance(profile, DeviceProfile)
    assert profile.captured_at
    assert profile.operating_system
    assert profile.machine
    assert profile.cpu.logical_processors is None or profile.cpu.logical_processors >= 1
    assert profile.memory.total_bytes is None or profile.memory.total_bytes > 0
    assert profile.memory.available_bytes is None or profile.memory.available_bytes >= 0
    assert profile.disk.requested_path == str(tmp_path)
    assert profile.disk.measured_path is not None
    assert profile.disk.total_bytes is not None
    assert profile.disk.free_bytes is not None
    assert profile.python.version
    assert profile.python.executable


def test_profile_is_json_serializable(tmp_path):
    profile = DeviceProfiler().profile(target_path=tmp_path)
    encoded = json.dumps(profile.to_dict())
    decoded = json.loads(encoded)

    assert decoded["operating_system"] == profile.operating_system
    assert decoded["cpu"]["logical_processors"] == profile.cpu.logical_processors


def test_docker_can_be_omitted(tmp_path):
    profile = DeviceProfiler().profile(target_path=tmp_path, include_docker=False)

    assert profile.docker is None


def test_nonexistent_target_uses_existing_parent(tmp_path):
    missing = tmp_path / "not-created" / "nested"
    profile = DeviceProfiler().profile(target_path=missing)

    assert profile.disk.requested_path == str(missing)
    assert profile.disk.measured_path is not None
    assert Path(profile.disk.measured_path).exists()


def test_memory_failure_does_not_abort_profile(tmp_path, monkeypatch):
    def fail():
        raise OSError("simulated memory query failure")

    monkeypatch.setattr("agent.device.device_profiler.platform.system", lambda: "Windows")
    monkeypatch.setattr(DeviceProfiler, "_windows_memory", staticmethod(fail))

    profile = DeviceProfiler().profile(target_path=tmp_path)

    assert profile.operating_system
    assert profile.memory.total_bytes is None
    assert any("memory profiling unavailable" in warning for warning in profile.warnings)
