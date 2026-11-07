from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent.world.environment_contracts import (
    EnvironmentActionRequest,
    EnvironmentActionResult,
    EnvironmentContractError,
    EnvironmentDescriptor,
    EnvironmentMode,
    EnvironmentObservation,
)


def test_descriptor_accepts_simulation_and_real_world_modes():
    simulation = EnvironmentDescriptor("room", "physical_space", EnvironmentMode.SIMULATION)
    real = EnvironmentDescriptor("robot", "physical_space", EnvironmentMode.REAL_WORLD)
    assert simulation.mode is EnvironmentMode.SIMULATION
    assert real.mode is EnvironmentMode.REAL_WORLD


def test_descriptor_rejects_untrusted_mode_string():
    with pytest.raises(EnvironmentContractError):
        EnvironmentDescriptor("room", "physical_space", "simulation")


def test_descriptor_metadata_is_defensively_copied_and_read_only():
    source = {"provider": {"name": "test"}}
    descriptor = EnvironmentDescriptor(
        "room", "physical_space", EnvironmentMode.SIMULATION, metadata=source
    )
    source["provider"]["name"] = "changed"
    assert descriptor.metadata["provider"]["name"] == "test"
    with pytest.raises(TypeError):
        descriptor.metadata["new"] = "value"
    with pytest.raises(TypeError):
        descriptor.metadata["provider"]["name"] = "mutated"


def test_observation_requires_identity_and_json_state():
    observation = EnvironmentObservation(
        "env-1", "instance-1", "r1", {"door": "locked"}
    )
    assert observation.state["door"] == "locked"
    with pytest.raises(EnvironmentContractError):
        EnvironmentObservation("env-1", "instance-1", "r1", {"bad": object()})


def test_action_request_validates_action_and_arguments():
    request = EnvironmentActionRequest(
        "door.unlock", {"force": False}, "idem-1", "r7"
    )
    assert request.action_id == "door.unlock"
    with pytest.raises(EnvironmentContractError):
        EnvironmentActionRequest(" ", {})


def test_action_result_unknown_outcome_cannot_claim_success():
    with pytest.raises(EnvironmentContractError):
        EnvironmentActionResult(
            True, "env-1", "instance-1", "r1", "r2",
            outcome_known=False,
        )


def test_action_result_supports_unknown_outcome_without_success_claim():
    result = EnvironmentActionResult(
        False, "env-1", "instance-1", "r1", "r1",
        {"reason": "timeout"}, error_code="outcome_unknown",
        outcome_known=False,
    )
    assert result.outcome_known is False
    assert result.success is False


def test_nested_non_finite_numbers_are_rejected():
    with pytest.raises(EnvironmentContractError):
        EnvironmentObservation(
            "env-1", "instance-1", "r1", {"measurement": float("nan")}
        )
