
from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app


class FakeCognitiveRuntime:
    """Minimal runtime stub for API lifecycle tests."""

    agent = object()


def make_client() -> TestClient:
    return TestClient(
        create_app(
            runtime_factory=FakeCognitiveRuntime,
        )
    )


def test_execution_state_returns_initial_room_state():
    with make_client() as client:
        response = client.get("/api/v1/execution/state")

    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["state"]["door"] == "locked"
    assert body["state"]["box"] == "closed"
    assert body["state"]["key"] == "inside_box"

    assert set(body["allowed_actions"]) == {
        "open_box",
        "take_key",
        "find_spare_key",
        "unlock_door",
    }


def test_execution_run_executes_single_action_and_records_telemetry():
    with make_client() as client:
        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

        telemetry_response = client.get(
            "/api/v1/telemetry/capabilities"
        )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["success"] is True
    assert body["goal_satisfied"] is True
    assert body["state_before"]["box"] == "closed"
    assert body["state_after"]["box"] == "open"

    assert telemetry_response.status_code == 200
    telemetry = telemetry_response.json()

    assert telemetry["source"] == "execution_runtime"
    assert telemetry["snapshot"]["event_count"] == 2
    assert telemetry["snapshot"]["capabilities"]["execution"] == {
        "attempts": 1,
        "successes": 1,
        "failures": 0,
    }
    assert telemetry["snapshot"]["capabilities"]["goal_achievement"] == {
        "attempts": 1,
        "successes": 1,
        "failures": 0,
    }


def test_execution_run_executes_ordered_plan():
    with make_client() as client:
        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": [
                    "open_box",
                    "take_key",
                    "unlock_door",
                ],
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
            },
        )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["success"] is True
    assert body["goal_satisfied"] is True
    assert body["state_after"]["box"] == "open"
    assert body["state_after"]["key"] == "agent"
    assert body["state_after"]["door"] == "unlocked"
    assert len(body["steps"]) == 3


def test_execution_run_rejects_unknown_action():
    with make_client() as client:
        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["run_arbitrary_code"],
                "goal_description": "Invalid action",
                "target_state": {"door": "unlocked"},
            },
        )

    assert response.status_code == 422


def test_execution_run_rejects_empty_target_state():
    with make_client() as client:
        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {},
            },
        )

    assert response.status_code == 422


def test_execution_reset_restores_state_and_retains_telemetry():
    with make_client() as client:
        execution_response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

        telemetry_before = client.get(
            "/api/v1/telemetry/capabilities"
        ).json()

        reset_response = client.post(
            "/api/v1/execution/reset"
        )

        state_response = client.get(
            "/api/v1/execution/state"
        )

        telemetry_after = client.get(
            "/api/v1/telemetry/capabilities"
        ).json()

    assert execution_response.status_code == 200
    assert reset_response.status_code == 200
    assert reset_response.json()["state"]["box"] == "closed"

    state = state_response.json()["state"]
    assert state["box"] == "closed"
    assert state["door"] == "locked"
    assert state["key"] == "inside_box"

    assert (
        telemetry_after["snapshot"]["event_count"]
        == telemetry_before["snapshot"]["event_count"]
    )
