
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture(autouse=True)
def isolate_plan_run_idempotency_db(monkeypatch, tmp_path):
    """Keep each test's mutation gate isolated from the developer's durable DB."""
    database_path = tmp_path / "plan-run-idempotency.sqlite3"
    monkeypatch.setenv("HAI_PLAN_RUN_IDEMPOTENCY_DB", str(database_path))


def _client() -> TestClient:
    from tests.test_api_execution import FakeCognitiveRuntime

    return TestClient(create_app(runtime_factory=FakeCognitiveRuntime))


def test_plan_run_rejects_negative_max_replans(monkeypatch):
    from tests.test_api_execution import FakeCognitiveRuntime, make_plan

    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Unlock the door", ["open_box"]),
    )

    with _client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
                "adaptive_replanning": True,
                "max_replans": -1,
            },
        )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "invalid_adaptive_replanning_config"
    assert body["detail"]["execution_started"] is False


def test_plan_run_rejects_boolean_max_replans(monkeypatch):
    from tests.test_api_execution import FakeCognitiveRuntime, make_plan

    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Unlock the door", ["open_box"]),
    )

    with _client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
                "adaptive_replanning": True,
                "max_replans": True,
            },
        )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "invalid_adaptive_replanning_config"
    assert body["detail"]["execution_started"] is False


def test_plan_run_rejects_max_replans_above_limit(monkeypatch):
    from tests.test_api_execution import FakeCognitiveRuntime, make_plan

    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Unlock the door", ["open_box"]),
    )

    with _client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
                "adaptive_replanning": True,
                "max_replans": 4,
            },
        )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["error"] == "invalid_adaptive_replanning_config"
    assert body["detail"]["execution_started"] is False


def test_plan_run_routes_valid_adaptive_request_to_adaptive_helper(
    monkeypatch,
):
    from tests.test_api_execution import FakeCognitiveRuntime, make_plan
    import api.routes.execution as execution_route

    supplied_plan = make_plan("Unlock the door", ["open_box"])
    monkeypatch.setattr(FakeCognitiveRuntime, "plan", supplied_plan)

    planner = SimpleNamespace(plan=lambda *args, **kwargs: supplied_plan)
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "orchestrator",
        SimpleNamespace(planner=planner),
        raising=False,
    )

    calls = []

    def fake_execute_adaptive_plan(**kwargs):
        calls.append(kwargs)
        raise execution_route.HTTPException(
            status_code=422,
            detail={
                "error": "adaptive_test_probe",
                "execution_started": False,
            },
        )

    monkeypatch.setattr(
        execution_route,
        "execute_adaptive_plan",
        fake_execute_adaptive_plan,
    )

    with _client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
                "adaptive_replanning": True,
                "max_replans": 1,
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "adaptive_test_probe"
    assert len(calls) == 1
    assert calls[0]["max_replans"] == 1
    assert calls[0]["planner"] is planner


def test_plan_run_adaptive_replanning_executes_replacement_plan(
    monkeypatch,
):
    """Exercise the real adaptive helper and simulation runtime."""
    from agent.llm.planning import LLMPlanner
    from agent.llm.service import LLMService
    from tests.test_api_execution import FakeCognitiveRuntime, make_plan

    # The provider is deliberately inert. The real LLMPlanner is used,
    # but its plan method returns deterministic test plans.
    provider = SimpleNamespace(name="adaptive-test-provider")
    planner = LLMPlanner(LLMService(provider))

    replacement_plan = make_plan(
        "Open the box",
        ["open_box"],
    )
    planner.plan = lambda *args, **kwargs: replacement_plan

    initial_plan = make_plan(
        "Open the box",
        ["take_key"],
    )

    monkeypatch.setattr(FakeCognitiveRuntime, "plan", initial_plan)
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "orchestrator",
        SimpleNamespace(planner=planner),
        raising=False,
    )

    with _client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
                "adaptive_replanning": True,
                "max_replans": 1,
            },
        )

    body = response.json()

    assert response.status_code == 200, body
    print(body)
    assert body["success"] is True
    assert body["adaptive_replanning"]["enabled"] is True
    assert body["adaptive_replanning"]["replanned"] is True