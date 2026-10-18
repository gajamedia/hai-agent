from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import create_app


class FakeCognitiveResult:
    def to_dict(self):
        return {
            "context": {"goal": "test goal"},
            "interpretation": {"summary": "test interpretation"},
            "reasoning": {"summary": "test reasoning"},
            "plan": {"steps": [{"action": "open_box"}]},
            "metacognition": None,
        }


class FakeCognitiveRuntime:
    last_trace = None

    def __init__(self, *, fail: bool = False):
        self.calls = []
        self.fail = fail

    def run(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        if self.fail:
            raise RuntimeError("test runtime failure")
        return FakeCognitiveResult()


class FakeExecutionRuntime:
    def __init__(self):
        self.calls = []

    def get_state(self):
        return {"box": "closed", "door": "locked", "key": "inside_box"}

    @property
    def allowed_actions(self):
        return {"open_box"}

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"success": True}

    def reset(self):
        return self.get_state()


def make_client(cognitive=None, execution=None):
    cognitive = cognitive or FakeCognitiveRuntime()
    execution = execution or FakeExecutionRuntime()
    app = create_app(
        runtime_factory=lambda: cognitive,
        execution_runtime_factory=lambda: execution,
    )
    return TestClient(app), cognitive, execution


def test_cognitive_contract_returns_result_and_does_not_execute_plan():
    client, cognitive, execution = make_client()
    with client:
        response = client.post(
            "/api/v1/cognitive/run",
            json={
                "observation": {"room": "room_a"},
                "goal": "Open the box",
                "include_trace": True,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"]["plan"]["steps"] == [{"action": "open_box"}]
    assert body["trace"] is None
    assert len(cognitive.calls) == 1
    assert execution.calls == []


def test_cognitive_contract_rejects_unknown_fields():
    client, _, execution = make_client()
    with client:
        response = client.post(
            "/api/v1/cognitive/run",
            json={
                "observation": {"room": "room_a"},
                "execute_actions": True,
            },
        )

    assert response.status_code == 422
    assert execution.calls == []


def test_cognitive_metacognition_requires_episode_and_reflection():
    client, cognitive, execution = make_client()
    with client:
        response = client.post(
            "/api/v1/cognitive/run",
            json={
                "observation": {"room": "room_a"},
                "include_metacognition": True,
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "episode_required"
    assert cognitive.calls == []
    assert execution.calls == []


def test_unexpected_cognitive_runtime_failure_maps_to_500():
    cognitive = FakeCognitiveRuntime(fail=True)
    client, _, execution = make_client(cognitive=cognitive)
    with client:
        response = client.post(
            "/api/v1/cognitive/run",
            json={"observation": {"room": "room_a"}},
        )

    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "cognitive_runtime_failed"
    assert execution.calls == []


def test_execution_plan_run_rejects_missing_required_fields():
    client, cognitive, execution = make_client()
    with client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={"observation": {"room": "room_a"}},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "goal_description_required"
    assert cognitive.calls == []
    assert execution.calls == []


def test_execution_run_rejects_empty_actions_before_runtime_call():
    client, _, execution = make_client()
    with client:
        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": [],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

    assert response.status_code == 422
    assert execution.calls == []


def test_execution_state_reports_unavailable_runtime():
    client, _, _ = make_client()
    with client:
        client.app.state.execution_runtime = None
        response = client.get("/api/v1/execution/state")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "execution_runtime_unavailable"


def test_health_endpoint_is_independent_of_execution_calls():
    client, _, execution = make_client()
    with client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "hai-agent-api",
    }
    assert execution.calls == []
