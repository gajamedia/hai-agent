from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import create_app
from api.dependencies import get_runtime


class FakeResult:
    def to_dict(self):
        return {
            "context": {"goal": "test goal"},
            "interpretation": {"summary": "test interpretation"},
            "reasoning": {"summary": "test reasoning"},
            "plan": {"steps": []},
            "metacognition": None,
        }


class FakeRuntime:
    last_trace = None

    def __init__(self):
        self.calls = []

    def run(self, observation, **kwargs):
        self.calls.append((observation, kwargs))
        return FakeResult()


def test_health_endpoint():
    runtime = FakeRuntime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_cognitive_endpoint_returns_pipeline_result():
    runtime = FakeRuntime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cognitive/run",
            json={
                "observation": {"room": "lab", "door": "closed"},
                "goal": "open the door",
                "metadata": {"request_id": "test-001"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["result"]["interpretation"]["summary"] == "test interpretation"
    assert body["result"]["plan"]["steps"] == []
    assert body["cycle"] is None
    assert body["trace"] is None

    observation, kwargs = runtime.calls[0]
    assert observation == {"room": "lab", "door": "closed"}
    assert kwargs["goal"] == "open the door"
    assert kwargs["metadata"] == {"request_id": "test-001"}


def test_observation_is_required():
    app = create_app(runtime_factory=FakeRuntime)

    with TestClient(app) as client:
        response = client.post("/api/v1/cognitive/run", json={"goal": "test"})

    assert response.status_code == 422


def test_unknown_request_fields_are_rejected():
    app = create_app(runtime_factory=FakeRuntime)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cognitive/run",
            json={"observation": "test", "execute_actions": True},
        )

    assert response.status_code == 422
