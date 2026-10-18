from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from api.main import create_app


@dataclass(frozen=True)
class FakeEvent:
    event_id: int
    capability: str
    success: bool | None
    source: str


class FakeAgent:
    def __init__(self, name: str, event_count: int):
        self.name = name
        self.events = [
            FakeEvent(i + 1, name, True, name)
            for i in range(event_count)
        ]

    def get_capability_telemetry_snapshot(self):
        return {
            "capabilities": (
                {self.name: {
                    "attempts": len(self.events),
                    "successes": len(self.events),
                    "failures": 0,
                }}
                if self.events else {}
            ),
            "event_count": len(self.events),
            "total_attempts": len(self.events),
            "total_successes": len(self.events),
            "total_failures": 0,
        }

    def get_capability_telemetry_events(self):
        return list(self.events)


class FakeRuntime:
    def __init__(self, agent):
        self.agent = agent


def build_client(cognitive_count=1, execution_count=2, *,
                 cognitive_available=True, execution_available=True):
    cognitive_factory = (
        (lambda: FakeRuntime(FakeAgent("cognitive_reasoning", cognitive_count)))
        if cognitive_available else (lambda: None)
    )
    execution_factory = (
        (lambda: FakeRuntime(FakeAgent("simulation_execution", execution_count)))
        if execution_available else (lambda: None)
    )
    # The execution factory is supported by the Step 42.2.4 runtime wiring.
    app = create_app(
        runtime_factory=cognitive_factory,
        execution_runtime_factory=execution_factory,
    )
    return TestClient(app)


def test_sources_endpoint_returns_cognitive_and_execution_telemetry_separately():
    with build_client(cognitive_count=1, execution_count=3) as client:
        response = client.get("/api/v1/telemetry/capabilities/sources?limit=20")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    sources = body["sources"]
    assert sources["cognitive_runtime"]["status"] == "ok"
    assert sources["execution_runtime"]["status"] == "ok"
    assert sources["cognitive_runtime"]["source"] == "cognitive_runtime"
    assert sources["execution_runtime"]["source"] == "execution_runtime"
    assert sources["cognitive_runtime"]["snapshot"]["event_count"] == 1
    assert sources["execution_runtime"]["snapshot"]["event_count"] == 3
    assert sources["cognitive_runtime"]["events"][0]["capability"] == "cognitive_reasoning"
    assert sources["execution_runtime"]["events"][0]["capability"] == "simulation_execution"


def test_explicit_cognitive_source_is_not_replaced_by_execution_evidence():
    with build_client(cognitive_count=0, execution_count=5) as client:
        response = client.get(
            "/api/v1/telemetry/capabilities?source=cognitive_runtime"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "cognitive_runtime"
    assert body["snapshot"]["event_count"] == 0
    assert body["events"] == []


def test_explicit_execution_source_selects_simulation_telemetry():
    with build_client(cognitive_count=1, execution_count=4) as client:
        response = client.get(
            "/api/v1/telemetry/capabilities?source=execution_runtime"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "execution_runtime"
    assert body["snapshot"]["event_count"] == 4
    assert all(event["capability"] == "simulation_execution" for event in body["events"])


def test_sources_endpoint_reports_missing_runtime_without_hiding_other_source():
    with build_client(
        cognitive_count=2,
        execution_count=0,
        cognitive_available=True,
        execution_available=False,
    ) as client:
        response = client.get("/api/v1/telemetry/capabilities/sources")

    assert response.status_code == 200
    sources = response.json()["sources"]
    assert sources["cognitive_runtime"]["status"] == "ok"
    assert sources["cognitive_runtime"]["snapshot"]["event_count"] == 2
    assert sources["execution_runtime"]["status"] == "unavailable"
    assert sources["execution_runtime"]["error"] == "runtime_unavailable"


def test_sources_endpoint_rejects_invalid_limit():
    with build_client() as client:
        response = client.get("/api/v1/telemetry/capabilities/sources?limit=0")
    assert response.status_code == 422


def test_source_endpoint_limit_does_not_change_snapshot_event_count():
    with build_client(cognitive_count=5, execution_count=1) as client:
        response = client.get(
            "/api/v1/telemetry/capabilities?source=cognitive_runtime&limit=2"
        )

    assert response.status_code == 200
    body = response.json()
    assert body["snapshot"]["event_count"] == 5
    assert body["events_returned"] == 2
    assert body["events_limit"] == 2
