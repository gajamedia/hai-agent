from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from api.main import create_app


@dataclass(frozen=True)
class FakeTelemetryEvent:
    event_id: int
    capability: str
    success: bool | None
    source: str


@dataclass(frozen=True)
class FakeTelemetrySnapshot:
    capabilities: dict[str, dict[str, int]]
    event_count: int
    total_attempts: int
    total_successes: int
    total_failures: int


class FakeAgent:
    def __init__(self, events=None, capabilities=None):
        self._events = list(events or [])
        self._capabilities = dict(capabilities or {})

    def get_capability_telemetry_snapshot(self):
        return FakeTelemetrySnapshot(
            capabilities=self._capabilities,
            event_count=len(self._events),
            total_attempts=sum(item["attempts"] for item in self._capabilities.values()),
            total_successes=sum(item["successes"] for item in self._capabilities.values()),
            total_failures=sum(item["failures"] for item in self._capabilities.values()),
        )

    def get_capability_telemetry_events(self):
        return list(self._events)


class FakeRuntime:
    def __init__(self, agent):
        self.agent = agent


def test_capability_telemetry_returns_snapshot_and_newest_events_first():
    events = [
        FakeTelemetryEvent(1, "planning", True, "runtime"),
        FakeTelemetryEvent(2, "execution", False, "decide_and_act"),
        FakeTelemetryEvent(3, "goal_achievement", None, "runtime"),
    ]
    agent = FakeAgent(
        events=events,
        capabilities={
            "planning": {"attempts": 1, "successes": 1, "failures": 0},
            "execution": {"attempts": 1, "successes": 0, "failures": 1},
            "goal_achievement": {"attempts": 1, "successes": 0, "failures": 0},
        },
    )
    app = create_app(runtime_factory=lambda: FakeRuntime(agent))

    with TestClient(app) as client:
        response = client.get("/api/v1/telemetry/capabilities?limit=2")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["snapshot"]["capability_count"] == 3
    assert body["snapshot"]["event_count"] == 3
    assert body["snapshot"]["total_attempts"] == 3
    assert body["snapshot"]["total_successes"] == 1
    assert body["snapshot"]["total_failures"] == 1
    assert [event["event_id"] for event in body["events"]] == [3, 2]
    assert body["events_returned"] == 2
    assert body["events_limit"] == 2


def test_capability_telemetry_supports_empty_snapshot():
    app = create_app(runtime_factory=lambda: FakeRuntime(FakeAgent()))

    with TestClient(app) as client:
        response = client.get("/api/v1/telemetry/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["snapshot"]["capabilities"] == {}
    assert body["snapshot"]["capability_count"] == 0
    assert body["snapshot"]["has_evidence"] is False
    assert body["events"] == []


def test_capability_telemetry_rejects_invalid_limit():
    app = create_app(runtime_factory=lambda: FakeRuntime(FakeAgent()))

    with TestClient(app) as client:
        response = client.get("/api/v1/telemetry/capabilities?limit=0")

    assert response.status_code == 422


def test_capability_telemetry_reports_missing_agent():
    app = create_app(runtime_factory=lambda: object())

    with TestClient(app) as client:
        response = client.get("/api/v1/telemetry/capabilities")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "agent_unavailable"
