
from __future__ import annotations

from dataclasses import dataclass, field

from fastapi.testclient import TestClient

from api.main import create_app


@dataclass
class FakeStrategy:
    name: str
    goal: str
    steps: list[str] = field(default_factory=list)
    status: str = "active"
    success_count: int = 0
    failure_count: int = 0
    total_actions: int = 0
    reflection_penalty: float = 0.0
    source_experiences: list[str] = field(default_factory=list)
    total_progress: float = 0.0
    progress_count: int = 0
    positive_progress_count: int = 0
    negative_progress_count: int = 0

    @property
    def attempts(self):
        return self.success_count + self.failure_count

    @property
    def success_rate(self):
        return self.success_count / self.attempts if self.attempts else 0.0

    @property
    def average_actions(self):
        return self.total_actions / self.attempts if self.attempts else 0.0

    @property
    def average_progress(self):
        return self.total_progress / self.progress_count if self.progress_count else 0.0

    @property
    def is_active(self):
        return self.status == "active"


@dataclass
class FakeEvidence:
    strategy_name: str
    source_strategy_name: str
    source_experience_id: str
    generation: int = 1
    status: str = "experimental"
    trial_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_progress: float = 0.0
    source_experience_ids: list[str] = field(default_factory=list)

    @property
    def success_rate(self):
        return self.success_count / self.trial_count if self.trial_count else 0.0

    @property
    def failure_rate(self):
        return self.failure_count / self.trial_count if self.trial_count else 0.0

    @property
    def average_progress(self):
        return self.total_progress / self.trial_count if self.trial_count else 0.0


class FakeStrategyStore:
    def __init__(self, strategies=None):
        self._strategies = list(strategies or [])

    def all(self):
        return list(self._strategies)


class FakeAgent:
    name = "test-agent"

    def __init__(self, strategies=None, active_strategy=None, evidence=None):
        self.strategy_store = FakeStrategyStore(strategies)
        self.active_strategy = active_strategy
        self._evidence = list(evidence or [])

    def get_all_strategy_evolution_evidence(self):
        return list(self._evidence)


class FakeRuntime:
    def __init__(self, agent):
        self.agent = agent


def test_strategy_endpoint_returns_registry_and_evolution_evidence():
    active = FakeStrategy(
        name="open_door",
        goal="open the door",
        steps=["find_key", "unlock_door"],
        success_count=3,
        failure_count=1,
        total_actions=8,
        source_experiences=["exp-1", "exp-2"],
        total_progress=2.0,
        progress_count=4,
        positive_progress_count=3,
        negative_progress_count=1,
    )
    retired = FakeStrategy(
        name="legacy_open",
        goal="open the door",
        steps=["force_door"],
        status="retired",
    )
    evidence = FakeEvidence(
        strategy_name="open_door_v2",
        source_strategy_name="open_door",
        source_experience_id="exp-9",
        generation=2,
        trial_count=4,
        success_count=3,
        failure_count=1,
        total_progress=2.0,
        source_experience_ids=["exp-9", "exp-10"],
    )
    app = create_app(
        runtime_factory=lambda: FakeRuntime(
            FakeAgent(
                strategies=[active, retired],
                active_strategy=active,
                evidence=[evidence],
            )
        )
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/strategies")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["source"] == "cognitive_runtime"
    assert body["agent_name"] == "test-agent"
    assert body["strategy_count"] == 2
    assert body["active_strategy_name"] == "open_door"

    strategies = {item["name"]: item for item in body["strategies"]}
    assert strategies["open_door"]["is_currently_selected"] is True
    assert strategies["open_door"]["success_rate"] == 0.75
    assert strategies["open_door"]["average_actions"] == 2.0
    assert strategies["legacy_open"]["status"] == "retired"
    assert strategies["legacy_open"]["is_active"] is False

    assert body["evolution_evidence_count"] == 1
    assert body["evolution_evidence"][0]["strategy_name"] == "open_door_v2"
    assert body["evolution_evidence"][0]["success_rate"] == 0.75
    assert body["evolution_evidence_persistence"] == "in_memory"


def test_strategy_endpoint_supports_empty_registry():
    app = create_app(runtime_factory=lambda: FakeRuntime(FakeAgent()))

    with TestClient(app) as client:
        response = client.get("/api/v1/strategies")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy_count"] == 0
    assert body["strategies"] == []
    assert body["active_strategy_name"] is None
    assert body["evolution_evidence"] == []


def test_strategy_endpoint_reports_missing_agent():
    app = create_app(runtime_factory=lambda: FakeRuntime(agent=None))

    with TestClient(app) as client:
        response = client.get("/api/v1/strategies")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "agent_unavailable"


def test_strategy_endpoint_reports_missing_registry():
    app = create_app(runtime_factory=lambda: FakeRuntime(agent=object()))

    with TestClient(app) as client:
        response = client.get("/api/v1/strategies")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "strategy_registry_unavailable"
