
from __future__ import annotations

from fastapi.testclient import TestClient

from agent.core.agent import Agent
from api.execution_runtime import create_execution_runtime
from api.main import create_app


class FakeCognitiveRuntime:
    """Cognitive runtime test fixture with a real strategy registry."""

    def __init__(self, memory_path):
        self.agent = Agent(
            "hai-test-cognitive-agent",
            memory_path=memory_path,
        )


def make_client(database_path) -> TestClient:
    """Create an API client using isolated runtime databases."""
    cognitive_database_path = (
        database_path.parent / "cognitive_memory.db"
    )

    return TestClient(
        create_app(
            runtime_factory=lambda: FakeCognitiveRuntime(
                cognitive_database_path,
            ),
            execution_runtime_factory=lambda: create_execution_runtime(
                memory_path=database_path,
            ),
        )
    )

def get_simulation_strategy(
    client: TestClient,
    *,
    goal: str,
    steps: list[str],
) -> dict:
    """Find a simulation strategy by its goal and ordered steps."""
    response = client.get("/api/v1/strategies")
    assert response.status_code == 200, response.text

    body = response.json()
    simulation = body.get("simulation")

    assert simulation is not None, body

    strategies = simulation.get("strategies", [])

    matches = [
        strategy
        for strategy in strategies
        if strategy.get("goal") == goal
        and strategy.get("steps") == steps
    ]

    assert len(matches) == 1, {
        "goal": goal,
        "steps": steps,
        "matches": matches,
        "all_strategies": strategies,
    }

    return matches[0]


def test_simulation_agent_persists_strategy_across_restart(
    tmp_path,
):
    database_path = tmp_path / "simulation_memory.db"

    first_agent = Agent(
        "hai-simulation-agent",
        memory_path=database_path,
    )

    strategy = first_agent.strategy_store.get_or_create(
        goal="Open the box",
        steps=["open_box"],
    )

    strategy.record_success(
        experience_id="simulation-exp-1",
        action_count=1,
        progress_delta=1.0,
    )
    first_agent.strategy_store.save(strategy)

    restarted_agent = Agent(
        "hai-simulation-agent",
        memory_path=database_path,
    )

    restored = restarted_agent.strategy_store.find_exact(
        goal="Open the box",
        steps=["open_box"],
    )

    assert restored is not None
    assert restored.success_count == 1
    assert restored.failure_count == 0
    assert "simulation-exp-1" in restored.source_experiences


def test_execution_api_creates_strategy_and_persists_across_restart(
    tmp_path,
):
    database_path = tmp_path / "execution_integration.db"

    # First runtime: execute a real request through the API.
    with make_client(database_path) as client:
        execution_response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

        assert execution_response.status_code == 200, (
            execution_response.text
        )

        execution = execution_response.json()

        assert execution["success"] is True
        assert execution["goal_satisfied"] is True
        assert execution["state_after"]["box"] == "open"

        # Verify that the execution lifecycle updated the strategy registry.
        strategy_before_restart = get_simulation_strategy(
            client,
            goal="Open the box",
            steps=["open_box"],
        )

        assert strategy_before_restart["success_count"] == 1
        assert strategy_before_restart["failure_count"] == 0
        assert strategy_before_restart["attempts"] == 1
        assert len(strategy_before_restart["source_experiences"]) == 1

        original_experience_ids = set(
            strategy_before_restart["source_experiences"]
        )

    # The TestClient context has closed its app runtime.
    # A new app/runtime is created with the same database path.
    with make_client(database_path) as restarted_client:
        strategy_after_restart = get_simulation_strategy(
            restarted_client,
            goal="Open the box",
            steps=["open_box"],
        )

        assert strategy_after_restart["success_count"] == 1
        assert strategy_after_restart["failure_count"] == 0
        assert strategy_after_restart["attempts"] == 1

        restored_experience_ids = set(
            strategy_after_restart["source_experiences"]
        )

        assert restored_experience_ids == original_experience_ids
