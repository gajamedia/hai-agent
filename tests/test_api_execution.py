
from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.llm.planning import LLMPlan, LLMPlanStep
from api.main import create_app


class FakeCognitiveRuntime:
    """Minimal cognitive runtime stub for API integration tests."""

    agent = object()
    plan: LLMPlan | None = None

    def run(
        self,
        observation,
        *,
        goal=None,
        state=None,
        metadata=None,
        **kwargs,
    ):
        if self.plan is None:
            raise RuntimeError("Test plan has not been configured.")

        return SimpleNamespace(plan=self.plan)


def make_client() -> TestClient:
    return TestClient(
        create_app(runtime_factory=FakeCognitiveRuntime)
    )


def make_plan(goal: str, actions: list[str]) -> LLMPlan:
    """Build a real LLMPlan using the project's planning contract."""

    steps = tuple(
        LLMPlanStep(
            index=index,
            action=action,
            purpose=f"Execute {action} toward the goal.",
            preconditions=(),
            expected_observation=f"Observe the result of {action}.",
            success_criteria=f"{action} completes successfully.",
            risk=None,
        )
        for index, action in enumerate(actions, start=1)
    )

    return LLMPlan(
        goal=goal,
        rationale="Integration test plan.",
        steps=steps,
        assumptions=(),
        constraints=(),
        stop_conditions=("Stop when an action fails.",),
        confidence=0.9,
        provider="test",
        model="test-model",
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


def test_plan_run_executes_cognitive_plan_and_verifies_goal(
    monkeypatch,
):
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan(
            "Unlock the door",
            ["open_box", "take_key", "unlock_door"],
        ),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": {
                    "description": "A locked door and a closed box."
                },
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
            },
        )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["status"] == "ok"
    assert body["mode"] == "cognitive_plan_execution"
    assert body["success"] is True

    assert body["plan"]["goal"] == "Unlock the door"
    assert body["plan"]["steps"][0]["action"] == "open_box"

    assert body["execution"]["success"] is True
    assert body["execution"]["planned_step_count"] == 3
    assert body["execution"]["executed_step_count"] == 3

    assert body["goal"]["satisfied"] is True
    assert body["state_before"]["door"] == "locked"
    assert body["state_after"]["box"] == "open"
    assert body["state_after"]["key"] == "agent"
    assert body["state_after"]["door"] == "unlocked"

    assert body["audit"]["source"] == "execution_plan_run"
    assert body["audit"]["execution_mode"] == "isolated_simulation"


def test_plan_run_rejects_unknown_action_before_execution(
    monkeypatch,
):
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan(
            "Unlock the door",
            ["run_arbitrary_code"],
        ),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "Try an invalid plan.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
            },
        )

        state_response = client.get(
            "/api/v1/execution/state"
        )

    assert response.status_code == 422, response.text

    # The invalid plan must be rejected before any simulation action runs.
    state = state_response.json()["state"]
    assert state["box"] == "closed"
    assert state["key"] == "inside_box"
    assert state["door"] == "locked"


def test_plan_run_stops_when_target_state_is_satisfied(monkeypatch):
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan(
            "Open the box",
            ["open_box", "take_key"],
        ),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["success"] is True
    assert body["goal"]["satisfied"] is True
    assert body["execution"]["success"] is True
    assert body["execution"]["completed"] is False
    assert body["execution"]["stopped"] is True
    assert body["execution"]["stop_reason"] == "goal_satisfied"
    assert body["execution"]["planned_step_count"] == 2
    assert body["execution"]["executed_step_count"] == 1
    assert [step["action"] for step in body["execution"]["steps"]] == [
        "open_box"
    ]
    assert body["state_after"]["box"] == "open"
    assert body["state_after"]["key"] == "inside_box"


def test_plan_run_stops_after_failed_action(monkeypatch):
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan(
            "Unlock the door",
            ["take_key", "open_box"],
        ),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
            },
        )

    assert response.status_code == 200, response.text

    body = response.json()

    # A failed action must not be reported as a successful overall run.
    assert body["success"] is False
    assert body["execution"]["success"] is False
    assert body["goal"]["satisfied"] is False

    # The second action should not run after the first action fails.
    assert body["execution"]["executed_step_count"] == 1
    assert body["execution"]["steps"][0]["action"] == "take_key"
    assert body["execution"]["steps"][0]["success"] is False

    assert body["state_after"]["door"] == "locked"


def test_plan_run_reports_unsatisfied_target_state(monkeypatch):
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan(
            "Unlock the door",
            ["open_box"],
        ),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
            },
        )

    assert response.status_code == 200, response.text

    body = response.json()

    # The action succeeds, but the requested final goal is not achieved.
    assert body["execution"]["success"] is True
    assert body["goal"]["satisfied"] is False
    assert body["success"] is False

    assert body["state_after"]["box"] == "open"
    assert body["state_after"]["door"] == "locked"



def test_plan_run_persists_execution_security_audit(monkeypatch, tmp_path):
    from agent.llm.execution_security import SQLiteExecutionSecurityAuditSink

    audit_path = tmp_path / "execution-security.sqlite3"
    monkeypatch.setenv("HAI_EXECUTION_SECURITY_AUDIT_DB", str(audit_path))
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Open the box", ["open_box"]),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["execution"]["success"] is True
    assert audit_path.exists()

    events = SQLiteExecutionSecurityAuditSink(audit_path).read_all()
    assert any(
        event.action == "open_box"
        and event.allowed is True
        and event.event_type == "authorization"
        for event in events
    )
    assert any(
        event.action == "open_box"
        and event.event_type == "execution_outcome"
        and event.execution_success is True
        for event in events
    )


def test_execution_run_persists_authorization_and_outcome_audit(monkeypatch, tmp_path):
    audit_path = tmp_path / "execution-audit.sqlite3"
    monkeypatch.setenv("HAI_EXECUTION_SECURITY_AUDIT_DB", str(audit_path))

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

    assert response.status_code == 200
    from agent.llm.execution_security import SQLiteExecutionSecurityAuditSink

    events = SQLiteExecutionSecurityAuditSink(audit_path).read_all()
    assert [event.event_type for event in events] == [
        "authorization",
        "execution_outcome",
    ]
    assert events[0].allowed is True
    assert events[1].execution_success is True


def test_execution_run_audit_failure_blocks_action(monkeypatch, tmp_path):
    audit_path = tmp_path / "missing-parent" / "audit.sqlite3"
    monkeypatch.setenv("HAI_EXECUTION_SECURITY_AUDIT_DB", str(audit_path))
    # Replace the app-scoped sink with a failing sink after app startup.
    with make_client() as client:
        app = client.app
        class BrokenSink:
            def record(self, event):
                raise OSError("audit unavailable")
        app.state.execution_security_audit_sink = BrokenSink()

        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )
        state_response = client.get("/api/v1/execution/state")

    assert response.status_code == 503
    assert response.json()["detail"]["retry_safe"] is False
    assert state_response.json()["state"]["box"] == "closed"


def test_execution_run_outcome_audit_failure_reports_ambiguous_result():
    with make_client() as client:
        class BrokenOutcomeSink:
            def record(self, event):
                if event.event_type == "execution_outcome":
                    raise OSError("outcome audit unavailable")

        client.app.state.execution_security_audit_sink = BrokenOutcomeSink()

        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box"],
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )
        state_response = client.get("/api/v1/execution/state")

    assert response.status_code == 503
    assert response.json()["detail"]["retry_safe"] is False
    assert state_response.status_code == 200
    assert state_response.json()["state"]["box"] == "open"


def test_execution_run_records_unknown_outcome_for_missing_step_evidence(
    monkeypatch,
):
    with make_client() as client:
        runtime = client.app.state.execution_runtime

        def partial_run(*, actions, goal_description, target_state):
            return {
                "success": False,
                "steps": [
                    {
                        "action": actions[0],
                        "success": True,
                        "message": "First action completed",
                    }
                ],
            }

        monkeypatch.setattr(runtime, "run", partial_run)

        class RecordingSink:
            def __init__(self):
                self.events = []

            def record(self, event):
                self.events.append(event)

        sink = RecordingSink()
        client.app.state.execution_security_audit_sink = sink

        response = client.post(
            "/api/v1/execution/run",
            json={
                "actions": ["open_box", "take_key"],
                "goal_description": "Open the box and take the key",
                "target_state": {"box": "open", "key": "agent"},
            },
        )

    assert response.status_code == 200

    outcomes = [
        event for event in sink.events
        if event.event_type == "execution_outcome"
    ]
    assert len(outcomes) == 2
    assert outcomes[0].action == "open_box"
    assert outcomes[0].execution_success is True
    assert outcomes[0].reason == "execution_run_result"
    assert outcomes[1].action == "take_key"
    assert outcomes[1].execution_success is None
    assert outcomes[1].reason == "execution_outcome_unknown"


def test_plan_run_rejects_unreachable_goal_without_changing_state(
    monkeypatch,
):
    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Unlock the door", ["unlock_door"]),
    )

    with make_client() as client:
        before_response = client.get("/api/v1/execution/state")
        assert before_response.status_code == 200
        state_before = before_response.json()["state"]

        # The default room state has a key inside the box, so this test
        # first changes the key state to make the target unreachable.
        state_update = client.put(
            "/api/v1/execution/state",
            json={
                "door": "locked",
                "box": "closed",
                "key": "missing",
                "spare_key": "unavailable",
                "agent_location": "room",
            },
        )

        if state_update.status_code not in (200, 204):
            # Do not silently pass if this API does not support state updates.
            raise AssertionError(
                "Could not set test world state: "
                f"{state_update.status_code} {state_update.text}"
            )

        state_before_run = client.get("/api/v1/execution/state").json()["state"]

        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The key is missing and no spare key is available.",
                "goal_description": "Unlock the door",
                "target_state": {"door": "unlocked"},
            },
        )

        state_after_run = client.get("/api/v1/execution/state").json()["state"]

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["error"] == "goal_unreachable"
    assert detail["goal_achieved"] is False
    assert detail["execution_started"] is False
    assert state_after_run == state_before_run

def test_plan_run_continues_when_feasibility_is_inconclusive(
    monkeypatch,
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        FakeCognitiveRuntime,
        "plan",
        make_plan("Open the box", ["open_box"]),
    )

    # Simulate an inconclusive assessment. This must not be treated
    # as proof that the goal is unreachable.
    monkeypatch.setattr(
        "api.routes.execution.assess_goal_feasibility",
        lambda **kwargs: SimpleNamespace(
            status="inconclusive",
            explored_states=1,
            action_sequence=(),
            reason="Test exploration limit reached.",
        ),
    )

    with make_client() as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json={
                "observation": "The box is closed.",
                "goal_description": "Open the box",
                "target_state": {"box": "open"},
            },
        )

    assert response.status_code == 200, response.text

    body = response.json()
    assert body["execution"]["success"] is True
    assert body["goal"]["satisfied"] is True
    assert body["state_after"]["box"] == "open"
