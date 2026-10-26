from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.llm.planning import LLMPlan, LLMPlanStep
from api.main import create_app


class FakeCognitiveRuntime:
    agent = object()
    plan: LLMPlan | None = None

    def run(self, observation, *, goal=None, state=None, metadata=None, **kwargs):
        if self.plan is None:
            raise RuntimeError("Test plan has not been configured.")
        return SimpleNamespace(plan=self.plan)


def make_client() -> TestClient:
    return TestClient(create_app(runtime_factory=FakeCognitiveRuntime))


def make_plan(goal: str, actions: list[str]) -> LLMPlan:
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
        rationale="Idempotency API integration test.",
        steps=steps,
        assumptions=(),
        constraints=(),
        stop_conditions=("Stop when an action fails.",),
        confidence=0.9,
        provider="test",
        model="test-model",
    )


def request_body(goal: str = "Open the box") -> dict:
    return {
        "observation": {"description": "A closed box."},
        "goal_description": goal,
        "target_state": {"box": "open"},
        "actions": ["open_box"],
    }


def test_plan_run_idempotency_replays_completed_response_without_reexecution(monkeypatch, tmp_path):
    monkeypatch.setenv("HAI_PLAN_RUN_IDEMPOTENCY_DB", str(tmp_path / "plan-run.sqlite3"))
    monkeypatch.setattr(FakeCognitiveRuntime, "plan", None)
    body = request_body()
    headers = {"Idempotency-Key": "open-box-request-001"}

    with make_client() as client:
        first = client.post("/api/v1/execution/plan-run", json=body, headers=headers)
        state_after_first = client.get("/api/v1/execution/state").json()["state"]
        second = client.post("/api/v1/execution/plan-run", json=body, headers=headers)
        state_after_second = client.get("/api/v1/execution/state").json()["state"]

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["idempotency"] == {"key": headers["Idempotency-Key"], "replayed": False}
    assert second.json()["idempotency"] == {"key": headers["Idempotency-Key"], "replayed": True}
    assert first.json()["state_after"] == second.json()["state_after"]
    assert state_after_first == state_after_second
    assert state_after_second["box"] == "open"


def test_plan_run_idempotency_rejects_same_key_for_different_request(monkeypatch, tmp_path):
    monkeypatch.setenv("HAI_PLAN_RUN_IDEMPOTENCY_DB", str(tmp_path / "plan-run.sqlite3"))
    headers = {"Idempotency-Key": "same-key-different-request"}

    with make_client() as client:
        first = client.post("/api/v1/execution/plan-run", json=request_body(), headers=headers)
        second_body = request_body("Unlock the door")
        second = client.post("/api/v1/execution/plan-run", json=second_body, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 409, second.text
    assert second.json()["detail"]["error"] == "idempotency_key_conflict"


def test_plan_run_idempotency_replays_after_app_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("HAI_PLAN_RUN_IDEMPOTENCY_DB", str(tmp_path / "plan-run.sqlite3"))
    body = request_body()
    headers = {"Idempotency-Key": "persisted-key-001"}

    with make_client() as first_client:
        first = first_client.post("/api/v1/execution/plan-run", json=body, headers=headers)
    with make_client() as second_client:
        replay = second_client.post("/api/v1/execution/plan-run", json=body, headers=headers)
        fresh_app_state = second_client.get("/api/v1/execution/state").json()["state"]

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotency"]["replayed"] is True
    assert replay.json()["state_after"] == first.json()["state_after"]
    # The new app has a fresh simulation; the replay did not execute it again.
    assert fresh_app_state["box"] == "closed"


def test_plan_run_idempotency_requires_reconciliation_after_unknown_result(monkeypatch, tmp_path):
    from api.plan_run_idempotency import PlanRunIdempotencyStore

    monkeypatch.setenv("HAI_PLAN_RUN_IDEMPOTENCY_DB", str(tmp_path / "plan-run.sqlite3"))
    body = request_body()
    key = "interrupted-request-001"
    fingerprint = __import__("hashlib").sha256(
        __import__("json").dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    ).hexdigest()
    store = PlanRunIdempotencyStore(tmp_path / "plan-run.sqlite3")
    store.claim(idempotency_key=key, request_fingerprint=fingerprint)
    store.mark_unknown(idempotency_key=key, request_fingerprint=fingerprint)

    with make_client() as client:
        response = client.post("/api/v1/execution/plan-run", json=body, headers={"Idempotency-Key": key})

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "idempotency_outcome_unknown"
    assert response.json()["detail"]["reconciliation_required"] is True


def test_plan_run_ledger_concurrent_claim_only_reserves_once(tmp_path):
    """Two workers racing for one key must never both obtain a fresh claim."""
    from concurrent.futures import ThreadPoolExecutor

    from api.plan_run_idempotency import (
        PlanRunIdempotencyInProgress,
        PlanRunIdempotencyStore,
    )

    store_path = tmp_path / "concurrent-plan-run.sqlite3"
    first_store = PlanRunIdempotencyStore(store_path)
    second_store = PlanRunIdempotencyStore(store_path)
    barrier = __import__("threading").Barrier(2)
    fingerprint = "a" * 64

    def claim(store):
        barrier.wait(timeout=5)
        try:
            result = store.claim(
                idempotency_key="concurrent-key",
                request_fingerprint=fingerprint,
            )
            return result.status
        except PlanRunIdempotencyInProgress:
            return "in_progress"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, (first_store, second_store)))

    assert sorted(outcomes) == ["in_progress", "running"]
    record = first_store.get_record("concurrent-key")
    assert record is not None
    assert record.status == "running"


def test_plan_run_ledger_concurrent_different_payload_conflicts(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from api.plan_run_idempotency import (
        PlanRunIdempotencyConflict,
        PlanRunIdempotencyStore,
    )

    store_path = tmp_path / "concurrent-conflict.sqlite3"
    stores = [PlanRunIdempotencyStore(store_path) for _ in range(2)]
    barrier = __import__("threading").Barrier(2)

    def claim(args):
        store, fingerprint = args
        barrier.wait(timeout=5)
        try:
            return store.claim(
                idempotency_key="one-key",
                request_fingerprint=fingerprint,
            ).status
        except PlanRunIdempotencyConflict:
            return "conflict"
        except Exception as exc:
            # If the other request won with a different fingerprint, conflict is
            # the only valid result for this worker; propagate unexpected failures.
            raise AssertionError(f"Unexpected claim failure: {exc}") from exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(claim, ((stores[0], "fingerprint-a"), (stores[1], "fingerprint-b"))))

    assert sorted(outcomes) == ["conflict", "running"]


def test_plan_run_storage_errors_do_not_mask_original_completion_error(monkeypatch, tmp_path):
    from api.routes import execution as execution_route
    from api.plan_run_idempotency import PlanRunIdempotencyStore

    monkeypatch.setenv("HAI_PLAN_RUN_IDEMPOTENCY_DB", str(tmp_path / "failure-plan-run.sqlite3"))
    monkeypatch.setattr(
        execution_route,
        "_run_cognitive_plan_impl",
        lambda request, payload: {"success": True, "state_after": {"box": "open"}},
    )

    def fail_complete(*args, **kwargs):
        raise OSError("original completion persistence failure")

    def fail_mark_unknown(*args, **kwargs):
        raise OSError("secondary unknown persistence failure")

    monkeypatch.setattr(PlanRunIdempotencyStore, "complete", fail_complete)
    monkeypatch.setattr(PlanRunIdempotencyStore, "mark_unknown", fail_mark_unknown)

    with TestClient(create_app(runtime_factory=FakeCognitiveRuntime), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/execution/plan-run",
            json=request_body(),
            headers={"Idempotency-Key": "completion-failure-key"},
        )

    assert response.status_code == 500
    # The primary completion failure remains visible in the server error logs;
    # the secondary ledger failure must not escape and replace it.
    store = PlanRunIdempotencyStore(tmp_path / "failure-plan-run.sqlite3")
    record = store.get_record("completion-failure-key")
    assert record is not None
    assert record.status == "running"  # fail-closed when both writes are unavailable


def test_plan_run_idempotency_key_remains_valid_after_simulation_reset(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "HAI_PLAN_RUN_IDEMPOTENCY_DB",
        str(tmp_path / "reset-contract.sqlite3"),
    )
    body = request_body()
    headers = {"Idempotency-Key": "reset-contract-key-001"}

    with make_client() as client:
        first = client.post(
            "/api/v1/execution/plan-run",
            json=body,
            headers=headers,
        )
        assert first.status_code == 200, first.text

        reset = client.post("/api/v1/execution/reset")
        assert reset.status_code == 200, reset.text
        assert reset.json()["state"]["box"] == "closed"

        replay = client.post(
            "/api/v1/execution/plan-run",
            json=body,
            headers=headers,
        )
        state_after_replay = client.get(
            "/api/v1/execution/state"
        ).json()["state"]

    assert replay.status_code == 200, replay.text
    assert replay.json()["idempotency"]["replayed"] is True
    assert replay.json()["state_after"] == first.json()["state_after"]

    # Reset changes simulation state, but does not invalidate the
    # completed request key or execute the original plan again.
    assert state_after_replay["box"] == "closed"
