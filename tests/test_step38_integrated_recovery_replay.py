from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.llm.execution_trace_runtime_observer import ExecutionTraceRuntimeObserver
from agent.llm.execution_trace_store import ExecutionTraceStore, ExecutionTraceIntegrityError
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class FakeResult:
    success: bool
    result: object


class FakeAPI:
    def __init__(self, results):
        self.calls = []
        self.agent = type("AgentStub", (), {"actions": {"open_box": object(), "take_key": object()}})()
        self.results = list(results)

    def execute(self, request):
        self.calls.append(request)
        if self.results:
            return self.results.pop(0)
        return FakeResult(True, {})


def make_plan():
    return LLMPlan(
        goal="open the box",
        rationale="Open the box and take the key.",
        steps=(
            LLMPlanStep(index=1, action="open_box", purpose="Open box", expected_observation="box open", success_criteria="box open", risk="low"),
            LLMPlanStep(index=2, action="take_key", purpose="Take key", expected_observation="key held", success_criteria="key held", risk="low"),
        ),
        assumptions=(), constraints=(), stop_conditions=(), confidence=0.9,
        provider="test", model="test-model",
    )


def make_runtime(api, observer):
    return LLMPlanRuntimeExecutor.for_bridge(
        LLMPlanSingleActionBridge.for_api(api),
        runtime_event_observer=observer,
    )


def test_failed_attempt_and_recovery_attempt_share_execution_without_event_id_collision(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    api = FakeAPI([FakeResult(False, {"error": "blocked"}), FakeResult(True, {}), FakeResult(True, {})])
    runtime = make_runtime(api, observer)
    plan = make_plan()

    failed = runtime.execute(plan, metadata={"execution_id": "run-recovery", "generation": 1})
    recovered = runtime.execute(plan, metadata={"execution_id": "run-recovery", "generation": 2})

    assert failed.stopped and not failed.success
    assert recovered.success
    records = store.replay("run-recovery")
    assert len(records) == 7
    assert [row["payload"]["generation"] for row in records] == [1, 1, 1, 2, 2, 2, 2]
    assert len({row["event_id"] for row in records}) == len(records)
    assert store.verify()["valid"] is True
    assert observer.error_count == 0


def test_replay_after_recovery_does_not_execute_actions_again(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    api = FakeAPI([FakeResult(False, {"error": "blocked"}), FakeResult(True, {}), FakeResult(True, {})])
    runtime = make_runtime(api, observer)
    plan = make_plan()
    runtime.execute(plan, metadata={"execution_id": "run-readonly", "generation": 1})
    runtime.execute(plan, metadata={"execution_id": "run-readonly", "generation": 2})
    calls_before_replay = len(api.calls)

    replayed = store.replay("run-readonly")

    assert replayed
    assert len(api.calls) == calls_before_replay == 3
    assert all(not callable(item) for item in replayed)


def test_coordinator_events_use_generation_scoped_ids(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    plan = make_plan()

    observer(SimpleNamespace(phase="coordinated", generation=1, goal_id="goal-a", plan=plan,
                             runtime_result=None, world_state={"private": "excluded"}, metadata={"execution_id": "multi-goal-run"}))
    observer(SimpleNamespace(phase="coordinated", generation=2, goal_id="goal-a", plan=plan,
                             runtime_result=None, world_state={"private": "excluded"}, metadata={"execution_id": "multi-goal-run"}))

    records = store.replay("multi-goal-run")
    assert len(records) == 2
    assert records[0]["event_id"] != records[1]["event_id"]
    assert records[0]["payload"]["generation"] == 1
    assert records[1]["payload"]["generation"] == 2
    assert all("world_state" not in row["payload"] for row in records)


def test_identical_event_retry_is_idempotent_within_generation(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    event = SimpleNamespace(phase="segment_started", generation=3, goal_id="goal-a", plan=make_plan(),
                            runtime_result=None, metadata={"execution_id": "run-idempotent", "generation": 3})

    observer(event)
    observer(event)

    assert len(store.replay("run-idempotent")) == 1
    assert observer.error_count == 0


def test_conflicting_event_for_same_generation_is_reported(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store, raise_on_error=True)
    base = dict(phase="segment_started", generation=3, goal_id="goal-a", plan=make_plan(),
                runtime_result=None, metadata={"execution_id": "run-conflict", "generation": 3})
    observer(SimpleNamespace(**base))
    changed = dict(base)
    changed["goal_id"] = "goal-b"

    with pytest.raises(ExecutionTraceIntegrityError):
        observer(SimpleNamespace(**changed))

    assert len(store.replay("run-conflict")) == 1
    assert observer.error_count == 1


def test_runtime_failure_trace_records_stop_before_recovery_generation(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    api = FakeAPI([FakeResult(False, {"error": "precondition failed"}), FakeResult(True, {}), FakeResult(True, {})])
    runtime = make_runtime(api, observer)
    plan = make_plan()
    runtime.execute(plan, metadata={"execution_id": "run-order", "generation": 1})
    runtime.execute(plan, metadata={"execution_id": "run-order", "generation": 2})

    records = store.replay("run-order")
    phases_by_generation = {
        generation: [row["event_type"] for row in records if row["payload"].get("generation") == generation]
        for generation in (1, 2)
    }
    assert phases_by_generation[1] == ["started", "step_recorded", "stopped"]
    assert phases_by_generation[2] == ["started", "step_recorded", "step_recorded", "completed"]
