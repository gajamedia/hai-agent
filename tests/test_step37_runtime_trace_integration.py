from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from agent.llm.execution_trace_runtime_observer import ExecutionTraceRuntimeObserver
from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.llm_plan_runtime_execution import LLMPlanRuntimeExecutor
from agent.llm.llm_plan_single_action_bridge import LLMPlanSingleActionBridge
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass
class FakeResult:
    success: bool
    result: object


class FakeAPI:
    def __init__(self, actions=("open_box", "take_key"), results=None):
        self.calls = []
        self.agent = type("AgentStub", (), {"actions": {name: object() for name in actions}})()
        self.results = list(results or [FakeResult(True, {}), FakeResult(True, {})])

    def execute(self, request):
        self.calls.append(request)
        if self.results:
            return self.results.pop(0)
        return FakeResult(True, {})


def make_plan():
    return LLMPlan(
        goal="open the box",
        rationale="Complete the goal in order.",
        steps=(
            LLMPlanStep(index=1, action="open_box", purpose="Open box", expected_observation="box open", success_criteria="box open", risk="low"),
            LLMPlanStep(index=2, action="take_key", purpose="Take key", expected_observation="key held", success_criteria="key held", risk="low"),
        ),
        assumptions=(), constraints=(), stop_conditions=(), confidence=0.9,
        provider="test", model="test-model",
    )


def make_runtime(api, observer):
    runtime = LLMPlanRuntimeExecutor.for_bridge(
        LLMPlanSingleActionBridge.for_api(api),
        runtime_event_observer=observer,
    )
    return runtime


def test_runtime_events_are_persisted_in_order_and_actions_execute_once(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    api = FakeAPI()
    runtime = make_runtime(api, observer)

    report = runtime.execute(make_plan(), metadata={"execution_id": "run-success-1", "source": "test"})

    assert report.success
    records = store.replay("run-success-1")
    assert [item["event_type"] for item in records] == [
        "started", "step_recorded", "step_recorded", "completed"
    ]
    assert [item["sequence"] for item in records] == [1, 2, 3, 4]
    assert len({item["event_id"] for item in records}) == 4
    assert len(api.calls) == 2
    assert observer.error_count == 0
    assert store.verify()["valid"] is True


def test_failed_step_and_stop_are_persisted_without_running_next_action(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    api = FakeAPI(results=[FakeResult(False, {"error": "blocked"})])
    runtime = make_runtime(api, observer)

    report = runtime.execute(make_plan(), metadata={"execution_id": "run-failure-1"})

    assert report.stopped and not report.success
    records = store.replay("run-failure-1")
    assert [item["event_type"] for item in records] == ["started", "step_recorded", "stopped"]
    step_record = records[1]["payload"]["step"]
    assert step_record["success"] is False
    assert len(api.calls) == 1
    assert observer.error_count == 0


def test_duplicate_delivery_is_idempotent(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    observer = ExecutionTraceRuntimeObserver(store)
    event = SimpleNamespace(
        phase="started", plan=SimpleNamespace(goal="test goal"), step=None, record=None,
        stop_reason=None, next_step_index=None, world_state={"secret": "not included by default"},
        metadata={"execution_id": "run-duplicate"},
    )

    observer(event)
    observer(event)

    assert len(store.read_all()) == 1
    assert store.read_all()[0]["event_id"] == "run-duplicate::started::plan"
    assert "world_state" not in store.read_all()[0]["payload"]
    assert observer.error_count == 0


def test_missing_execution_id_is_reported_without_raising_by_default(tmp_path):
    store = ExecutionTraceStore(tmp_path / "trace.jsonl", fsync=False)
    reported = []
    observer = ExecutionTraceRuntimeObserver(store, on_error=reported.append)
    event = SimpleNamespace(
        phase="started", plan=SimpleNamespace(goal="test goal"), step=None, record=None,
        stop_reason=None, next_step_index=None, metadata={},
    )

    observer(event)

    assert observer.error_count == 1
    assert "execution_id" in observer.last_error
    assert len(reported) == 1
    assert store.read_all() == []


def test_trace_store_failure_does_not_mask_runtime_event_by_default():
    class BrokenStore:
        def append(self, event):
            raise OSError("disk unavailable")

    observer = ExecutionTraceRuntimeObserver(BrokenStore())
    event = SimpleNamespace(
        phase="started", plan=SimpleNamespace(goal="test goal"), step=None, record=None,
        stop_reason=None, next_step_index=None, metadata={"execution_id": "run-disk-error"},
    )

    observer(event)  # best-effort default is non-raising

    assert observer.error_count == 1
    assert "disk unavailable" in observer.last_error


def test_strict_mode_can_propagate_trace_failure():
    class BrokenStore:
        def append(self, event):
            raise OSError("disk unavailable")

    observer = ExecutionTraceRuntimeObserver(BrokenStore(), raise_on_error=True)
    event = SimpleNamespace(
        phase="started", plan=SimpleNamespace(goal="test goal"), step=None, record=None,
        stop_reason=None, next_step_index=None, metadata={"execution_id": "run-strict"},
    )

    with pytest.raises(OSError, match="disk unavailable"):
        observer(event)
