from __future__ import annotations

import pytest

from agent.llm.capability_execution_contracts import CapabilityAuthorizationError
from agent.llm.execution_trace_store import ExecutionTraceStore
from agent.llm.traced_authenticated_capability_execution_gateway import (
    CapabilityTracePersistenceError,
    TracedAuthenticatedCapabilityExecutionGateway,
)


class Principal:
    actor_id = "operator-17"


class Request:
    target = "business_database"
    capability_id = "business_database.select"
    arguments = {"table": "orders", "filters": {"customer": "private-value"}}


class FakeGateway:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else {"rows": [{"id": 1}]}
        self.error = error
        self.calls = 0

    def read(self, principal, request):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def make(tmp_path, gateway=None, store=None):
    store = store or ExecutionTraceStore(tmp_path / "capability-trace.jsonl", fsync=False)
    gateway = gateway or FakeGateway()
    return TracedAuthenticatedCapabilityExecutionGateway(gateway, store), gateway, store


def test_success_records_ordered_lifecycle_without_arguments_or_result(tmp_path):
    traced, gateway, store = make(tmp_path)
    result = traced.read(Principal(), Request(), execution_id="read-001")

    assert result == {"rows": [{"id": 1}]}
    assert gateway.calls == 1
    rows = store.replay("read-001")
    assert [row["event_type"] for row in rows] == [
        "started", "authorized", "completed"
    ]
    payloads = [row["payload"] for row in rows]
    assert all(p["actor_id"] == "operator-17" for p in payloads)
    assert all(p["target"] == "business_database" for p in payloads)
    assert all("arguments" not in p and "rows" not in p for p in payloads)
    assert all("private-value" not in str(p) for p in payloads)
    assert store.verify()["valid"] is True


def test_authorization_denial_is_traced_and_adapter_gateway_is_not_retried(tmp_path):
    gateway = FakeGateway(error=CapabilityAuthorizationError("denied"))
    traced, gateway, store = make(tmp_path, gateway=gateway)

    with pytest.raises(CapabilityAuthorizationError):
        traced.read(Principal(), Request(), execution_id="read-denied")

    assert gateway.calls == 1
    assert [r["event_type"] for r in store.replay("read-denied")] == [
        "started", "denied"
    ]


def test_non_authorization_gateway_error_is_traced_as_failed(tmp_path):
    gateway = FakeGateway(error=RuntimeError("adapter unavailable"))
    traced, gateway, store = make(tmp_path, gateway=gateway)

    with pytest.raises(RuntimeError, match="adapter unavailable"):
        traced.read(Principal(), Request(), execution_id="read-failed")

    assert [r["event_type"] for r in store.replay("read-failed")] == [
        "started", "failed"
    ]
    assert store.replay("read-failed")[-1]["payload"]["outcome"] == "RuntimeError"


def test_start_trace_failure_prevents_gateway_access(tmp_path):
    class BrokenStore:
        def append(self, event):
            raise OSError("disk unavailable")

    gateway = FakeGateway()
    traced, _, _ = make(tmp_path, gateway=gateway, store=BrokenStore())

    with pytest.raises(CapabilityTracePersistenceError):
        traced.read(Principal(), Request(), execution_id="read-no-trace")
    assert gateway.calls == 0


def test_terminal_trace_failure_withholds_read_result(tmp_path):
    class FailCompletedStore:
        def __init__(self):
            self.inner = ExecutionTraceStore(tmp_path / "partial.jsonl", fsync=False)

        def append(self, event):
            if event["event_type"] == "completed":
                raise OSError("disk full")
            return self.inner.append(event)

    gateway = FakeGateway()
    store = FailCompletedStore()
    traced, _, _ = make(tmp_path, gateway=gateway, store=store)

    with pytest.raises(CapabilityTracePersistenceError):
        traced.read(Principal(), Request(), execution_id="read-terminal-fail")
    assert gateway.calls == 1
    assert [r["event_type"] for r in store.inner.replay("read-terminal-fail")] == [
        "started", "authorized"
    ]


def test_identical_retry_is_trace_idempotent(tmp_path):
    traced, gateway, store = make(tmp_path)
    traced.read(Principal(), Request(), execution_id="read-retry")
    traced.read(Principal(), Request(), execution_id="read-retry")

    assert gateway.calls == 2
    assert len(store.replay("read-retry")) == 3
    assert store.verify()["valid"] is True


def test_reusing_execution_id_for_different_actor_is_integrity_conflict(tmp_path):
    traced, _, store = make(tmp_path)
    traced.read(Principal(), Request(), execution_id="read-conflict")

    class OtherPrincipal:
        actor_id = "operator-18"

    from agent.llm.execution_trace_store import ExecutionTraceIntegrityError
    with pytest.raises(CapabilityTracePersistenceError):
        traced.read(OtherPrincipal(), Request(), execution_id="read-conflict")

    assert len(store.replay("read-conflict")) == 3
