from __future__ import annotations

import traceback

import pytest

from agent.llm.traced_authenticated_capability_execution_gateway import (
    CapabilityTracePersistenceError,
    TracedAuthenticatedCapabilityExecutionGateway,
)


class Principal:
    actor_id = "test-operator"


class Request:
    target = "business_database"
    capability_id = "business_database.select"


class FakeGateway:
    def __init__(self):
        self.calls = 0

    def read(self, principal, request):
        self.calls += 1
        return {"ok": True}


class FailingStore:
    def __init__(self, *, fail_on="started", secret="DB_PASSWORD=super-secret"):
        self.fail_on = fail_on
        self.secret = secret
        self.events = []

    def append(self, event):
        if event["event_type"] == self.fail_on:
            raise OSError(self.secret)
        self.events.append(event)


def test_trace_start_failure_sanitizes_diagnostic_and_blocks_gateway():
    gateway = FakeGateway()
    store = FailingStore(fail_on="started")
    traced = TracedAuthenticatedCapabilityExecutionGateway(gateway, store)

    with pytest.raises(CapabilityTracePersistenceError) as caught:
        traced.read(Principal(), Request(), execution_id="safe-start")

    assert gateway.calls == 0
    assert traced.error_count == 1
    assert traced.last_error == "OSError"
    assert store.secret not in traced.last_error
    assert store.secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    assert store.secret not in rendered


def test_terminal_trace_failure_withholds_result_and_sanitizes_diagnostic():
    gateway = FakeGateway()
    store = FailingStore(fail_on="completed", secret="postgres://user:password@private-host/db")
    traced = TracedAuthenticatedCapabilityExecutionGateway(gateway, store)

    with pytest.raises(CapabilityTracePersistenceError) as caught:
        traced.read(Principal(), Request(), execution_id="safe-terminal")

    assert gateway.calls == 1
    assert [event["event_type"] for event in store.events] == ["started", "authorized"]
    assert traced.error_count == 1
    assert traced.last_error == "OSError"
    assert store.secret not in traced.last_error
    assert store.secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_success_clears_previous_error_diagnostic():
    gateway = FakeGateway()
    store = FailingStore()
    store.fail_on = "never"
    traced = TracedAuthenticatedCapabilityExecutionGateway(gateway, store)

    traced.last_error = "OSError"
    result = traced.read(Principal(), Request(), execution_id="safe-success")

    assert result == {"ok": True}
    assert traced.last_error is None
    assert traced.error_count == 0
    assert traced.persisted_count == 3
