from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.resource_lifecycle import (
    CapabilityExecutionResourceBundle,
    close_owned_resources,
)


class SyncResource:
    def __init__(self, name, events, *, fail=False):
        self.name = name
        self.events = events
        self.fail = fail

    def close(self):
        self.events.append(("close", self.name))
        if self.fail:
            raise RuntimeError(f"{self.name} close failed")


class AsyncResource:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    async def aclose(self):
        self.events.append(("aclose", self.name))


class DualCloser:
    def __init__(self, events):
        self.events = events

    async def aclose(self):
        self.events.append(("aclose", "dual"))

    def close(self):
        self.events.append(("close", "dual"))


class FakeGateway:
    def read(self, principal, request, execution_id=None):
        return {"ok": True}


class BorrowedGateway(FakeGateway):
    def __init__(self, events):
        self.events = events

    def close(self):
        self.events.append(("close", "borrowed-gateway"))


def _make_app(*, gateway_factory, proposal_store_factory=None, provider=None, repository=None):
    return create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        cognitive_capability_runtime_factory=lambda app, runtime: None,
        authentication_provider_factory=lambda app: provider,
        authorization_policy_factory=lambda app: object(),
        capability_execution_gateway_factory=gateway_factory,
        capability_store_factory=proposal_store_factory or (lambda: object()),
        autonomous_testing_repository_factory=lambda path: repository,
        autonomous_testing_manager_factory=lambda repository: object(),
    )


def test_cleanup_closes_in_reverse_order_deduplicates_and_continues(caplog):
    events = []
    first = SyncResource("first", events)
    broken = SyncResource("broken", events, fail=True)
    last = AsyncResource("last", events)
    dual = DualCloser(events)

    with caplog.at_level(logging.ERROR):
        asyncio.run(close_owned_resources(
            [first, broken, last, dual, first],
            logger=logging.getLogger("test.lifecycle"),
        ))

    assert events == [
        ("aclose", "dual"),
        ("aclose", "last"),
        ("close", "broken"),
        ("close", "first"),
    ]
    assert "Failed to close application-owned resource" in caplog.text


def test_cleanup_accepts_sync_close_returning_awaitable():
    events = []

    class AwaitableClose:
        def close(self):
            async def finish():
                events.append("finished")
            return finish()

    asyncio.run(close_owned_resources([AwaitableClose()]))
    assert events == ["finished"]


def test_bundle_requires_explicit_tuple_ownership():
    with pytest.raises(TypeError, match="must be a tuple"):
        CapabilityExecutionResourceBundle(FakeGateway(), owned_resources=[object()])


def test_owned_gateway_resources_and_app_resources_close_on_normal_shutdown(tmp_path, monkeypatch):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    events = []
    provider = SyncResource("provider", events)
    gateway_dependency = SyncResource("gateway-dependency", events)
    repository = SyncResource("repository", events)
    gateway = FakeGateway()

    app = _make_app(
        gateway_factory=lambda app: CapabilityExecutionResourceBundle(
            gateway=gateway,
            owned_resources=(gateway_dependency,),
        ),
        provider=provider,
        repository=repository,
    )

    with TestClient(app):
        assert app.state.capability_execution_gateway is gateway

    assert events == [
        ("close", "repository"),
        ("close", "gateway-dependency"),
        ("close", "provider"),
    ]
    assert app.state.capability_execution_gateway is None
    assert app.state.execution_trace_store is None


def test_legacy_bare_gateway_is_borrowed_and_not_closed(monkeypatch):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    events = []
    gateway = BorrowedGateway(events)
    app = _make_app(gateway_factory=lambda app: gateway)

    with TestClient(app):
        assert app.state.capability_execution_gateway is gateway

    assert events == []


def test_startup_failure_after_gateway_factory_still_closes_transferred_resources(monkeypatch):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    events = []
    resource = SyncResource("gateway-resource", events)

    def fail_store_factory():
        raise RuntimeError("proposal store startup failed")

    app = _make_app(
        gateway_factory=lambda app: CapabilityExecutionResourceBundle(
            gateway=FakeGateway(),
            owned_resources=(resource,),
        ),
        proposal_store_factory=fail_store_factory,
    )

    with pytest.raises(RuntimeError, match="proposal store startup failed"):
        with TestClient(app):
            pass

    assert events == [("close", "gateway-resource")]


def test_invalid_bundle_gateway_still_closes_transferred_resources(monkeypatch):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    events = []
    resource = SyncResource("owned-before-validation", events)

    app = _make_app(
        gateway_factory=lambda app: CapabilityExecutionResourceBundle(
            gateway=object(),
            owned_resources=(resource,),
        )
    )

    with pytest.raises(RuntimeError, match="must return an object"):
        with TestClient(app):
            pass

    assert events == [("close", "owned-before-validation")]


def test_bundle_with_none_gateway_fails_and_closes_resources(monkeypatch):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    events = []
    resource = SyncResource("owned-with-none-gateway", events)

    app = _make_app(
        gateway_factory=lambda app: CapabilityExecutionResourceBundle(
            gateway=None,
            owned_resources=(resource,),
        )
    )

    with pytest.raises(RuntimeError, match="must contain a gateway"):
        with TestClient(app):
            pass

    assert events == [("close", "owned-with-none-gateway")]


def test_failure_after_repository_creation_closes_repository(monkeypatch):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    events = []
    repository = SyncResource("repository", events)

    def fail_manager_factory(repository):
        raise RuntimeError("batch manager startup failed")

    app = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        cognitive_capability_runtime_factory=lambda app, runtime: None,
        authentication_provider_factory=lambda app: None,
        authorization_policy_factory=lambda app: object(),
        capability_store_factory=lambda: object(),
        autonomous_testing_repository_factory=lambda path: repository,
        autonomous_testing_manager_factory=fail_manager_factory,
    )

    with pytest.raises(RuntimeError, match="batch manager startup failed"):
        with TestClient(app):
            pass

    assert events == [("close", "repository")]
