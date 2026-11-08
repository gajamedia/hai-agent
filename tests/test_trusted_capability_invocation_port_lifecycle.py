
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent.llm.trusted_capability_invocation_port import CapabilityInvocationSpec
from api.main import create_app
from api.security.authorization import AuthorizationPolicy
from api.trusted_capability_invocation_port_factory import (
    TrustedCapabilityInvocationPortFactoryError,
    build_trusted_capability_invocation_port,
)


class FakeGateway:
    def __init__(self):
        self.close_calls = 0
        self.read_calls = 0

    def read(self, principal, request, *, execution_id):
        self.read_calls += 1
        raise AssertionError("The lifecycle test must not invoke gateway.read().")

    def close(self):
        self.close_calls += 1


class FakeRepository:
    def close(self):
        pass


def valid_specs():
    return {
        "read_business_data": CapabilityInvocationSpec(
            action_name="read_business_data",
            target="business_database",
            capability_id="business_database.select",
        )
    }


def make_lifecycle_app(
    monkeypatch,
    *,
    gateway_factory=None,
    specs_factory=None,
):
    monkeypatch.setenv("HAI_AUTH_ENABLED", "false")
    monkeypatch.setenv("HAI_AUTH_REQUIRED", "false")
    monkeypatch.setenv("HAI_ENV", "development")

    return create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
        capability_execution_gateway_factory=gateway_factory,
        trusted_capability_invocation_specs_factory=specs_factory,
        authentication_provider_factory=lambda app: None,
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
        autonomous_testing_repository_factory=lambda path: FakeRepository(),
        autonomous_testing_manager_factory=lambda repository: object(),
    )


# ---------------------------------------------------------------------------
# Existing factory tests
# ---------------------------------------------------------------------------

def test_factory_binds_the_exact_existing_gateway_instance():
    gateway = FakeGateway()
    port = build_trusted_capability_invocation_port(gateway, valid_specs())

    assert port._gateway is gateway
    assert port.supports_action("read_business_data")
    assert port.action_names == ("read_business_data",)


def test_factory_does_not_invoke_gateway_during_binding():
    gateway = FakeGateway()

    build_trusted_capability_invocation_port(gateway, valid_specs())

    assert gateway.read_calls == 0


def test_factory_rejects_missing_gateway():
    with pytest.raises(TrustedCapabilityInvocationPortFactoryError):
        build_trusted_capability_invocation_port(None, valid_specs())


def test_factory_rejects_gateway_without_read_contract():
    with pytest.raises(TrustedCapabilityInvocationPortFactoryError):
        build_trusted_capability_invocation_port(object(), valid_specs())


@pytest.mark.parametrize("specs", [{}, None, []])
def test_factory_rejects_missing_or_invalid_specs(specs):
    with pytest.raises(TrustedCapabilityInvocationPortFactoryError):
        build_trusted_capability_invocation_port(FakeGateway(), specs)


def test_factory_rejects_spec_key_mismatch():
    specs = {
        "wrong_action": CapabilityInvocationSpec(
            action_name="read_business_data",
            target="business_database",
            capability_id="business_database.select",
        )
    }

    with pytest.raises(TrustedCapabilityInvocationPortFactoryError):
        build_trusted_capability_invocation_port(FakeGateway(), specs)


# ---------------------------------------------------------------------------
# C.4 application lifecycle integration tests
# ---------------------------------------------------------------------------

def test_app_lifespan_binds_port_to_exact_existing_gateway(monkeypatch):
    gateway = FakeGateway()

    app = make_lifecycle_app(
        monkeypatch,
        gateway_factory=lambda app: gateway,
        specs_factory=lambda app: valid_specs(),
    )

    with TestClient(app):
        assert app.state.capability_execution_gateway is gateway

        port = app.state.trusted_capability_invocation_port

        assert port is not None
        assert port._gateway is gateway
        assert port.supports_action("read_business_data")
        assert port.action_names == ("read_business_data",)

        # Binding must not perform a database/capability read.
        assert gateway.read_calls == 0


def test_app_lifespan_rejects_specs_factory_without_gateway(monkeypatch):
    app = make_lifecycle_app(
        monkeypatch,
        gateway_factory=None,
        specs_factory=lambda app: valid_specs(),
    )

    with pytest.raises(
        RuntimeError,
        match="Trusted capability invocation specs require an existing",
    ):
        with TestClient(app):
            pass


def test_app_lifespan_keeps_port_none_when_specs_factory_is_not_configured(
    monkeypatch,
):
    gateway = FakeGateway()

    app = make_lifecycle_app(
        monkeypatch,
        gateway_factory=lambda app: gateway,
        specs_factory=None,
    )

    with TestClient(app):
        assert app.state.capability_execution_gateway is gateway
        assert app.state.trusted_capability_invocation_port is None

    assert app.state.trusted_capability_invocation_port is None


def test_shutdown_clears_port_without_closing_borrowed_gateway(monkeypatch):
    gateway = FakeGateway()

    app = make_lifecycle_app(
        monkeypatch,
        gateway_factory=lambda app: gateway,
        specs_factory=lambda app: valid_specs(),
    )

    with TestClient(app):
        assert app.state.capability_execution_gateway is gateway
        assert app.state.trusted_capability_invocation_port is not None

    # Shutdown clears application references.
    assert app.state.trusted_capability_invocation_port is None
    assert app.state.capability_execution_gateway is None

    # A bare gateway is borrowed, not owned by the application.
    assert gateway.close_calls == 0
    assert gateway.read_calls == 0
