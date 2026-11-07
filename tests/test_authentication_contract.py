from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request

from api.security.authentication import (
    AuthenticationProviderUnavailable,
    AuthenticationRejected,
    AuthenticatedPrincipal,
    InvalidAuthenticatedPrincipal,
    attach_authenticated_principal,
    authenticate_request,
    get_attached_principal,
    get_authentication_provider,
)


def test_principal_normalizes_actor_roles_and_permissions():
    principal = AuthenticatedPrincipal(
        actor_id="  user-17  ",
        roles=("admin", "admin", " reviewer "),
        permissions=("capability:read", "capability:read"),
        issuer="  https://identity.example ",
        subject="  subject-17 ",
    )

    assert principal.actor_id == "user-17"
    assert principal.roles == ("admin", "reviewer")
    assert principal.permissions == ("capability:read",)
    assert principal.issuer == "https://identity.example"
    assert principal.subject == "subject-17"
    assert principal.is_authenticated is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actor_id": ""},
        {"actor_id": "   "},
        {"actor_id": "user", "roles": "admin"},
        {"actor_id": "user", "permissions": "capability:read"},
        {"actor_id": "user", "roles": ("",)},
        {"actor_id": "user", "is_authenticated": False},
    ],
)
def test_invalid_principal_claims_are_rejected(kwargs):
    with pytest.raises(InvalidAuthenticatedPrincipal):
        AuthenticatedPrincipal(**kwargs)


def test_provider_unavailable_is_explicit_and_fail_closed():
    app = FastAPI()
    app.state.authentication_provider = None

    async def run():
        scope = {"type": "http", "app": app, "state": {}}
        request = Request(scope)
        with pytest.raises(AuthenticationProviderUnavailable):
            get_authentication_provider(request)
        with pytest.raises(AuthenticationProviderUnavailable):
            await authenticate_request(request)

    import asyncio
    asyncio.run(run())


def test_provider_none_means_unauthenticated():
    app = FastAPI()

    class Provider:
        async def authenticate(self, request):
            return None

    app.state.authentication_provider = Provider()

    async def run():
        request = Request({"type": "http", "app": app, "state": {}})
        with pytest.raises(AuthenticationRejected):
            await authenticate_request(request)

    import asyncio
    asyncio.run(run())


def test_provider_must_return_exact_principal_contract():
    app = FastAPI()

    class Provider:
        async def authenticate(self, request):
            return {"actor_id": "forged-user"}

    app.state.authentication_provider = Provider()

    async def run():
        request = Request({"type": "http", "app": app, "state": {}})
        with pytest.raises(InvalidAuthenticatedPrincipal):
            await authenticate_request(request)

    import asyncio
    asyncio.run(run())


def test_verified_provider_result_is_returned():
    app = FastAPI()
    expected = AuthenticatedPrincipal(
        actor_id="verified-user",
        roles=("capability-admin",),
        permissions=("capability:approve",),
    )

    class Provider:
        async def authenticate(self, request):
            return expected

    app.state.authentication_provider = Provider()

    async def run():
        request = Request({"type": "http", "app": app, "state": {}})
        actual = await authenticate_request(request)
        assert actual is expected

    import asyncio
    asyncio.run(run())


def test_attach_principal_populates_compatibility_actor_from_verified_identity():
    app = FastAPI()
    request = Request({"type": "http", "app": app, "state": {}})
    principal = AuthenticatedPrincipal(actor_id="verified-actor")

    attach_authenticated_principal(request, principal)

    assert get_attached_principal(request) is principal
    assert request.state.capability_actor == "verified-actor"


def test_arbitrary_actor_state_is_not_an_authenticated_principal():
    app = FastAPI()
    request = Request({"type": "http", "app": app, "state": {"capability_actor": "forged"}})

    with pytest.raises(AuthenticationRejected):
        get_attached_principal(request)


def test_attach_rejects_untrusted_mapping():
    app = FastAPI()
    request = Request({"type": "http", "app": app, "state": {}})

    with pytest.raises(InvalidAuthenticatedPrincipal):
        attach_authenticated_principal(
            request,
            {"actor_id": "forged"},  # type: ignore[arg-type]
        )
