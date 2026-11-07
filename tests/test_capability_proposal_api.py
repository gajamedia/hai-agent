
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from agent.llm.capability_proposal_store import CapabilityProposalStore
from agent.llm.capability_proposal_validator import CapabilityProposalValidator
from api.routes.capabilities import router
from api.security.authentication import AuthenticatedPrincipal
from api.security.authentication_middleware import CapabilityAuthenticationMiddleware
from api.security.authorization import CAPABILITY_PERMISSIONS, AuthorizationPolicy


ALL_PROPOSAL_PERMISSIONS = tuple(
    CAPABILITY_PERMISSIONS[key]
    for key in (
        "proposal_read",
        "proposal_submit",
        "proposal_approve",
        "proposal_reject",
        "proposal_register",
    )
)


class StaticAuthenticationProvider:
    def __init__(self, principal):
        self.principal = principal

    async def authenticate(self, request):
        return self.principal


class FakeApprovalManager:
    def __init__(self, store):
        self.store = store

    def submit(self, proposal):
        return self.store.submit(proposal)

    def approve(self, proposal_id, *, approved_by):
        from agent.llm.capability_proposal import ProposalStatus

        return self.store.transition(
            proposal_id,
            expected_status=ProposalStatus.PENDING,
            new_status=ProposalStatus.APPROVED,
            actor=approved_by,
        )

    def reject(self, proposal_id, *, rejected_by, reason):
        from agent.llm.capability_proposal import ProposalStatus

        return self.store.transition(
            proposal_id,
            expected_status=ProposalStatus.PENDING,
            new_status=ProposalStatus.REJECTED,
            actor=rejected_by,
            reason=reason,
        )

    def register_approved_read_only(self, proposal_id, *, registered_by):
        raise RuntimeError(
            "registration is not part of this route-contract test"
        )


def make_client(
    *,
    with_manager=True,
    actor="reviewer",
    permissions=ALL_PROPOSAL_PERMISSIONS,
    with_provider=True,
    with_policy=True,
):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    store = CapabilityProposalStore(
        validator=CapabilityProposalValidator()
    )

    app.state.capability_proposal_store = store

    app.state.capability_approval_manager = (
        FakeApprovalManager(store) if with_manager else None
    )

    app.state.authentication_provider = (
        StaticAuthenticationProvider(
            AuthenticatedPrincipal(
                actor_id=actor,
                permissions=tuple(permissions),
            )
        )
        if with_provider and actor is not None
        else StaticAuthenticationProvider(None)
        if with_provider
        else None
    )

    app.state.authorization_policy = (
        AuthorizationPolicy() if with_policy else None
    )

    @app.middleware("http")
    async def spoofable_legacy_actor(request: Request, call_next):
        # Deliberately forge the old compatibility field. The central
        # capability middleware must clear it and use only provider identity.
        request.state.capability_actor = "forged-from-request-state"
        return await call_next(request)

    app.add_middleware(CapabilityAuthenticationMiddleware)

    return TestClient(app), store, app


def proposal_payload(**overrides):
    payload = {
        "capability_id": "app.environment.observe_extra",
        "action_name": "observe_extra",
        "description": "Read an additional application-owned observation.",
        "category": "read_only",
        "risk_level": "low",
        "permissions": ["read:environment"],
        "preconditions": {},
        "effects": {},
        "parameters": {},
    }

    payload.update(overrides)
    return payload


def test_submit_list_and_get_proposal():
    client, _, _ = make_client()

    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    )

    assert response.status_code == 201

    proposal = response.json()["proposal"]

    assert proposal["status"] == "pending"
    assert proposal["proposal_id"]
    assert proposal["version"] == 1

    listed = client.get("/api/v1/capability-proposals")

    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    fetched = client.get(
        f"/api/v1/capability-proposals/{proposal['proposal_id']}"
    )

    assert fetched.status_code == 200
    assert (
        fetched.json()["proposal"]["capability_id"]
        == "app.environment.observe_extra"
    )


def test_submission_rejects_executable_or_system_fields():
    client, store, _ = make_client()

    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(handler="print('unsafe')"),
    )

    assert response.status_code == 422
    assert store.list_all() == ()

    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(
            status="approved",
            approved_by="attacker",
        ),
    )

    assert response.status_code == 422
    assert store.list_all() == ()


def test_submission_rejects_unknown_category_and_invalid_metadata():
    client, _, _ = make_client()

    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(category="executable"),
    )

    assert response.status_code == 422

    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(effects=[]),
    )

    assert response.status_code == 422


def test_filter_by_status_and_reject_invalid_status():
    client, _, _ = make_client()

    client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    )

    pending = client.get(
        "/api/v1/capability-proposals?status=pending"
    )

    assert pending.status_code == 200
    assert pending.json()["count"] == 1

    invalid = client.get(
        "/api/v1/capability-proposals?status=active"
    )

    assert invalid.status_code == 422


def test_approve_uses_verified_principal_actor_not_legacy_state_value():
    client, _, _ = make_client()

    created = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    ).json()["proposal"]

    approved = client.post(
        f"/api/v1/capability-proposals/{created['proposal_id']}/approve"
    )

    assert approved.status_code == 200
    assert approved.json()["proposal"]["status"] == "approved"
    assert approved.json()["proposal"]["approved_by"] == "reviewer"

    assert (
        approved.json()["proposal"]["approved_by"]
        != "forged-from-request-state"
    )


def test_reject_requires_reason_and_records_verified_actor():
    client, _, _ = make_client()

    created = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    ).json()["proposal"]

    missing_reason = client.post(
        f"/api/v1/capability-proposals/{created['proposal_id']}/reject",
        json={},
    )

    assert missing_reason.status_code == 422

    rejected = client.post(
        f"/api/v1/capability-proposals/{created['proposal_id']}/reject",
        json={
            "reason": "Not in the approved application catalog."
        },
    )

    assert rejected.status_code == 200
    assert rejected.json()["proposal"]["status"] == "rejected"
    assert rejected.json()["proposal"]["rejected_by"] == "reviewer"


def test_approval_fails_closed_when_manager_is_not_configured():
    client, _, _ = make_client(with_manager=False)

    created = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    ).json()["proposal"]

    response = client.post(
        f"/api/v1/capability-proposals/{created['proposal_id']}/approve"
    )

    assert response.status_code == 503
    assert (
        response.json()["detail"]["error"]
        == "capability_approval_unavailable"
    )


def test_missing_authentication_provider_is_unavailable():
    client, _, _ = make_client(with_provider=False)

    response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 503

    # STAGE 2.8B.11A:
    # Match the current production middleware error contract.
    assert (
        response.json()["detail"]["error"]
        == "authentication_provider_unavailable"
    )


def test_provider_rejecting_request_returns_401():
    client, _, _ = make_client(actor=None)

    response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 401
    assert (
        response.json()["detail"]["error"]
        == "authentication_required"
    )


def test_forged_legacy_actor_state_does_not_authenticate_request():
    client, _, _ = make_client(actor=None)

    # Middleware deliberately writes a plausible-looking actor string, but
    # the trusted provider rejects the request and no principal is attached.
    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    )

    assert response.status_code == 401
    assert (
        response.json()["detail"]["error"]
        == "authentication_required"
    )


def test_missing_authorization_policy_fails_closed():
    client, _, _ = make_client(with_policy=False)

    response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 503
    assert (
        response.json()["detail"]["error"]
        == "authorization_policy_unavailable"
    )


def test_authenticated_principal_without_required_permission_gets_403():
    client, _, _ = make_client(
        permissions=(
            CAPABILITY_PERMISSIONS["proposal_read"],
        )
    )

    response = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    )

    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "permission_denied"

    assert (
        response.json()["detail"]["permission"]
        == CAPABILITY_PERMISSIONS["proposal_submit"]
    )


def test_read_permission_does_not_grant_approval_permission():
    client, _, _ = make_client(
        permissions=(
            CAPABILITY_PERMISSIONS["proposal_read"],
        )
    )

    response = client.post(
        "/api/v1/capability-proposals/some-id/approve"
    )

    assert response.status_code == 403


def test_submit_list_and_detail_have_independent_permission_checks():
    submit_only = (
        CAPABILITY_PERMISSIONS["proposal_submit"],
    )

    client, _, _ = make_client(permissions=submit_only)

    created = client.post(
        "/api/v1/capability-proposals",
        json=proposal_payload(),
    )

    assert created.status_code == 201

    assert (
        client.get("/api/v1/capability-proposals").status_code
        == 403
    )

    proposal_id = created.json()["proposal"]["proposal_id"]

    assert (
        client.get(
            f"/api/v1/capability-proposals/{proposal_id}"
        ).status_code
        == 403
    )


def test_unknown_proposal_returns_404_after_authorization():
    client, _, _ = make_client()

    response = client.get(
        "/api/v1/capability-proposals/does-not-exist"
    )

    assert response.status_code == 404


def test_main_lifespan_initializes_policy_provider_store_and_manager():
    from api.main import create_app

    captured = {}

    def store_factory():
        store = CapabilityProposalStore(
            validator=CapabilityProposalValidator()
        )
        captured["store"] = store
        return store

    shared_components = {
        "registry": object(),
        "adapter": object(),
        "action_builder": object(),
        "agent": object(),
    }
    capability_runtime = type(
        "FakeCognitiveCapabilityRuntime",
        (),
        {
            **shared_components,
            "lifecycle_manager": None,
        },
    )()

    def manager_factory(app, store):
        captured["app"] = app
        captured["manager_store"] = store
        manager = FakeApprovalManager(store)
        # The app requires every approval manager to share the exact
        # application-owned capability components. This fixture models
        # that wiring without constructing the full cognitive runtime.
        for name, component in shared_components.items():
            setattr(manager, name, component)
        return manager

    def provider_factory(app):
        return StaticAuthenticationProvider(
            AuthenticatedPrincipal(
                actor_id="lifespan-reviewer",
                permissions=ALL_PROPOSAL_PERMISSIONS,
            )
        )

    def policy_factory(app):
        return AuthorizationPolicy()

    application = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=store_factory,
        capability_approval_manager_factory=manager_factory,
        cognitive_capability_runtime_factory=lambda app, runtime: capability_runtime,
        authentication_provider_factory=provider_factory,
        authorization_policy_factory=policy_factory,
    )

    with TestClient(application) as client:
        assert (
            application.state.capability_proposal_store
            is captured["store"]
        )

        assert application.state.capability_approval_manager is not None
        assert application.state.authentication_provider is not None
        assert application.state.authorization_policy is not None

        assert (
            captured["manager_store"]
            is captured["store"]
        )

        response = client.post(
            "/api/v1/capability-proposals",
            json=proposal_payload(),
        )

        assert response.status_code == 201

    assert application.state.authentication_provider is None
    assert application.state.authorization_policy is None
    assert application.state.capability_proposal_store is None
    assert application.state.capability_approval_manager is None


def test_capability_authentication_middleware_clears_forged_principal_and_actor():
    client, _, app = make_client()

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 200


def test_capability_authentication_middleware_rejects_when_provider_is_missing():
    client, _, _ = make_client(with_provider=False)

    with client:
        response = client.get("/api/v1/capability-proposals")

    assert response.status_code == 503

    # STAGE 2.8B.11A:
    # Match the current production middleware error contract.
    assert (
        response.json()["detail"]["error"]
        == "authentication_provider_unavailable"
    )

def test_main_lifespan_rejects_approval_manager_without_shared_capability_runtime():
    from api.main import create_app

    application = create_app(
        runtime_factory=lambda: object(),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: CapabilityProposalStore(
            validator=CapabilityProposalValidator()
        ),
        capability_approval_manager_factory=lambda app, store: FakeApprovalManager(store),
        cognitive_capability_runtime_factory=lambda app, runtime: None,
        authentication_provider_factory=lambda app: None,
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
    )

    with pytest.raises(
        RuntimeError,
        match="cognitive capability runtime is unavailable",
    ):
        with TestClient(application):
            pass

