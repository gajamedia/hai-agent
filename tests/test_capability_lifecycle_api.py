from __future__ import annotations

from types import SimpleNamespace


from fastapi.testclient import TestClient

from agent.core.action import Action
from agent.llm.capability_registry import CapabilityStatus, TrustedCapabilityRegistry
from agent.llm.dynamic_workspace_runtime_adapter import DynamicWorkspaceRuntimeAdapter
from agent.llm.cognitive_capability_runtime import compose_cognitive_capability_runtime
from api.main import create_app
from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationPolicy, CAPABILITY_PERMISSIONS


ALL_LIFECYCLE_PERMISSIONS = tuple(
    CAPABILITY_PERMISSIONS[key]
    for key in (
        "lifecycle_read",
        "lifecycle_activate",
        "lifecycle_disable",
        "lifecycle_revoke",
    )
)


class PrincipalMiddleware:
    """Test-only trusted identity injection; production uses AuthenticationProvider."""

    def __init__(self, app, *, actor="operator", permissions=ALL_LIFECYCLE_PERMISSIONS,
                 actor_only=False):
        self.app = app
        self.actor = actor
        self.permissions = permissions
        self.actor_only = actor_only

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            state = scope.setdefault("state", {})
            if self.actor is not None:
                state["capability_actor"] = self.actor
                if not self.actor_only:
                    state["authenticated_principal"] = AuthenticatedPrincipal(
                        actor_id=self.actor,
                        permissions=tuple(self.permissions),
                    )
        await self.app(scope, receive, send)


def make_runtime():
    action = Action(
        name="inspect_room",
        description="Inspect the room",
        handler=lambda **kwargs: {"success": True},
    )

    class Planner:
        def __init__(self, actions):
            self.actions = list(actions)

    class FakeAgent:
        def __init__(self):
            self.actions = {"inspect_room": action}
            self.planner = Planner(self.actions.values())
            self.adaptive_planner = SimpleNamespace(planner=self.planner)
            self.hierarchical_planner = SimpleNamespace(planner=self.planner)
            self.risk_aware_hierarchical_planner = SimpleNamespace(
                hierarchical_planner=self.hierarchical_planner
            )
            self.unified_executor = object()
            self._planner_sync_hook = None

        def set_planner_sync_hook(self, hook):
            self._planner_sync_hook = hook

        def clear_planner_sync_hook(self, hook=None):
            if hook is not None and hook is not self._planner_sync_hook:
                return False
            self._planner_sync_hook = None
            return True

        def register_action(self, action):
            """Match the Agent registration contract used by the runtime builder."""
            self.actions[action.name] = action
            self.configure_planner()

        def configure_planner(self):
            self.planner = Planner(self.actions.values())
            self.adaptive_planner.planner = self.planner
            self.hierarchical_planner.planner = self.planner
            self.risk_aware_hierarchical_planner.hierarchical_planner = (
                self.hierarchical_planner
            )
            if self._planner_sync_hook:
                self._planner_sync_hook()

    fake_agent = FakeAgent()
    capability_runtime = compose_cognitive_capability_runtime(fake_agent)
    capability_runtime.registry.register_trusted(
        capability_id="application.inspect_room",
        action=action,
        risk_level="low",
        permissions=("read:environment",),
    )
    # The action already exists on the fake Agent. Synchronization binds the
    # registry record to that exact action and verifies the shared planner.
    capability_runtime.adapter.synchronize()

    runtime = SimpleNamespace(
        agent=fake_agent,
        capability_runtime=capability_runtime,
    )
    return runtime, capability_runtime.lifecycle_manager


class StaticAuthenticationProvider:
    def __init__(self, principal):
        self.principal = principal

    async def authenticate(self, request):
        return self.principal


def make_client(*, manager=True, actor="operator", permissions=ALL_LIFECYCLE_PERMISSIONS,
                policy=True, actor_only=False):
    runtime, lifecycle_manager = make_runtime()

    def manager_factory(app, cognitive_runtime):
        assert cognitive_runtime is runtime
        return lifecycle_manager if manager else None

    app = create_app(
        runtime_factory=lambda: runtime,
        cognitive_capability_runtime_factory=(
            lambda app, cognitive_runtime: runtime.capability_runtime
        ),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
        capability_lifecycle_manager_factory=manager_factory,
        authorization_policy_factory=(
            (lambda app: AuthorizationPolicy()) if policy else None
        ),
        authentication_provider_factory=(
            lambda app: StaticAuthenticationProvider(
                None if actor is None or actor_only else AuthenticatedPrincipal(
                    actor_id=actor,
                    permissions=tuple(permissions),
                )
            )
        ),
    )
    app.add_middleware(
        PrincipalMiddleware,
        actor=actor,
        permissions=permissions,
        actor_only=actor_only,
    )
    return TestClient(app), lifecycle_manager


def test_lifecycle_routes_fail_closed_when_manager_is_missing():
    client, _ = make_client(manager=False)
    with client:
        response = client.get("/api/v1/capabilities")
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "capability_lifecycle_unavailable"


def test_unauthorized_mutations_are_denied_before_missing_manager_is_resolved():
    client, _ = make_client(manager=False, permissions=())
    with client:
        responses = [
            client.post(f"/api/v1/capabilities/application.inspect_room/{operation}")
            for operation in ("activate", "disable", "revoke")
        ]
    assert [response.status_code for response in responses] == [403, 403, 403]
    assert all(
        response.json()["detail"]["error"] == "capability_lifecycle_forbidden"
        for response in responses
    )


def test_authorized_mutation_still_reports_missing_manager():
    client, _ = make_client(manager=False)
    with client:
        response = client.post("/api/v1/capabilities/application.inspect_room/disable")
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "capability_lifecycle_unavailable"


def test_lifecycle_routes_require_verified_principal_not_actor_state():
    client, _ = make_client(actor="operator", actor_only=True)
    with client:
        response = client.get("/api/v1/capabilities")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "authentication_required"


def test_lifecycle_routes_reject_missing_principal():
    client, _ = make_client(actor=None)
    with client:
        response = client.get("/api/v1/capabilities")
    assert response.status_code == 401


def test_lifecycle_routes_fail_closed_without_authorization_policy():
    client, _ = make_client(policy=False)
    with client:
        response = client.get("/api/v1/capabilities")
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == (
        "capability_lifecycle_authorization_unavailable"
    )


def test_read_permission_is_required_for_list_detail_and_inspection():
    client, _ = make_client(permissions=())
    with client:
        listed = client.get("/api/v1/capabilities")
        detail = client.get("/api/v1/capabilities/application.inspect_room")
        inspection = client.get("/api/v1/capabilities/inspection")
    assert listed.status_code == detail.status_code == inspection.status_code == 403


def test_each_mutation_requires_its_exact_permission():
    cases = (
        ("activate", "lifecycle_activate"),
        ("disable", "lifecycle_disable"),
        ("revoke", "lifecycle_revoke"),
    )
    for operation, permission_key in cases:
        other_permissions = tuple(
            permission
            for key, permission in CAPABILITY_PERMISSIONS.items()
            if key.startswith("lifecycle_") and key != permission_key
        )
        client, _ = make_client(permissions=other_permissions)
        with client:
            response = client.post(
                f"/api/v1/capabilities/application.inspect_room/{operation}"
            )
        assert response.status_code == 403


def test_list_and_detail_include_inactive_capability_records():
    client, _ = make_client()
    with client:
        disabled = client.post("/api/v1/capabilities/application.inspect_room/disable")
        listed = client.get("/api/v1/capabilities")
        detail = client.get("/api/v1/capabilities/application.inspect_room")
    assert disabled.status_code == 200
    assert listed.status_code == 200
    assert listed.json()["capabilities"][0]["status"] == "disabled"
    assert detail.status_code == 200
    assert detail.json()["capability"]["capability_id"] == "application.inspect_room"


def test_disable_then_activate_updates_planner_and_registry():
    client, manager = make_client()
    with client:
        disabled = client.post("/api/v1/capabilities/application.inspect_room/disable")
        activated = client.post("/api/v1/capabilities/application.inspect_room/activate")
    assert disabled.status_code == 200
    assert disabled.json()["result"]["status"] == "disabled"
    assert activated.status_code == 200
    assert activated.json()["result"]["status"] == "active"
    assert manager.registry.get("application.inspect_room").status == CapabilityStatus.ACTIVE


def test_revoke_is_terminal_over_api():
    client, _ = make_client()
    with client:
        revoked = client.post("/api/v1/capabilities/application.inspect_room/revoke")
        activated = client.post("/api/v1/capabilities/application.inspect_room/activate")
    assert revoked.status_code == 200
    assert revoked.json()["result"]["status"] == "revoked"
    assert activated.status_code == 409


def test_builtin_cannot_be_disabled_over_api():
    runtime, manager = make_runtime()
    builtin_action = Action(
        name="open_box",
        description="Open box",
        handler=lambda **kwargs: {"success": True},
    )
    manager.registry.register_builtin(builtin_action)
    manager.agent.actions["open_box"] = builtin_action
    manager.agent.configure_planner()
    app = create_app(
        runtime_factory=lambda: runtime,
        cognitive_capability_runtime_factory=(
            lambda app, cognitive_runtime: runtime.capability_runtime
        ),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
        capability_lifecycle_manager_factory=lambda app, runtime: manager,
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
        authentication_provider_factory=lambda app: StaticAuthenticationProvider(
            AuthenticatedPrincipal(actor_id="operator", permissions=ALL_LIFECYCLE_PERMISSIONS)
        ),
    )
    app.add_middleware(PrincipalMiddleware)
    with TestClient(app) as client:
        response = client.post("/api/v1/capabilities/builtin.open_box/disable")
    assert response.status_code == 422


def test_unknown_capability_returns_404():
    client, _ = make_client()
    with client:
        response = client.post("/api/v1/capabilities/application.unknown/disable")
    assert response.status_code == 404


def test_inspection_endpoint_reports_consistency():
    client, _ = make_client()
    with client:
        response = client.get("/api/v1/capabilities/inspection")
    assert response.status_code == 200
    assert response.json()["inspection"]["is_consistent"] is True


def test_policy_permission_catalog_is_used_for_mutation_authorization():
    permissions_seen = []

    class RecordingPolicy(AuthorizationPolicy):
        def require(self, principal, permission):
            permissions_seen.append((principal.actor_id, permission))
            return super().require(principal, permission)

    runtime, manager = make_runtime()
    app = create_app(
        runtime_factory=lambda: runtime,
        cognitive_capability_runtime_factory=(
            lambda app, cognitive_runtime: runtime.capability_runtime
        ),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
        capability_lifecycle_manager_factory=lambda app, runtime: manager,
        authorization_policy_factory=lambda app: RecordingPolicy(),
        authentication_provider_factory=lambda app: StaticAuthenticationProvider(
            AuthenticatedPrincipal(actor_id="operator", permissions=ALL_LIFECYCLE_PERMISSIONS)
        ),
    )
    app.add_middleware(PrincipalMiddleware)
    with TestClient(app) as client:
        response = client.post("/api/v1/capabilities/application.inspect_room/disable")
    assert response.status_code == 200
    assert permissions_seen == [("operator", CAPABILITY_PERMISSIONS["lifecycle_disable"])]


def test_lifecycle_manager_factory_receives_cognitive_runtime_not_execution_runtime():
    runtime, manager = make_runtime()
    captured = {}
    app = create_app(
        runtime_factory=lambda: runtime,
        cognitive_capability_runtime_factory=(
            lambda app, cognitive_runtime: runtime.capability_runtime
        ),
        execution_runtime_factory=lambda: object(),
        capability_store_factory=lambda: object(),
        capability_lifecycle_manager_factory=lambda app, value: (
            captured.update(runtime=value) or manager
        ),
        authorization_policy_factory=lambda app: AuthorizationPolicy(),
    )
    with TestClient(app):
        assert captured["runtime"] is runtime


def test_lifecycle_authentication_uses_provider_not_preexisting_actor_state():
    client, _ = make_client(actor="operator", actor_only=True)
    with client:
        response = client.get("/api/v1/capabilities")
    assert response.status_code == 401
    assert response.json()["detail"]["error"] == "authentication_required"
