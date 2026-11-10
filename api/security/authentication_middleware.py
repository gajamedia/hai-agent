from __future__ import annotations

import logging
import re

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from api.security.authentication import (
    AuthenticationError,
    AuthenticationProviderUnavailable,
    authenticate_request,
    attach_authenticated_principal,
)
from api.security.authorization import AuthorizationPolicy, AuthorizationDenied

logger = logging.getLogger("hai_agent.authentication")

_CAPABILITY_PREFIXES = (
    "/api/v1/capability-proposals",
    "/api/v1/capabilities",
)
_API_PREFIX = "/api/v1"

# Exact route contracts. New API routes must be added here before they can be
# used when HAI_AUTH_ENABLED=true. Capability routes retain their endpoint-level
# permission checks because each operation already has a distinct permission.
_ROUTE_PERMISSION_RULES: tuple[tuple[str, str, str], ...] = (
    ("GET", r"^/api/v1/execution/state$", "execution:state:read"),
    ("PUT", r"^/api/v1/execution/state$", "execution:state:write"),
    ("POST", r"^/api/v1/execution/run$", "execution:action:execute"),
    ("POST", r"^/api/v1/execution/plan-run$", "execution:plan:execute"),
    ("POST", r"^/api/v1/execution/reset$", "execution:state:reset"),
    ("POST", r"^/api/v1/execution/reconcile$", "execution:reconciliation:write"),
    ("POST", r"^/api/v1/capability-reads$", "capability:read:execute"),
    ("POST", r"^/api/v1/trusted-capability-invocations$", "capability:canonical:invoke"),
    ("POST", r"^/api/v1/cognitive/run$", "cognitive:cycle:run"),
    ("GET", r"^/api/v1/strategies$", "strategy:registry:read"),
    ("GET", r"^/api/v1/telemetry/capabilities(?:/sources)?$", "telemetry:capability:read"),
    ("GET", r"^/api/v1/database-onboarding/schema$", "database:onboarding:read"),
    ("POST", r"^/api/v1/database-onboarding/allowlists$", "database:onboarding:write"),
    ("POST", r"^/api/v1/autonomous-testing/batches$", "autonomous-testing:batch:create"),
    ("GET", r"^/api/v1/autonomous-testing/batches$", "autonomous-testing:batch:read"),
    ("POST", r"^/api/v1/autonomous-testing/batches/[^/]+/run$", "autonomous-testing:batch:run"),
    ("POST", r"^/api/v1/autonomous-testing/batches/[^/]+/reconcile$", "autonomous-testing:batch:reconcile"),
    ("GET", r"^/api/v1/autonomous-testing/batches/[^/]+/(?:results|reconciliation)$", "autonomous-testing:batch:read"),
    ("GET", r"^/api/v1/autonomous-testing/batches/[^/]+$", "autonomous-testing:batch:read"),
    # Data reports: plan, run, cancel and download all need the report permission.
    # Reading the business data itself is enforced again, per read, by the gateway.
    ("POST", r"^/api/v1/table-reports$", "report:table:run"),
    ("POST", r"^/api/v1/table-reports/plans$", "report:table:run"),
    ("POST", r"^/api/v1/table-reports/plans/[^/]+/run$", "report:table:run"),
    ("DELETE", r"^/api/v1/table-reports/plans/[^/]+$", "report:table:run"),
    ("GET", r"^/api/v1/table-reports/reports/[^/]+/download$", "report:table:run"),
    # Connecting databases and approving tables.
    ("GET", r"^/api/v1/database-connections$", "database:connection:manage"),
    ("POST", r"^/api/v1/database-connections$", "database:connection:manage"),
    ("GET", r"^/api/v1/database-connections/profiles$", "database:connection:manage"),
    ("GET", r"^/api/v1/database-connections/profiles/[^/]+/databases$", "database:connection:manage"),
    ("GET", r"^/api/v1/database-connections/files$", "database:connection:manage"),
    ("GET", r"^/api/v1/database-connections/[^/]+/schema$", "database:connection:manage"),
    ("POST", r"^/api/v1/database-connections/[^/]+/approve$", "database:connection:manage"),
    ("POST", r"^/api/v1/database-connections/[^/]+/refresh$", "database:connection:manage"),
    ("POST", r"^/api/v1/database-connections/servers$", "database:connection:manage"),
    ("POST", r"^/api/v1/database-connections/servers/inspect$", "database:connection:manage"),
    ("POST", r"^/api/v1/database-connections/servers/auto$", "database:connection:manage"),
    ("DELETE", r"^/api/v1/database-connections/servers/[^/]+$", "database:connection:manage"),
    ("DELETE", r"^/api/v1/database-connections/[^/]+$", "database:connection:manage"),
    # Data changes: planning/cancelling and running are the writer's; approving is a second person's.
    ("GET", r"^/api/v1/data-changes/config$", "data:change:plan"),
    ("POST", r"^/api/v1/data-changes/plans$", "data:change:plan"),
    ("GET", r"^/api/v1/data-changes/plans/[^/]+$", "data:change:plan"),
    ("DELETE", r"^/api/v1/data-changes/plans/[^/]+$", "data:change:plan"),
    ("POST", r"^/api/v1/data-changes/plans/[^/]+/run$", "data:change:execute"),
    ("GET", r"^/api/v1/data-changes/approvals$", "data:change:approve"),
    ("POST", r"^/api/v1/data-changes/approvals/[^/]+/(?:approve|reject)$", "data:change:approve"),
    # Recording new data (insert only): one permission for planning, cancelling and running.
    ("GET", r"^/api/v1/data-entries/config$", "data:entry:record"),
    ("GET", r"^/api/v1/data-entries/databases$", "data:entry:record"),
    ("POST", r"^/api/v1/data-entries/plans$", "data:entry:record"),
    ("GET", r"^/api/v1/data-entries/plans/[^/]+$", "data:entry:record"),
    ("DELETE", r"^/api/v1/data-entries/plans/[^/]+$", "data:entry:record"),
    ("POST", r"^/api/v1/data-entries/plans/[^/]+/run$", "data:entry:record"),
    # Creating databases: one permission for planning and running; a fresh login is checked by the handler.
    ("GET", r"^/api/v1/database-creation/config$", "database:create:manage"),
    ("POST", r"^/api/v1/database-creation/plans$", "database:create:manage"),
    ("GET", r"^/api/v1/database-creation/plans/[^/]+$", "database:create:manage"),
    ("DELETE", r"^/api/v1/database-creation/plans/[^/]+$", "database:create:manage"),
    ("POST", r"^/api/v1/database-creation/plans/[^/]+/run$", "database:create:manage"),
    # Task history: your own tasks. The permission comes with the roles that run features; the handler
    # then shows only your tasks, and only for operations you may still use.
    ("GET", r"^/api/v1/tasks$", "task:history:read"),
    ("GET", r"^/api/v1/tasks/[^/]+$", "task:history:read"),
    # Engine requests: anyone who may ask can list (the handler narrows it to their own
    # unless they are a reviewer); only reviewers can change a request.
    ("POST", r"^/api/v1/database-engine-requests$", "database:engine:request"),
    ("GET", r"^/api/v1/database-engine-requests$", "database:engine:request"),
    ("PATCH", r"^/api/v1/database-engine-requests/[^/]+$", "database:engine:review"),
)

_COMPILED_RULES = tuple(
    (method, re.compile(pattern), permission)
    for method, pattern, permission in _ROUTE_PERMISSION_RULES
)


def required_permission_for_route(method: str, path: str) -> str | None:
    """Return the exact permission for a protected API route, or None."""
    normalized_method = (method or "").upper()
    for rule_method, pattern, permission in _COMPILED_RULES:
        if normalized_method == rule_method and pattern.fullmatch(path):
            return permission
    return None


class APIAuthenticationAuthorizationMiddleware:
    """Central authentication boundary for the versioned API.

    In local development with HAI_AUTH_ENABLED=false, non-capability routes
    remain available for compatibility. Capability routes preserve their
    existing always-authenticated behavior. When authentication is enabled,
    every /api/v1 route must be explicitly mapped and authorized; unmapped
    routes fail closed. Production deployments should set HAI_AUTH_REQUIRED=true.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @staticmethod
    def _is_capability_path(path: str) -> bool:
        return any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in _CAPABILITY_PREFIXES
        )

    @staticmethod
    def _is_api_path(path: str) -> bool:
        return path == _API_PREFIX or path.startswith(_API_PREFIX + "/")

    @staticmethod
    async def _respond(
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        error: str,
        message: str,
        permission: str | None = None,
    ) -> None:
        detail = {"error": error, "message": message}
        if permission is not None:
            detail["permission"] = permission
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        if method == "OPTIONS":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if not self._is_api_path(path):
            await self.app(scope, receive, send)
            return

        app = scope.get("app")
        app_state = getattr(app, "state", None)
        auth_enabled = bool(getattr(app_state, "auth_enabled", False))
        capability_path = self._is_capability_path(path)

        # Preserve local development compatibility for non-capability routes.
        # Production must opt into the startup guard HAI_AUTH_REQUIRED=true.
        if not auth_enabled and not capability_path:
            await self.app(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state.pop("authenticated_principal", None)
        state.pop("capability_actor", None)
        request = Request(scope, receive=receive)

        try:
            principal = await authenticate_request(request)
            attach_authenticated_principal(request, principal)
        except AuthenticationProviderUnavailable:
            logger.error("Authentication provider unavailable: method=%s path=%s", method, path)
            await self._respond(
                scope, receive, send,
                status_code=503,
                error="authentication_provider_unavailable",
                message="Authentication service is temporarily unavailable.",
            )
            return
        except AuthenticationError as exc:
            logger.info(
                "Authentication rejected: method=%s path=%s error_type=%s",
                method, path, type(exc).__name__,
            )
            await self._respond(
                scope, receive, send,
                status_code=401,
                error="authentication_required",
                message="A verified authenticated principal is required.",
            )
            return
        except Exception:
            logger.exception("Unexpected authentication middleware failure: method=%s path=%s", method, path)
            await self._respond(
                scope, receive, send,
                status_code=500,
                error="authentication_internal_error",
                message="An internal authentication error occurred.",
            )
            return

        # Capability endpoints perform operation-specific PBAC inside their
        # handlers. All other API routes use this central route permission map.
        if not capability_path:
            permission = required_permission_for_route(method, path)
            if permission is None:
                await self._respond(
                    scope, receive, send,
                    status_code=403,
                    error="authorization_route_unmapped",
                    message="This API route has no explicit authorization contract.",
                )
                return

            policy = getattr(app_state, "authorization_policy", None)
            if not isinstance(policy, AuthorizationPolicy):
                await self._respond(
                    scope, receive, send,
                    status_code=503,
                    error="authorization_policy_unavailable",
                    message="The application authorization policy is not configured.",
                )
                return
            try:
                policy.require(principal, permission)
            except AuthorizationDenied:
                await self._respond(
                    scope, receive, send,
                    status_code=403,
                    error="permission_denied",
                    message="The authenticated principal does not have the required permission.",
                    permission=permission,
                )
                return
            except Exception:
                logger.exception("Authorization policy evaluation failed for %s %s", method, path)
                await self._respond(
                    scope, receive, send,
                    status_code=503,
                    error="authorization_policy_invalid",
                    message="The configured authorization policy could not evaluate this request.",
                )
                return

        await self.app(scope, receive, send)


# Backward-compatible import name used by existing application code/tests.
CapabilityAuthenticationMiddleware = APIAuthenticationAuthorizationMiddleware

__all__ = [
    "APIAuthenticationAuthorizationMiddleware",
    "CapabilityAuthenticationMiddleware",
    "required_permission_for_route",
]
