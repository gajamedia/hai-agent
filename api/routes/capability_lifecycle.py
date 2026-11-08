from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityRegistrationError,
    CapabilityStatus,
    CapabilityValidationError,
)
from agent.llm.capability_lifecycle_manager import (
    CapabilityLifecycleManagerError,
    CapabilityLifecycleSynchronizationError,
    CapabilityLifecycleValidationError,
)
from api.security.authentication import (
    AuthenticationError,
    AuthenticatedPrincipal,
    get_attached_principal,
)
from api.security.authorization import (
    CAPABILITY_PERMISSIONS,
    AuthorizationDenied,
    AuthorizationPolicy,
)

router = APIRouter(prefix="/capabilities", tags=["capability-lifecycle"])


def _get_manager(request: Request) -> Any:
    manager = getattr(request.app.state, "capability_lifecycle_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "capability_lifecycle_unavailable",
                "message": (
                    "Capability lifecycle management is not configured. "
                    "Supply an application-owned lifecycle manager factory."
                ),
            },
        )
    return manager


def _get_principal(request: Request) -> AuthenticatedPrincipal:
    try:
        return get_attached_principal(request)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "authenticated_principal_required",
                "message": (
                    "A verified AuthenticatedPrincipal must be attached by "
                    "trusted authentication middleware."
                ),
            },
        ) from exc


def _get_authorization_policy(request: Request) -> AuthorizationPolicy:
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "capability_lifecycle_authorization_unavailable",
                "message": (
                    "No valid application-owned authorization policy is configured. "
                    "Capability lifecycle access is disabled."
                ),
            },
        )
    return policy


def _authorize(
    request: Request,
    principal: AuthenticatedPrincipal,
    permission_key: str,
) -> None:
    policy = _get_authorization_policy(request)
    permission = CAPABILITY_PERMISSIONS[permission_key]
    try:
        policy.require(principal, permission)
    except AuthorizationDenied as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "capability_lifecycle_forbidden",
                "permission": permission,
                "message": "The authenticated principal lacks the required permission.",
            },
        ) from exc


def _record_payload(record: Any) -> dict[str, Any]:
    return {
        "capability_id": record.capability_id,
        "action_name": record.action_name,
        "description": record.description,
        "trust_level": record.trust_level.value,
        "risk_level": record.risk_level,
        "permissions": list(record.permissions),
        "status": record.status.value,
        "version": record.version,
    }


def _inspection_payload(report: Any) -> dict[str, Any]:
    adapter = report.adapter_report
    return {
        "is_consistent": report.is_consistent,
        "registry_version": report.registry_version,
        "planner_sync_hook_attached": report.planner_sync_hook_attached,
        "planner_action_names": list(report.planner_action_names),
        "active_registry_action_names": list(report.active_registry_action_names),
        "missing_active_action_names": list(report.missing_active_action_names),
        "inactive_planner_action_names": list(report.inactive_planner_action_names),
        "adapter": {
            "registry_version": adapter.registry_version,
            "ready_action_names": list(adapter.ready_action_names),
            "already_registered_action_names": list(adapter.already_registered_action_names),
            "inactive_registered_action_names": list(adapter.inactive_registered_action_names),
            "conflicts": list(adapter.conflicts),
            "summary": dict(adapter.summary),
        },
    }


def _raise_lifecycle_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CapabilityNotFoundError):
        return HTTPException(
            status_code=404,
            detail={"error": "capability_not_found", "message": str(exc)},
        )
    if isinstance(exc, (CapabilityLifecycleValidationError, CapabilityValidationError)):
        return HTTPException(
            status_code=422,
            detail={"error": "capability_lifecycle_validation_failed", "message": str(exc)},
        )
    if isinstance(exc, CapabilityRegistrationError):
        return HTTPException(
            status_code=409,
            detail={"error": "capability_lifecycle_conflict", "message": str(exc)},
        )
    if isinstance(exc, CapabilityLifecycleSynchronizationError):
        return HTTPException(
            status_code=409,
            detail={"error": "capability_lifecycle_sync_failed", "message": str(exc)},
        )
    if isinstance(exc, CapabilityLifecycleManagerError):
        return HTTPException(
            status_code=409,
            detail={"error": "capability_lifecycle_failed", "message": str(exc)},
        )
    return HTTPException(
        status_code=500,
        detail={
            "error": "capability_lifecycle_internal_error",
            "message": "Capability lifecycle operation failed.",
        },
    )


# Authenticate and authorize before resolving lifecycle infrastructure. This
# ensures unauthorized callers receive 403 even while the manager is absent.
# Keep the static route before /{capability_id}.
@router.get("/inspection")
def inspect_capability_lifecycle(request: Request) -> dict[str, Any]:
    principal = _get_principal(request)
    _authorize(request, principal, "lifecycle_read")
    manager = _get_manager(request)
    try:
        report = manager.inspect()
    except Exception as exc:
        raise _raise_lifecycle_error(exc) from exc
    return {"status": "ok", "inspection": _inspection_payload(report)}


@router.get("")
def list_capabilities(request: Request) -> dict[str, Any]:
    principal = _get_principal(request)
    _authorize(request, principal, "lifecycle_read")
    manager = _get_manager(request)
    try:
        records = manager.registry.list_all()
    except Exception as exc:
        raise _raise_lifecycle_error(exc) from exc
    return {
        "status": "ok",
        "count": len(records),
        "capabilities": [_record_payload(item) for item in records],
    }


@router.get("/{capability_id}")
def get_capability(capability_id: str, request: Request) -> dict[str, Any]:
    principal = _get_principal(request)
    _authorize(request, principal, "lifecycle_read")
    manager = _get_manager(request)
    try:
        record = manager.registry.get(capability_id, include_inactive=True)
    except Exception as exc:
        raise _raise_lifecycle_error(exc) from exc
    return {"status": "ok", "capability": _record_payload(record)}


def _transition(
    capability_id: str,
    request: Request,
    permission_key: str,
    target_status: CapabilityStatus,
) -> dict[str, Any]:
    principal = _get_principal(request)
    _authorize(request, principal, permission_key)
    manager = _get_manager(request)
    try:
        result = manager.transition(capability_id, target_status)
    except Exception as exc:
        raise _raise_lifecycle_error(exc) from exc
    return {
        "status": "ok",
        "result": {
            "capability_id": result.capability_id,
            "action_name": result.action_name,
            "previous_status": result.previous_status.value,
            "status": result.status.value,
            "registry_version": result.registry_version,
            "record_version": result.record_version,
            "changed": result.changed,
            "planner_action_names": list(result.planner_action_names),
        },
    }


@router.post("/{capability_id}/activate")
def activate_capability(capability_id: str, request: Request) -> dict[str, Any]:
    return _transition(capability_id, request, "lifecycle_activate", CapabilityStatus.ACTIVE)


@router.post("/{capability_id}/disable")
def disable_capability(capability_id: str, request: Request) -> dict[str, Any]:
    return _transition(capability_id, request, "lifecycle_disable", CapabilityStatus.DISABLED)


@router.post("/{capability_id}/revoke")
def revoke_capability(capability_id: str, request: Request) -> dict[str, Any]:
    return _transition(capability_id, request, "lifecycle_revoke", CapabilityStatus.REVOKED)


__all__ = ["router"]
