from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent.world.environment_capability_discovery import (
    EnvironmentCapabilityDiscoveryError,
)
from api.security.authentication import (
    AuthenticationError, AuthenticatedPrincipal, get_attached_principal,
)
from api.security.authorization import (
    CAPABILITY_PERMISSIONS, AuthorizationDenied, AuthorizationPolicy,
)

router = APIRouter(prefix="/capabilities/environment-workspace", tags=["environment-workspace"])


def _principal(request: Request) -> AuthenticatedPrincipal:
    try:
        return get_attached_principal(request)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail={
            "error": "authenticated_principal_required",
            "message": "A verified AuthenticatedPrincipal is required.",
        }) from exc


def _authorize(request: Request, principal: AuthenticatedPrincipal) -> None:
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise HTTPException(status_code=503, detail={
            "error": "environment_workspace_authorization_unavailable",
            "message": "No valid application-owned authorization policy is configured.",
        })
    permission = CAPABILITY_PERMISSIONS["environment_workspace_read"]
    try:
        policy.require(principal, permission)
    except AuthorizationDenied as exc:
        raise HTTPException(status_code=403, detail={
            "error": "environment_workspace_forbidden",
            "permission": permission,
        }) from exc


def _descriptor(value: Any) -> dict[str, Any]:
    d = value.descriptor
    return {
        "environment_type": d.environment_type, "domain": d.domain,
        "mode": getattr(d.mode, "value", d.mode), "version": d.version,
        "supports_observation": d.supports_observation,
        "supports_reset": d.supports_reset,
        "supports_idempotency": d.supports_idempotency,
        "metadata": dict(d.metadata),
    }


@router.get("/providers")
def list_environment_providers(request: Request) -> dict[str, Any]:
    principal = _principal(request)
    _authorize(request, principal)
    manager = getattr(request.app.state, "environment_lifecycle_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail={"error": "environment_lifecycle_unavailable"})
    providers = manager.discover_providers()
    return {"items": [{
        "provider_id": item.provider_id, "descriptor": _descriptor(item),
        "active_instance_count": item.active_instance_count,
    } for item in providers], "count": len(providers)}


@router.get("/instances")
def list_environment_instances(request: Request, include_closed: bool = False) -> dict[str, Any]:
    principal = _principal(request)
    _authorize(request, principal)
    manager = getattr(request.app.state, "environment_lifecycle_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail={"error": "environment_lifecycle_unavailable"})
    items = manager.list_instances(include_closed=include_closed)
    return {"items": [{
        "instance_id": item.instance_id, "provider_id": item.provider_id,
        "status": item.status.value, "descriptor": _descriptor(item),
    } for item in items], "count": len(items)}


@router.get("/instances/{instance_id}/capabilities")
def discover_instance_capabilities(instance_id: str, request: Request) -> dict[str, Any]:
    principal = _principal(request)
    _authorize(request, principal)
    discovery = getattr(request.app.state, "environment_capability_discovery", None)
    if discovery is None:
        raise HTTPException(status_code=503, detail={
            "error": "environment_capability_discovery_unconfigured",
            "message": "Configure HAI_ENVIRONMENT_CAPABILITY_BINDINGS with trusted application-owned JSON bindings.",
        })
    try:
        report = discovery.discover(instance_id)
    except EnvironmentCapabilityDiscoveryError as exc:
        raise HTTPException(status_code=404, detail={
            "error": "environment_capability_discovery_failed", "message": str(exc),
        }) from exc
    return {
        "environment_instance_id": report.environment_instance_id,
        "environment_type": report.environment_type,
        "registry_version": report.registry_version,
        "available_capability_ids": list(report.available_capability_ids),
        "excluded_capability_ids": list(report.excluded_capability_ids),
        "candidates": [{
            "capability_id": item.capability_id, "action_name": item.action_name,
            "description": item.description, "risk_level": item.risk_level,
            "permissions": list(item.permissions), "state": item.state.value,
        } for item in report.candidates],
    }
