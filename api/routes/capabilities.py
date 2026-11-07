from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request, status

from agent.llm.capability_proposal import (
    CapabilityProposal,
    CapabilityProposalError,
    CapabilityProposalValidationError,
    ProposalCategory,
    ProposalStatus,
)
from agent.llm.capability_proposal_store import CapabilityProposalStore
from agent.llm.capability_approval_manager import (
    CapabilityApprovalAuthorizationError,
    CapabilityApprovalError,
    CapabilityApprovalRegistrationError,
    CapabilityApprovalValidationError,
)
from agent.llm.capability_proposal_validator import CapabilityProposalValidator
from api.security.authentication import (
    AuthenticationError,
    get_attached_principal,
)
from api.security.authorization import (
    CAPABILITY_PERMISSIONS,
    AuthorizationDenied,
    AuthorizationPolicy,
    AuthorizationPolicyConfigurationError,
)

router = APIRouter(prefix="/capability-proposals", tags=["capability-proposals"])

_ALLOWED_SUBMISSION_FIELDS = frozenset({
    "capability_id",
    "action_name",
    "description",
    "category",
    "risk_level",
    "permissions",
    "preconditions",
    "effects",
    "parameters",
})


def _authorize(request: Request, permission_key: str) -> str:
    """Enforce PBAC for the principal attached by central capability middleware."""
    try:
        principal = get_attached_principal(request)
    except AuthenticationError as exc:
        # This is a defensive invariant check. The central middleware should
        # reject the request before the route is reached.
        raise HTTPException(
            status_code=401,
            detail={
                "error": "authentication_required",
                "message": "A verified authenticated principal is required.",
            },
        ) from exc

    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "authorization_policy_unavailable",
                "message": "The application authorization policy is not configured.",
            },
        )

    permission = CAPABILITY_PERMISSIONS[permission_key]
    try:
        policy.require(principal, permission)
    except AuthorizationDenied as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "permission_denied",
                "permission": permission,
                "message": "The authenticated principal does not have the required permission.",
            },
        ) from exc
    except AuthorizationPolicyConfigurationError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "authorization_policy_invalid",
                "message": "The configured authorization policy is invalid.",
            },
        ) from exc

    return principal.actor_id


def _get_store(request: Request) -> CapabilityProposalStore:
    store = getattr(request.app.state, "capability_proposal_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "capability_proposal_store_unavailable",
                "message": "The persistent capability proposal store is not initialized.",
            },
        )
    return store


def _get_manager(request: Request) -> Any:
    manager = getattr(request.app.state, "capability_approval_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "capability_approval_unavailable",
                "message": (
                    "Capability approval is not configured. Configure an application-owned "
                    "approval manager and trusted authorization policy before enabling it."
                ),
            },
        )
    return manager


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def _proposal_payload(proposal: CapabilityProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "capability_id": proposal.capability_id,
        "action_name": proposal.action_name,
        "description": proposal.description,
        "category": proposal.category.value,
        "risk_level": proposal.risk_level,
        "permissions": list(proposal.permissions),
        "preconditions": _json_value(proposal.preconditions),
        "effects": _json_value(proposal.effects),
        "parameters": _json_value(proposal.parameters),
        "status": proposal.status.value,
        "created_at": proposal.created_at,
        "approved_by": proposal.approved_by,
        "rejected_by": proposal.rejected_by,
        "rejection_reason": proposal.rejection_reason,
        "registered_by": proposal.registered_by,
        "version": proposal.version,
    }


def _validate_submission_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_proposal", "message": "Request body must be a JSON object."},
        )

    unknown = sorted(set(payload) - _ALLOWED_SUBMISSION_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unsupported_proposal_fields",
                "message": (
                    "Proposal submission accepts declarative metadata only. "
                    "Unsupported fields: " + ", ".join(unknown)
                ),
            },
        )

    required = ("capability_id", "action_name", "description", "category", "risk_level")
    missing = [field for field in required if field not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_proposal_fields",
                "message": "Missing required fields: " + ", ".join(missing),
            },
        )

    for field in ("capability_id", "action_name", "description", "risk_level"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_proposal_field",
                    "field": field,
                    "message": f"{field} must be a non-empty string.",
                },
            )

    try:
        category = ProposalCategory(payload["category"])
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_proposal_category",
                "message": "category must be 'read_only' or 'state_changing'.",
            },
        ) from exc

    permissions = payload.get("permissions", [])
    if not isinstance(permissions, list) or any(not isinstance(item, str) for item in permissions):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_proposal_permissions",
                "message": "permissions must be an array of strings.",
            },
        )

    normalized: dict[str, Any] = {}
    for field in ("preconditions", "effects", "parameters"):
        value = payload.get(field, {})
        if not isinstance(value, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_proposal_metadata",
                    "field": field,
                    "message": f"{field} must be a JSON object.",
                },
            )
        normalized[field] = value

    return {
        "capability_id": payload["capability_id"].strip(),
        "action_name": payload["action_name"].strip(),
        "description": payload["description"].strip(),
        "category": category,
        "risk_level": payload["risk_level"].strip(),
        "permissions": tuple(permissions),
        **normalized,
    }


def _raise_store_error(exc: Exception) -> HTTPException:
    message = str(exc)
    if "not found" in message.lower():
        return HTTPException(
            status_code=404,
            detail={"error": "capability_proposal_not_found", "message": message},
        )
    return HTTPException(
        status_code=409,
        detail={"error": "capability_proposal_conflict", "message": message},
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def submit_capability_proposal(payload: dict[str, Any], request: Request) -> dict[str, Any]:
    """Submit declarative metadata; no executable handler or source is accepted."""
    _authorize(request, "proposal_submit")
    values = _validate_submission_payload(payload)
    proposal = CapabilityProposal(proposal_id=uuid4().hex, **values)
    store = _get_store(request)
    manager = getattr(request.app.state, "capability_approval_manager", None)

    try:
        if manager is not None:
            stored = manager.submit(proposal)
        else:
            validator = getattr(store, "validator", None) or CapabilityProposalValidator()
            validator.validate(proposal)
            stored = store.submit(proposal)
    except CapabilityProposalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_capability_proposal", "message": str(exc)},
        ) from exc
    except (CapabilityProposalError, ValueError) as exc:
        raise _raise_store_error(exc) from exc
    except CapabilityApprovalError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "capability_proposal_rejected", "message": str(exc)},
        ) from exc

    return {"status": "ok", "proposal": _proposal_payload(stored)}


@router.get("")
def list_capability_proposals(
    request: Request,
    proposal_status: str | None = Query(default=None, alias="status"),
) -> dict[str, Any]:
    _authorize(request, "proposal_read")
    store = _get_store(request)
    try:
        proposals = store.list_all()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "capability_proposal_read_failed", "message": "Unable to read proposal records."},
        ) from exc

    if proposal_status is not None:
        try:
            selected = ProposalStatus(proposal_status)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_proposal_status",
                    "message": "status must be pending, approved, rejected, or registered.",
                },
            ) from exc
        proposals = tuple(item for item in proposals if item.status == selected)

    return {
        "status": "ok",
        "count": len(proposals),
        "proposals": [_proposal_payload(item) for item in proposals],
    }


@router.get("/{proposal_id}")
def get_capability_proposal(proposal_id: str, request: Request) -> dict[str, Any]:
    _authorize(request, "proposal_read")
    store = _get_store(request)
    try:
        proposal = store.get(proposal_id)
    except CapabilityProposalError as exc:
        raise _raise_store_error(exc) from exc
    return {"status": "ok", "proposal": _proposal_payload(proposal)}


@router.post("/{proposal_id}/approve")
def approve_capability_proposal(proposal_id: str, request: Request) -> dict[str, Any]:
    actor = _authorize(request, "proposal_approve")
    manager = _get_manager(request)
    try:
        proposal = manager.approve(proposal_id, approved_by=actor)
    except CapabilityApprovalAuthorizationError as exc:
        raise HTTPException(
            status_code=403,
            detail={"error": "capability_approval_forbidden", "message": str(exc)},
        ) from exc
    except CapabilityApprovalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "capability_approval_validation_failed", "message": str(exc)},
        ) from exc
    except CapabilityProposalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_capability_proposal", "message": str(exc)},
        ) from exc
    except CapabilityProposalError as exc:
        raise _raise_store_error(exc) from exc
    except CapabilityApprovalError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "capability_approval_conflict", "message": str(exc)},
        ) from exc
    return {"status": "ok", "proposal": _proposal_payload(proposal)}


@router.post("/{proposal_id}/reject")
def reject_capability_proposal(
    proposal_id: str,
    payload: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    actor = _authorize(request, "proposal_reject")
    manager = _get_manager(request)
    reason = payload.get("reason") if isinstance(payload, dict) else None
    if not isinstance(reason, str) or not reason.strip():
        raise HTTPException(
            status_code=422,
            detail={"error": "rejection_reason_required", "message": "A non-empty reason is required."},
        )
    try:
        proposal = manager.reject(proposal_id, rejected_by=actor, reason=reason)
    except CapabilityApprovalAuthorizationError as exc:
        raise HTTPException(
            status_code=403,
            detail={"error": "capability_approval_forbidden", "message": str(exc)},
        ) from exc
    except CapabilityApprovalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "capability_rejection_validation_failed", "message": str(exc)},
        ) from exc
    except CapabilityProposalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_capability_proposal", "message": str(exc)},
        ) from exc
    except CapabilityProposalError as exc:
        raise _raise_store_error(exc) from exc
    except CapabilityApprovalError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "capability_rejection_conflict", "message": str(exc)},
        ) from exc
    return {"status": "ok", "proposal": _proposal_payload(proposal)}


@router.post("/{proposal_id}/register")
def register_capability_proposal(proposal_id: str, request: Request) -> dict[str, Any]:
    """Register only approved read-only proposals through the trusted manager."""
    actor = _authorize(request, "proposal_register")
    manager = _get_manager(request)
    try:
        record = manager.register_approved_read_only(proposal_id, registered_by=actor)
    except CapabilityApprovalAuthorizationError as exc:
        raise HTTPException(
            status_code=403,
            detail={"error": "capability_registration_forbidden", "message": str(exc)},
        ) from exc
    except CapabilityApprovalRegistrationError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "capability_registration_failed", "message": str(exc)},
        ) from exc
    except CapabilityApprovalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "capability_registration_validation_failed", "message": str(exc)},
        ) from exc
    except CapabilityProposalValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_capability_proposal", "message": str(exc)},
        ) from exc
    except CapabilityProposalError as exc:
        raise _raise_store_error(exc) from exc
    except CapabilityApprovalError as exc:
        raise HTTPException(
            status_code=409,
            detail={"error": "capability_registration_conflict", "message": str(exc)},
        ) from exc

    return {
        "status": "ok",
        "capability": {
            "capability_id": record.capability_id,
            "action_name": record.action_name,
            "description": record.description,
            "trust_level": record.trust_level.value,
            "risk_level": record.risk_level,
            "permissions": list(record.permissions),
            "status": record.status.value,
            "version": record.version,
        },
    }


__all__ = ["router"]
