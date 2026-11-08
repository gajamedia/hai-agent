from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.encoders import jsonable_encoder

from api.security.authentication import AuthenticationError, get_attached_principal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy
from api.security.trusted_execution_context import TrustedExecutionContext
from agent.llm.agent_execution_boundary import AgentExecutionRequest
from agent.llm.capability_execution_contracts import (
    CapabilityAdapterError,
    CapabilityAuthorizationError,
    CapabilityReadResult,
    CapabilityRequestValidationError,
    CapabilityResolutionError,
)
from agent.llm.execution_security import ExecutionSecurityError
from agent.llm.traced_authenticated_capability_execution_gateway import CapabilityTracePersistenceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trusted-capability-invocations", tags=["trusted-capability-invocation"])
_PERMISSION = "capability:canonical:invoke"
_ALLOWED_FIELDS = frozenset({"action", "arguments"})


def _principal(request: Request):
    try:
        return get_attached_principal(request)
    except AuthenticationError as exc:
        raise HTTPException(401, detail={
            "error": "authentication_required",
            "message": "A verified authenticated principal is required.",
        }) from exc


def _authorize(request: Request, principal) -> None:
    policy = getattr(request.app.state, "authorization_policy", None)
    if not isinstance(policy, AuthorizationPolicy):
        raise HTTPException(503, detail={
            "error": "authorization_policy_unavailable",
            "message": "The application authorization policy is not configured.",
        })
    try:
        policy.require(principal, _PERMISSION)
    except AuthorizationDenied as exc:
        raise HTTPException(403, detail={
            "error": "permission_denied",
            "message": "The authenticated principal lacks canonical invocation permission.",
            "permission": _PERMISSION,
        }) from exc
    except Exception as exc:
        logger.exception("Canonical invocation authorization policy failed.")
        raise HTTPException(503, detail={
            "error": "authorization_policy_invalid",
            "message": "The configured authorization policy could not evaluate this request.",
        }) from exc


def _parse_payload(payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        raise HTTPException(422, detail={
            "error": "invalid_canonical_invocation",
            "message": "Unsupported request field(s): " + ", ".join(sorted(map(str, unknown))),
        })
    action = payload.get("action")
    arguments = payload.get("arguments", {})
    if not isinstance(action, str) or not action.strip():
        raise HTTPException(422, detail={
            "error": "invalid_canonical_invocation",
            "message": "action must be a non-empty string.",
        })
    if not isinstance(arguments, Mapping):
        raise HTTPException(422, detail={
            "error": "invalid_canonical_invocation",
            "message": "arguments must be a JSON object.",
        })
    return action.strip(), dict(arguments)


@router.post("")
def invoke_trusted_capability(request: Request, payload: dict[str, Any] = Body(...)):
    """Invoke exactly one server-mapped trusted capability through the canonical boundary."""
    principal = _principal(request)
    _authorize(request, principal)
    action, arguments = _parse_payload(payload)

    state = request.app.state
    port = getattr(state, "trusted_capability_invocation_port", None)
    if port is None or not callable(getattr(port, "supports_action", None)):
        raise HTTPException(503, detail={
            "error": "trusted_invocation_unavailable",
            "message": "The trusted capability invocation port is not configured.",
        })
    if not port.supports_action(action):
        raise HTTPException(404, detail={
            "error": "trusted_action_not_found",
            "message": "The requested trusted action is not configured.",
        })

    agent = getattr(getattr(state, "cognitive_runtime", None), "agent", None)
    if agent is None:
        raise HTTPException(503, detail={
            "error": "canonical_agent_unavailable",
            "message": "The application cognitive Agent is not available.",
        })

    try:
        from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI

        # Reuse the Agent's configured policy. An absent or unconfigured policy
        # keeps the canonical boundary fail-closed; this endpoint creates no approval.
        canonical_api = AgentCanonicalSingleActionAPI.for_agent(
            agent,
            decision_factory=lambda _request: None,
            security_policy=getattr(agent, "execution_security_policy", None),
            trusted_capability_invocation_port=port,
        )
        execution_context = TrustedExecutionContext(principal)
        boundary_request = AgentExecutionRequest(
            action=action,
            step={"arguments": arguments},
            position=1,
            goal=None,
            world_state=None,
            experience_id=None,
            metadata={},
        )
        result = canonical_api.execute_from_boundary_request(
            boundary_request, execution_context=execution_context
        )
        read_result = result.report.result
        if not isinstance(read_result, CapabilityReadResult):
            raise TypeError("Canonical capability invocation returned an invalid result.")
    except ExecutionSecurityError as exc:
        raise HTTPException(403, detail={
            "error": "canonical_execution_denied",
            "message": "The canonical execution security policy denied this action.",
        }) from exc
    except CapabilityTracePersistenceError as exc:
        logger.error("Canonical capability trace persistence failed; result withheld.", exc_info=True)
        raise HTTPException(503, detail={
            "error": "capability_trace_unavailable",
            "message": "Required capability execution evidence could not be persisted. The result is not available.",
        }) from exc
    except CapabilityAuthorizationError as exc:
        raise HTTPException(403, detail={
            "error": "capability_read_forbidden",
            "message": "The authenticated principal is not authorized for this capability.",
        }) from exc
    except CapabilityRequestValidationError as exc:
        raise HTTPException(422, detail={
            "error": "capability_read_invalid",
            "message": "The structured capability arguments are invalid.",
        }) from exc
    except CapabilityResolutionError as exc:
        raise HTTPException(404, detail={
            "error": "capability_read_not_found",
            "message": "The configured capability target or operation is unavailable.",
        }) from exc
    except CapabilityAdapterError as exc:
        logger.warning("Canonical capability adapter failed.", exc_info=True)
        raise HTTPException(502, detail={
            "error": "capability_adapter_failed",
            "message": "The capability adapter could not complete the read.",
        }) from exc
    except Exception as exc:
        logger.exception("Canonical trusted capability invocation failed.")
        raise HTTPException(503, detail={
            "error": "canonical_invocation_failed",
            "message": "The canonical trusted invocation could not be completed.",
        }) from exc

    return {
        "status": "ok",
        "execution_id": execution_context.execution_id,
        "action": action,
        "target": read_result.target,
        "capability_id": read_result.capability_id,
        "data": jsonable_encoder(read_result.data),
        "metadata": jsonable_encoder(dict(read_result.metadata)),
        "read_only": True,
    }


__all__ = ["router"]
