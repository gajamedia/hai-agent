from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.encoders import jsonable_encoder

from api.security.authentication import AuthenticationError, get_attached_principal
from agent.llm.authenticated_capability_execution_gateway import AuthenticatedCapabilityReadRequest
from agent.llm.capability_execution_contracts import (
    CapabilityAdapterError,
    CapabilityAuthorizationError,
    CapabilityRequestValidationError,
    CapabilityResolutionError,
)
from agent.llm.traced_authenticated_capability_execution_gateway import CapabilityTracePersistenceError

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/capability-reads', tags=['capability-execution'])
_ALLOWED_FIELDS = frozenset({'target', 'capability_id', 'arguments'})


def _principal(request: Request):
    try:
        return get_attached_principal(request)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail={
            'error': 'authentication_required',
            'message': 'A verified authenticated principal is required.',
        }) from exc


def _gateway(request: Request):
    gateway = getattr(request.app.state, 'capability_execution_gateway', None)
    if gateway is None or not callable(getattr(gateway, 'read', None)):
        raise HTTPException(status_code=503, detail={
            'error': 'capability_execution_unavailable',
            'message': 'The application has not configured a traced capability execution gateway.',
        })
    return gateway


def _read_request(payload: Mapping[str, Any]) -> AuthenticatedCapabilityReadRequest:
    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        raise HTTPException(status_code=422, detail={
            'error': 'invalid_capability_read_request',
            'message': 'Unsupported request field(s): ' + ', '.join(sorted(map(str, unknown))),
        })
    target = payload.get('target')
    capability_id = payload.get('capability_id')
    arguments = payload.get('arguments', {})
    if not isinstance(target, str) or not target.strip():
        raise HTTPException(status_code=422, detail={
            'error': 'invalid_capability_read_request', 'message': 'target must be a non-empty string.',
        })
    if not isinstance(capability_id, str) or not capability_id.strip():
        raise HTTPException(status_code=422, detail={
            'error': 'invalid_capability_read_request', 'message': 'capability_id must be a non-empty string.',
        })
    if not isinstance(arguments, Mapping):
        raise HTTPException(status_code=422, detail={
            'error': 'invalid_capability_read_request', 'message': 'arguments must be a JSON object.',
        })
    try:
        return AuthenticatedCapabilityReadRequest(
            target=target, capability_id=capability_id, arguments=arguments,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={
            'error': 'invalid_capability_read_request', 'message': 'The capability read request is invalid.',
        }) from exc


@router.post('')
def read_capability(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Execute one authenticated, authorized, traced read-only capability call."""
    principal = _principal(request)
    gateway = _gateway(request)
    read_request = _read_request(payload)
    execution_id = uuid4().hex
    try:
        result = gateway.read(principal, read_request, execution_id=execution_id)
    except CapabilityTracePersistenceError as exc:
        logger.error('Required capability trace persistence failed; execution_id=%s', execution_id, exc_info=True)
        raise HTTPException(status_code=503, detail={
            'error': 'capability_trace_unavailable',
            'message': 'Required capability execution evidence could not be persisted. The read result is not available.',
            'execution_id': execution_id,
        }) from exc
    except CapabilityAuthorizationError as exc:
        raise HTTPException(status_code=403, detail={
            'error': 'capability_read_forbidden',
            'message': 'The authenticated principal is not authorized for this capability.',
            'execution_id': execution_id,
        }) from exc
    except CapabilityRequestValidationError as exc:
        raise HTTPException(status_code=422, detail={
            'error': 'capability_read_invalid', 'message': str(exc), 'execution_id': execution_id,
        }) from exc
    except CapabilityResolutionError as exc:
        raise HTTPException(status_code=404, detail={
            'error': 'capability_read_not_found',
            'message': 'The requested capability target or operation is unavailable.',
            'execution_id': execution_id,
        }) from exc
    except CapabilityAdapterError as exc:
        logger.warning('Capability adapter read failed; execution_id=%s', execution_id, exc_info=True)
        raise HTTPException(status_code=502, detail={
            'error': 'capability_adapter_failed',
            'message': 'The capability adapter could not complete the read.',
            'execution_id': execution_id,
        }) from exc
    except Exception as exc:
        logger.exception('Unexpected capability execution failure; execution_id=%s', execution_id)
        raise HTTPException(status_code=500, detail={
            'error': 'capability_execution_internal_error',
            'message': 'Capability execution failed.',
            'execution_id': execution_id,
        }) from exc
    return {
        'status': 'ok',
        'execution_id': execution_id,
        'target': result.target,
        'capability_id': result.capability_id,
        'data': jsonable_encoder(result.data),
        'metadata': jsonable_encoder(dict(result.metadata)),
        'read_only': True,
    }


__all__ = ['router']
