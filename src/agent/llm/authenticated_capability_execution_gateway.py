from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import (
    AuthorizationDenied,
    AuthorizationPolicy,
)
from agent.llm.capability_execution_contracts import (
    CapabilityAuthorizationError,
    CapabilityReadRequest,
    CapabilityReadResult,
)
from agent.llm.capability_execution_gateway import CapabilityExecutionGateway
from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)
from agent.llm.execution_security import (
    ExecutionSecurityError,
    ExecutionSecurityPolicy,
)


@dataclass(frozen=True, slots=True)
class AuthenticatedCapabilityReadRequest:
    """Read request without a caller-controlled actor identity."""

    target: str
    capability_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("target", "capability_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a mapping.")
        object.__setattr__(self, "arguments", dict(self.arguments))


class AuthenticatedCapabilityExecutionGateway:
    """Authenticated, permission-checked facade over the P2.2 read gateway.

    The only public read entrypoint requires an AuthenticatedPrincipal supplied
    by the trusted API authentication layer. No actor id is accepted from the
    capability request. Registry status/trust, explicit capability permissions,
    the application's exact authorization policy, and the existing execution
    risk policy are checked before an adapter can be called.

    The underlying P2.2 gateway and adapters remain private implementation
    details. Application code must not separately publish adapter instances or
    the raw gateway to request handlers.
    """

    _ALLOWED_TRUST_LEVELS = frozenset({
        CapabilityTrustLevel.BUILTIN,
        CapabilityTrustLevel.APPLICATION,
        CapabilityTrustLevel.APPROVED_PROPOSAL,
    })

    def __init__(
        self,
        adapters,
        *,
        capability_registry: TrustedCapabilityRegistry,
        authorization_policy: AuthorizationPolicy,
        execution_security_policy: ExecutionSecurityPolicy,
    ) -> None:
        if not isinstance(capability_registry, TrustedCapabilityRegistry):
            raise TypeError("capability_registry must be TrustedCapabilityRegistry.")
        if not isinstance(authorization_policy, AuthorizationPolicy):
            raise TypeError("authorization_policy must be AuthorizationPolicy.")
        if not isinstance(execution_security_policy, ExecutionSecurityPolicy):
            raise TypeError("execution_security_policy must be ExecutionSecurityPolicy.")

        # The inner gateway's callback is deliberately trivial: this facade
        # performs the complete trusted authorization sequence first.
        self._gateway = CapabilityExecutionGateway(
            adapters,
            authorizer=lambda _actor, _target, _capability: True,
        )
        self._capability_registry = capability_registry
        self._authorization_policy = authorization_policy
        self._execution_security_policy = execution_security_policy

    @property
    def targets(self) -> tuple[str, ...]:
        return self._gateway.targets

    def read(
        self,
        principal: AuthenticatedPrincipal,
        request: AuthenticatedCapabilityReadRequest,
    ) -> CapabilityReadResult:
        if not isinstance(principal, AuthenticatedPrincipal):
            raise CapabilityAuthorizationError(
                "A verified AuthenticatedPrincipal is required."
            )
        if principal.is_authenticated is not True:
            raise CapabilityAuthorizationError("The principal is not authenticated.")
        if not isinstance(request, AuthenticatedCapabilityReadRequest):
            raise TypeError("request must be AuthenticatedCapabilityReadRequest.")

        try:
            # The lease is acquired before checking trust/permissions and stays
            # held through the adapter read. A concurrent disable/revoke waits
            # for this operation; once the transition returns, no earlier
            # leased read can still be running.
            with self._capability_registry.execution_lease(
                request.capability_id
            ) as record:
                return self._read_with_active_record(principal, request, record)
        except CapabilityNotFoundError as exc:
            raise CapabilityAuthorizationError(
                "Capability is unknown, disabled, or revoked."
            ) from exc

    def _read_with_active_record(
        self,
        principal: AuthenticatedPrincipal,
        request: AuthenticatedCapabilityReadRequest,
        record,
    ) -> CapabilityReadResult:
        if record.trust_level not in self._ALLOWED_TRUST_LEVELS:
            raise CapabilityAuthorizationError(
                "Capability trust level is not permitted for execution."
            )
        if not record.permissions:
            raise CapabilityAuthorizationError(
                "Capability has no explicit application permission."
            )

        try:
            for permission in record.permissions:
                self._authorization_policy.require(principal, permission)
        except AuthorizationDenied as exc:
            raise CapabilityAuthorizationError(
                "The authenticated principal lacks a required capability permission."
            ) from exc
        except Exception as exc:
            raise CapabilityAuthorizationError(
                "Capability authorization policy failed."
            ) from exc

        # The risk policy uses only the registry-owned capability id. The
        # caller cannot supply or lower risk through request arguments.
        try:
            self._execution_security_policy.authorize(
                _CapabilityRiskRequest(action=record.capability_id, position=1)
            )
        except ExecutionSecurityError as exc:
            raise CapabilityAuthorizationError(
                "Capability execution security policy denied this read."
            ) from exc
        except Exception as exc:
            raise CapabilityAuthorizationError(
                "Capability execution security policy is unavailable."
            ) from exc

        gateway_request = CapabilityReadRequest(
            target=request.target,
            capability_id=request.capability_id,
            actor_id=principal.actor_id,
            arguments=request.arguments,
        )
        return self._gateway.read(gateway_request)


@dataclass(frozen=True, slots=True)
class _CapabilityRiskRequest:
    action: str
    position: int


__all__ = [
    "AuthenticatedCapabilityReadRequest",
    "AuthenticatedCapabilityExecutionGateway",
]
