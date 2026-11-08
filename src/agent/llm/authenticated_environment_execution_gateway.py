"""Authenticated execution boundary for trusted generic environment adapters.

This gateway is deliberately separate from the legacy read-only capability
gateway. It accepts only application-resolved targets and registry-owned
capabilities. It does not register capabilities, construct adapters, mutate an
Agent/Planner, or execute plans; the canonical plan runtime remains responsible
for invoking one action at a time through this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from api.security.authentication import AuthenticatedPrincipal
from api.security.authorization import AuthorizationDenied, AuthorizationPolicy
from agent.llm.capability_execution_contracts import CapabilityAuthorizationError
from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)
from agent.llm.execution_security import (
    ExecutionSecurityError,
    ExecutionSecurityPolicy,
)
from agent.world.environment_lifecycle import (
    EnvironmentInstanceError,
    EnvironmentLifecycleManager,
)
from agent.world.environment_contracts import (
    EnvironmentActionRequest,
    EnvironmentActionResult,
    EnvironmentAdapter,
    EnvironmentContractError,
    EnvironmentDescriptor,
    EnvironmentObservation,
)


class EnvironmentExecutionError(RuntimeError):
    """Base error raised by the authenticated environment execution boundary."""


class EnvironmentExecutionDenied(CapabilityAuthorizationError, EnvironmentExecutionError):
    """Raised when authentication, trust, permission, or policy checks deny execution."""


class EnvironmentExecutionOutcomeUnknown(EnvironmentExecutionError):
    """Raised when an action may have produced side effects but cannot be verified."""


class EnvironmentExecutionConfigurationError(EnvironmentExecutionError):
    """Raised when the trusted application target resolver is misconfigured."""


@dataclass(frozen=True, slots=True)
class AuthenticatedEnvironmentExecutionRequest:
    """Caller request; actor identity and risk are never caller-controlled."""

    instance_id: str
    capability_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    expected_revision: str | None = None
    position: int = 1

    def __post_init__(self) -> None:
        for name in ("instance_id", "capability_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.arguments, Mapping):
            raise ValueError("arguments must be a mapping.")
        # EnvironmentActionRequest performs a deep JSON-only validation and freeze.
        validated = EnvironmentActionRequest(
            action_id=self.capability_id,
            arguments=self.arguments,
            idempotency_key=self.idempotency_key,
            expected_revision=self.expected_revision,
        )
        object.__setattr__(self, "arguments", validated.arguments)
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 1:
            raise ValueError("position must be a positive integer.")


@dataclass(frozen=True, slots=True)
class EnvironmentExecutionTarget:
    """Private application-resolved adapter target for one active instance."""

    instance_id: str
    descriptor: EnvironmentDescriptor
    adapter: EnvironmentAdapter

    def __post_init__(self) -> None:
        if not isinstance(self.instance_id, str) or not self.instance_id.strip():
            raise ValueError("instance_id must be a non-empty string.")
        if not isinstance(self.descriptor, EnvironmentDescriptor):
            raise TypeError("descriptor must be EnvironmentDescriptor.")
        if not isinstance(self.adapter, EnvironmentAdapter):
            raise TypeError("adapter must implement EnvironmentAdapter.")
        if self.adapter.descriptor != self.descriptor:
            raise ValueError("adapter descriptor does not match the trusted target descriptor.")


@dataclass(frozen=True, slots=True)
class AuthenticatedEnvironmentExecutionResult:
    """Execution evidence; success is true only after a matching fresh observation."""

    capability_id: str
    instance_id: str
    action_result: EnvironmentActionResult
    observation_before: EnvironmentObservation
    observation_after: EnvironmentObservation

    @property
    def success(self) -> bool:
        return self.action_result.success and self.action_result.outcome_known


@dataclass(frozen=True, slots=True)
class _EnvironmentRiskRequest:
    action: str
    position: int


class AuthenticatedEnvironmentExecutionGateway:
    """Fail-closed gateway for executing trusted environment capabilities.

    The application-owned lifecycle manager resolves the adapter internally.
    The gateway holds both a capability execution lease and an environment
    instance lease through pre-observation, adapter invocation, post-observation,
    and durable outcome audit. Adapters are never accepted from request data.
    """

    _ALLOWED_TRUST_LEVELS = frozenset({
        CapabilityTrustLevel.BUILTIN,
        CapabilityTrustLevel.APPLICATION,
        CapabilityTrustLevel.APPROVED_PROPOSAL,
    })

    def __init__(
        self,
        *,
        environment_lifecycle_manager: EnvironmentLifecycleManager,
        capability_binding_validator: Callable[[str, str], bool],
        capability_registry: TrustedCapabilityRegistry,
        authorization_policy: AuthorizationPolicy,
        execution_security_policy: ExecutionSecurityPolicy,
    ) -> None:
        if not isinstance(environment_lifecycle_manager, EnvironmentLifecycleManager):
            raise TypeError("environment_lifecycle_manager must be EnvironmentLifecycleManager.")
        if not callable(capability_binding_validator):
            raise TypeError("capability_binding_validator must be callable.")
        if not isinstance(capability_registry, TrustedCapabilityRegistry):
            raise TypeError("capability_registry must be TrustedCapabilityRegistry.")
        if not isinstance(authorization_policy, AuthorizationPolicy):
            raise TypeError("authorization_policy must be AuthorizationPolicy.")
        if not isinstance(execution_security_policy, ExecutionSecurityPolicy):
            raise TypeError("execution_security_policy must be ExecutionSecurityPolicy.")
        self._environment_lifecycle_manager = environment_lifecycle_manager
        self._capability_binding_validator = capability_binding_validator
        self._capability_registry = capability_registry
        self._authorization_policy = authorization_policy
        self._execution_security_policy = execution_security_policy

    def execute(
        self,
        principal: AuthenticatedPrincipal,
        request: AuthenticatedEnvironmentExecutionRequest,
    ) -> AuthenticatedEnvironmentExecutionResult:
        if not isinstance(principal, AuthenticatedPrincipal) or principal.is_authenticated is not True:
            raise EnvironmentExecutionDenied("A verified authenticated principal is required.")
        if not isinstance(request, AuthenticatedEnvironmentExecutionRequest):
            raise TypeError("request must be AuthenticatedEnvironmentExecutionRequest.")

        # Acquire the environment lease first. This makes CLOSING instances reject
        # new work immediately instead of allowing a new request to queue behind
        # an in-flight capability lease while shutdown waits for that same work.
        # No adapter method is called until capability trust and permissions pass.
        try:
            with self._environment_lifecycle_manager.lease_instance(
                request.instance_id
            ) as instance_lease:
                target = EnvironmentExecutionTarget(
                    instance_id=instance_lease.snapshot.instance_id,
                    descriptor=instance_lease.snapshot.descriptor,
                    adapter=instance_lease.adapter,
                )
                try:
                    with self._capability_registry.execution_lease(request.capability_id) as record:
                        self._authorize_record(principal, record)
                        self._validate_environment_binding(target, request)
                        self._authorize_risk(request)
                        return self._execute_on_target(target, request)
                except CapabilityNotFoundError as exc:
                    raise EnvironmentExecutionDenied(
                        "Capability is unknown, disabled, or revoked."
                    ) from exc
        except EnvironmentInstanceError as exc:
            raise EnvironmentExecutionDenied(
                "Environment instance is unknown, closing, or unavailable."
            ) from exc

    def _authorize_record(self, principal: AuthenticatedPrincipal, record: Any) -> None:
        if record.trust_level not in self._ALLOWED_TRUST_LEVELS:
            raise EnvironmentExecutionDenied("Capability trust level is not permitted for execution.")
        if not record.permissions:
            raise EnvironmentExecutionDenied("Capability has no explicit application permission.")
        try:
            for permission in record.permissions:
                self._authorization_policy.require(principal, permission)
        except AuthorizationDenied as exc:
            raise EnvironmentExecutionDenied(
                "The authenticated principal lacks a required capability permission."
            ) from exc
        except EnvironmentExecutionDenied:
            raise
        except Exception as exc:
            raise EnvironmentExecutionDenied("Capability authorization policy failed.") from exc

    def _validate_environment_binding(
        self,
        target: EnvironmentExecutionTarget,
        request: AuthenticatedEnvironmentExecutionRequest,
    ) -> None:
        try:
            allowed = self._capability_binding_validator(
                target.descriptor.environment_type,
                request.capability_id,
            )
        except Exception as exc:
            raise EnvironmentExecutionDenied(
                "Trusted environment capability binding could not be verified."
            ) from exc
        if allowed is not True:
            raise EnvironmentExecutionDenied(
                "Capability is not registered for this environment type."
            )

    def _authorize_risk(self, request: AuthenticatedEnvironmentExecutionRequest) -> None:
        try:
            self._execution_security_policy.authorize(
                _EnvironmentRiskRequest(
                    action=request.capability_id,
                    position=request.position,
                )
            )
        except ExecutionSecurityError as exc:
            raise EnvironmentExecutionDenied(
                "Environment capability execution security policy denied this action."
            ) from exc
        except Exception as exc:
            raise EnvironmentExecutionDenied(
                "Environment capability execution security policy is unavailable."
            ) from exc

    def _execute_on_target(
        self,
        target: EnvironmentExecutionTarget,
        request: AuthenticatedEnvironmentExecutionRequest,
    ) -> AuthenticatedEnvironmentExecutionResult:
        descriptor = target.descriptor
        if not descriptor.supports_observation:
            raise EnvironmentExecutionDenied(
                "Environment provider must support observation before action execution."
            )
        if request.idempotency_key is not None and not descriptor.supports_idempotency:
            raise EnvironmentExecutionDenied(
                "Environment provider does not declare idempotency support."
            )

        try:
            before = target.adapter.observe(instance_id=target.instance_id)
            if not isinstance(before, EnvironmentObservation):
                raise EnvironmentContractError("adapter.observe must return EnvironmentObservation.")
            if before.instance_id != target.instance_id:
                raise EnvironmentContractError("pre-action observation instance_id mismatch.")
            if before.environment_id != descriptor.environment_type:
                raise EnvironmentContractError("pre-action observation environment_id mismatch.")
            if request.expected_revision is not None and request.expected_revision != before.revision:
                raise EnvironmentExecutionDenied("Environment revision does not match expected_revision.")
        except EnvironmentExecutionDenied:
            raise
        except Exception as exc:
            raise EnvironmentExecutionDenied("Environment could not be safely observed before execution.") from exc

        adapter_request = EnvironmentActionRequest(
            action_id=request.capability_id,
            arguments=request.arguments,
            idempotency_key=request.idempotency_key,
            expected_revision=request.expected_revision or before.revision,
        )
        try:
            action_result = target.adapter.execute(
                instance_id=target.instance_id,
                request=adapter_request,
            )
        except Exception as exc:
            self._record_outcome(request, success=None, reason="adapter_execution_exception")
            raise EnvironmentExecutionOutcomeUnknown(
                "Adapter execution raised after invocation; outcome may be unknown. Do not retry blindly."
            ) from exc

        if not isinstance(action_result, EnvironmentActionResult):
            self._record_outcome(request, success=None, reason="invalid_adapter_result")
            raise EnvironmentExecutionOutcomeUnknown(
                "Adapter returned an invalid result after invocation; outcome is unknown."
            )
        if action_result.instance_id != target.instance_id or action_result.environment_id != descriptor.environment_type:
            self._record_outcome(request, success=None, reason="adapter_result_identity_mismatch")
            raise EnvironmentExecutionOutcomeUnknown(
                "Adapter result identity mismatch; outcome cannot be verified."
            )
        if action_result.previous_revision != before.revision:
            self._record_outcome(request, success=None, reason="adapter_previous_revision_mismatch")
            raise EnvironmentExecutionOutcomeUnknown(
                "Adapter result does not match the pre-action revision; outcome cannot be verified."
            )

        try:
            after = target.adapter.observe(instance_id=target.instance_id)
            if not isinstance(after, EnvironmentObservation):
                raise EnvironmentContractError("adapter.observe must return EnvironmentObservation.")
            if after.instance_id != target.instance_id or after.environment_id != descriptor.environment_type:
                raise EnvironmentContractError("post-action observation identity mismatch.")
            if after.revision != action_result.current_revision:
                raise EnvironmentContractError("post-action observation revision mismatch.")
        except Exception as exc:
            self._record_outcome(request, success=None, reason="post_action_observation_unavailable")
            raise EnvironmentExecutionOutcomeUnknown(
                "Action was invoked but its resulting state cannot be verified. Do not retry blindly."
            ) from exc

        if not action_result.outcome_known:
            self._record_outcome(request, success=None, reason="adapter_reported_unknown_outcome")
        else:
            self._record_outcome(
                request,
                success=action_result.success,
                reason="environment_action_result",
            )
        return AuthenticatedEnvironmentExecutionResult(
            capability_id=request.capability_id,
            instance_id=request.instance_id,
            action_result=action_result,
            observation_before=before,
            observation_after=after,
        )

    def _record_outcome(
        self,
        request: AuthenticatedEnvironmentExecutionRequest,
        *,
        success: bool | None,
        reason: str,
    ) -> None:
        try:
            self._execution_security_policy.record_execution_outcome(
                _EnvironmentRiskRequest(
                    action=request.capability_id,
                    position=request.position,
                ),
                success=success,
                reason=reason,
            )
        except Exception as exc:
            raise EnvironmentExecutionOutcomeUnknown(
                "Execution outcome could not be durably audited; Do not retry blindly."
            ) from exc


__all__ = [
    "AuthenticatedEnvironmentExecutionRequest",
    "AuthenticatedEnvironmentExecutionResult",
    "AuthenticatedEnvironmentExecutionGateway",
    "EnvironmentExecutionConfigurationError",
    "EnvironmentExecutionDenied",
    "EnvironmentExecutionError",
    "EnvironmentExecutionOutcomeUnknown",
    "EnvironmentExecutionTarget",
]
