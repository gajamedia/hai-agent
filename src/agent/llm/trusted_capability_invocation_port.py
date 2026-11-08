from __future__ import annotations

"""Canonical invocation port for trusted, read-only capabilities.

The port never resolves a capability from LLM-controlled target/capability IDs.
Application configuration owns that mapping. Only the request's structured
``step["arguments"]`` is caller data, and the downstream adapter validates it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from api.security.trusted_execution_context import TrustedExecutionContext
from agent.llm.authenticated_capability_execution_gateway import (
    AuthenticatedCapabilityReadRequest,
)
from agent.llm.capability_execution_contracts import CapabilityReadResult


class TrustedCapabilityInvocationError(RuntimeError):
    """Base error for a trusted capability invocation."""


class TrustedCapabilityInvocationConfigurationError(
    TrustedCapabilityInvocationError
):
    """Raised when the invocation port is not safely configured."""


class TrustedCapabilityInvocationRequestError(TrustedCapabilityInvocationError):
    """Raised when a canonical action cannot be mapped to a read request."""


@dataclass(frozen=True, slots=True)
class CapabilityInvocationSpec:
    """Server-owned mapping from one canonical action to one capability."""

    action_name: str
    target: str
    capability_id: str
    arguments_key: str = "arguments"

    def __post_init__(self) -> None:
        for field_name in ("action_name", "target", "capability_id", "arguments_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())


@dataclass(frozen=True, slots=True)
class TrustedCapabilityInvocationDecision:
    """Non-sensitive decision metadata for the canonical execution report."""

    action_name: str
    target: str
    capability_id: str
    allowed: bool = True


class TrustedCapabilityInvocationPort:
    """Invoke only application-mapped capabilities through a traced gateway.

    The gateway must be the existing application-lifecycle gateway, normally
    ``app.state.capability_execution_gateway``. This class does not create,
    own, close, or replace a gateway or database connection.
    """

    def __init__(self, gateway: Any, specs: Mapping[str, CapabilityInvocationSpec]):
        if gateway is None or not callable(getattr(gateway, "read", None)):
            raise TypeError("gateway must expose callable read(...).")
        if not isinstance(specs, Mapping) or not specs:
            raise ValueError("specs must be a non-empty mapping.")

        normalized: dict[str, CapabilityInvocationSpec] = {}
        for key, spec in specs.items():
            if not isinstance(spec, CapabilityInvocationSpec):
                raise TypeError("Every spec must be CapabilityInvocationSpec.")
            if key != spec.action_name:
                raise ValueError("Each spec key must equal spec.action_name.")
            if key in normalized:
                raise ValueError(f"Duplicate action mapping: {key}")
            normalized[key] = spec

        self._gateway = gateway
        self._specs = dict(normalized)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def supports_action(self, action_name: str) -> bool:
        return isinstance(action_name, str) and action_name in self._specs

    def describe_action(self, action_name: str) -> TrustedCapabilityInvocationDecision:
        spec = self._specs.get(action_name)
        if spec is None:
            raise TrustedCapabilityInvocationRequestError(
                "Action is not mapped to a trusted capability."
            )
        return TrustedCapabilityInvocationDecision(
            action_name=spec.action_name,
            target=spec.target,
            capability_id=spec.capability_id,
        )

    def invoke(self, request: Any, *, execution_context: Any) -> CapabilityReadResult:
        """Invoke exactly one mapped capability using the exact trusted context."""
        from agent.llm.agent_execution_boundary import AgentExecutionRequest

        if not isinstance(request, AgentExecutionRequest):
            raise TypeError("request must be AgentExecutionRequest.")
        if type(execution_context) is not TrustedExecutionContext:
            raise TrustedCapabilityInvocationConfigurationError(
                "A valid TrustedExecutionContext is required."
            )
        if execution_context.principal.is_authenticated is not True:
            raise TrustedCapabilityInvocationConfigurationError(
                "The trusted principal is not authenticated."
            )

        spec = self._specs.get(request.action)
        if spec is None:
            raise TrustedCapabilityInvocationRequestError(
                "Action is not mapped to a trusted capability."
            )

        step = request.step
        if not isinstance(step, Mapping):
            raise TrustedCapabilityInvocationRequestError(
                "Capability step must be a mapping containing structured arguments."
            )
        arguments = step.get(spec.arguments_key)
        if not isinstance(arguments, Mapping):
            raise TrustedCapabilityInvocationRequestError(
                f"Capability step must contain a mapping at '{spec.arguments_key}'."
            )

        # Target and capability ID are server-owned. The LLM cannot override
        # them through step, metadata, goal, or world_state.
        gateway_request = AuthenticatedCapabilityReadRequest(
            target=spec.target,
            capability_id=spec.capability_id,
            arguments=dict(arguments),
        )
        result = self._gateway.read(
            execution_context.principal,
            gateway_request,
            execution_id=execution_context.execution_id,
        )
        if not isinstance(result, CapabilityReadResult):
            raise TrustedCapabilityInvocationConfigurationError(
                "Trusted gateway returned an invalid CapabilityReadResult."
            )
        return result


__all__ = [
    "CapabilityInvocationSpec",
    "TrustedCapabilityInvocationDecision",
    "TrustedCapabilityInvocationError",
    "TrustedCapabilityInvocationConfigurationError",
    "TrustedCapabilityInvocationRequestError",
    "TrustedCapabilityInvocationPort",
]
