from __future__ import annotations

"""Build the C.3 invocation port around an already-owned API gateway."""

from collections.abc import Mapping
from typing import Any

from agent.llm.trusted_capability_invocation_port import (
    CapabilityInvocationSpec,
    TrustedCapabilityInvocationPort,
)


class TrustedCapabilityInvocationPortFactoryError(RuntimeError):
    """Raised when the lifecycle gateway/specification contract is invalid."""


def build_trusted_capability_invocation_port(
    gateway: Any,
    specs: Mapping[str, CapabilityInvocationSpec],
) -> TrustedCapabilityInvocationPort:
    """Bind server-owned action specs to the existing traced gateway."""
    if gateway is None or not callable(getattr(gateway, "read", None)):
        raise TrustedCapabilityInvocationPortFactoryError(
            "An existing capability gateway with callable read(...) is required."
        )
    if not isinstance(specs, Mapping) or not specs:
        raise TrustedCapabilityInvocationPortFactoryError(
            "The invocation specs factory must return a non-empty mapping."
        )
    try:
        return TrustedCapabilityInvocationPort(gateway=gateway, specs=specs)
    except (TypeError, ValueError) as exc:
        raise TrustedCapabilityInvocationPortFactoryError(
            "The trusted capability invocation specifications are invalid."
        ) from exc


__all__ = [
    "TrustedCapabilityInvocationPortFactoryError",
    "build_trusted_capability_invocation_port",
]
