from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable

from .capability_execution_contracts import (
    CapabilityAdapterError,
    CapabilityAuthorizationError,
    CapabilityReadAdapter,
    CapabilityReadRequest,
    CapabilityReadResult,
    CapabilityResolutionError,
)


class CapabilityExecutionGateway:
    """Application-owned, fail-closed gateway for read-only capabilities.

    The gateway resolves only adapters supplied at construction. It does not
    register Agent Actions, invoke Action handlers, execute plans, or expose
    mutation methods. The authorizer must explicitly return True; missing,
    failing, or non-boolean authorization always denies the request.
    """

    def __init__(
        self,
        adapters: Iterable[CapabilityReadAdapter],
        *,
        authorizer: Callable[[str, str, str], bool] | None = None,
    ) -> None:
        if isinstance(adapters, (str, bytes)):
            raise TypeError("adapters must be an iterable of adapter objects.")
        self._adapters: dict[str, CapabilityReadAdapter] = {}
        for adapter in adapters:
            target = getattr(adapter, "target", None)
            supports = getattr(adapter, "supports", None)
            read = getattr(adapter, "read", None)
            if not isinstance(target, str) or not target.strip():
                raise TypeError("Every adapter must expose a non-empty target.")
            if not callable(supports) or not callable(read):
                raise TypeError("Every adapter must implement supports() and read().")
            target = target.strip()
            if target in self._adapters:
                raise ValueError(f"Duplicate capability adapter target: {target}")
            self._adapters[target] = adapter
        if not self._adapters:
            raise ValueError("At least one capability adapter is required.")
        if authorizer is not None and not callable(authorizer):
            raise TypeError("authorizer must be callable or None.")
        self._authorizer = authorizer

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def read(self, request: CapabilityReadRequest) -> CapabilityReadResult:
        if not isinstance(request, CapabilityReadRequest):
            raise TypeError("request must be a CapabilityReadRequest.")
        if self._authorizer is None:
            raise CapabilityAuthorizationError(
                "Capability gateway is fail-closed: no application authorizer is configured."
            )
        try:
            authorized = self._authorizer(
                request.actor_id,
                request.target,
                request.capability_id,
            )
        except Exception as exc:
            raise CapabilityAuthorizationError(
                "Application authorization check failed."
            ) from exc
        if authorized is not True:
            raise CapabilityAuthorizationError("Actor is not authorized for this capability.")

        adapter = self._adapters.get(request.target)
        if adapter is None:
            raise CapabilityResolutionError(
                f"Unknown capability target: {request.target}"
            )
        try:
            if adapter.supports(request.capability_id) is not True:
                raise CapabilityResolutionError(
                    f"Capability '{request.capability_id}' is not registered for target '{request.target}'."
                )
            data = adapter.read(request.capability_id, request.arguments)
            return CapabilityReadResult(
                target=request.target,
                capability_id=request.capability_id,
                data=deepcopy(data),
                metadata={"read_only": True},
            )
        except CapabilityResolutionError:
            raise
        except Exception as exc:
            if isinstance(exc, CapabilityAdapterError):
                raise
            raise CapabilityAdapterError(
                f"Capability read failed for '{request.target}:{request.capability_id}'."
            ) from exc


__all__ = ["CapabilityExecutionGateway"]
