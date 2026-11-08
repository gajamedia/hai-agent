from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from threading import RLock
from typing import Any, Mapping

from .capability_execution_contracts import (
    CapabilityAdapterError,
    CapabilityRequestValidationError,
)


class RoomEnvironmentAdapter:
    """Read-only adapter for observing the isolated RoomEnvironment.

    This adapter intentionally exposes no execute(), reset(), set_state(),
    Agent, Action, or handler access. Mutating simulation remains on the
    existing canonical execution path and is not routed through this gateway.
    """

    TARGET = "room_environment"
    OBSERVE_CAPABILITY = "room_environment.observe_state"

    def __init__(self, environment: Any, lock: Any | None = None) -> None:
        observe = getattr(environment, "observe", None)
        if environment is None or not callable(observe):
            raise TypeError("environment must provide a callable observe() method.")
        if lock is not None and not callable(getattr(lock, "__enter__", None)):
            raise TypeError("lock must be a context manager when supplied.")
        self._environment = environment
        self._lock = lock if lock is not None else RLock()

    @property
    def target(self) -> str:
        return self.TARGET

    def supports(self, capability_id: str) -> bool:
        return capability_id == self.OBSERVE_CAPABILITY

    def read(self, capability_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if not self.supports(capability_id):
            raise CapabilityRequestValidationError(
                f"Unsupported RoomEnvironment capability: {capability_id}"
            )
        if arguments:
            raise CapabilityRequestValidationError(
                "room_environment.observe_state does not accept arguments."
            )
        try:
            with self._lock:
                snapshot = self._environment.observe()
            if not isinstance(snapshot, Mapping):
                raise TypeError("RoomEnvironment.observe() must return a mapping.")
            return deepcopy(dict(snapshot))
        except CapabilityRequestValidationError:
            raise
        except Exception as exc:
            raise CapabilityAdapterError(
                "RoomEnvironment state could not be read."
            ) from exc


__all__ = ["RoomEnvironmentAdapter"]
