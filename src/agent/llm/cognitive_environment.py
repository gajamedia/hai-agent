from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from threading import RLock
from types import MappingProxyType
from typing import Any


class CognitiveEnvironmentError(RuntimeError):
    """Base error for the isolated cognitive environment."""


class CognitiveEnvironmentValidationError(CognitiveEnvironmentError, ValueError):
    """Raised when an observation cannot be safely published."""


def _freeze(value: Any) -> Any:
    """Return a detached, recursively read-only observation value."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


class CognitiveEnvironment:
    """Read-only observation boundary, deliberately separate from simulation.

    This object does not dispatch actions, mutate Agent state, or wrap the
    RoomEnvironment. Trusted application code may publish observations;
    capability handlers may only call observe().
    """

    def __init__(self, initial_observation: Mapping[str, Any] | None = None) -> None:
        self._lock = RLock()
        self._version = 0
        self._observation: Mapping[str, Any] = MappingProxyType({})
        if initial_observation is not None:
            self.publish(initial_observation)

    @staticmethod
    def _validate_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise CognitiveEnvironmentValidationError(
                "observation must be a mapping."
            )
        result = dict(observation)
        for key in result:
            if not isinstance(key, str) or not key.strip():
                raise CognitiveEnvironmentValidationError(
                    "observation keys must be non-empty strings."
                )
        try:
            deepcopy(result)
        except Exception as exc:
            raise CognitiveEnvironmentValidationError(
                "observation must contain values that can be safely copied."
            ) from exc
        return result

    def publish(self, observation: Mapping[str, Any]) -> int:
        """Publish a detached observation from trusted application code."""
        candidate = self._validate_observation(observation)
        frozen = _freeze(candidate)
        with self._lock:
            self._observation = frozen
            self._version += 1
            return self._version

    def observe(self) -> Mapping[str, Any]:
        """Return an immutable snapshot; no environment state is changed."""
        with self._lock:
            return MappingProxyType({
                "version": self._version,
                "observation": _freeze(self._observation),
            })

    @property
    def version(self) -> int:
        with self._lock:
            return self._version
