"""Registry of features and their operations, with an operational on/off switch.

Registering a feature does not make anything run. The registry only answers
"is this operation known and currently enabled?"; the pipeline (a later step)
asks it before every run. Unknown or disabled operations fail closed.
"""

from __future__ import annotations

import threading
from typing import Iterable

from .contracts import ContractError, Feature, Operation


class RegistryError(RuntimeError):
    """Base class for registry errors."""


class UnknownOperation(RegistryError):
    """No such operation is registered."""


class OperationDisabled(RegistryError):
    """The operation (or its whole feature) is switched off."""


class OperationRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._features: dict[str, Feature] = {}
        self._operations: dict[str, Operation] = {}
        self._disabled_features: set[str] = set()
        self._disabled_operations: set[str] = set()

    def register(self, feature: Feature) -> None:
        if not isinstance(feature, Feature):
            raise ContractError("register() needs a Feature")
        with self._lock:
            if feature.name in self._features:
                raise RegistryError(f"feature already registered: {feature.name}")
            clash = [op.id for op in feature.operations if op.id in self._operations]
            if clash:  # cannot happen with unique feature names; kept as a guard
                raise RegistryError(f"operation already registered: {clash[0]}")
            self._features[feature.name] = feature
            for op in feature.operations:
                self._operations[op.id] = op

    def features(self) -> tuple[Feature, ...]:
        with self._lock:
            return tuple(self._features.values())

    def operations(self) -> tuple[Operation, ...]:
        with self._lock:
            return tuple(self._operations.values())

    def get(self, operation_id: str) -> Operation:
        """Return an enabled operation; unknown or disabled ones raise."""
        with self._lock:
            op = self._operations.get(operation_id)
            if op is None:
                raise UnknownOperation(f"unknown operation: {operation_id}")
            if op.feature in self._disabled_features or op.id in self._disabled_operations:
                raise OperationDisabled(f"operation is disabled: {operation_id}")
            return op

    def is_enabled(self, operation_id: str) -> bool:
        try:
            self.get(operation_id)
        except RegistryError:
            return False
        return True

    def disable_feature(self, name: str) -> None:
        with self._lock:
            if name not in self._features:
                raise UnknownOperation(f"unknown feature: {name}")
            self._disabled_features.add(name)

    def enable_feature(self, name: str) -> None:
        with self._lock:
            if name not in self._features:
                raise UnknownOperation(f"unknown feature: {name}")
            self._disabled_features.discard(name)

    def disable_operation(self, operation_id: str) -> None:
        with self._lock:
            if operation_id not in self._operations:
                raise UnknownOperation(f"unknown operation: {operation_id}")
            self._disabled_operations.add(operation_id)

    def enable_operation(self, operation_id: str) -> None:
        with self._lock:
            if operation_id not in self._operations:
                raise UnknownOperation(f"unknown operation: {operation_id}")
            self._disabled_operations.discard(operation_id)

    def enabled_operations(self) -> Iterable[Operation]:
        with self._lock:
            return tuple(op for op in self._operations.values() if self.is_enabled(op.id))


__all__ = ["OperationDisabled", "OperationRegistry", "RegistryError", "UnknownOperation"]
