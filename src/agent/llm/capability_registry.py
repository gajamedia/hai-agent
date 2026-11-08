
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping

from agent.core.action import Action


class CapabilityRegistryError(RuntimeError):
    """Base exception for capability registry errors."""


class CapabilityValidationError(CapabilityRegistryError):
    """Raised when capability metadata is invalid."""


class CapabilityRegistrationError(CapabilityRegistryError):
    """Raised when capability registration or lifecycle change is rejected."""


class CapabilityNotFoundError(CapabilityRegistryError):
    """Raised when a capability does not exist."""


class CapabilityStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    REVOKED = "revoked"


class CapabilityTrustLevel(str, Enum):
    BUILTIN = "builtin"
    APPLICATION = "application"
    APPROVED_PROPOSAL = "approved_proposal"


@dataclass(frozen=True)
class CapabilityRecord:
    capability_id: str
    action_name: str
    description: str
    action: Action
    trust_level: CapabilityTrustLevel
    risk_level: str
    permissions: tuple[str, ...]
    status: CapabilityStatus
    version: int = 1


@dataclass(frozen=True)
class CapabilityRegistrySnapshot:
    version: int
    records: Mapping[str, CapabilityRecord]


class TrustedCapabilityRegistry:
    """
    Application-controlled registry for executable capabilities.

    Important:
    - LLM proposals cannot register themselves.
    - Only application code may call register_trusted().
    - Existing Agent.actions are not mutated by this registry.
    - Execution remains the responsibility of the canonical executor.
    - Revoked capabilities are terminal and cannot be reactivated.
    """

    ACTION_NAME_PATTERN = re.compile(
        r"^[a-z][a-z0-9_]{0,63}$"
    )

    CAPABILITY_ID_PATTERN = re.compile(
        r"^[a-z][a-z0-9_.-]{0,127}$"
    )

    ALLOWED_RISK_LEVELS = frozenset({
        "low",
        "medium",
        "high",
        "critical",
    })

    def __init__(
        self,
        builtin_actions: Iterable[Action] | None = None,
    ) -> None:
        self._lock = RLock()
        # Per-capability locks serialize reads against lifecycle changes without
        # blocking unrelated capabilities. Lock objects are retained for the
        # registry lifetime so all callers coordinate on the same lock.
        self._execution_locks: dict[str, RLock] = {}
        self._records: dict[str, CapabilityRecord] = {}
        self._action_index: dict[str, str] = {}
        self._version = 0

        if builtin_actions is not None:
            for action in builtin_actions:
                self.register_builtin(action)

    @classmethod
    def _validate_action_name(cls, name: str) -> None:
        if (
            not isinstance(name, str)
            or not cls.ACTION_NAME_PATTERN.fullmatch(name)
        ):
            raise CapabilityValidationError(
                f"Invalid action name: {name!r}"
            )

    @classmethod
    def _validate_capability_id(
        cls,
        capability_id: str,
    ) -> None:
        if (
            not isinstance(capability_id, str)
            or not cls.CAPABILITY_ID_PATTERN.fullmatch(capability_id)
        ):
            raise CapabilityValidationError(
                f"Invalid capability ID: {capability_id!r}"
            )

    @classmethod
    def _validate_risk_level(cls, risk_level: str) -> None:
        if (
            not isinstance(risk_level, str)
            or risk_level not in cls.ALLOWED_RISK_LEVELS
        ):
            raise CapabilityValidationError(
                f"Invalid risk level: {risk_level!r}"
            )

    @staticmethod
    def _normalize_permissions(
        permissions: Iterable[str],
    ) -> tuple[str, ...]:
        if isinstance(permissions, (str, bytes)):
            raise CapabilityValidationError(
                "permissions must be an iterable of strings."
            )

        try:
            values = tuple(permissions)
        except TypeError as exc:
            raise CapabilityValidationError(
                "permissions must be iterable."
            ) from exc

        normalized: list[str] = []

        for permission in values:
            if (
                not isinstance(permission, str)
                or not permission.strip()
                or permission != permission.strip()
            ):
                raise CapabilityValidationError(
                    "Invalid permission value."
                )

            if permission in normalized:
                raise CapabilityValidationError(
                    f"Duplicate permission: {permission}"
                )

            normalized.append(permission)

        return tuple(sorted(normalized))

    @staticmethod
    def _validate_action(action: Action) -> None:
        if not isinstance(action, Action):
            raise CapabilityValidationError(
                "action must be an agent.core.action.Action."
            )

        if not callable(getattr(action, "handler", None)):
            raise CapabilityValidationError(
                "Action handler must be callable."
            )

        if (
            not isinstance(action.description, str)
            or not action.description.strip()
        ):
            raise CapabilityValidationError(
                "Action description cannot be empty."
            )

        if not isinstance(action.preconditions, Mapping):
            raise CapabilityValidationError(
                "Action preconditions must be a mapping."
            )

        if not isinstance(action.effects, Mapping):
            raise CapabilityValidationError(
                "Action effects must be a mapping."
            )

    def _register(
        self,
        *,
        capability_id: str,
        action: Action,
        trust_level: CapabilityTrustLevel,
        risk_level: str,
        permissions: Iterable[str],
    ) -> CapabilityRecord:
        self._validate_capability_id(capability_id)
        self._validate_action(action)
        self._validate_action_name(action.name)
        self._validate_risk_level(risk_level)

        normalized_permissions = self._normalize_permissions(
            permissions
        )

        with self._lock:
            if capability_id in self._records:
                raise CapabilityRegistrationError(
                    f"Capability already exists: {capability_id}"
                )

            existing_id = self._action_index.get(action.name)

            if existing_id is not None:
                raise CapabilityRegistrationError(
                    f"Action name already registered: {action.name}"
                )

            record = CapabilityRecord(
                capability_id=capability_id,
                action_name=action.name,
                description=action.description,
                action=action,
                trust_level=trust_level,
                risk_level=risk_level,
                permissions=normalized_permissions,
                status=CapabilityStatus.ACTIVE,
            )

            self._records[capability_id] = record
            self._action_index[action.name] = capability_id
            self._version += 1

            return record

    def register_builtin(
        self,
        action: Action,
        *,
        risk_level: str = "low",
        permissions: Iterable[str] = (),
    ) -> CapabilityRecord:
        """
        Register an application-defined built-in action.

        The default risk is intended only for compatibility with the
        existing four simulation actions. Applications should provide
        an explicit risk level when the built-in has different risk.
        """
        if not isinstance(action, Action):
            raise CapabilityValidationError(
                "action must be an Action."
            )

        capability_id = f"builtin.{action.name}"

        return self._register(
            capability_id=capability_id,
            action=action,
            trust_level=CapabilityTrustLevel.BUILTIN,
            risk_level=risk_level,
            permissions=permissions,
        )

    def register_trusted(
        self,
        *,
        capability_id: str,
        action: Action,
        risk_level: str,
        permissions: Iterable[str] = (),
    ) -> CapabilityRecord:
        """
        Register a capability implemented and supplied by the application.

        This method must never be exposed directly as an LLM tool.
        """
        return self._register(
            capability_id=capability_id,
            action=action,
            trust_level=CapabilityTrustLevel.APPLICATION,
            risk_level=risk_level,
            permissions=permissions,
        )

    def register_approved_proposal(
        self,
        *,
        capability_id: str,
        action: Action,
        risk_level: str,
        permissions: Iterable[str] = (),
    ) -> CapabilityRecord:
        """
        Register an application-built Action after an application approval.

        The proposal itself supplies metadata only. The caller must resolve the
        handler from an application-owned catalog before calling this method.
        This method must never be exposed directly as an LLM tool.
        """
        self._validate_capability_id(capability_id)
        self._validate_action(action)
        self._validate_action_name(action.name)
        self._validate_risk_level(risk_level)
        normalized_permissions = self._normalize_permissions(permissions)

        return self._register(
            capability_id=capability_id,
            action=action,
            trust_level=CapabilityTrustLevel.APPROVED_PROPOSAL,
            risk_level=risk_level,
            permissions=normalized_permissions,
        )

    def rollback_approved_proposal_registration(
        self,
        capability_id: str,
        *,
        expected_action: Action,
    ) -> bool:
        """
        Remove a just-created approved-proposal registration during rollback.

        Only APPROVED_PROPOSAL records can be removed. This is intended for
        application transaction recovery when Agent/Planner synchronization
        fails immediately after registration; it is not a user-facing delete.
        """
        execution_lock = self._execution_lock_for(capability_id)
        with execution_lock, self._lock:
            record = self._records.get(capability_id)
            if record is None:
                return False
            if record.trust_level != CapabilityTrustLevel.APPROVED_PROPOSAL:
                raise CapabilityRegistrationError(
                    "Only approved-proposal registrations can be rolled back."
                )
            if record.action is not expected_action:
                raise CapabilityRegistrationError(
                    "Approved-proposal rollback action identity mismatch."
                )
            self._records.pop(capability_id)
            self._action_index.pop(record.action_name, None)
            self._version += 1
            return True

    def _execution_lock_for(self, capability_id: str) -> RLock:
        """Return the stable per-capability lock, creating it if needed."""
        with self._lock:
            lock = self._execution_locks.get(capability_id)
            if lock is None:
                lock = RLock()
                self._execution_locks[capability_id] = lock
            return lock

    @contextmanager
    def execution_lease(self, capability_id: str) -> Iterator[CapabilityRecord]:
        """Hold an active capability against disable/revoke for one operation.

        The global registry lock is held only while resolving the record. The
        per-capability lock remains held for the caller's entire context, so a
        lifecycle transition for this capability waits until the operation
        finishes. Other capability IDs remain independently executable.

        A lease does not make a capability permanently active: once the
        context exits, a waiting disable/revoke may immediately take effect.
        """
        if not isinstance(capability_id, str) or not capability_id:
            raise CapabilityValidationError(
                "capability_id must be a non-empty string."
            )
        execution_lock = self._execution_lock_for(capability_id)
        with execution_lock:
            record = self.get(capability_id)
            yield record

    def get(
        self,
        capability_id: str,
        *,
        include_inactive: bool = False,
    ) -> CapabilityRecord:
        with self._lock:
            record = self._records.get(capability_id)

            if record is None:
                raise CapabilityNotFoundError(
                    f"Capability not found: {capability_id}"
                )

            if (
                not include_inactive
                and record.status != CapabilityStatus.ACTIVE
            ):
                raise CapabilityNotFoundError(
                    f"Capability is not active: {capability_id}"
                )

            return record

    def get_by_action(
        self,
        action_name: str,
        *,
        include_inactive: bool = False,
    ) -> CapabilityRecord:
        with self._lock:
            capability_id = self._action_index.get(action_name)

            if capability_id is None:
                raise CapabilityNotFoundError(
                    f"Action is not registered: {action_name}"
                )

            return self.get(
                capability_id,
                include_inactive=include_inactive,
            )

    def list_active(self) -> tuple[CapabilityRecord, ...]:
        with self._lock:
            return tuple(
                record
                for record in self._records.values()
                if record.status == CapabilityStatus.ACTIVE
            )

    def list_all(self) -> tuple[CapabilityRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def disable(self, capability_id: str) -> CapabilityRecord:
        return self._change_status(
            capability_id,
            CapabilityStatus.DISABLED,
        )

    def activate(self, capability_id: str) -> CapabilityRecord:
        """
        Reactivate a disabled capability.

        Only DISABLED -> ACTIVE is allowed.
        REVOKED capabilities are permanently terminal.
        """
        return self._change_status(
            capability_id,
            CapabilityStatus.ACTIVE,
        )

    def revoke(self, capability_id: str) -> CapabilityRecord:
        return self._change_status(
            capability_id,
            CapabilityStatus.REVOKED,
        )

    def apply_lifecycle_transition(
        self,
        capability_id: str,
        status: CapabilityStatus,
        *,
        synchronize,
    ) -> CapabilityRecord:
        """
        Apply a lifecycle transition and synchronize dependent runtime state.

        The registry lock is held while the transition and synchronization
        callback run. If synchronization fails, the previous record is
        restored. Registry versions remain monotonic: a failed transition
        that had changed the record advances the registry version once more
        when rolling back, while the restored record keeps its prior version.

        This method is intended for trusted application lifecycle
        orchestration, not for exposure as an LLM tool.
        """
        if not callable(synchronize):
            raise CapabilityValidationError(
                "synchronize must be callable."
            )

        execution_lock = self._execution_lock_for(capability_id)
        with execution_lock, self._lock:
            previous = self._records.get(capability_id)
            if previous is None:
                raise CapabilityNotFoundError(
                    f"Capability not found: {capability_id}"
                )

            updated = self._change_status(capability_id, status)
            changed = updated is not previous

            try:
                synchronize()
            except BaseException:
                if changed:
                    self._records[capability_id] = previous
                    self._version += 1
                raise

            return self._records[capability_id]

    def _change_status(
        self,
        capability_id: str,
        status: CapabilityStatus,
    ) -> CapabilityRecord:
        if not isinstance(status, CapabilityStatus):
            raise CapabilityValidationError(
                "status must be a CapabilityStatus."
            )

        execution_lock = self._execution_lock_for(capability_id)
        with execution_lock, self._lock:
            record = self._records.get(capability_id)

            if record is None:
                raise CapabilityNotFoundError(
                    f"Capability not found: {capability_id}"
                )

            if record.trust_level == CapabilityTrustLevel.BUILTIN:
                raise CapabilityRegistrationError(
                    "Built-in capabilities cannot be disabled "
                    "or revoked through this registry."
                )

            if record.status == CapabilityStatus.REVOKED:
                if status == CapabilityStatus.REVOKED:
                    return record

                raise CapabilityRegistrationError(
                    "A revoked capability cannot be reactivated."
                )

            if record.status == status:
                return record

            allowed_transitions = {
                CapabilityStatus.ACTIVE: {
                    CapabilityStatus.DISABLED,
                    CapabilityStatus.REVOKED,
                },
                CapabilityStatus.DISABLED: {
                    CapabilityStatus.ACTIVE,
                    CapabilityStatus.REVOKED,
                },
            }

            if status not in allowed_transitions.get(record.status, set()):
                raise CapabilityRegistrationError(
                    f"Invalid capability lifecycle transition: "
                    f"{record.status.value} -> {status.value}"
                )

            updated = CapabilityRecord(
                capability_id=record.capability_id,
                action_name=record.action_name,
                description=record.description,
                action=record.action,
                trust_level=record.trust_level,
                risk_level=record.risk_level,
                permissions=record.permissions,
                status=status,
                version=record.version + 1,
            )

            self._records[capability_id] = updated
            self._version += 1

            return updated

    def snapshot(self) -> CapabilityRegistrySnapshot:
        with self._lock:
            return CapabilityRegistrySnapshot(
                version=self._version,
                records=MappingProxyType(
                    dict(self._records)
                ),
            )

    def is_active(self, capability_id: str) -> bool:
        with self._lock:
            record = self._records.get(capability_id)

            return (
                record is not None
                and record.status == CapabilityStatus.ACTIVE
            )

    def contains_action(self, action_name: str) -> bool:
        with self._lock:
            capability_id = self._action_index.get(action_name)

            if capability_id is None:
                return False

            record = self._records.get(capability_id)

            return (
                record is not None
                and record.status == CapabilityStatus.ACTIVE
            )


__all__ = [
    "CapabilityRegistryError",
    "CapabilityValidationError",
    "CapabilityRegistrationError",
    "CapabilityNotFoundError",
    "CapabilityStatus",
    "CapabilityTrustLevel",
    "CapabilityRecord",
    "CapabilityRegistrySnapshot",
    "TrustedCapabilityRegistry",
]
