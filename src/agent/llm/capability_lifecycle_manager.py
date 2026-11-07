from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any

from agent.llm.capability_registry import (
    CapabilityNotFoundError,
    CapabilityRegistrationError,
    CapabilityRecord,
    CapabilityStatus,
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)
from agent.llm.dynamic_workspace_runtime_adapter import (
    DynamicWorkspaceAdapterReport,
    DynamicWorkspaceRuntimeAdapter,
)


class CapabilityLifecycleManagerError(RuntimeError):
    """Base error for capability lifecycle orchestration."""


class CapabilityLifecycleValidationError(CapabilityLifecycleManagerError):
    """Raised when lifecycle manager dependencies or requests are invalid."""


class CapabilityLifecycleSynchronizationError(CapabilityLifecycleManagerError):
    """Raised when a lifecycle transition cannot be synchronized safely."""


@dataclass(frozen=True)
class CapabilityLifecycleResult:
    capability_id: str
    action_name: str
    previous_status: CapabilityStatus
    status: CapabilityStatus
    registry_version: int
    record_version: int
    changed: bool
    planner_action_names: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityLifecycleInspection:
    registry_version: int
    adapter_report: DynamicWorkspaceAdapterReport
    planner_action_names: tuple[str, ...]
    active_registry_action_names: tuple[str, ...]
    missing_active_action_names: tuple[str, ...]
    inactive_planner_action_names: tuple[str, ...]
    planner_sync_hook_attached: bool

    @property
    def is_consistent(self) -> bool:
        return (
            self.planner_sync_hook_attached
            and not self.adapter_report.ready
            and not self.adapter_report.conflicts
            and not self.missing_active_action_names
            and not self.inactive_planner_action_names
        )


class CapabilityLifecycleManager:
    """
    Coordinates trusted capability status changes with planner rebuilding.

    The manager changes registry lifecycle state transactionally and asks the
    existing Agent.configure_planner() method to rebuild the planner graph.
    The DynamicWorkspaceRuntimeAdapter hook then filters inactive capability
    actions. It does not execute actions or alter the canonical execution path.
    """

    def __init__(
        self,
        *,
        registry: TrustedCapabilityRegistry,
        adapter: DynamicWorkspaceRuntimeAdapter,
    ) -> None:
        if not isinstance(registry, TrustedCapabilityRegistry):
            raise CapabilityLifecycleValidationError(
                "registry must be a TrustedCapabilityRegistry."
            )
        if not isinstance(adapter, DynamicWorkspaceRuntimeAdapter):
            raise CapabilityLifecycleValidationError(
                "adapter must be a DynamicWorkspaceRuntimeAdapter."
            )
        if adapter.registry is not registry:
            raise CapabilityLifecycleValidationError(
                "adapter must use the same capability registry."
            )

        agent = adapter.agent
        configure_planner = getattr(agent, "configure_planner", None)
        if not callable(configure_planner):
            raise CapabilityLifecycleValidationError(
                "Adapter Agent must expose callable configure_planner()."
            )
        if getattr(agent, "_planner_sync_hook", None) is not adapter._planner_sync_callback:
            raise CapabilityLifecycleValidationError(
                "Adapter planner synchronization hook is not installed."
            )

        self.registry = registry
        self.adapter = adapter
        self.agent = agent
        self._lock = RLock()

    @staticmethod
    def _validate_target_status(status: CapabilityStatus) -> None:
        if not isinstance(status, CapabilityStatus):
            raise CapabilityLifecycleValidationError(
                "status must be a CapabilityStatus."
            )
        if status == CapabilityStatus.ACTIVE:
            return
        if status in (CapabilityStatus.DISABLED, CapabilityStatus.REVOKED):
            return
        raise CapabilityLifecycleValidationError(
            f"Unsupported lifecycle status: {status!r}"
        )

    def _planner_sync_hook_attached(self) -> bool:
        return (
            getattr(self.agent, "_planner_sync_hook", None)
            is self.adapter._planner_sync_callback
        )

    def _require_planner_sync_hook(self) -> None:
        if not self._planner_sync_hook_attached():
            raise CapabilityLifecycleSynchronizationError(
                "Capability lifecycle transition rejected because the adapter "
                "planner synchronization hook is no longer installed."
            )

    def _planner_action_names(self) -> tuple[str, ...]:
        planner = getattr(self.agent, "planner", None)
        actions = getattr(planner, "actions", None)
        if not isinstance(actions, list):
            raise CapabilityLifecycleSynchronizationError(
                "Agent planner is unavailable or has an invalid actions list."
            )
        return tuple(action.name for action in actions)

    def transition(
        self,
        capability_id: str,
        status: CapabilityStatus,
    ) -> CapabilityLifecycleResult:
        self._validate_target_status(status)

        if not isinstance(capability_id, str) or not capability_id.strip():
            raise CapabilityLifecycleValidationError(
                "capability_id must be a non-empty string."
            )

        with self._lock:
            self._require_planner_sync_hook()
            try:
                previous = self.registry.get(
                    capability_id,
                    include_inactive=True,
                )
            except CapabilityNotFoundError:
                raise

            if previous.trust_level == CapabilityTrustLevel.BUILTIN:
                if previous.status != status:
                    raise CapabilityLifecycleValidationError(
                        "Built-in capabilities cannot be changed by the lifecycle manager."
                    )
                snapshot = self.registry.snapshot()
                return CapabilityLifecycleResult(
                    capability_id=previous.capability_id,
                    action_name=previous.action_name,
                    previous_status=previous.status,
                    status=previous.status,
                    registry_version=snapshot.version,
                    record_version=previous.version,
                    changed=False,
                    planner_action_names=self._planner_action_names(),
                )

            try:
                updated = self.registry.apply_lifecycle_transition(
                    capability_id,
                    status,
                    synchronize=self.agent.configure_planner,
                )
            except (CapabilityNotFoundError, CapabilityRegistrationError):
                raise
            except BaseException as exc:
                raise CapabilityLifecycleSynchronizationError(
                    "Capability lifecycle transition failed. Registry status "
                    "was restored when a transition had been applied, and the "
                    "Agent planner graph was protected by configure_planner()."
                ) from exc

            snapshot = self.registry.snapshot()
            planner_names = self._planner_action_names()

            return CapabilityLifecycleResult(
                capability_id=updated.capability_id,
                action_name=updated.action_name,
                previous_status=previous.status,
                status=updated.status,
                registry_version=snapshot.version,
                record_version=updated.version,
                changed=updated is not previous,
                planner_action_names=planner_names,
            )

    def activate(self, capability_id: str) -> CapabilityLifecycleResult:
        return self.transition(capability_id, CapabilityStatus.ACTIVE)

    def disable(self, capability_id: str) -> CapabilityLifecycleResult:
        return self.transition(capability_id, CapabilityStatus.DISABLED)

    def revoke(self, capability_id: str) -> CapabilityLifecycleResult:
        return self.transition(capability_id, CapabilityStatus.REVOKED)

    def get_status(self, capability_id: str) -> CapabilityStatus:
        return self.registry.get(
            capability_id,
            include_inactive=True,
        ).status

    def inspect(self) -> CapabilityLifecycleInspection:
        with self._lock:
            snapshot = self.registry.snapshot()
            report = self.adapter.inspect()
            planner_names = self._planner_action_names()
            hook_attached = self._planner_sync_hook_attached()
            planner_name_set = set(planner_names)

            # Derive this list from the registry alone. Filtering it through
            # Agent.actions here would hide active capabilities whose action
            # mapping has gone missing or has been replaced.
            active_records = tuple(
                record
                for record in snapshot.records.values()
                if record.status == CapabilityStatus.ACTIVE
            )
            active_names = tuple(record.action_name for record in active_records)
            inactive_names = {
                record.action_name
                for record in snapshot.records.values()
                if record.status != CapabilityStatus.ACTIVE
            }

            missing = tuple(
                record.action_name
                for record in active_records
                if (
                    self.agent.actions.get(record.action_name) is not record.action
                    or record.action_name not in planner_name_set
                )
            )
            inactive_in_planner = tuple(
                name for name in planner_names
                if name in inactive_names
            )

            return CapabilityLifecycleInspection(
                registry_version=snapshot.version,
                adapter_report=report,
                planner_action_names=planner_names,
                active_registry_action_names=active_names,
                missing_active_action_names=missing,
                inactive_planner_action_names=inactive_in_planner,
                planner_sync_hook_attached=hook_attached,
            )


__all__ = [
    "CapabilityLifecycleManagerError",
    "CapabilityLifecycleValidationError",
    "CapabilityLifecycleSynchronizationError",
    "CapabilityLifecycleResult",
    "CapabilityLifecycleInspection",
    "CapabilityLifecycleManager",
]
