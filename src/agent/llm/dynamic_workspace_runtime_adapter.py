from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping

from agent.core.action import Action
from agent.llm.capability_registry import (
    CapabilityRecord,
    CapabilityStatus,
    CapabilityTrustLevel,
    TrustedCapabilityRegistry,
)


class DynamicWorkspaceAdapterError(RuntimeError):
    """Base error for Dynamic Workspace runtime adapter."""


class DynamicWorkspaceAdapterValidationError(DynamicWorkspaceAdapterError):
    """Raised when adapter dependencies or registry data are invalid."""


class DynamicWorkspaceSynchronizationError(DynamicWorkspaceAdapterError):
    """Raised when capability synchronization cannot be completed."""


@dataclass(frozen=True)
class DynamicWorkspaceAdapterReport:
    """Read-only consistency report between registry and Agent."""

    registry_version: int
    ready: tuple[CapabilityRecord, ...]
    already_registered: tuple[CapabilityRecord, ...]
    inactive_registered: tuple[CapabilityRecord, ...]
    conflicts: tuple[str, ...]

    @property
    def ready_action_names(self) -> tuple[str, ...]:
        return tuple(record.action_name for record in self.ready)

    @property
    def already_registered_action_names(self) -> tuple[str, ...]:
        return tuple(record.action_name for record in self.already_registered)

    @property
    def inactive_registered_action_names(self) -> tuple[str, ...]:
        return tuple(record.action_name for record in self.inactive_registered)

    @property
    def can_synchronize(self) -> bool:
        # Inactive records are not registration conflicts. They are handled
        # by synchronize_planner(), which removes them from planning only.
        return not self.conflicts

    @property
    def summary(self) -> Mapping[str, int]:
        return MappingProxyType({
            "ready": len(self.ready),
            "already_registered": len(self.already_registered),
            "inactive_registered": len(self.inactive_registered),
            "conflicts": len(self.conflicts),
        })


@dataclass(frozen=True)
class DynamicWorkspaceSyncResult:
    """Result of registering active capabilities into Agent.actions."""

    registry_version: int
    synchronized_action_names: tuple[str, ...]
    already_registered_action_names: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.synchronized_action_names)

    @property
    def total_active_actions(self) -> int:
        return len(self.synchronized_action_names) + len(
            self.already_registered_action_names
        )


@dataclass(frozen=True)
class DynamicWorkspacePlannerSyncResult:
    """Result of applying registry status to the shared Planner action list."""

    registry_version: int
    available_action_names: tuple[str, ...]
    excluded_action_names: tuple[str, ...]
    changed: bool


class DynamicWorkspaceRuntimeAdapter:
    """
    Bridges trusted registry metadata to Agent registration and Planner filtering.

    synchronize():
        Registers active trusted Action objects in Agent.actions and rebuilds
        the Agent planner when new actions were inserted.

    synchronize_planner():
        Filters the existing shared Planner.actions list according to registry
        status. Disabled/revoked actions remain in Agent.actions for learning,
        reflection, and other consumers, but are excluded from planning.

    Neither operation executes Action handlers, approves proposals, modifies
    runtime allowlists, or changes the canonical execution path.
    """

    _ALLOWED_TRUST_LEVELS = frozenset({
        CapabilityTrustLevel.BUILTIN,
        CapabilityTrustLevel.APPLICATION,
        CapabilityTrustLevel.APPROVED_PROPOSAL,
    })

    def __init__(
        self,
        *,
        registry: TrustedCapabilityRegistry,
        agent: Any,
    ) -> None:
        if not isinstance(registry, TrustedCapabilityRegistry):
            raise DynamicWorkspaceAdapterValidationError(
                "registry must be a TrustedCapabilityRegistry."
            )

        if agent is None:
            raise DynamicWorkspaceAdapterValidationError("agent cannot be None.")

        if not isinstance(getattr(agent, "actions", None), MutableMapping):
            raise DynamicWorkspaceAdapterValidationError(
                "Agent must expose a mutable actions mapping."
            )

        self.registry = registry
        self.agent = agent
        self._lock = RLock()
        self._planner_sync_callback = self.synchronize_planner

        set_hook = getattr(agent, "set_planner_sync_hook", None)
        if not callable(set_hook):
            raise DynamicWorkspaceAdapterValidationError(
                "Agent must expose set_planner_sync_hook() for automatic "
                "planner lifecycle synchronization."
            )

        try:
            set_hook(self._planner_sync_callback)
        except Exception as exc:
            raise DynamicWorkspaceAdapterValidationError(
                "Could not install the planner synchronization hook."
            ) from exc

    def detach(self) -> bool:
        """Detach this adapter's callback without changing Agent actions."""
        clear_hook = getattr(self.agent, "clear_planner_sync_hook", None)
        if not callable(clear_hook):
            return False
        return bool(clear_hook(self._planner_sync_callback))

    @classmethod
    def _validate_record(cls, record: CapabilityRecord) -> None:
        if not isinstance(record, CapabilityRecord):
            raise DynamicWorkspaceAdapterValidationError(
                "Registry returned an invalid capability record."
            )
        if not isinstance(record.action, Action):
            raise DynamicWorkspaceAdapterValidationError(
                f"Capability '{record.capability_id}' does not contain a valid Action."
            )
        if not isinstance(record.action_name, str):
            raise DynamicWorkspaceAdapterValidationError(
                "Capability action_name must be a string."
            )
        if record.action.name != record.action_name:
            raise DynamicWorkspaceAdapterValidationError(
                f"Capability '{record.capability_id}' has inconsistent action metadata."
            )
        if not isinstance(record.status, CapabilityStatus):
            raise DynamicWorkspaceAdapterValidationError(
                f"Capability '{record.capability_id}' has an invalid status."
            )
        if record.trust_level not in cls._ALLOWED_TRUST_LEVELS:
            raise DynamicWorkspaceAdapterValidationError(
                f"Capability '{record.capability_id}' does not have an accepted trust level."
            )

    def _inspect_snapshot(self, snapshot) -> DynamicWorkspaceAdapterReport:
        ready = []
        already_registered = []
        inactive_registered = []
        conflicts = []
        seen_names: set[str] = set()

        for record in tuple(snapshot.records.values()):
            self._validate_record(record)

            if record.action_name in seen_names:
                conflicts.append(f"duplicate_registry_action:{record.action_name}")
                continue
            seen_names.add(record.action_name)

            existing = self.agent.actions.get(record.action_name)
            if record.status == CapabilityStatus.ACTIVE:
                if existing is None:
                    ready.append(record)
                elif existing is record.action:
                    already_registered.append(record)
                else:
                    conflicts.append(
                        f"agent_action_conflict:{record.action_name}"
                    )
            elif existing is record.action:
                inactive_registered.append(record)
            elif existing is not None:
                conflicts.append(
                    f"inactive_action_name_conflict:{record.action_name}"
                )

        return DynamicWorkspaceAdapterReport(
            registry_version=snapshot.version,
            ready=tuple(ready),
            already_registered=tuple(already_registered),
            inactive_registered=tuple(inactive_registered),
            conflicts=tuple(sorted(set(conflicts))),
        )

    def inspect(self) -> DynamicWorkspaceAdapterReport:
        """Inspect registry/Agent consistency without mutating either."""
        with self._lock:
            return self._inspect_snapshot(self.registry.snapshot())

    def _capture_agent_state(self) -> tuple[dict, dict]:
        actions_snapshot = dict(self.agent.actions)
        planner_attributes = (
            "planner",
            "adaptive_planner",
            "hierarchical_planner",
            "risk_aware_hierarchical_planner",
            "unified_executor",
        )
        planner_snapshot = {
            name: (hasattr(self.agent, name), getattr(self.agent, name, None))
            for name in planner_attributes
        }
        return actions_snapshot, planner_snapshot

    def _restore_agent_state(
        self,
        actions_snapshot: dict,
        planner_snapshot: dict,
    ) -> None:
        self.agent.actions.clear()
        self.agent.actions.update(actions_snapshot)
        for name, (existed, value) in planner_snapshot.items():
            if existed:
                setattr(self.agent, name, value)
            elif hasattr(self.agent, name):
                delattr(self.agent, name)

    def synchronize(self) -> DynamicWorkspaceSyncResult:
        """
        Register active trusted capabilities into Agent.actions.

        Inactive actions already present in Agent.actions are deliberately
        retained. The installed planner hook filters them whenever
        Agent.configure_planner() rebuilds the planner graph.
        """
        with self._lock:
            initial_snapshot = self.registry.snapshot()
            report = self._inspect_snapshot(initial_snapshot)

            if report.conflicts:
                raise DynamicWorkspaceSynchronizationError(
                    "Cannot synchronize because action conflicts exist: "
                    + ", ".join(report.conflicts)
                )

            configure_planner = getattr(self.agent, "configure_planner", None)
            if report.ready and not callable(configure_planner):
                raise DynamicWorkspaceSynchronizationError(
                    "Agent must expose callable configure_planner() "
                    "to synchronize new actions."
                )

            if self.registry.snapshot().version != initial_snapshot.version:
                raise DynamicWorkspaceSynchronizationError(
                    "Capability registry changed during synchronization. "
                    "Retry with a fresh inspection."
                )

            if not report.ready:
                return DynamicWorkspaceSyncResult(
                    registry_version=initial_snapshot.version,
                    synchronized_action_names=(),
                    already_registered_action_names=(
                        report.already_registered_action_names
                    ),
                )

            actions_snapshot, planner_snapshot = self._capture_agent_state()
            names = tuple(record.action_name for record in report.ready)

            try:
                for record in report.ready:
                    if record.action_name in self.agent.actions:
                        raise DynamicWorkspaceSynchronizationError(
                            "Action name became occupied during synchronization: "
                            f"{record.action_name}"
                        )
                    self.agent.actions[record.action_name] = record.action

                configure_planner()

                final_snapshot = self.registry.snapshot()
                if final_snapshot.version != initial_snapshot.version:
                    raise DynamicWorkspaceSynchronizationError(
                        "Capability registry changed while the Agent planner "
                        "was being rebuilt."
                    )

                for record in report.ready:
                    if self.agent.actions.get(record.action_name) is not record.action:
                        raise DynamicWorkspaceSynchronizationError(
                            "Agent action mapping changed unexpectedly for "
                            f"'{record.action_name}'."
                        )
            except BaseException as exc:
                try:
                    self._restore_agent_state(actions_snapshot, planner_snapshot)
                except BaseException as rollback_error:
                    raise DynamicWorkspaceSynchronizationError(
                        "Synchronization failed and Agent rollback also failed."
                    ) from rollback_error

                if isinstance(exc, DynamicWorkspaceSynchronizationError):
                    raise
                raise DynamicWorkspaceSynchronizationError(
                    "Synchronization failed. Agent state was restored."
                ) from exc

            return DynamicWorkspaceSyncResult(
                registry_version=initial_snapshot.version,
                synchronized_action_names=names,
                already_registered_action_names=(
                    report.already_registered_action_names
                ),
            )

    def _shared_planner(self):
        planner = getattr(self.agent, "planner", None)
        adaptive = getattr(self.agent, "adaptive_planner", None)
        hierarchical = getattr(self.agent, "hierarchical_planner", None)
        risk_aware = getattr(self.agent, "risk_aware_hierarchical_planner", None)

        if planner is None:
            raise DynamicWorkspaceSynchronizationError(
                "Agent has no configured planner."
            )
        if getattr(adaptive, "planner", None) is not planner:
            raise DynamicWorkspaceSynchronizationError(
                "AdaptivePlanner does not share Agent.planner."
            )
        if getattr(hierarchical, "planner", None) is not planner:
            raise DynamicWorkspaceSynchronizationError(
                "HierarchicalPlanner does not share Agent.planner."
            )
        if getattr(risk_aware, "hierarchical_planner", None) is not hierarchical:
            raise DynamicWorkspaceSynchronizationError(
                "RiskAwareHierarchicalPlanner does not share Agent.hierarchical_planner."
            )

        actions = getattr(planner, "actions", None)
        if not isinstance(actions, list):
            raise DynamicWorkspaceSynchronizationError(
                "Agent.planner.actions must be a mutable list."
            )
        return planner, actions

    def synchronize_planner(self) -> DynamicWorkspacePlannerSyncResult:
        """
        Filter the shared Planner.actions list using the current registry.

        Non-registry actions are preserved. Active registry actions must be
        the exact objects registered in Agent.actions. Inactive registry
        actions are removed from Planner.actions but retained in Agent.actions.
        The operation validates everything before mutating the list.
        """
        with self._lock:
            snapshot = self.registry.snapshot()
            report = self._inspect_snapshot(snapshot)
            if report.conflicts:
                raise DynamicWorkspaceSynchronizationError(
                    "Cannot filter Planner because action conflicts exist: "
                    + ", ".join(report.conflicts)
                )

            planner, planner_actions = self._shared_planner()
            del planner  # The validated planner object is not otherwise needed.

            records_by_name = {
                record.action_name: record
                for record in snapshot.records.values()
            }
            candidate = []
            excluded = []

            for action in planner_actions:
                if not isinstance(action, Action):
                    raise DynamicWorkspaceSynchronizationError(
                        "Planner.actions contains a non-Action entry."
                    )

                record = records_by_name.get(action.name)
                if record is None:
                    candidate.append(action)
                    continue

                registered = self.agent.actions.get(record.action_name)
                if registered is not record.action or action is not record.action:
                    raise DynamicWorkspaceSynchronizationError(
                        "Planner/Agent action identity conflict for "
                        f"'{record.action_name}'."
                    )

                if record.status == CapabilityStatus.ACTIVE:
                    candidate.append(action)
                else:
                    excluded.append(action.name)

            # Detect active trusted actions missing from the Planner. A planner
            # built from Agent.actions should contain them; silently accepting
            # a mismatch could make registry state and planning state diverge.
            present_names = {action.name for action in candidate}
            for record in snapshot.records.values():
                if (
                    record.status == CapabilityStatus.ACTIVE
                    and self.agent.actions.get(record.action_name) is record.action
                    and record.action_name not in present_names
                ):
                    raise DynamicWorkspaceSynchronizationError(
                        "Active registered action is missing from Planner.actions: "
                        f"{record.action_name}."
                    )

            if self.registry.snapshot().version != snapshot.version:
                raise DynamicWorkspaceSynchronizationError(
                    "Capability registry changed during Planner synchronization."
                )

            changed = (
                len(planner_actions) != len(candidate)
                or any(
                    current is not proposed
                    for current, proposed in zip(planner_actions, candidate)
                )
            )
            planner_actions[:] = candidate

            return DynamicWorkspacePlannerSyncResult(
                registry_version=snapshot.version,
                available_action_names=tuple(action.name for action in candidate),
                excluded_action_names=tuple(excluded),
                changed=changed,
            )


__all__ = [
    "DynamicWorkspaceAdapterError",
    "DynamicWorkspaceAdapterValidationError",
    "DynamicWorkspaceSynchronizationError",
    "DynamicWorkspaceAdapterReport",
    "DynamicWorkspaceSyncResult",
    "DynamicWorkspacePlannerSyncResult",
    "DynamicWorkspaceRuntimeAdapter",
]
