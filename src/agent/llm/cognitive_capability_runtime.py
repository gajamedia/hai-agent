from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.llm.capability_lifecycle_manager import CapabilityLifecycleManager
from agent.llm.dynamic_workspace_runtime_adapter import DynamicWorkspaceRuntimeAdapter

from agent.llm.capability_registry import TrustedCapabilityRegistry
from agent.llm.cognitive_environment import CognitiveEnvironment
from agent.llm.dynamic_workspace_actions import DynamicWorkspaceActionBuilder


class CognitiveCapabilityRuntimeError(RuntimeError):
    """Raised when the isolated capability runtime cannot be constructed."""


@dataclass(frozen=True)
class CognitiveCapabilityRuntime:
    """Infrastructure container for cognitive capabilities.

    Stage 2.8A intentionally does not attach DynamicWorkspaceRuntimeAdapter,
    register capabilities, or modify Agent.actions/Planner. Those operations
    remain disabled until a controlled execution-context design is approved.
    """

    agent: Any
    environment: CognitiveEnvironment
    registry: TrustedCapabilityRegistry
    action_builder: DynamicWorkspaceActionBuilder
    adapter: DynamicWorkspaceRuntimeAdapter | None = None
    lifecycle_manager: CapabilityLifecycleManager | None = None

    @property
    def active_capability_count(self) -> int:
        return len(self.registry.snapshot().records)

    def inspect(self) -> dict[str, Any]:
        snapshot = self.registry.snapshot()
        return {
            "registry_version": snapshot.version,
            "capability_count": len(snapshot.records),
            "active_capability_count": sum(
                1 for record in snapshot.records.values()
                if getattr(record.status, "value", record.status) == "active"
            ),
            "adapter_attached": self.adapter is not None,
            "lifecycle_manager_enabled": self.lifecycle_manager is not None,
            "execution_enabled": False,
            "composition_ready": (
                self.adapter is not None
                and self.lifecycle_manager is not None
                and self.lifecycle_manager.inspect().is_consistent
            ),
        }


def create_cognitive_capability_runtime(agent: Any) -> CognitiveCapabilityRuntime:
    """Create isolated cognitive capability infrastructure without Agent mutation."""
    if agent is None:
        raise CognitiveCapabilityRuntimeError("A cognitive Agent is required.")

    try:
        registry = TrustedCapabilityRegistry()
        environment = CognitiveEnvironment()
        builder = DynamicWorkspaceActionBuilder(agent)
    except Exception as exc:
        raise CognitiveCapabilityRuntimeError(
            "Could not initialize cognitive capability infrastructure."
        ) from exc

    return CognitiveCapabilityRuntime(
        agent=agent,
        environment=environment,
        registry=registry,
        action_builder=builder,
    )


__all__ = [
    "CognitiveCapabilityRuntimeError",
    "CognitiveCapabilityRuntime",
    "create_cognitive_capability_runtime",
    "compose_cognitive_capability_runtime",
]



def compose_cognitive_capability_runtime(
    agent: Any,
) -> CognitiveCapabilityRuntime:
    """Compose the trusted registry, workspace adapter, and lifecycle manager.

    This enables capability registration/planner lifecycle wiring only. It does
    not enable arbitrary execution or merge the cognitive Agent with the
    isolated RoomEnvironment execution Agent.
    """
    runtime = create_cognitive_capability_runtime(agent)
    adapter = DynamicWorkspaceRuntimeAdapter(
        registry=runtime.registry,
        agent=runtime.agent,
    )
    try:
        adapter.synchronize()
        lifecycle_manager = CapabilityLifecycleManager(
            registry=runtime.registry,
            adapter=adapter,
        )
        inspection = lifecycle_manager.inspect()
        if not inspection.is_consistent:
            raise CognitiveCapabilityRuntimeError(
                "Capability runtime composition is inconsistent."
            )
    except Exception:
        adapter.detach()
        raise

    return CognitiveCapabilityRuntime(
        agent=runtime.agent,
        environment=runtime.environment,
        registry=runtime.registry,
        action_builder=runtime.action_builder,
        adapter=adapter,
        lifecycle_manager=lifecycle_manager,
    )
