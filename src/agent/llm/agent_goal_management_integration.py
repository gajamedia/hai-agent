from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .goal_management import (
    GoalLifecycleManager,
    GoalLifecycleError,
    GoalManagementError,
    GoalSelectionPolicy,
    GoalSelection,
    GoalStatus,
    ManagedGoal,
)


class AgentGoalManagementIntegrationError(GoalManagementError):
    """Base error for Agent/goal-manager integration."""


class AgentGoalManagementIntegrationConfigurationError(
    AgentGoalManagementIntegrationError
):
    """Raised when the integration target is invalid or unsafe."""


class AgentGoalProjectionError(AgentGoalManagementIntegrationError):
    """Raised when the optional Agent goal projection cannot be applied."""


@dataclass(frozen=True)
class AgentGoalManagementSnapshot:
    """Stable view of Agent-facing goal state."""

    current_goal_id: Optional[str]
    current_goal: Optional[ManagedGoal]
    goal_status: Optional[GoalStatus]
    goal_progress: Optional[float]
    manager_sequence: int


class AgentGoalManagementIntegrator:
    """Bind GoalLifecycleManager to an Agent without replacing Agent behavior.

    The manager remains the source of truth. Integration only provides a
    dedicated ``agent.goal_manager`` reference plus optional scalar/object
    projections on the Agent itself. It never calls ``decide_and_act``, the LLM,
    an execution engine, or memory/learning APIs.
    """

    def __init__(
        self,
        agent: Any,
        manager: GoalLifecycleManager,
        *,
        project_to_agent: bool = True,
    ) -> None:
        if agent is None:
            raise AgentGoalManagementIntegrationConfigurationError(
                "agent must not be None."
            )
        if not isinstance(manager, GoalLifecycleManager):
            raise AgentGoalManagementIntegrationConfigurationError(
                "manager must be a GoalLifecycleManager."
            )
        self.agent = agent
        self.manager = manager
        self.project_to_agent = bool(project_to_agent)
        self._bound_attribute_original = getattr(agent, "goal_manager", _MISSING)
        self._projected_attributes: set[str] = set()

        try:
            setattr(self.agent, "goal_manager", self.manager)
        except Exception as exc:  # pragma: no cover - defensive for exotic agents
            raise AgentGoalManagementIntegrationConfigurationError(
                "Unable to attach goal_manager to agent."
            ) from exc

        self._sync_projection()

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        *,
        manager: Optional[GoalLifecycleManager] = None,
        selection_policy: Optional[GoalSelectionPolicy] = None,
        project_to_agent: bool = True,
    ) -> "AgentGoalManagementIntegrator":
        if manager is None:
            manager = GoalLifecycleManager(selection_policy=selection_policy)
        elif selection_policy is not None:
            raise AgentGoalManagementIntegrationConfigurationError(
                "selection_policy cannot be supplied when manager is provided."
            )
        return cls(
            agent,
            manager,
            project_to_agent=project_to_agent,
        )

    @property
    def current_goal(self) -> Optional[ManagedGoal]:
        goal_id = self.manager.current_goal_id
        if goal_id is None:
            return None
        return self.manager.get_goal(goal_id)

    def add_goal(
        self,
        goal_id: str,
        description: str,
        *,
        priority: float = 0.5,
        parent_id: Optional[str] = None,
        payload: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ManagedGoal:
        goal = self.manager.add_goal(
            goal_id,
            description,
            priority=priority,
            parent_id=parent_id,
            payload=payload,
            metadata=metadata,
        )
        self._sync_projection()
        return goal

    def select_next_goal(self) -> GoalSelection:
        selection = self.manager.select_next_goal()
        self._sync_projection()
        return selection

    def set_current_goal(self, goal_id: Optional[str]) -> Optional[ManagedGoal]:
        goal = self.manager.set_current_goal(goal_id)
        self._sync_projection()
        return goal

    def activate(self, goal_id: str) -> ManagedGoal:
        goal = self.manager.activate(goal_id)
        self._sync_projection()
        return goal

    def pause(self, goal_id: str) -> ManagedGoal:
        goal = self.manager.pause(goal_id)
        self._sync_projection()
        return goal

    def resume(self, goal_id: str) -> ManagedGoal:
        goal = self.manager.resume(goal_id)
        self._sync_projection()
        return goal

    def complete(self, goal_id: str) -> ManagedGoal:
        goal = self.manager.complete(goal_id)
        self._sync_projection()
        return goal

    def fail(self, goal_id: str) -> ManagedGoal:
        goal = self.manager.fail(goal_id)
        self._sync_projection()
        return goal

    def cancel(self, goal_id: str) -> ManagedGoal:
        goal = self.manager.cancel(goal_id)
        self._sync_projection()
        return goal

    def update_progress(self, goal_id: str, progress: float) -> ManagedGoal:
        goal = self.manager.update_progress(goal_id, progress)
        self._sync_projection()
        return goal

    def snapshot(self) -> AgentGoalManagementSnapshot:
        current = self.current_goal
        return AgentGoalManagementSnapshot(
            current_goal_id=self.manager.current_goal_id,
            current_goal=current,
            goal_status=current.status if current is not None else None,
            goal_progress=current.progress if current is not None else None,
            manager_sequence=self.manager.snapshot().sequence,
        )

    def restore_agent_projection(self) -> AgentGoalManagementSnapshot:
        """Re-apply the current manager state to the Agent-facing projection."""
        self._sync_projection()
        return self.snapshot()

    def detach(self) -> None:
        """Detach only the dedicated ``goal_manager`` attribute and projection."""
        for name in (
            "current_goal_id",
            "current_goal",
            "goal_status",
            "goal_progress",
            "goal_management_snapshot",
        ):
            # Only remove fields if we created them during projection.
            if name in self._projected_attributes:
                try:
                    delattr(self.agent, name)
                except AttributeError:
                    pass
        self._projected_attributes.clear()

        if self._bound_attribute_original is _MISSING:
            try:
                delattr(self.agent, "goal_manager")
            except AttributeError:
                pass
        else:
            try:
                setattr(self.agent, "goal_manager", self._bound_attribute_original)
            except Exception as exc:  # pragma: no cover - defensive
                raise AgentGoalProjectionError(
                    "Unable to restore the previous goal_manager binding."
                ) from exc

    def _sync_projection(self) -> None:
        if not self.project_to_agent:
            return

        current = self.current_goal
        values = {
            "current_goal_id": self.manager.current_goal_id,
            "current_goal": current,
            "goal_status": current.status if current is not None else None,
            "goal_progress": current.progress if current is not None else None,
            "goal_management_snapshot": self.snapshot(),
        }
        for name, value in values.items():
            try:
                setattr(self.agent, name, value)
            except Exception as exc:
                raise AgentGoalProjectionError(
                    f"Unable to project goal field {name!r} to agent."
                ) from exc
            self._projected_attributes.add(name)


class _Missing:
    pass


_MISSING = _Missing()


__all__ = [
    "AgentGoalManagementIntegrator",
    "AgentGoalManagementSnapshot",
    "AgentGoalManagementIntegrationConfigurationError",
    "AgentGoalManagementIntegrationError",
    "AgentGoalProjectionError",
]
