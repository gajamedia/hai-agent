from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class GoalManagementError(Exception):
    """Base error for goal lifecycle management."""


class GoalManagementConfigurationError(GoalManagementError):
    """Raised for invalid manager or policy configuration."""


class GoalNotFoundError(GoalManagementError):
    """Raised when a requested goal does not exist."""


class GoalLifecycleError(GoalManagementError):
    """Raised when a lifecycle transition is invalid."""


class GoalHierarchyError(GoalManagementError):
    """Raised for invalid parent/child relationships."""


class GoalStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.CANCELLED,
        }


@dataclass(frozen=True)
class ManagedGoal:
    """Provider-neutral goal record owned by the goal manager."""

    goal_id: str
    description: str
    priority: float = 0.5
    parent_id: Optional[str] = None
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sequence: int = 0

    def __post_init__(self) -> None:
        if not str(self.goal_id).strip():
            raise ValueError("goal_id must not be empty.")
        if not str(self.description).strip():
            raise ValueError("description must not be empty.")
        object.__setattr__(self, "priority", _bounded_number(self.priority, "priority"))
        object.__setattr__(self, "progress", _bounded_number(self.progress, "progress"))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class GoalSelection:
    """Result of choosing the next goal to work on."""

    selected_goal_id: Optional[str]
    previous_goal_id: Optional[str]
    reason: str
    preempted: bool
    eligible_goal_ids: tuple[str, ...]


@dataclass(frozen=True)
class GoalManagerSnapshot:
    """Immutable diagnostic snapshot."""

    goals: tuple[ManagedGoal, ...]
    current_goal_id: Optional[str]
    sequence: int


class GoalSelectionPolicy:
    """Deterministic priority-based goal selector.

    Higher priority wins. Ties prefer more progress, then older insertion order.
    Terminal goals are never eligible. A candidate may preempt the current goal
    only when its priority exceeds the configured margin.
    """

    def __init__(self, preempt_priority_margin: float = 0.0) -> None:
        self.preempt_priority_margin = _bounded_number(
            preempt_priority_margin,
            "preempt_priority_margin",
        )

    def select(
        self,
        goals: Iterable[ManagedGoal],
        *,
        current_goal_id: Optional[str] = None,
    ) -> GoalSelection:
        candidates = [goal for goal in goals if not goal.status.terminal]
        candidates.sort(
            key=lambda goal: (
                -goal.priority,
                -goal.progress,
                goal.sequence,
            )
        )
        eligible_ids = tuple(goal.goal_id for goal in candidates)

        if not candidates:
            return GoalSelection(
                selected_goal_id=None,
                previous_goal_id=current_goal_id,
                reason="no_eligible_goals",
                preempted=False,
                eligible_goal_ids=eligible_ids,
            )

        top = candidates[0]
        if current_goal_id is None:
            return GoalSelection(
                selected_goal_id=top.goal_id,
                previous_goal_id=None,
                reason="highest_priority_goal",
                preempted=False,
                eligible_goal_ids=eligible_ids,
            )

        current = next(
            (goal for goal in candidates if goal.goal_id == current_goal_id),
            None,
        )
        if current is None:
            return GoalSelection(
                selected_goal_id=top.goal_id,
                previous_goal_id=current_goal_id,
                reason="current_goal_not_eligible",
                preempted=False,
                eligible_goal_ids=eligible_ids,
            )

        if top.goal_id == current.goal_id:
            return GoalSelection(
                selected_goal_id=current.goal_id,
                previous_goal_id=current.goal_id,
                reason="continue_current_goal",
                preempted=False,
                eligible_goal_ids=eligible_ids,
            )

        required_priority = current.priority + self.preempt_priority_margin
        if top.priority > required_priority:
            return GoalSelection(
                selected_goal_id=top.goal_id,
                previous_goal_id=current.goal_id,
                reason="higher_priority_goal_preempts_current",
                preempted=True,
                eligible_goal_ids=eligible_ids,
            )

        return GoalSelection(
            selected_goal_id=current.goal_id,
            previous_goal_id=current.goal_id,
            reason="current_goal_retained",
            preempted=False,
            eligible_goal_ids=eligible_ids,
        )


class GoalLifecycleManager:
    """Owns goal lifecycle and deterministic current-goal selection.

    This module is intentionally independent from Agent, LLM, memory, and
    execution. It provides a safe boundary that later stages can integrate.
    """

    _ALLOWED_TRANSITIONS = {
        GoalStatus.PENDING: {GoalStatus.ACTIVE, GoalStatus.CANCELLED},
        GoalStatus.ACTIVE: {
            GoalStatus.PAUSED,
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.CANCELLED,
        },
        GoalStatus.PAUSED: {
            GoalStatus.ACTIVE,
            GoalStatus.CANCELLED,
            GoalStatus.FAILED,
        },
        GoalStatus.COMPLETED: set(),
        GoalStatus.FAILED: set(),
        GoalStatus.CANCELLED: set(),
    }

    def __init__(
        self,
        *,
        selection_policy: Optional[GoalSelectionPolicy] = None,
    ) -> None:
        self.selection_policy = selection_policy or GoalSelectionPolicy()
        self._goals: dict[str, ManagedGoal] = {}
        self._sequence = 0
        self._current_goal_id: Optional[str] = None

    @property
    def current_goal_id(self) -> Optional[str]:
        return self._current_goal_id

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
        goal_id = str(goal_id).strip()
        if not goal_id:
            raise ValueError("goal_id must not be empty.")
        if goal_id in self._goals:
            raise GoalManagementError(f"Goal already exists: {goal_id!r}.")
        if parent_id is not None:
            parent_id = str(parent_id).strip()
            if parent_id not in self._goals:
                raise GoalHierarchyError(f"Parent goal does not exist: {parent_id!r}.")
            if self._goals[parent_id].status == GoalStatus.CANCELLED:
                raise GoalHierarchyError("A cancelled goal cannot receive a child.")

        self._sequence += 1
        goal = ManagedGoal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            parent_id=parent_id,
            payload=payload,
            metadata=metadata or {},
            sequence=self._sequence,
        )
        self._goals[goal_id] = goal
        return goal

    def get_goal(self, goal_id: str) -> ManagedGoal:
        try:
            return self._goals[goal_id]
        except KeyError as exc:
            raise GoalNotFoundError(f"Unknown goal: {goal_id!r}.") from exc

    def list_goals(self, *, include_terminal: bool = True) -> tuple[ManagedGoal, ...]:
        goals = list(self._goals.values())
        if not include_terminal:
            goals = [goal for goal in goals if not goal.status.terminal]
        goals.sort(key=lambda goal: goal.sequence)
        return tuple(goals)

    def children_of(self, goal_id: str) -> tuple[ManagedGoal, ...]:
        self.get_goal(goal_id)
        return tuple(
            goal
            for goal in self.list_goals()
            if goal.parent_id == goal_id
        )

    def activate(self, goal_id: str) -> ManagedGoal:
        return self._transition(goal_id, GoalStatus.ACTIVE)

    def pause(self, goal_id: str) -> ManagedGoal:
        goal = self._transition(goal_id, GoalStatus.PAUSED)
        if self._current_goal_id == goal_id:
            self._current_goal_id = None
        return goal

    def resume(self, goal_id: str) -> ManagedGoal:
        return self._transition(goal_id, GoalStatus.ACTIVE)

    def complete(self, goal_id: str) -> ManagedGoal:
        goal = self._transition(goal_id, GoalStatus.COMPLETED)
        goal = self._replace(goal, progress=1.0)
        if self._current_goal_id == goal_id:
            self._current_goal_id = None
        return goal

    def fail(self, goal_id: str) -> ManagedGoal:
        goal = self._transition(goal_id, GoalStatus.FAILED)
        if self._current_goal_id == goal_id:
            self._current_goal_id = None
        return goal

    def cancel(self, goal_id: str) -> ManagedGoal:
        goal = self._transition(goal_id, GoalStatus.CANCELLED)
        if self._current_goal_id == goal_id:
            self._current_goal_id = None
        return goal

    def update_progress(self, goal_id: str, progress: float) -> ManagedGoal:
        goal = self.get_goal(goal_id)
        if goal.status.terminal:
            raise GoalLifecycleError(
                f"Cannot update terminal goal: {goal_id!r}."
            )
        return self._replace(goal, progress=_bounded_number(progress, "progress"))

    def select_next_goal(self) -> GoalSelection:
        selection = self.selection_policy.select(
            self.list_goals(include_terminal=False),
            current_goal_id=self._current_goal_id,
        )
        selected = selection.selected_goal_id
        if selected is not None:
            selected_goal = self.get_goal(selected)
            if selected_goal.status == GoalStatus.PENDING:
                self.activate(selected)
            self._current_goal_id = selected
        elif self._current_goal_id is not None:
            self._current_goal_id = None
        return selection

    def set_current_goal(self, goal_id: Optional[str]) -> Optional[ManagedGoal]:
        if goal_id is None:
            self._current_goal_id = None
            return None
        goal = self.get_goal(goal_id)
        if goal.status.terminal:
            raise GoalLifecycleError("A terminal goal cannot be current.")
        if goal.status == GoalStatus.PENDING:
            goal = self.activate(goal_id)
        self._current_goal_id = goal_id
        return goal

    def snapshot(self) -> GoalManagerSnapshot:
        return GoalManagerSnapshot(
            goals=self.list_goals(),
            current_goal_id=self._current_goal_id,
            sequence=self._sequence,
        )

    def _transition(self, goal_id: str, target: GoalStatus) -> ManagedGoal:
        goal = self.get_goal(goal_id)
        if target not in self._ALLOWED_TRANSITIONS[goal.status]:
            raise GoalLifecycleError(
                f"Invalid goal transition: {goal.status.value} -> {target.value}."
            )
        return self._replace(goal, status=target)

    def _replace(self, goal: ManagedGoal, **changes: Any) -> ManagedGoal:
        replacement = ManagedGoal(
            goal_id=changes.get("goal_id", goal.goal_id),
            description=changes.get("description", goal.description),
            priority=changes.get("priority", goal.priority),
            parent_id=changes.get("parent_id", goal.parent_id),
            status=changes.get("status", goal.status),
            progress=changes.get("progress", goal.progress),
            payload=changes.get("payload", goal.payload),
            metadata=changes.get("metadata", goal.metadata),
            sequence=goal.sequence,
        )
        self._goals[goal.goal_id] = replacement
        return replacement


def _bounded_number(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return number
