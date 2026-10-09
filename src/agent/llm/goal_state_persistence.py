from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .goal_management import (
    GoalManagerSnapshot,
    GoalSelectionPolicy,
    GoalStatus,
    GoalLifecycleManager,
    ManagedGoal,
)


class PersistentGoalStateError(Exception):
    """Base exception for persistent goal state."""


class PersistentGoalStateConfigurationError(PersistentGoalStateError):
    """Raised for invalid persistence configuration."""


class PersistentGoalStateSerializationError(PersistentGoalStateError):
    """Raised when goal state cannot be represented as JSON."""


class PersistentGoalStateNotFoundError(PersistentGoalStateError):
    """Raised when no persisted goal state exists."""


class PersistentGoalStateVersionError(PersistentGoalStateError):
    """Raised when a persisted snapshot uses an unsupported schema."""


@dataclass(frozen=True)
class PersistentGoalStateSnapshot:
    """Serializable envelope around the immutable goal manager snapshot."""

    manager_snapshot: GoalManagerSnapshot
    preempt_priority_margin: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            margin = float(self.preempt_priority_margin)
        except (TypeError, ValueError) as exc:
            raise ValueError("preempt_priority_margin must be numeric.") from exc
        if not 0.0 <= margin <= 1.0:
            raise ValueError("preempt_priority_margin must be between 0.0 and 1.0.")
        object.__setattr__(self, "preempt_priority_margin", margin)
        if int(self.schema_version) != self.schema_version or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer.")


@dataclass(frozen=True)
class PersistentGoalStateLoadResult:
    """Result returned by a successful load/restore operation."""

    snapshot: PersistentGoalStateSnapshot
    manager: GoalLifecycleManager


class PersistentGoalStateStore:
    """SQLite-backed persistence for the Stage 25 goal lifecycle state.

    The store is independent from episodic/semantic memory and self-awareness
    persistence. It preserves goals, lifecycle status, progress, hierarchy,
    metadata, payload, insertion sequence, current goal, and selection policy.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        if path is None:
            raise PersistentGoalStateConfigurationError("path is required.")
        self.path = Path(path).expanduser()
        if self.path.exists() and self.path.is_dir():
            self.path = self.path / "goal_state.sqlite3"
        if not self.path.name:
            raise PersistentGoalStateConfigurationError("path must name a file.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save(self, snapshot: GoalManagerSnapshot, *, preempt_priority_margin: float = 0.0) -> PersistentGoalStateSnapshot:
        persisted = PersistentGoalStateSnapshot(
            manager_snapshot=snapshot,
            preempt_priority_margin=preempt_priority_margin,
            schema_version=self.SCHEMA_VERSION,
        )
        payload = _snapshot_to_dict(persisted)
        encoded = _json_dumps(payload)
        with self._connect() as connection:
            connection.execute("DELETE FROM goal_state WHERE singleton_id = 1")
            connection.execute(
                "INSERT INTO goal_state (singleton_id, schema_version, payload) VALUES (1, ?, ?)",
                (persisted.schema_version, encoded),
            )
        return persisted

    def save_manager(self, manager: GoalLifecycleManager) -> PersistentGoalStateSnapshot:
        return self.save(
            manager.snapshot(),
            preempt_priority_margin=manager.selection_policy.preempt_priority_margin,
        )

    def load(self) -> PersistentGoalStateSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT schema_version, payload FROM goal_state WHERE singleton_id = 1"
            ).fetchone()
        if row is None:
            raise PersistentGoalStateNotFoundError("No persisted goal state exists.")
        schema_version, encoded = row
        if int(schema_version) != self.SCHEMA_VERSION:
            raise PersistentGoalStateVersionError(
                f"Unsupported persisted schema version: {schema_version}."
            )
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise PersistentGoalStateSerializationError(
                "Persisted goal state contains invalid JSON."
            ) from exc
        return _snapshot_from_dict(decoded)

    def restore_manager(self) -> PersistentGoalStateLoadResult:
        snapshot = self.load()
        manager = GoalLifecycleManager(
            selection_policy=GoalSelectionPolicy(snapshot.preempt_priority_margin)
        )
        _restore_snapshot_into_manager(manager, snapshot.manager_snapshot)
        return PersistentGoalStateLoadResult(snapshot=snapshot, manager=manager)

    def exists(self) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM goal_state WHERE singleton_id = 1"
            ).fetchone()
        return row is not None

    def delete(self) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM goal_state WHERE singleton_id = 1"
            )
            return cursor.rowcount > 0

    def clear(self) -> bool:
        return self.delete()

    def count(self) -> int:
        return 1 if self.exists() else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS goal_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    schema_version INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )


def _snapshot_to_dict(snapshot: PersistentGoalStateSnapshot) -> dict[str, Any]:
    manager = snapshot.manager_snapshot
    return {
        "schema_version": snapshot.schema_version,
        "preempt_priority_margin": snapshot.preempt_priority_margin,
        "current_goal_id": manager.current_goal_id,
        "sequence": manager.sequence,
        "goals": [_goal_to_dict(goal) for goal in manager.goals],
    }


def _snapshot_from_dict(data: Mapping[str, Any]) -> PersistentGoalStateSnapshot:
    schema_version = data.get("schema_version")
    if int(schema_version) != 1:
        raise PersistentGoalStateVersionError(
            f"Unsupported persisted schema version: {schema_version}."
        )
    goals = tuple(_goal_from_dict(item) for item in data.get("goals", []))
    sequence = int(data.get("sequence", 0))
    if sequence < 0:
        raise PersistentGoalStateSerializationError("sequence cannot be negative.")
    current_goal_id = data.get("current_goal_id")
    if current_goal_id is not None:
        current_goal_id = str(current_goal_id)
        if current_goal_id not in {goal.goal_id for goal in goals}:
            raise PersistentGoalStateSerializationError(
                "current_goal_id does not reference a persisted goal."
            )
    if goals and sequence < max(goal.sequence for goal in goals):
        raise PersistentGoalStateSerializationError(
            "sequence is smaller than a persisted goal sequence."
        )
    manager_snapshot = GoalManagerSnapshot(
        goals=goals,
        current_goal_id=current_goal_id,
        sequence=sequence,
    )
    return PersistentGoalStateSnapshot(
        manager_snapshot=manager_snapshot,
        preempt_priority_margin=float(data.get("preempt_priority_margin", 0.0)),
        schema_version=schema_version,
    )


def _goal_to_dict(goal: ManagedGoal) -> dict[str, Any]:
    return {
        "goal_id": goal.goal_id,
        "description": goal.description,
        "priority": goal.priority,
        "parent_id": goal.parent_id,
        "status": goal.status.value,
        "progress": goal.progress,
        "payload": goal.payload,
        "metadata": dict(goal.metadata),
        "sequence": goal.sequence,
    }


def _goal_from_dict(data: Mapping[str, Any]) -> ManagedGoal:
    try:
        status = GoalStatus(str(data["status"]))
    except (KeyError, ValueError) as exc:
        raise PersistentGoalStateSerializationError("Invalid persisted goal status.") from exc
    try:
        sequence = int(data["sequence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistentGoalStateSerializationError("Invalid persisted goal sequence.") from exc
    if sequence <= 0:
        raise PersistentGoalStateSerializationError("Goal sequence must be positive.")
    try:
        return ManagedGoal(
            goal_id=str(data["goal_id"]),
            description=str(data["description"]),
            priority=float(data.get("priority", 0.5)),
            parent_id=(None if data.get("parent_id") is None else str(data["parent_id"])),
            status=status,
            progress=float(data.get("progress", 0.0)),
            payload=data.get("payload"),
            metadata=data.get("metadata") or {},
            sequence=sequence,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersistentGoalStateSerializationError("Invalid persisted goal record.") from exc


def _restore_snapshot_into_manager(
    manager: GoalLifecycleManager,
    snapshot: GoalManagerSnapshot,
) -> None:
    goals = tuple(sorted(snapshot.goals, key=lambda goal: goal.sequence))
    for goal in goals:
        manager.add_goal(
            goal.goal_id,
            goal.description,
            priority=goal.priority,
            parent_id=goal.parent_id,
            payload=goal.payload,
            metadata=goal.metadata,
        )
        if goal.progress:
            manager.update_progress(goal.goal_id, goal.progress)
        if goal.status == GoalStatus.PENDING:
            continue
        if goal.status in {GoalStatus.ACTIVE, GoalStatus.PAUSED, GoalStatus.COMPLETED, GoalStatus.FAILED}:
            manager.activate(goal.goal_id)
        if goal.status == GoalStatus.PAUSED:
            manager.pause(goal.goal_id)
        elif goal.status == GoalStatus.COMPLETED:
            manager.complete(goal.goal_id)
        elif goal.status == GoalStatus.FAILED:
            manager.fail(goal.goal_id)
        elif goal.status == GoalStatus.CANCELLED:
            manager.cancel(goal.goal_id)
    manager.set_current_goal(snapshot.current_goal_id)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise PersistentGoalStateSerializationError(
            "Goal payload or metadata is not JSON serializable."
        ) from exc
