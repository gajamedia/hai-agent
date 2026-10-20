from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


FeasibilityStatus = Literal[
    "reachable",
    "unreachable",
    "inconclusive",
]


@dataclass(frozen=True)
class GoalFeasibilityResult:
    status: FeasibilityStatus
    explored_states: int
    action_sequence: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "explored_states": self.explored_states,
            "action_sequence": list(self.action_sequence),
            "reason": self.reason,
        }


def assess_goal_feasibility(
    initial_state: Mapping[str, Any],
    goal: Mapping[str, Any],
    action_catalog: Sequence[Mapping[str, Any]],
    *,
    max_states: int = 10_000,
) -> GoalFeasibilityResult:
    """Search trusted action transitions without executing runtime actions."""
    if not isinstance(initial_state, Mapping):
        raise TypeError("initial_state must be a mapping")
    if not isinstance(goal, Mapping) or not goal:
        raise ValueError("goal must be a non-empty mapping")
    if not isinstance(action_catalog, Sequence) or isinstance(
        action_catalog, (str, bytes, bytearray)
    ):
        raise TypeError("action_catalog must be a sequence")
    if max_states < 1:
        raise ValueError("max_states must be >= 1")

    actions: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for item in action_catalog:
        if not isinstance(item, Mapping):
            continue

        action_id = item.get("id", item.get("name"))
        preconditions = item.get("preconditions", {})
        effects = item.get("effects", {})

        if not isinstance(action_id, str) or not action_id.strip():
            continue
        if not isinstance(preconditions, Mapping) or not isinstance(effects, Mapping):
            continue

        actions.append(
            (action_id.strip(), dict(preconditions), dict(effects))
        )

    start = dict(initial_state)

    def goal_matches(state: Mapping[str, Any]) -> bool:
        return all(state.get(key) == value for key, value in goal.items())

    def state_key(state: Mapping[str, Any]) -> tuple[tuple[str, str], ...] | None:
        try:
            return tuple(
                sorted(
                    (str(key), repr(value))
                    for key, value in state.items()
                )
            )
        except Exception:
            return None

    if goal_matches(start):
        return GoalFeasibilityResult(
            status="reachable",
            explored_states=1,
            reason="Goal already satisfied by the initial state.",
        )

    queue = deque([(start, ())])
    initial_key = state_key(start)
    if initial_key is None:
        return GoalFeasibilityResult(
            status="inconclusive",
            explored_states=0,
            reason="Initial state cannot be represented safely.",
        )

    visited = {initial_key}
    explored = 0

    while queue:
        if explored >= max_states:
            return GoalFeasibilityResult(
                status="inconclusive",
                explored_states=explored,
                reason="State exploration limit reached.",
            )

        state, path = queue.popleft()
        explored += 1

        for action_id, preconditions, effects in actions:
            if not all(
                state.get(key) == expected
                for key, expected in preconditions.items()
            ):
                continue

            next_state = dict(state)
            next_state.update(effects)
            next_path = path + (action_id,)

            if goal_matches(next_state):
                return GoalFeasibilityResult(
                    status="reachable",
                    explored_states=explored,
                    action_sequence=next_path,
                    reason="A valid action sequence reaches the goal.",
                )

            key = state_key(next_state)
            if key is None:
                return GoalFeasibilityResult(
                    status="inconclusive",
                    explored_states=explored,
                    reason="A resulting state cannot be represented safely.",
                )

            if key not in visited:
                visited.add(key)
                queue.append((next_state, next_path))

    return GoalFeasibilityResult(
        status="unreachable",
        explored_states=explored,
        reason=(
            "No path found in the supplied action catalog. "
            "This conclusion assumes the catalog is complete and its "
            "preconditions and effects fully describe the world transitions."
        ),
    )
