"""Deterministic, data-only scenario generation for autonomous testing."""
from __future__ import annotations

import random
from typing import Optional

from .contracts import (
    AutonomousTestCase,
    Difficulty,
    FaultInjectionPolicy,
    FaultType,
    ScenarioDefinition,
)

# Declarative transitions only; no runtime or action handler is invoked here.
_ACTIONS = ("open_box", "take_key", "find_spare_key", "unlock_door")
# Stable order is part of the seeded fault schedule. Do not reorder casually.
_FAULT_SCHEDULE = (
    FaultType.ACTION_FAILURE,
    FaultType.TEMPORARY_UNAVAILABILITY,
    FaultType.OBSERVATION_INTERRUPTION,
)


def _apply_model_action(state: dict[str, str], action: str) -> Optional[dict[str, str]]:
    nxt = dict(state)
    if action == "open_box":
        if nxt["box"] != "closed":
            return None
        nxt["box"] = "open"
    elif action == "take_key":
        if nxt["box"] != "open" or nxt["key"] != "inside_box":
            return None
        nxt["key"] = "agent"
    elif action == "find_spare_key":
        if nxt["spare_key"] != "available":
            return None
        nxt["spare_key"] = "found"
    elif action == "unlock_door":
        if nxt["door"] != "locked" or not (
            nxt["key"] == "agent" or nxt["spare_key"] == "found"
        ):
            return None
        nxt["door"] = "unlocked"
    else:
        return None
    return nxt


def _shortest_path(initial: dict[str, str], goal: dict[str, str]) -> Optional[list[str]]:
    """Small deterministic BFS over a declarative state model."""
    queue: list[tuple[dict[str, str], list[str]]] = [(dict(initial), [])]
    seen = {tuple(sorted(initial.items()))}
    while queue:
        state, path = queue.pop(0)
        if all(state.get(k) == v for k, v in goal.items()):
            return path
        for action in _ACTIONS:
            nxt = _apply_model_action(state, action)
            if nxt is None:
                continue
            key = tuple(sorted(nxt.items()))
            if key not in seen:
                seen.add(key)
                queue.append((nxt, path + [action]))
    return None


class AutonomousScenarioGenerator:
    """Generate reproducible scenario definitions without executing actions."""

    def __init__(self, seed: int = 0):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.seed = seed
        self._rng = random.Random(seed)

    @staticmethod
    def _goal_for(initial: dict[str, str]) -> dict[str, str]:
        return {"door": "unlocked"}

    def generate_scenario(
        self,
        index: int,
        difficulty: Difficulty | str = Difficulty.MEDIUM,
        *,
        scenario_id: Optional[str] = None,
        inject_fault: bool = False,
        fault_type: FaultType | str = FaultType.ACTION_FAILURE,
    ) -> ScenarioDefinition:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("index must be a non-negative integer")
        difficulty = Difficulty(difficulty)
        try:
            selected_fault_type = FaultType(fault_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported fault_type") from exc

        seed = self.seed + index
        rng = random.Random(seed)
        templates = {
            Difficulty.EASY: [
                {"door": "locked", "box": "open", "key": "agent", "spare_key": "unavailable", "agent_location": "room"},
                {"door": "locked", "box": "closed", "key": "missing", "spare_key": "found", "agent_location": "room"},
            ],
            Difficulty.MEDIUM: [
                {"door": "locked", "box": "closed", "key": "inside_box", "spare_key": "unavailable", "agent_location": "room"},
                {"door": "locked", "box": "closed", "key": "missing", "spare_key": "available", "agent_location": "room"},
                {"door": "locked", "box": "open", "key": "inside_box", "spare_key": "unavailable", "agent_location": "room"},
            ],
            Difficulty.HARD: [
                {"door": "locked", "box": "closed", "key": "inside_box", "spare_key": "available", "agent_location": "room"},
                {"door": "locked", "box": "open", "key": "missing", "spare_key": "available", "agent_location": "room"},
                {"door": "locked", "box": "closed", "key": "missing", "spare_key": "available", "agent_location": "room"},
            ],
        }
        candidates = templates[difficulty][:]
        rng.shuffle(candidates)
        selected = next(
            (state for state in candidates if _shortest_path(state, self._goal_for(state)) is not None),
            None,
        )
        if selected is None:
            raise RuntimeError(f"No reachable template for difficulty {difficulty.value}")

        goal = self._goal_for(selected)
        path = _shortest_path(selected, goal) or []
        fault_policy = None
        if inject_fault:
            fault_policy = FaultInjectionPolicy(
                enabled=True,
                fault_type=selected_fault_type,
                target_action=path[0] if path else "unlock_door",
                trigger_step=1,
                max_occurrences=1,
            )

        sid = scenario_id or f"scenario-{seed:08d}"
        return ScenarioDefinition(
            scenario_id=sid,
            name=f"{difficulty.value.title()} room scenario {index + 1}",
            difficulty=difficulty,
            seed=seed,
            initial_state=selected,
            goal=goal,
            goal_description="Unlock the room door",
            fault_policy=fault_policy,
            metadata={
                "generator": "AutonomousScenarioGenerator",
                "generator_version": 2,
                "solution_path": path,
                "solution_length": len(path),
                "model_verified_reachable": True,
                "fault_schedule": [x.value for x in _FAULT_SCHEDULE],
            },
        )

    def generate_cases(
        self,
        count: int,
        difficulty: Difficulty | str = Difficulty.MEDIUM,
        *,
        case_id_prefix: str = "case",
        inject_fault_every: int = 0,
    ) -> tuple[AutonomousTestCase, ...]:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        if not isinstance(inject_fault_every, int) or isinstance(inject_fault_every, bool) or inject_fault_every < 0:
            raise ValueError("inject_fault_every must be a non-negative integer")
        if not isinstance(case_id_prefix, str) or not case_id_prefix.strip():
            raise ValueError("case_id_prefix must be non-empty")

        cases = []
        injected_count = 0
        for i in range(count):
            inject = bool(inject_fault_every and (i + 1) % inject_fault_every == 0)
            fault_type = _FAULT_SCHEDULE[injected_count % len(_FAULT_SCHEDULE)] if inject else FaultType.ACTION_FAILURE
            if inject:
                injected_count += 1
            scenario = self.generate_scenario(
                i, difficulty, inject_fault=inject, fault_type=fault_type,
            )
            cases.append(AutonomousTestCase(
                case_id=f"{case_id_prefix}-{i + 1:04d}",
                scenario=scenario,
                metadata={
                    "batch_index": i,
                    "generator_seed": self.seed,
                    "fault_schedule_index": injected_count - 1 if inject else None,
                },
            ))
        return tuple(cases)


def generate_scenarios(
    count: int,
    *,
    seed: int = 0,
    difficulty: Difficulty | str = Difficulty.MEDIUM,
) -> tuple[ScenarioDefinition, ...]:
    generator = AutonomousScenarioGenerator(seed=seed)
    return tuple(generator.generate_scenario(i, difficulty) for i in range(count))


__all__ = ["AutonomousScenarioGenerator", "generate_scenarios"]
