
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any

from agent.cognition.agent_decision import (
    AgentDecision,
    DecisionMode,
)
from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.world.room import RoomEnvironment

ALLOWED_ACTIONS = (
    "open_box",
    "take_key",
    "find_spare_key",
    "unlock_door",
)


def _goal_satisfied_for_runtime(
    state: dict[str, Any],
    target_state: dict[str, Any],
) -> bool:
    return bool(target_state) and all(
        state.get(key) == expected
        for key, expected in target_state.items()
    )


def _simulation_handler(**kwargs: Any) -> dict[str, Any]:
    """
    Intentionally inert handler.

    RoomEnvironment is responsible for applying simulated
    state changes when the registered action is executed.
    """
    return {"success": True}


def create_simulation_agent(
    memory_path: str | Path | None = None,
) -> Agent:
    """Create a dedicated, persistent Agent for room simulation."""
    if memory_path is None:
        memory_path = os.getenv(
            "HAI_SIMULATION_MEMORY_PATH",
            "data/simulation_memory.db",
        )

    agent = Agent(
        "hai-simulation-agent",
        memory_path=memory_path,
    )

    actions = [
        Action(
            name="open_box",
            description="Open the closed box.",
            handler=_simulation_handler,
            preconditions={"box": "closed"},
            effects={"box": "open"},
        ),
        Action(
            name="take_key",
            description="Take the key from the open box.",
            handler=_simulation_handler,
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find the available spare key.",
            handler=_simulation_handler,
            preconditions={"spare_key": "available"},
            effects={"key": "agent"},
        ),
        Action(
            name="unlock_door",
            description="Unlock the door using the key.",
            handler=_simulation_handler,
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={"door": "unlocked"},
        ),
    ]

    agent.register_actions(actions)
    return agent


class ExecutionRuntime:
    """
    Process-local execution runtime for RoomEnvironment.

    This runtime is independent of the cognitive API runtime.
    Its lock serializes execution, state reads, and resets.
    """

    def __init__(
        self,
        memory_path: str | Path | None = None,
        environment: Any | None = None,
    ) -> None:
        self.agent = create_simulation_agent(
            memory_path=memory_path,
        )
        self.environment = (
            environment if environment is not None
            else RoomEnvironment()
        )
        self.lock = RLock()

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return ALLOWED_ACTIONS

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.environment.observe())

    def set_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply a validated state to the isolated Room simulation."""
        if not isinstance(state, dict):
            raise ValueError("state must be a JSON object.")

        allowed_values = {
            "door": {"locked", "unlocked"},
            "box": {"closed", "open"},
            "key": {"inside_box", "agent", "missing"},
            "spare_key": {"available", "found", "unavailable"},
            "agent_location": {"room"},
        }
        expected_keys = set(allowed_values)
        received_keys = set(state)
        if received_keys != expected_keys:
            missing = sorted(expected_keys - received_keys)
            unknown = sorted(received_keys - expected_keys)
            details = []
            if missing:
                details.append("missing fields: " + ", ".join(missing))
            if unknown:
                details.append("unknown fields: " + ", ".join(unknown))
            raise ValueError("Invalid world state (" + "; ".join(details) + ").")

        normalized: dict[str, str] = {}
        for field, choices in allowed_values.items():
            value = state[field]
            if not isinstance(value, str) or value not in choices:
                raise ValueError(
                    f"Invalid value for '{field}'. Allowed values: {', '.join(sorted(choices))}."
                )
            normalized[field] = value

        with self.lock:
            # Mutate only the sandbox environment, under the same lock used by
            # state reads, resets, and execution. This is setup, not an action.
            self.environment.state = dict(normalized)
            return dict(self.environment.observe())

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.environment.reset()
            return dict(self.environment.observe())

    def run(
        self,
        actions: list[str],
        goal_description: str,
        target_state: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not actions:
            raise ValueError("At least one action is required.")

        if len(actions) > len(ALLOWED_ACTIONS):
            raise ValueError("Too many actions were supplied.")

        if not target_state:
            raise ValueError("target_state must not be empty.")

        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                raise ValueError("idempotency_key must be a non-empty string.")
            if len(actions) != 1:
                raise ValueError(
                    "idempotency_key is supported only for single-action requests."
                )
            execute_idempotent = getattr(
                self.environment, "execute_idempotent", None
            )
            if not callable(execute_idempotent):
                raise ValueError(
                    "The configured environment does not support single-action idempotency."
                )

        unknown = [
            name for name in actions
            if name not in ALLOWED_ACTIONS
        ]
        if unknown:
            raise ValueError(
                f"Unsupported action(s): {', '.join(unknown)}"
            )

        with self.lock:
            state_before = dict(self.environment.observe())

            if idempotency_key is not None:
                fingerprint_payload = {
                    "actions": list(actions),
                    "goal_description": goal_description,
                    "target_state": target_state,
                }
                request_fingerprint = hashlib.sha256(
                    json.dumps(
                        fingerprint_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                idempotent_result = self.environment.execute_idempotent(
                    action=actions[0],
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                )
                state_after = dict(self.environment.observe())
                goal_satisfied = _goal_satisfied_for_runtime(
                    state_after, target_state
                )
                result_payload = dict(idempotent_result)
                replayed = bool(result_payload.pop("replayed", False))
                action_success = bool(result_payload.get("success", False))
                return {
                    "status": "ok",
                    "mode": DecisionMode.EXPLOIT.value,
                    "reason": "Explicit allowlisted simulation execution request.",
                    "actions": list(actions),
                    "success": action_success,
                    "goal_satisfied": goal_satisfied,
                    "goal": {
                        "description": goal_description,
                        "target_state": dict(target_state),
                        "completed": goal_satisfied,
                        "failed": not goal_satisfied,
                    },
                    "state_before": state_before,
                    "state_after": state_after,
                    "steps": [{
                        "action": actions[0],
                        "success": action_success,
                        "message": result_payload.get("message"),
                    }],
                    "telemetry_events": [],
                    "replayed": replayed,
                    "idempotency_key": idempotency_key,
                }

            goal = Goal(
                description=goal_description,
                target_state=dict(target_state),
            )

            # The caller explicitly requests these actions.
            # This does not claim the agent autonomously chose them.
            decision = AgentDecision(
                mode=DecisionMode.EXPLOIT,
                reason=(
                    "Explicit allowlisted simulation execution request."
                ),
                uncertainty=0.1,
                expected_information_gain=0.0,
                risk=0.1,
            )

            registered_actions = [
                self.agent.actions[name]
                for name in actions
            ]

            if len(registered_actions) == 1:
                result = self.agent.unified_executor.execute_action(
                    decision=decision,
                    action=registered_actions[0],
                    agent=self.agent,
                    environment=self.environment,
                    goal=goal,
                    world_state=state_before,
                )
            else:
                result = self.agent.unified_executor.execute_plan(
                    decision=decision,
                    agent=self.agent,
                    environment=self.environment,
                    goal=goal,
                    actions=registered_actions,
                    world_state=state_before,
                )

            state_after = dict(self.environment.observe())
            execution_success = bool(
                getattr(result, "success", False)
            )
            goal_satisfied = goal.is_satisfied(state_after)

            if goal_satisfied:
                goal.complete()

            telemetry_events: list[dict[str, Any]] = []
            recorder = getattr(
                self.agent,
                "record_capability_telemetry",
                None,
            )

            if callable(recorder):
                recorder(
                    "execution",
                    execution_success,
                    source="execution_api",
                )
                telemetry_events.append({
                    "capability": "execution",
                    "success": execution_success,
                })

                recorder(
                    "goal_achievement",
                    goal_satisfied,
                    source="execution_api",
                )
                telemetry_events.append({
                    "capability": "goal_achievement",
                    "success": goal_satisfied,
                })

                if len(actions) > 1:
                    recorder(
                        "exploitation",
                        execution_success,
                        source="execution_api",
                    )
                    telemetry_events.append({
                        "capability": "exploitation",
                        "success": execution_success,
                    })

            steps: list[dict[str, Any]] = []

            for step in (
                getattr(result, "plan_execution_results", ()) or ()
            ):
                steps.append({
                    "action": getattr(step, "action", None),
                    "success": bool(
                        getattr(step, "success", False)
                    ),
                    "message": getattr(step, "message", None),
                })

            direct_result = getattr(result, "direct_result", None)

            if isinstance(direct_result, dict) and not steps:
                steps.append({
                    "action": actions[0],
                    "success": bool(
                        direct_result.get(
                            "success",
                            execution_success,
                        )
                    ),
                    "message": direct_result.get("message"),
                })

            return {
                "status": "ok",
                "mode": decision.mode.value,
                "reason": decision.reason,
                "actions": list(actions),
                "success": execution_success,
                "goal_satisfied": goal_satisfied,
                "goal": {
                    "description": goal.description,
                    "target_state": dict(goal.target_state),
                    "completed": goal.completed,
                    "failed": goal.failed,
                },
                "state_before": state_before,
                "state_after": state_after,
                "steps": steps,
                "telemetry_events": telemetry_events,
            }


def create_execution_runtime(
    memory_path: str | Path | None = None,
    environment: Any | None = None,
) -> ExecutionRuntime:
    return ExecutionRuntime(
        memory_path=memory_path,
        environment=environment,
    )