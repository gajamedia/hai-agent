
from __future__ import annotations

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
    ) -> None:
        self.agent = create_simulation_agent(
            memory_path=memory_path,
        )
        self.environment = RoomEnvironment()
        self.lock = RLock()

    @property
    def allowed_actions(self) -> tuple[str, ...]:
        return ALLOWED_ACTIONS

    def get_state(self) -> dict[str, Any]:
        with self.lock:
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
    ) -> dict[str, Any]:
        if not actions:
            raise ValueError("At least one action is required.")

        if len(actions) > len(ALLOWED_ACTIONS):
            raise ValueError("Too many actions were supplied.")

        if not target_state:
            raise ValueError("target_state must not be empty.")

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
) -> ExecutionRuntime:
    return ExecutionRuntime(memory_path=memory_path)