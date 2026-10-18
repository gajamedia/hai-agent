"""Step 39.5: controlled P0 scenario provider for the repository's RoomEnvironment.

This provider is a deterministic benchmark fixture for the built-in room domain.
It is not a production approval service and must not be used to authorize real actions.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Iterable

from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import DecisionMode
from agent.llm.p0_canonical_runtime_adapter import (
    P0RuntimeScenario,
    P0RuntimeScenarioProvider,
)
from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.tools.room_actions import create_room_actions
from agent.world.room import RoomEnvironment


_ACTION_RISKS = {
    "open_box": 0.10,
    "take_key": 0.10,
    "find_spare_key": 0.15,
    "unlock_door": 0.40,
}


class RoomP0ScenarioProvider(P0RuntimeScenarioProvider):
    """Build 30 isolated, deterministic tasks using the project's real room domain."""

    def tasks(self) -> Iterable[P0BenchmarkTask]:
        tasks: list[P0BenchmarkTask] = []
        for number in range(1, 31):
            # Keep the matrix deterministic and varied without intentionally
            # injecting failures into the first acceptance benchmark.
            if number <= 10:
                route = "box_key"
                target = {"door": "unlocked"}
                description = "Open the box, take its key, and unlock the door."
                spare_key_available = False
            elif number <= 20:
                route = "spare_key"
                target = {"door": "unlocked"}
                description = "Find the spare key and unlock the door."
                spare_key_available = True
            elif number <= 25:
                route = "open_box"
                target = {"box": "open"}
                description = "Open the box and verify its state."
                spare_key_available = True
            else:
                route = "take_key"
                target = {"key": "agent"}
                description = "Open the box, take the key, and verify possession."
                spare_key_available = False

            tasks.append(
                P0BenchmarkTask(
                    task_id=f"task-{number:02d}",
                    description=description,
                    payload={
                        "route": route,
                        "target_state": target,
                        "spare_key_available": spare_key_available,
                        "scenario_version": "room-p0-v1",
                    },
                )
            )
        return tuple(tasks)

    def build_scenario(self, task: P0BenchmarkTask, run_index: int) -> P0RuntimeScenario:
        if not 1 <= run_index <= 3:
            raise ValueError("run_index must be in [1, 3].")
        if task.task_id not in {item.task_id for item in self.tasks()}:
            raise ValueError(f"Unknown P0 task: {task.task_id}")

        payload = task.payload
        environment = RoomEnvironment(
            spare_key_available=bool(payload["spare_key_available"])
        )
        agent = Agent(name=f"p0-{task.task_id}-run-{run_index:02d}")
        agent.register_actions(create_room_actions(environment))

        target_state = dict(payload["target_state"])
        goal = Goal(description=task.description, target_state=target_state)
        agent.set_goal(goal)

        route = payload["route"]
        if route == "box_key":
            action_names = ("open_box", "take_key", "unlock_door")
        elif route == "spare_key":
            action_names = ("find_spare_key", "unlock_door")
        elif route == "open_box":
            action_names = ("open_box",)
        elif route == "take_key":
            action_names = ("open_box", "take_key")
        else:
            raise ValueError(f"Unsupported route: {route}")

        steps = tuple(
            LLMPlanStep(
                index=index,
                action=action_name,
                purpose=f"Execute {action_name} for {task.task_id}",
                preconditions=(),
                expected_observation=f"State reflects action {action_name}",
                success_criteria=f"Action {action_name} reports success",
                risk="low",
            )
            for index, action_name in enumerate(action_names, start=1)
        )
        plan = LLMPlan(
            goal=task.description,
            rationale="Deterministic controlled plan for the built-in RoomEnvironment.",
            steps=steps,
            assumptions=("RoomEnvironment starts from its documented reset state.",),
            constraints=("Use registered Agent actions through the canonical runtime only.",),
            stop_conditions=("Stop immediately on a failed action.",),
            confidence=1.0,
            provider="p0-room-fixture",
            model="deterministic-fixture-v1",
            raw_content="",
        )

        # The canonical binding requires an AgentDecision-like value. The executor
        # consumes mode/action/reason; the action object itself is resolved from the
        # Agent's registered action map by the canonical binding.
        def decision_factory(request):
            return SimpleNamespace(
                mode=DecisionMode.EXPLOIT,
                action=request.action,
                reason="P0 deterministic room scenario",
                confidence=1.0,
                explore_allowed=False,
                exploit_allowed=True,
            )

        # This in-memory ledger records duplicate state-changing action attempts.
        # It is intentionally isolated per scenario and is not a production ledger.
        side_effects: list[str] = []
        original_execute = environment.execute

        def tracked_execute(action: str):
            if action in {"open_box", "take_key", "find_spare_key", "unlock_door"}:
                before = environment.observe()
                result = original_execute(action)
                after = environment.observe()
                if result.get("success") and before != after:
                    side_effects.append(action)
                return result
            return original_execute(action)

        environment.execute = tracked_execute

        return P0RuntimeScenario(
            agent=agent,
            environment=environment,
            plan=plan,
            goal=goal,
            decision_factory=decision_factory,
            action_risks=dict(_ACTION_RISKS),
            approval_required_actions=(),
            approval_checker=None,
            risk_threshold=0.65,
            duplicate_harmful_side_effect_count=lambda: sum(
                count - 1
                for count in __import__("collections").Counter(side_effects).values()
                if count > 1
            ),
            metadata={
                "scenario_version": "room-p0-v1",
                "route": route,
                "run_index": run_index,
                "side_effect_ledger_kind": "isolated_room_fixture",
            },
        )


def build_provider() -> RoomP0ScenarioProvider:
    """Factory path used by Step 39.4 readiness and Step 39.2 benchmark CLIs."""
    return RoomP0ScenarioProvider()


__all__ = ["RoomP0ScenarioProvider", "build_provider"]
