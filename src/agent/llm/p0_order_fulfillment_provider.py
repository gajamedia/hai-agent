"""Step 41.4: second-domain P0 provider for deterministic order fulfillment.

This is an isolated in-memory benchmark fixture. It is intentionally distinct from
RoomEnvironment and is not a production inventory/shipping integration.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Iterable

from agent.core.action import Action
from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.cognition.agent_decision import DecisionMode
from agent.llm.p0_canonical_runtime_adapter import P0RuntimeScenario
from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.world.environment import Environment


_ACTION_RISKS = {
    "receive_stock": 0.10,
    "pick_order": 0.15,
    "pack_order": 0.20,
    "dispatch_order": 0.40,
}


class OrderFulfillmentEnvironment(Environment):
    """Small deterministic order-fulfillment world with observable state."""

    def __init__(self, *, stock_status: str = "expected", order_status: str = "created"):
        super().__init__()
        self.state = {
            "stock_status": stock_status,
            "order_status": order_status,
            "picked": False,
            "packed": False,
            "dispatched": False,
        }

    def execute(self, action: str) -> dict[str, object]:
        if action == "receive_stock":
            if self.state["stock_status"] != "expected":
                return {"success": False, "message": "Stock is not expected."}
            self.state["stock_status"] = "available"
            return {"success": True, "message": "Stock received and available."}

        if action == "pick_order":
            if self.state["stock_status"] != "available":
                return {"success": False, "message": "Stock is not available."}
            if self.state["order_status"] != "created" or self.state["picked"]:
                return {"success": False, "message": "Order cannot be picked."}
            self.state["picked"] = True
            self.state["order_status"] = "picked"
            return {"success": True, "message": "Order picked."}

        if action == "pack_order":
            if not self.state["picked"] or self.state["packed"]:
                return {"success": False, "message": "Order cannot be packed."}
            self.state["packed"] = True
            self.state["order_status"] = "packed"
            return {"success": True, "message": "Order packed."}

        if action == "dispatch_order":
            if not self.state["packed"] or self.state["dispatched"]:
                return {"success": False, "message": "Order cannot be dispatched."}
            self.state["dispatched"] = True
            self.state["order_status"] = "dispatched"
            return {"success": True, "message": "Order dispatched."}

        return {"success": False, "message": f"Unknown order-fulfillment action: {action}"}


def _actions(environment: OrderFulfillmentEnvironment) -> list[Action]:
    return [
        Action(
            name="receive_stock",
            description="Receive expected stock into available inventory",
            handler=lambda: environment.execute("receive_stock"),
            preconditions={"stock_status": "expected"},
            effects={"stock_status": "available"},
        ),
        Action(
            name="pick_order",
            description="Pick the order from available stock",
            handler=lambda: environment.execute("pick_order"),
            preconditions={"stock_status": "available", "order_status": "created"},
            effects={"picked": True, "order_status": "picked"},
        ),
        Action(
            name="pack_order",
            description="Pack a picked order",
            handler=lambda: environment.execute("pack_order"),
            preconditions={"picked": True, "packed": False},
            effects={"packed": True, "order_status": "packed"},
        ),
        Action(
            name="dispatch_order",
            description="Dispatch a packed order",
            handler=lambda: environment.execute("dispatch_order"),
            preconditions={"packed": True, "dispatched": False},
            effects={"dispatched": True, "order_status": "dispatched"},
        ),
    ]


class OrderFulfillmentP0ScenarioProvider:
    """30 deterministic tasks; 20 development tasks and 10 held-out variants.

    Holdout tasks use a different initial condition (stock already received) and
    require a different action composition. They are not used to tune this package.
    """

    DOMAIN = "order_fulfillment"
    DEVELOPMENT_IDS = frozenset(f"task-{i:02d}" for i in range(1, 21))
    HOLDOUT_IDS = frozenset(f"task-{i:02d}" for i in range(21, 31))

    def tasks(self) -> Iterable[P0BenchmarkTask]:
        tasks: list[P0BenchmarkTask] = []
        for number in range(1, 31):
            if number <= 10:
                route = ("receive_stock",)
                target = {"stock_status": "available"}
                initial_stock = "expected"
                description = "Receive incoming stock and verify inventory availability."
            elif number <= 20:
                route = ("receive_stock", "pick_order", "pack_order", "dispatch_order")
                target = {"order_status": "dispatched", "dispatched": True}
                initial_stock = "expected"
                description = "Receive stock and fulfill an order through dispatch."
            else:
                # Held-out task composition: inventory is already received. The agent
                # must begin from a different initial state and complete fulfillment.
                route = ("pick_order", "pack_order", "dispatch_order")
                target = {"order_status": "dispatched", "dispatched": True}
                initial_stock = "available"
                description = "Fulfill an order from already available inventory."
            split = "development" if number <= 20 else "holdout"
            tasks.append(
                P0BenchmarkTask(
                    task_id=f"task-{number:02d}",
                    description=description,
                    payload={
                        "domain": self.DOMAIN,
                        "split": split,
                        "route": route,
                        "target_state": target,
                        "initial_stock_status": initial_stock,
                        "scenario_version": "order-fulfillment-p0-v1",
                    },
                )
            )
        return tuple(tasks)

    def build_scenario(self, task: P0BenchmarkTask, run_index: int) -> P0RuntimeScenario:
        if run_index not in (1, 2, 3):
            raise ValueError("run_index must be in [1, 3].")
        task_by_id = {item.task_id: item for item in self.tasks()}
        canonical_task = task_by_id.get(task.task_id)
        if canonical_task is None or canonical_task.payload != task.payload:
            raise ValueError(f"Unknown or altered order-fulfillment task: {task.task_id}")

        payload = task.payload
        environment = OrderFulfillmentEnvironment(
            stock_status=str(payload["initial_stock_status"])
        )
        agent = Agent(name=f"p0-fulfillment-{task.task_id}-run-{run_index:02d}")
        agent.register_actions(_actions(environment))

        goal = Goal(description=task.description, target_state=dict(payload["target_state"]))
        agent.set_goal(goal)

        action_names = tuple(payload["route"])
        steps = tuple(
            LLMPlanStep(
                index=index,
                action=action_name,
                purpose=f"Execute {action_name} for {task.task_id}",
                preconditions=(),
                expected_observation=f"Order fulfillment state reflects {action_name}",
                success_criteria=f"Action {action_name} reports success",
                risk="low",
            )
            for index, action_name in enumerate(action_names, start=1)
        )
        plan = LLMPlan(
            goal=task.description,
            rationale="Deterministic order-fulfillment scenario using a separate domain model.",
            steps=steps,
            assumptions=("Order-fulfillment fixture starts in the task's declared state.",),
            constraints=("Use registered Agent actions through the canonical runtime only.",),
            stop_conditions=("Stop immediately on a failed action.",),
            confidence=1.0,
            provider="p0-order-fulfillment-fixture",
            model="deterministic-fixture-v1",
            raw_content="",
        )

        def decision_factory(request):
            return SimpleNamespace(
                mode=DecisionMode.EXPLOIT,
                action=request.action,
                reason="Step 41.4 deterministic order-fulfillment scenario",
                confidence=1.0,
                explore_allowed=False,
                exploit_allowed=True,
            )

        # Count repeated successful state-changing actions as fixture-level duplicates.
        side_effects: list[str] = []
        original_execute = environment.execute

        def tracked_execute(action: str):
            before = environment.observe()
            result = original_execute(action)
            after = environment.observe()
            if result.get("success") and before != after:
                side_effects.append(action)
            return result

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
                count - 1 for count in Counter(side_effects).values() if count > 1
            ),
            metadata={
                "domain": self.DOMAIN,
                "split": str(payload["split"]),
                "scenario_version": "order-fulfillment-p0-v1",
                "run_index": run_index,
                "ledger_kind": "isolated_in_memory_fixture",
            },
        )


def build_provider() -> OrderFulfillmentP0ScenarioProvider:
    return OrderFulfillmentP0ScenarioProvider()


__all__ = [
    "OrderFulfillmentEnvironment",
    "OrderFulfillmentP0ScenarioProvider",
    "build_provider",
]
