"""Step 39.6B: deterministic recovery/replay fixture for the canonical runtime.

This provider is test-only. It deliberately injects one recoverable failure before
any state-changing side effect, then supplies an explicit recovery plan. It is not a
production recovery policy or a general-purpose retry mechanism.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from typing import Iterable

from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.agent_decision import DecisionMode
from agent.llm.p0_canonical_runtime_adapter import (
    P0RuntimeScenario,
    P0RuntimeScenarioProvider,
)
from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.planning import LLMPlan, LLMPlanStep


class _Environment:
    def __init__(self) -> None:
        self.state = {"box": "closed"}

    def observe(self) -> dict[str, str]:
        return dict(self.state)


class _UnifiedExecutor:
    def execute(self, **kwargs):
        # The canonical binding requires the Agent executor to expose execute().
        # The binding should then dispatch through execute_action below.
        raise AssertionError(
            "Canonical binding should dispatch registered actions through execute_action."
        )

    def execute_action(self, *, action, environment, **kwargs):
        return action.execute(environment=environment)


class _Agent:
    def __init__(self, action: Action) -> None:
        self.actions = {action.name: action}
        self.unified_executor = _UnifiedExecutor()
        self.execution_security_policy = None


class RecoveryReplayScenarioProvider(P0RuntimeScenarioProvider):
    """30 canonical tasks; each scenario fails safely once, then recovers explicitly."""

    def tasks(self) -> Iterable[P0BenchmarkTask]:
        return tuple(
            P0BenchmarkTask(
                task_id=f"task-{number:02d}",
                description="Open the box, recovering after one transient safe failure.",
                payload={"target_state": {"box": "open"}, "scenario_version": "recovery-replay-v1"},
            )
            for number in range(1, 31)
        )

    @staticmethod
    def _plan(label: str, goal_description: str) -> LLMPlan:
        return LLMPlan(
            goal=goal_description,
            rationale=label,
            steps=(
                LLMPlanStep(
                    index=1,
                    action="open_box",
                    purpose=label,
                    preconditions=(),
                    expected_observation="box is open",
                    success_criteria="box == open",
                    risk="low",
                ),
            ),
            assumptions=("The transient failure occurs before a state mutation.",),
            constraints=("Use the canonical runtime; never retry implicitly.",),
            stop_conditions=("Stop the current plan on action failure.",),
            confidence=1.0,
            provider="p0-recovery-replay-fixture",
            model="deterministic-fixture-v1",
            raw_content="",
        )

    def build_scenario(self, task: P0BenchmarkTask, run_index: int) -> P0RuntimeScenario:
        if not 1 <= run_index <= 3:
            raise ValueError("run_index must be in [1, 3].")
        if task.task_id not in {row.task_id for row in self.tasks()}:
            raise ValueError(f"Unknown P0 task: {task.task_id}")

        environment = _Environment()
        attempts = {"count": 0}
        successful_effects: list[str] = []

        def open_box(*, environment):
            attempts["count"] += 1
            # Inject exactly one transient failure without mutating world state.
            if attempts["count"] == 1:
                return {"success": False, "action": "open_box", "error": "transient_safe_failure"}
            before = environment.observe()
            environment.state["box"] = "open"
            after = environment.observe()
            if before != after:
                successful_effects.append("open_box")
            return {"success": True, "action": "open_box"}

        action = Action(
            name="open_box",
            description="Open the box after a transient safe failure",
            handler=open_box,
            preconditions={"box": "closed"},
            effects={"box": "open"},
        )
        agent = _Agent(action)
        goal = Goal(description=task.description, target_state=dict(task.payload["target_state"]))
        plan = self._plan(
            "initial plan; expected to stop on the injected transient failure",
            goal.description,
        )

        def decision_factory(request):
            # Match the canonical single-action API contract: the decision must
            # carry the exact action requested by the runtime. SimpleNamespace is
            # intentional because the project decision model may not expose the
            # same constructor fields across versions.
            return SimpleNamespace(
                mode=DecisionMode.EXPLOIT,
                action=request.action,
                reason="deterministic recovery/replay fixture",
                confidence=1.0,
                explore_allowed=False,
                exploit_allowed=True,
            )

        def is_recoverable_failure(report, final_state) -> bool:
            return (
                not bool(getattr(report, "success", False))
                and final_state.get("box") == "closed"
                and attempts["count"] == 1
            )

        def recovery_plan_factory(scenario, report, final_state):
            if final_state.get("box") != "closed":
                raise RuntimeError("Refusing recovery because the world state is not the expected safe state.")
            return self._plan(
                "explicit recovery plan after verified safe transient failure",
                scenario.goal.description,
            )

        def duplicate_count() -> int:
            return sum(count - 1 for count in Counter(successful_effects).values() if count > 1)

        return P0RuntimeScenario(
            agent=agent,
            environment=environment,
            plan=plan,
            goal=goal,
            decision_factory=decision_factory,
            action_risks={"open_box": 0.10},
            approval_required_actions=(),
            approval_checker=None,
            risk_threshold=0.65,
            duplicate_harmful_side_effect_count=duplicate_count,
            is_recoverable_failure=is_recoverable_failure,
            recovery_plan_factory=recovery_plan_factory,
            metadata={
                "scenario_version": "recovery-replay-v1",
                "task_id": task.task_id,
                "run_index": run_index,
                "injected_failure": "one safe transient failure before mutation",
                "successful_side_effects": successful_effects,
                "action_attempt_count": attempts,
                "side_effect_ledger_kind": "in_memory_test_fixture",
            },
        )


def build_provider() -> RecoveryReplayScenarioProvider:
    """Factory path for tests and explicit benchmark experiments."""
    return RecoveryReplayScenarioProvider()


__all__ = ["RecoveryReplayScenarioProvider", "build_provider"]
