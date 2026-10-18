"""Step 41.5 controlled fault-injection provider for recovery testing.

Wraps the existing Room P0 fixture, injects one deterministic transient action
failure, then asks the canonical adapter to execute the original plan as recovery.
This is a benchmark fixture, not a production retry policy.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from agent.core.action import Action
from agent.llm.p0_canonical_runtime_adapter import P0RuntimeScenario
from agent.llm.p0_room_scenario_provider import RoomP0ScenarioProvider
from agent.llm.p0_runtime_benchmark import P0BenchmarkTask
from agent.llm.planning import LLMPlan, LLMPlanStep


FAILURE_ACTION = "inject_transient_failure"


class RecoveryFaultInjectionProvider:
    """Injects exactly one registered, observable failed action per run."""

    def __init__(self) -> None:
        self._base = RoomP0ScenarioProvider()

    def tasks(self) -> Iterable[P0BenchmarkTask]:
        return tuple(self._base.tasks())

    def build_scenario(self, task: P0BenchmarkTask, run_index: int) -> P0RuntimeScenario:
        scenario = self._base.build_scenario(task, run_index)
        original_plan = scenario.plan

        # The action is registered like every other Agent action and returns an
        # explicit unsuccessful result without mutating the environment.
        scenario.agent.register_actions([
            Action(
                name=FAILURE_ACTION,
                description="Controlled Step 41.5 transient failure injection",
                handler=lambda: {
                    "success": False,
                    "message": "Injected transient failure for recovery benchmark.",
                },
                preconditions={},
                effects={},
            )
        ])

        failure_plan = LLMPlan(
            goal=original_plan.goal,
            rationale="Inject one controlled transient failure before recovery.",
            steps=(
                LLMPlanStep(
                    index=1,
                    action=FAILURE_ACTION,
                    purpose="Exercise explicit recoverable-failure handling",
                    preconditions=(),
                    expected_observation="Action returns a failed result and leaves state unchanged",
                    success_criteria="The adapter detects the failed execution and initiates recovery",
                    risk="low",
                ),
            ),
            assumptions=("The injected failure is deterministic and has no world-state side effect.",),
            constraints=("Recovery must use the same canonical runtime and explicit recovery plan.",),
            stop_conditions=("The initial failed plan must not be treated as goal completion.",),
            confidence=1.0,
            provider="step41-5-fault-injection",
            model="deterministic-fixture-v1",
            raw_content="",
        )

        def is_recoverable_failure(report, final_state) -> bool:
            return (
                not bool(getattr(report, "success", False))
                and not bool(scenario.goal.is_satisfied(final_state))
            )

        def recovery_plan_factory(_scenario, _report, _final_state):
            return original_plan

        risks = dict(scenario.action_risks)
        risks[FAILURE_ACTION] = 0.05
        metadata = {
            **dict(scenario.metadata or {}),
            "scenario_version": "room-p0-recovery-fault-injection-v1",
            "fault_injection": "one_transient_action_failure",
            "recovery_expected": True,
        }
        return replace(
            scenario,
            plan=failure_plan,
            action_risks=risks,
            is_recoverable_failure=is_recoverable_failure,
            recovery_plan_factory=recovery_plan_factory,
            metadata=metadata,
        )


def build_provider() -> RecoveryFaultInjectionProvider:
    return RecoveryFaultInjectionProvider()


__all__ = ["FAILURE_ACTION", "RecoveryFaultInjectionProvider", "build_provider"]
