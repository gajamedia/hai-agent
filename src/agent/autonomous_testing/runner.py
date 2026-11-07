"""Sequential autonomous test-case runner.

The runner orchestrates existing AT-02 components. It does not create an
executor or call action handlers: every action is delegated to the isolated
environment's ExecutionRuntime.run() path.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .contracts import (
    CaseStatus,
    RecoveryPolicy,
    ScenarioDefinition,
    TestCaseResult,
)
from .environment import IsolatedEnvironmentError, IsolatedTestEnvironment
from .evaluator import IndependentEvaluator
from .fault_injection import (
    FaultInjectionAdapter,
    UnknownExecutionOutcome,
)
from .recovery import RecoveryAction, RecoveryController


class SequentialTestRunner:
    """Run one case at a time, with isolated runtime and bounded recovery."""

    def __init__(
        self,
        *,
        environment_factory=IsolatedTestEnvironment,
        evaluator: Optional[IndependentEvaluator] = None,
        recovery_controller: Optional[RecoveryController] = None,
    ) -> None:
        self.environment_factory = environment_factory
        self.evaluator = evaluator or IndependentEvaluator()
        self.recovery_controller = recovery_controller or RecoveryController()
        self._running = False

    def run_case(
        self,
        test_case,
        *,
        recovery_policy: Optional[RecoveryPolicy] = None,
        environment_config=None,
    ) -> TestCaseResult:
        from .contracts import AutonomousTestCase, TestEnvironmentConfig

        if not isinstance(test_case, AutonomousTestCase):
            raise TypeError("test_case must be AutonomousTestCase")
        if self._running:
            raise RuntimeError("SequentialTestRunner does not allow concurrent runs")

        policy = recovery_policy or RecoveryPolicy()
        if not isinstance(policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be RecoveryPolicy")
        config = environment_config or TestEnvironmentConfig()
        if not isinstance(config, TestEnvironmentConfig):
            raise TypeError("environment_config must be TestEnvironmentConfig")

        scenario = test_case.scenario
        started_at = datetime.now(timezone.utc)
        self._running = True
        environment = None
        fault_adapter = None
        recovery_attempts = 0
        error: Optional[str] = None
        evidence: list[str] = []
        final_state = None
        blocked_reason = None

        try:
            actions = self._validated_solution_path(scenario)
            if not actions:
                blocked_reason = (
                    "Scenario has no executable solution_path metadata; "
                    "runner will not invent actions."
                )
                return self._blocked_result(
                    test_case.case_id, scenario, started_at, blocked_reason
                )

            environment = self.environment_factory(
                initial_state=dict(scenario.initial_state),
                config=config,
            )
            fault_adapter = FaultInjectionAdapter(
                environment,
                policy=scenario.fault_policy,
            )

            for action_index, action in enumerate(actions, start=1):
                attempt = 1
                while True:
                    try:
                        response = fault_adapter.run(
                            actions=[action],
                            goal_description=scenario.goal_description,
                            target_state=dict(scenario.goal),
                        )
                        if response.get("status") == "injected_fault":
                            fault = response.get("fault") or {}
                            decision = self.recovery_controller.decide(
                                policy=policy,
                                fault_type=fault.get("fault_type"),
                                attempt=attempt,
                                outcome_known=bool(fault.get("outcome_known")),
                                action_executed=fault.get("action_executed"),
                            )
                            evidence.append(
                                f"Action {action_index} ({action}) fault: "
                                f"{fault.get('fault_type')}; recovery={decision.action.value}."
                            )
                            if decision.action == RecoveryAction.RETRY:
                                recovery_attempts += 1
                                attempt += 1
                                continue
                            if decision.action == RecoveryAction.RECONCILE:
                                blocked_reason = decision.reason
                            else:
                                error = decision.reason
                            break

                        # A successful HTTP-like/runtime response is not proof
                        # of goal completion. Continue with the planned steps.
                        evidence.append(
                            f"Action {action_index} ({action}) dispatched through isolated runtime."
                        )
                        break

                    except UnknownExecutionOutcome as exc:
                        evidence.append(
                            f"Action {action_index} ({action}) outcome unknown; "
                            "automatic retry prohibited."
                        )
                        blocked_reason = str(exc)
                        break
                    except Exception as exc:
                        # Unexpected errors are not retried. Reconciliation by
                        # observed state is attempted once after the loop.
                        error = f"{type(exc).__name__}: {exc}"
                        evidence.append(
                            f"Action {action_index} ({action}) raised an exception; no automatic retry."
                        )
                        blocked_reason = "Execution interrupted; final state must be reconciled."
                        break

                if blocked_reason is not None or error is not None:
                    break

            try:
                final_state = environment.observe()
                evidence.append("Final state obtained from isolated environment observation.")
            except Exception as exc:
                error = error or f"Final observation failed: {type(exc).__name__}: {exc}"
                final_state = None
                evidence.append("Final state observation unavailable.")

            # Unknown outcome is not retried. A successful independent
            # observation may nevertheless establish the actual final state.
            result = self.evaluator.evaluate(
                scenario,
                final_state,
                case_id=test_case.case_id,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                recovery_attempts=recovery_attempts,
                error=error,
                evidence=tuple(evidence),
            )
            metadata = dict(result.metadata)
            metadata.update({
                "runner": "SequentialTestRunner",
                "runner_version": 1,
                "actions_planned": list(actions),
                "fault_records": [
                    record.to_dict() for record in (fault_adapter.records if fault_adapter else ())
                ],
                "reconciliation_required": blocked_reason is not None,
                "reconciliation_note": blocked_reason,
            })
            return TestCaseResult(
                case_id=result.case_id,
                status=result.status,
                started_at=result.started_at,
                finished_at=result.finished_at,
                evaluation=result.evaluation,
                recovery_attempts=result.recovery_attempts,
                error=result.error,
                metadata=metadata,
            )
        except IsolatedEnvironmentError as exc:
            return self._blocked_result(
                test_case.case_id,
                scenario,
                started_at,
                f"Isolated environment unavailable: {exc}",
            )
        finally:
            try:
                if environment is not None:
                    environment.close()
            finally:
                self._running = False

    @staticmethod
    def _validated_solution_path(scenario: ScenarioDefinition) -> list[str]:
        path = scenario.metadata.get("solution_path")
        if not isinstance(path, (list, tuple)) or not path:
            return []
        allowed = {"open_box", "take_key", "find_spare_key", "unlock_door"}
        if any(not isinstance(action, str) or action not in allowed for action in path):
            raise ValueError("Scenario solution_path contains unsupported actions.")
        return list(path)

    def _blocked_result(
        self,
        case_id: str,
        scenario: ScenarioDefinition,
        started_at: datetime,
        reason: str,
    ) -> TestCaseResult:
        return TestCaseResult(
            case_id=case_id,
            status=CaseStatus.BLOCKED,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            evaluation=None,
            error=reason,
            metadata={
                "runner": "SequentialTestRunner",
                "runner_version": 1,
                "decision_basis": "prerequisite_or_environment_block",
            },
        )


__all__ = ["SequentialTestRunner"]
