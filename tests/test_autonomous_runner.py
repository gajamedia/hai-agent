from pathlib import Path

import pytest

from agent.autonomous_testing.contracts import (
    AutonomousTestCase,
    CaseStatus,
    Difficulty,
    FaultInjectionPolicy,
    FaultType,
    RecoveryPolicy,
)
from agent.autonomous_testing.environment import IsolatedTestEnvironment
from agent.autonomous_testing.runner import SequentialTestRunner
from agent.autonomous_testing.scenario_generator import AutonomousScenarioGenerator


def _case(seed=31, difficulty=Difficulty.MEDIUM, fault_policy=None):
    scenario = AutonomousScenarioGenerator(seed).generate_scenario(0, difficulty)
    if fault_policy is not None:
        from dataclasses import replace
        from dataclasses import replace as dataclass_replace
        first_action = scenario.metadata["solution_path"][0]
        fault_policy = dataclass_replace(
            fault_policy,
            target_action=first_action,
        )
        scenario = replace(scenario, fault_policy=fault_policy)
    return AutonomousTestCase(case_id=f"case-{seed}", scenario=scenario)


def test_runs_one_case_and_independently_verifies_goal(tmp_path):
    case = _case()
    runner = SequentialTestRunner(
        environment_factory=lambda **kwargs: IsolatedTestEnvironment(
            **kwargs, temp_root=tmp_path
        )
    )
    result = runner.run_case(case)
    assert result.status == CaseStatus.PASSED
    assert result.evaluation.goal_satisfied is True
    assert result.metadata["runner"] == "SequentialTestRunner"
    assert list(result.metadata["actions_planned"]) == list(case.scenario.metadata["solution_path"])
    assert list(tmp_path.iterdir()) == []


def test_action_failure_retries_only_after_known_non_execution(tmp_path):
    policy = FaultInjectionPolicy(
        enabled=True,
        fault_type=FaultType.ACTION_FAILURE,
        target_action="open_box",
        trigger_step=1,
        max_occurrences=1,
    )
    case = _case(fault_policy=policy)
    runner = SequentialTestRunner(
        environment_factory=lambda **kwargs: IsolatedTestEnvironment(
            **kwargs, temp_root=tmp_path
        )
    )
    result = runner.run_case(case, recovery_policy=RecoveryPolicy(max_attempts=1))
    assert result.status == CaseStatus.PASSED
    assert result.recovery_attempts == 1
    assert len(result.metadata["fault_records"]) == 1


def test_unknown_observation_outcome_is_not_retried(tmp_path):
    policy = FaultInjectionPolicy(
        enabled=True,
        fault_type=FaultType.OBSERVATION_INTERRUPTION,
        target_action="open_box",
        trigger_step=1,
        max_occurrences=1,
    )
    case = _case(fault_policy=policy)
    runner = SequentialTestRunner(
        environment_factory=lambda **kwargs: IsolatedTestEnvironment(
            **kwargs, temp_root=tmp_path
        )
    )
    result = runner.run_case(case)
    assert result.recovery_attempts == 0
    assert result.metadata["reconciliation_required"] is True
    # State observation after interruption shows the first action happened,
    # but the remaining plan was deliberately not replayed.
    assert result.status in {CaseStatus.FAILED, CaseStatus.INCONCLUSIVE}
    assert list(tmp_path.iterdir()) == []


def test_disabled_recovery_does_not_retry_known_failure(tmp_path):
    policy = FaultInjectionPolicy(
        enabled=True,
        fault_type=FaultType.ACTION_FAILURE,
        target_action="open_box",
        trigger_step=1,
        max_occurrences=1,
    )
    case = _case(fault_policy=policy)
    runner = SequentialTestRunner(
        environment_factory=lambda **kwargs: IsolatedTestEnvironment(
            **kwargs, temp_root=tmp_path
        )
    )
    result = runner.run_case(case, recovery_policy=RecoveryPolicy(enabled=False))
    assert result.recovery_attempts == 0
    assert result.status == CaseStatus.FAILED


def test_environment_is_closed_when_execution_raises(tmp_path):
    created = []

    class BrokenEnvironment:
        def __init__(self, **kwargs):
            self.closed = False
            self.state = dict(kwargs["initial_state"])
            created.append(self)

        def run(self, **kwargs):
            raise RuntimeError("synthetic dispatch error")

        def observe(self):
            return dict(self.state)

        def close(self):
            self.closed = True

    runner = SequentialTestRunner(environment_factory=BrokenEnvironment)
    result = runner.run_case(_case())
    # The final observation proves the goal is still unmet, so the
    # independent evaluator correctly reports FAILED rather than INCONCLUSIVE.
    assert result.status == CaseStatus.FAILED
    assert created[0].closed is True


def test_runner_rejects_concurrent_reentry(tmp_path):
    runner = SequentialTestRunner()
    runner._running = True
    with pytest.raises(RuntimeError, match="concurrent"):
        runner.run_case(_case())
    runner._running = False


def test_runner_rejects_wrong_case_type():
    with pytest.raises(TypeError):
        SequentialTestRunner().run_case(object())


def test_runner_blocks_case_without_solution_path(tmp_path):
    from dataclasses import replace
    case = _case()
    scenario = replace(case.scenario, metadata={"generator": "manual"})
    case = AutonomousTestCase(case_id=case.case_id, scenario=scenario)
    runner = SequentialTestRunner()
    result = runner.run_case(case)
    assert result.status == CaseStatus.BLOCKED
    assert "no executable solution_path" in result.error


def test_runtime_dispatch_uses_single_action_calls(tmp_path):
    calls = []

    class RecordingEnvironment:
        def __init__(self, **kwargs):
            self.state = dict(kwargs["initial_state"])
            self.closed = False
            calls.append(self)

        def run(self, **kwargs):
            self.state.update({
                "door": "unlocked"
            }) if kwargs["actions"][0] == "unlock_door" else None
            if kwargs["actions"][0] == "open_box":
                self.state["box"] = "open"
            elif kwargs["actions"][0] == "take_key":
                self.state["key"] = "agent"
            elif kwargs["actions"][0] == "find_spare_key":
                self.state["spare_key"] = "found"
                self.state["key"] = "agent"
            return {"success": False}  # deliberately untrusted
        def observe(self):
            return dict(self.state)
        def close(self):
            self.closed = True

    runner = SequentialTestRunner(environment_factory=RecordingEnvironment)
    result = runner.run_case(_case())
    assert result.status == CaseStatus.PASSED
    assert calls[0].closed is True



def test_temporary_unavailability_retries_known_non_execution(tmp_path):
    policy = FaultInjectionPolicy(
        enabled=True,
        fault_type=FaultType.TEMPORARY_UNAVAILABILITY,
        target_action="open_box",
        trigger_step=1,
        max_occurrences=1,
    )
    case = _case(fault_policy=policy)
    runner = SequentialTestRunner(
        environment_factory=lambda **kwargs: IsolatedTestEnvironment(
            **kwargs, temp_root=tmp_path
        )
    )
    result = runner.run_case(case, recovery_policy=RecoveryPolicy(max_attempts=1))
    assert result.recovery_attempts == 1
    assert result.status == CaseStatus.PASSED
    assert result.metadata["fault_records"][0]["fault_type"] == "temporary_unavailability"


def test_observation_interruption_records_unknown_outcome_without_retry(tmp_path):
    policy = FaultInjectionPolicy(
        enabled=True,
        fault_type=FaultType.OBSERVATION_INTERRUPTION,
        target_action="open_box",
        trigger_step=1,
        max_occurrences=1,
    )
    case = _case(fault_policy=policy)
    runner = SequentialTestRunner(
        environment_factory=lambda **kwargs: IsolatedTestEnvironment(
            **kwargs, temp_root=tmp_path
        )
    )
    result = runner.run_case(case, recovery_policy=RecoveryPolicy(max_attempts=2))
    assert result.recovery_attempts == 0
    assert result.metadata["reconciliation_required"] is True
    assert result.metadata["fault_records"][0]["fault_type"] == "observation_interruption"
    assert result.metadata["fault_records"][0]["outcome_known"] is False
    assert result.metadata["fault_records"][0]["action_executed"] is None
    assert result.status in {CaseStatus.FAILED, CaseStatus.INCONCLUSIVE}
