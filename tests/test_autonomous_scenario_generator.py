import pytest
from agent.autonomous_testing.contracts import Difficulty, FaultType
from agent.autonomous_testing.scenario_generator import (
    AutonomousScenarioGenerator, generate_scenarios,
)


def test_generator_is_reproducible_for_same_seed():
    a = AutonomousScenarioGenerator(seed=42).generate_scenario(0, "medium")
    b = AutonomousScenarioGenerator(seed=42).generate_scenario(0, "medium")
    assert a.to_dict() == b.to_dict()


def test_different_indices_have_distinct_ids_and_seeds():
    cases = AutonomousScenarioGenerator(seed=10).generate_cases(4)
    assert len({c.scenario.scenario_id for c in cases}) == 4
    assert [c.scenario.seed for c in cases] == [10, 11, 12, 13]


def test_scenarios_follow_exact_room_state_schema():
    for difficulty in Difficulty:
        scenario = AutonomousScenarioGenerator(3).generate_scenario(0, difficulty)
        assert set(scenario.initial_state) == {
            "door", "box", "key", "spare_key", "agent_location"
        }
        assert scenario.goal == {"door": "unlocked"}


def test_every_generated_scenario_is_model_reachable():
    from agent.autonomous_testing.scenario_generator import _shortest_path
    for difficulty in Difficulty:
        for i in range(20):
            scenario = AutonomousScenarioGenerator(100).generate_scenario(i, difficulty)
            assert _shortest_path(dict(scenario.initial_state), dict(scenario.goal)) is not None
            assert scenario.metadata["model_verified_reachable"] is True


def test_fault_injection_is_opt_in_and_targets_first_solution_action():
    normal = AutonomousScenarioGenerator(5).generate_scenario(0, "hard")
    faulty = AutonomousScenarioGenerator(5).generate_scenario(0, "hard", inject_fault=True)
    assert normal.fault_policy is None
    assert faulty.fault_policy.enabled is True
    assert faulty.fault_policy.fault_type == FaultType.ACTION_FAILURE
    assert faulty.fault_policy.target_action == faulty.metadata["solution_path"][0]


def test_generate_cases_returns_contract_objects():
    cases = AutonomousScenarioGenerator(7).generate_cases(3, "easy")
    assert len(cases) == 3
    assert [c.case_id for c in cases] == ["case-0001", "case-0002", "case-0003"]


def test_fault_schedule_is_applied_every_nth_case():
    cases = AutonomousScenarioGenerator(9).generate_cases(5, inject_fault_every=2)
    assert [c.scenario.fault_policy is not None for c in cases] == [
        False, True, False, True, False
    ]


def test_invalid_inputs_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        AutonomousScenarioGenerator(-1)
    with pytest.raises(ValueError):
        AutonomousScenarioGenerator(1).generate_cases(0)
    with pytest.raises(ValueError):
        AutonomousScenarioGenerator(1).generate_scenario(-1)
    with pytest.raises(ValueError):
        AutonomousScenarioGenerator(1).generate_scenario(0, "impossible")


def test_convenience_function():
    scenarios = generate_scenarios(2, seed=22, difficulty="easy")
    assert len(scenarios) == 2
    assert all(s.difficulty == Difficulty.EASY for s in scenarios)


@pytest.mark.parametrize("seed", [0, 9, 42, 123])
def test_fault_schedule_cycles_all_supported_types_deterministically(seed):
    from agent.autonomous_testing.scenario_generator import _FAULT_SCHEDULE

    generator = AutonomousScenarioGenerator(seed)
    first = generator.generate_cases(12, inject_fault_every=1)
    second = AutonomousScenarioGenerator(seed).generate_cases(12, inject_fault_every=1)

    first_types = [case.scenario.fault_policy.fault_type for case in first]
    second_types = [case.scenario.fault_policy.fault_type for case in second]
    assert first_types == second_types
    assert first_types == list(_FAULT_SCHEDULE) * 4


def test_fault_schedule_with_every_two_cases_cycles_three_fault_types():
    cases = AutonomousScenarioGenerator(123).generate_cases(20, inject_fault_every=2)
    injected = [case for case in cases if case.scenario.fault_policy is not None]
    assert [case.case_id for case in injected] == [
        "case-0002", "case-0004", "case-0006", "case-0008", "case-0010",
        "case-0012", "case-0014", "case-0016", "case-0018", "case-0020",
    ]
    assert [case.scenario.fault_policy.fault_type for case in injected] == [
        FaultType.ACTION_FAILURE,
        FaultType.TEMPORARY_UNAVAILABILITY,
        FaultType.OBSERVATION_INTERRUPTION,
    ] * 3 + [FaultType.ACTION_FAILURE]


def test_generate_scenario_accepts_explicit_fault_type():
    scenario = AutonomousScenarioGenerator(5).generate_scenario(
        0, "medium", inject_fault=True,
        fault_type=FaultType.TEMPORARY_UNAVAILABILITY,
    )
    assert scenario.fault_policy.fault_type == FaultType.TEMPORARY_UNAVAILABILITY


def test_invalid_fault_type_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="unsupported fault_type"):
        AutonomousScenarioGenerator(1).generate_scenario(
            0, inject_fault=True, fault_type="not-a-fault",
        )
