from __future__ import annotations

import pytest

from agent.llm.multi_scenario_evaluation import (
    MultiScenarioEvaluationError,
    ScenarioCase,
    ScenarioExecution,
    evaluate_scenarios,
)


def cases():
    return [
        ScenarioCase("room-1", "room", "Open the door", {"door": "locked"},
                     {"door": "unlocked"}, ("get_key", "unlock")),
        ScenarioCase("inventory-1", "inventory", "Fulfil an order", {"stock": 2},
                     {"stock": 0, "order": "fulfilled"}, ("pick", "ship"), holdout=True),
    ]


def test_aggregates_domains_and_holdout_and_verifies_final_state():
    def runner(case, run_index):
        if case.domain == "room":
            return ScenarioExecution({"door": "unlocked"}, True, 2,
                                     ("get_key", "unlock"), adapter_name="test-adapter")
        # Runtime claims success, but target state is incomplete: evaluator must reject completion.
        return ScenarioExecution({"stock": 0}, True, 2,
                                 ("pick", "ship"), adapter_name="test-adapter")

    report = evaluate_scenarios(cases(), runner, runs_per_scenario=2)
    assert report.scenario_count == 2
    assert report.total_runs == 4
    assert report.aggregate["correct_completion_rate"] == 0.5
    assert report.by_domain["room"]["correct_completion_rate"] == 1.0
    assert report.by_domain["inventory"]["runtime_success_rate"] == 1.0
    assert report.by_domain["inventory"]["goal_verified_rate"] == 0.0
    assert report.holdout["scenario_count"] == 1
    assert report.holdout["correct_completion_rate"] == 0.0
    assert report.evaluation_mode == "real_runtime"


def test_rejects_duplicate_ids():
    one = cases()[0]
    duplicate = ScenarioCase(one.scenario_id, "other", "another", {}, {"x": 1}, ("act",))
    with pytest.raises(MultiScenarioEvaluationError, match="unique"):
        evaluate_scenarios([one, duplicate], lambda *_: None)


def test_rejects_invalid_runner_output():
    with pytest.raises(MultiScenarioEvaluationError, match="ScenarioExecution"):
        evaluate_scenarios(cases()[:1], lambda *_: {"success": True})


def test_requires_nonempty_goal_target_and_valid_actions():
    with pytest.raises(MultiScenarioEvaluationError, match="target_state"):
        ScenarioCase("bad", "room", "bad", {}, {}, ("act",))
    with pytest.raises(MultiScenarioEvaluationError, match="allowed_actions"):
        ScenarioCase("bad", "room", "bad", {}, {"x": 1}, ())


def test_distinguishes_fixture_mode_from_real_runtime_mode():
    case = cases()[0]
    report = evaluate_scenarios(
        [case],
        lambda *_: ScenarioExecution({"door": "unlocked"}, True, 2, adapter_name="fixture"),
        evaluation_mode="fixture",
    )
    assert report.evaluation_mode == "fixture"
    assert report.aggregate["correct_completion_rate"] == 1.0
    assert report.to_dict()["schema_version"] == "step41.multi-scenario.v1"


def test_rejects_unknown_evaluation_mode():
    with pytest.raises(MultiScenarioEvaluationError, match="evaluation_mode"):
        evaluate_scenarios(cases(), lambda *_: None, evaluation_mode="pretend")
