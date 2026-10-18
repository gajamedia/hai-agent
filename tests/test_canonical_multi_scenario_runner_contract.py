from __future__ import annotations

import pytest

from agent.llm.canonical_multi_scenario_runner import (
    CanonicalMultiScenarioRunner,
    CanonicalMultiScenarioRunnerError,
    CanonicalScenarioBinding,
)
from agent.llm.multi_scenario_evaluation import ScenarioCase


def case():
    return ScenarioCase("case-1", "room", "Open door", {"door": "locked"}, {"door": "unlocked"}, ("unlock_door",))


def test_runner_requires_callable_factory():
    with pytest.raises(TypeError, match="binding_factory"):
        CanonicalMultiScenarioRunner(None)


def test_runner_rejects_non_binding_factory_result(tmp_path):
    runner = CanonicalMultiScenarioRunner(lambda *_: object(), artifact_dir=tmp_path)
    with pytest.raises(CanonicalMultiScenarioRunnerError, match="CanonicalScenarioBinding"):
        runner(case(), 1)


def test_binding_requires_fresh_domain_objects_and_callable_decision_factory():
    with pytest.raises(TypeError):
        CanonicalScenarioBinding(
            agent=object(), environment=object(), plan=object(), goal=object(),
            decision_factory=None, action_risks={},
        )
