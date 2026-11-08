from __future__ import annotations

import pytest

from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager
from agent.autonomous_testing.runner import SequentialTestRunner


class BrokenCloseEnvironment:
    def __init__(self, *, initial_state, config):
        self.state = dict(initial_state)
        self.close_calls = 0

    def run(self, **kwargs):
        return {"status": "ok"}

    def observe(self):
        return dict(self.state)

    def close(self):
        self.close_calls += 1
        raise RuntimeError("runtime close failed")


def _case():
    return AutonomousTestBatchManager().create_batch(count=1, seed=26042).cases[0]


def test_runner_releases_running_flag_when_environment_close_fails():
    environments = []

    def factory(*, initial_state, config):
        environment = BrokenCloseEnvironment(initial_state=initial_state, config=config)
        environments.append(environment)
        return environment

    runner = SequentialTestRunner(environment_factory=factory)

    with pytest.raises(RuntimeError, match="runtime close failed"):
        runner.run_case(_case())

    assert runner._running is False
    assert environments[0].close_calls == 1


def test_runner_can_start_another_case_after_previous_cleanup_failure():
    environments = []

    def factory(*, initial_state, config):
        environment = BrokenCloseEnvironment(initial_state=initial_state, config=config)
        environments.append(environment)
        return environment

    runner = SequentialTestRunner(environment_factory=factory)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="runtime close failed"):
            runner.run_case(_case())
        assert runner._running is False

    assert len(environments) == 2
    assert [environment.close_calls for environment in environments] == [1, 1]


def test_runner_releases_running_flag_when_environment_construction_fails():
    def factory(*, initial_state, config):
        raise RuntimeError("runtime construction failed")

    runner = SequentialTestRunner(environment_factory=factory)

    with pytest.raises(RuntimeError, match="runtime construction failed"):
        runner.run_case(_case())

    assert runner._running is False
