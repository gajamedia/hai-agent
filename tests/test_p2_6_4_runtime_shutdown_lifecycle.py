from __future__ import annotations

import asyncio

from api.resource_lifecycle import BackgroundTaskRegistry, close_owned_resources


def test_background_registry_tracks_active_tasks():
    registry = BackgroundTaskRegistry()
    registry.register("x")
    assert registry.active_task_ids == ("x",)
    registry.complete("x")
    assert registry.wait_for_idle(timeout=0.1)


def test_cleanup_defers_cancellation_until_all_resources_are_attempted():
    async def scenario():
        events = []
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowResource:
            async def aclose(self):
                events.append("slow-start")
                started.set()
                await release.wait()
                events.append("slow-done")

        class LastResource:
            def close(self):
                events.append("last-close")

        task = asyncio.create_task(close_owned_resources([LastResource(), SlowResource()]))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        cancelled = await task
        return cancelled, events

    cancelled, events = asyncio.run(scenario())
    assert cancelled is True
    assert events == ["slow-start", "slow-done", "last-close"]


def test_runner_releases_running_flag_when_environment_close_fails():
    from agent.autonomous_testing.runner import SequentialTestRunner
    from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager

    class BrokenCloseEnvironment:
        def __init__(self, *, initial_state, config):
            self.state = dict(initial_state)

        def run(self, **kwargs):
            return {"status": "ok"}

        def observe(self):
            return dict(self.state)

        def close(self):
            raise RuntimeError("runtime close failed")

    runner = SequentialTestRunner(environment_factory=BrokenCloseEnvironment)
    case = AutonomousTestBatchManager().create_batch(count=1, seed=23).cases[0]
    try:
        runner.run_case(case)
    except RuntimeError as exc:
        assert "runtime close failed" in str(exc)
    else:
        raise AssertionError("Expected environment cleanup failure")
    assert runner._running is False


def test_environment_removes_private_directory_even_if_runtime_close_fails(tmp_path):
    from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager
    from agent.autonomous_testing.environment import IsolatedEnvironmentError, IsolatedTestEnvironment

    class BrokenRuntime:
        def set_state(self, state):
            return dict(state)

        def close(self):
            raise RuntimeError("runtime close failed")

    case = AutonomousTestBatchManager().create_batch(count=1, seed=29).cases[0]
    environment = IsolatedTestEnvironment(
        initial_state=dict(case.scenario.initial_state),
        runtime_factory=lambda *, memory_path: BrokenRuntime(),
        temp_root=tmp_path,
    )
    directory = environment.storage_directory
    try:
        environment.close()
    except IsolatedEnvironmentError as exc:
        assert "runtime close failed" in str(exc)
    else:
        raise AssertionError("Expected runtime close failure")
    assert not directory.exists()
    assert environment.is_closed is True
