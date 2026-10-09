from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.llm.llm_plan_agent_experience_integration import (
    LLMPlanAgentExperienceIntegrationConfigurationError,
    LLMPlanAgentExperienceIntegrationError,
    LLMPlanAgentExperienceIntegrator,
    LLMPlanAgentExperienceRecord,
    LLMPlanAgentExperienceRecordingError,
)
from agent.llm.llm_plan_runtime_execution import (
    LLMPlanRuntimeEvent,
    LLMPlanStepExecutionRecord,
    LLMPlanRuntimeExecutionError,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


@dataclass(frozen=True)
class FakeFinalState:
    plan_goal: str
    next_step_index: int | None
    progress: float


@dataclass(frozen=True)
class FakeRuntimeResult:
    success: bool
    completed: bool
    stopped: bool
    final_state: FakeFinalState


class FakeRuntimeExecutor:
    def __init__(self) -> None:
        self.observers = []

    def add_event_observer(self, callback):
        if callback not in self.observers:
            self.observers.append(callback)

    def remove_event_observer(self, callback):
        if callback in self.observers:
            self.observers.remove(callback)

    def emit(self, event):
        for callback in tuple(self.observers):
            callback(event)


class FakeLiveRuntime:
    def __init__(self, runtime_executor: FakeRuntimeExecutor, *, plan_events=None):
        self.runtime_executor = runtime_executor
        self.plan_events = list(plan_events or [])

    def execute(self, plan, **kwargs):
        for event in self.plan_events:
            self.runtime_executor.emit(event(plan))
        final_state = FakeFinalState(
            plan_goal=plan.goal,
            next_step_index=None,
            progress=1.0,
        )
        return FakeRuntimeResult(
            success=True,
            completed=True,
            stopped=False,
            final_state=final_state,
        )


class FakeAgent:
    def __init__(self):
        self.state = SimpleNamespace()
        self.current_experience = None
        self.calls = []
        self.episodes = []

    def act(self, action):
        self.calls.append(("act", action))

    def observe(self, observation):
        self.calls.append(("observe", observation))

    def step(self):
        self.calls.append(("step",))

    def record_experience(
        self,
        action,
        observation_before,
        observation_after,
        result,
    ):
        episode = {
            "action": action,
            "observation_before": observation_before,
            "observation_after": observation_after,
            "result": result,
        }
        self.calls.append(("record_experience", episode))
        self.episodes.append(episode)
        return episode


def make_step(index: int, action: str) -> LLMPlanStep:
    return LLMPlanStep(
        index=index,
        action=action,
        purpose=f"purpose-{index}",
        expected_observation=f"obs-{index}",
        success_criteria="success",
    )


def make_plan() -> LLMPlan:
    return LLMPlan(
        goal="reach_goal",
        rationale="test plan",
        steps=(
            make_step(1, "open_box"),
            make_step(2, "take_key"),
        ),
        confidence=0.9,
    )


def make_event_factory(records):
    def factory(plan):
        events = [
            LLMPlanRuntimeEvent(
                phase="started",
                plan=plan,
                world_state={"box": "closed"},
            )
        ]
        for record in records:
            events.append(
                LLMPlanRuntimeEvent(
                    phase="step_recorded",
                    plan=plan,
                    step=record.step,
                    record=record,
                    world_state=record.observation,
                    result=record.result,
                    metadata={"runtime_executor": "fake"},
                )
            )
        return events

    return factory


def setup_runtime(record_list):
    executor = FakeRuntimeExecutor()
    factory = make_event_factory(record_list)
    events = factory(make_plan())
    live = FakeLiveRuntime(
        executor,
        plan_events=[lambda _plan, event=event: event for event in events],
    )
    return executor, live


def test_requires_agent():
    executor, live = setup_runtime([])
    with pytest.raises(LLMPlanAgentExperienceIntegrationConfigurationError):
        LLMPlanAgentExperienceIntegrator(None, live)


def test_requires_record_experience():
    agent = SimpleNamespace(state=SimpleNamespace(), act=Mock(), observe=Mock(), step=Mock())
    executor, live = setup_runtime([])
    with pytest.raises(LLMPlanAgentExperienceIntegrationConfigurationError):
        LLMPlanAgentExperienceIntegrator(agent, live)


def test_requires_runtime():
    with pytest.raises(LLMPlanAgentExperienceIntegrationConfigurationError):
        LLMPlanAgentExperienceIntegrator(FakeAgent(), None)


def test_requires_runtime_executor_observer_api():
    live = SimpleNamespace(runtime_executor=SimpleNamespace(), execute=Mock())
    with pytest.raises(LLMPlanAgentExperienceIntegrationConfigurationError):
        LLMPlanAgentExperienceIntegrator(FakeAgent(), live)


def test_for_agent_factory():
    executor, live = setup_runtime([])
    agent = FakeAgent()
    integrator = LLMPlanAgentExperienceIntegrator.for_agent(agent, live)
    assert integrator.agent is agent
    assert integrator.live_runtime is live


def test_initial_state():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    assert not integrator.active
    assert not integrator.attached
    assert integrator.last_result is None
    assert integrator.experiences == ()


def test_attach_is_idempotent():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    assert integrator.attach() is integrator
    assert integrator.attach() is integrator
    assert len(executor.observers) == 1
    integrator.detach()
    assert len(executor.observers) == 0


def test_detach_is_idempotent():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    integrator.detach()
    integrator.attach()
    integrator.detach()
    integrator.detach()
    assert not integrator.attached


def test_records_one_episode_per_step():
    plan = make_plan()
    rec1 = LLMPlanStepExecutionRecord(
        step=plan.steps[0],
        request=None,
        result={"ok": True},
        success=True,
        observation={"box": "open"},
    )
    rec2 = LLMPlanStepExecutionRecord(
        step=plan.steps[1],
        request=None,
        result={"ok": True},
        success=True,
        observation={"key": "agent"},
    )
    executor, live = setup_runtime([rec1, rec2])
    agent = FakeAgent()
    integrator = LLMPlanAgentExperienceIntegrator(agent, live)
    result = integrator.execute(plan)
    assert len(result.experiences) == 2
    assert len(agent.episodes) == 2
    assert result.experiences[0].step_index == 1
    assert result.experiences[1].step_index == 2


def test_observation_before_and_after_are_chained():
    plan = make_plan()
    records = [
        LLMPlanStepExecutionRecord(plan.steps[0], None, {"n": 1}, True, {"n": 2}),
        LLMPlanStepExecutionRecord(plan.steps[1], None, {"n": 2}, True, {"n": 3}),
    ]
    executor, live = setup_runtime(records)
    agent = FakeAgent()
    integrator = LLMPlanAgentExperienceIntegrator(agent, live)
    integrator.execute(plan)
    assert agent.episodes[0]["observation_before"] == {"box": "closed"}
    assert agent.episodes[0]["observation_after"] == {"n": 2}
    assert agent.episodes[1]["observation_before"] == {"n": 2}
    assert agent.episodes[1]["observation_after"] == {"n": 3}


def test_projects_action_and_observation_by_default():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    LLMPlanAgentExperienceIntegrator(agent, live).execute(plan)
    assert ("act", "open_box") in agent.calls
    assert ("observe", {"done": True}) in agent.calls


def test_can_disable_agent_action_projection():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    LLMPlanAgentExperienceIntegrator(agent, live, project_agent_action=False).execute(plan)
    assert not any(call[0] == "act" for call in agent.calls)
    assert any(call[0] == "observe" for call in agent.calls)


def test_can_disable_agent_observation_projection():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    LLMPlanAgentExperienceIntegrator(agent, live, project_agent_observation=False).execute(plan)
    assert not any(call[0] == "observe" for call in agent.calls)
    assert any(call[0] == "act" for call in agent.calls)


def test_can_advance_agent_cycle():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    LLMPlanAgentExperienceIntegrator(agent, live, advance_agent_cycle=True).execute(plan)
    assert ("step",) in agent.calls


def test_result_wraps_runtime_result():
    plan = make_plan()
    executor, live = setup_runtime([])
    result = LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    assert result.success is True
    assert result.completed is True
    assert result.stopped is False
    assert result.plan_goal == plan.goal
    assert result.final_state.plan_goal == plan.goal


def test_last_result_is_retained():
    plan = make_plan()
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    result = integrator.execute(plan)
    assert integrator.last_result is result


def test_experience_record_contains_expected_fields():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    result = LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    item = result.experiences[0]
    assert isinstance(item, LLMPlanAgentExperienceRecord)
    assert item.action == "open_box"
    assert item.success is True
    assert item.result == {"ok": True}


def test_duplicate_step_event_is_ignored():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record, record])
    agent = FakeAgent()
    result = LLMPlanAgentExperienceIntegrator(agent, live).execute(plan)
    assert len(result.experiences) == 1
    assert len(agent.episodes) == 1


def test_failed_step_is_still_recorded_as_experience():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"error": "timeout"}, False, {"done": False})
    executor, live = setup_runtime([record])
    result = LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    assert result.experiences[0].success is False
    assert result.experiences[0].episode["result"]["error"] == "timeout"


def test_skipped_step_is_recorded_when_runtime_emits_it_as_a_step():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(
        plan.steps[0], None, {"skipped": True}, False, {"n": 1}, skipped=True
    )
    executor, live = setup_runtime([record])
    result = LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    assert len(result.experiences) == 1


def test_started_event_only_does_not_record_experience():
    plan = make_plan()
    executor, live = setup_runtime([])
    agent = FakeAgent()
    result = LLMPlanAgentExperienceIntegrator(agent, live).execute(plan)
    assert result.experiences == ()


def test_non_step_events_do_not_record_experience():
    plan = make_plan()
    executor = FakeRuntimeExecutor()
    live = FakeLiveRuntime(
        executor,
        plan_events=[
            lambda p: LLMPlanRuntimeEvent("started", p, world_state={"x": 0}),
            lambda p: LLMPlanRuntimeEvent("stopped", p, stop_reason="manual"),
            lambda p: LLMPlanRuntimeEvent("finished", p),
        ],
    )
    agent = FakeAgent()
    result = LLMPlanAgentExperienceIntegrator(agent, live).execute(plan)
    assert result.experiences == ()


def test_experience_observer_receives_episode_metadata():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    observed = []
    integrator = LLMPlanAgentExperienceIntegrator(
        FakeAgent(), live, experience_observer=lambda *args: observed.append(args)
    )
    integrator.execute(plan)
    assert len(observed) == 1
    episode, index, action, metadata = observed[0]
    assert episode["action"] == action
    assert index == 1
    assert action == "open_box"
    assert "runtime_executor" in metadata


def test_experience_observer_failure_propagates():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    with pytest.raises(ValueError, match="observer failed"):
        LLMPlanAgentExperienceIntegrator(
            FakeAgent(), live, experience_observer=lambda *_: (_ for _ in ()).throw(ValueError("observer failed"))
        ).execute(plan)


def test_agent_record_failure_is_wrapped():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    agent.record_experience = Mock(side_effect=RuntimeError("boom"))
    with pytest.raises(LLMPlanAgentExperienceRecordingError, match="step 1"):
        LLMPlanAgentExperienceIntegrator(agent, live).execute(plan)


def test_reset_clears_transient_trace():
    plan = make_plan()
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    result = integrator.execute(plan)
    assert result is not None
    assert integrator.last_result is result
    integrator.reset()
    assert integrator.last_result is None
    assert integrator.experiences == ()


def test_reset_while_active_is_rejected():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    integrator._active = True
    with pytest.raises(LLMPlanAgentExperienceIntegrationConfigurationError):
        integrator.reset()
    integrator._active = False


def test_execute_rejects_invalid_plan():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    with pytest.raises(TypeError):
        integrator.execute("not-a-plan")


def test_concurrent_execution_guard():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    integrator._active = True
    with pytest.raises(LLMPlanAgentExperienceIntegrationConfigurationError):
        integrator.execute(make_plan())
    integrator._active = False


def test_detach_occurs_after_successful_execution():
    executor, live = setup_runtime([])
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    integrator.execute(make_plan())
    assert not integrator.attached
    assert executor.observers == []


def test_detach_occurs_after_recording_error():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"done": True})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    agent.record_experience = Mock(side_effect=RuntimeError("boom"))
    integrator = LLMPlanAgentExperienceIntegrator(agent, live)
    with pytest.raises(LLMPlanAgentExperienceRecordingError):
        integrator.execute(plan)
    assert not integrator.attached
    assert not integrator.active


def test_recorded_indices_are_ordered():
    plan = make_plan()
    records = [
        LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"n": 1}),
        LLMPlanStepExecutionRecord(plan.steps[1], None, {"ok": True}, True, {"n": 2}),
    ]
    executor, live = setup_runtime(records)
    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), live)
    integrator.execute(plan)
    assert integrator.recorded_step_indices == (1, 2)


def test_runtime_error_from_live_runtime_is_preserved():
    executor = FakeRuntimeExecutor()

    class FailingLiveRuntime:
        runtime_executor = executor
        def execute(self, plan, **kwargs):
            raise RuntimeError("runtime failed")

    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), FailingLiveRuntime())
    with pytest.raises(RuntimeError, match="runtime failed"):
        integrator.execute(make_plan())


def test_experience_error_does_not_replace_runtime_error():
    executor = FakeRuntimeExecutor()

    class FailingLiveRuntime:
        runtime_executor = executor
        def execute(self, plan, **kwargs):
            raise LLMPlanRuntimeExecutionError("runtime failed")

    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), FailingLiveRuntime())
    with pytest.raises(LLMPlanRuntimeExecutionError, match="runtime failed"):
        integrator.execute(make_plan())


def test_last_result_is_not_set_when_execution_fails():
    executor = FakeRuntimeExecutor()

    class FailingLiveRuntime:
        runtime_executor = executor
        def execute(self, plan, **kwargs):
            raise RuntimeError("runtime failed")

    integrator = LLMPlanAgentExperienceIntegrator(FakeAgent(), FailingLiveRuntime())
    with pytest.raises(RuntimeError):
        integrator.execute(make_plan())
    assert integrator.last_result is None


def test_custom_cycle_projection_is_called_once_per_step():
    plan = make_plan()
    records = [
        LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"n": 1}),
        LLMPlanStepExecutionRecord(plan.steps[1], None, {"ok": True}, True, {"n": 2}),
    ]
    executor, live = setup_runtime(records)
    agent = FakeAgent()
    LLMPlanAgentExperienceIntegrator(agent, live, advance_agent_cycle=True).execute(plan)
    assert agent.calls.count(("step",)) == 2


def test_experience_payload_converts_non_mapping_result():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, "OK", True, {"n": 1})
    executor, live = setup_runtime([record])
    result = LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    assert result.experiences[0].episode["result"] == {"value": "OK"}


def test_reset_can_be_reused_for_second_execution():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"n": 1})
    executor, live = setup_runtime([record])
    agent = FakeAgent()
    integrator = LLMPlanAgentExperienceIntegrator(agent, live)
    first = integrator.execute(plan)
    integrator.reset()
    second = integrator.execute(plan)
    assert len(first.experiences) == 1
    assert len(second.experiences) == 1


def test_original_plan_is_not_mutated():
    plan = make_plan()
    original = plan.to_dict()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"n": 1})
    executor, live = setup_runtime([record])
    LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    assert plan.to_dict() == original


def test_experience_record_trace_is_immutable_tuple():
    plan = make_plan()
    record = LLMPlanStepExecutionRecord(plan.steps[0], None, {"ok": True}, True, {"n": 1})
    executor, live = setup_runtime([record])
    result = LLMPlanAgentExperienceIntegrator(FakeAgent(), live).execute(plan)
    assert isinstance(result.experiences, tuple)


def test_config_error_is_an_integration_error():
    assert issubclass(
        LLMPlanAgentExperienceIntegrationConfigurationError,
        LLMPlanAgentExperienceIntegrationError,
    )


def test_recording_error_is_an_integration_error():
    assert issubclass(
        LLMPlanAgentExperienceRecordingError,
        LLMPlanAgentExperienceIntegrationError,
    )
