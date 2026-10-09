from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.llm.llm_plan_agent_state_integration import (
    LLMPlanAgentStateIntegrationConfigurationError,
    LLMPlanAgentStateIntegrationProjectionError,
    LLMPlanAgentStateIntegrator,
    LLMPlanAgentStateIntegrationResult,
)
from agent.llm.llm_plan_live_runtime import LLMPlanLiveRuntimeExecutionResult


@dataclass(frozen=True)
class FakeReport:
    success: bool
    completed: bool
    stopped: bool

from agent.llm.llm_plan_runtime_state import (
    LLMPlanRuntimeState,
    LLMPlanRuntimeStatus,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


def build_plan() -> LLMPlan:
    return LLMPlan(
        goal="open door",
        rationale="Use the key to unlock the door.",
        steps=(
            LLMPlanStep(index=1, action="take_key", purpose="get the key"),
            LLMPlanStep(index=2, action="unlock_door", purpose="unlock the door"),
        ),
        confidence=0.9,
    )


def state(
    *,
    current=1,
    next_step=2,
    attempted=(1,),
    succeeded=(1,),
    failed=(),
    skipped=(),
    progress=0.5,
    status=LLMPlanRuntimeStatus.RUNNING,
    completed=False,
    stopped=False,
    success=False,
    obs=None,
    result=None,
):
    return LLMPlanRuntimeState(
        plan_goal="open door",
        planned_step_indices=(1, 2),
        current_step_index=current,
        next_step_index=next_step,
        attempted_step_indices=attempted,
        succeeded_step_indices=succeeded,
        failed_step_indices=failed,
        skipped_step_indices=skipped,
        progress=progress,
        success_rate=1.0 if attempted else 0.0,
        status=status,
        completed=completed,
        stopped=stopped,
        success=success,
        stop_reason=None,
        last_observation=obs,
        last_result=result,
        metadata={},
    )


@dataclass
class FakeState:
    cycle: int = 0
    llm_plan_runtime_state: object = None


class FakeWorkingMemory:
    def __init__(self):
        self.plan = None
        self.reasoning = None
        self.current_action = None
        self.observation = None


class FakeAgent:
    def __init__(self):
        self.state = FakeState()
        self.working_memory = FakeWorkingMemory()
        self.plan_calls = []
        self.reasoning_calls = []
        self.action_calls = []
        self.observation_calls = []
        self.step_calls = 0

    def set_plan(self, plan):
        self.plan_calls.append(list(plan))
        self.working_memory.plan = list(plan)

    def set_reasoning(self, reasoning):
        self.reasoning_calls.append(reasoning)
        self.working_memory.reasoning = reasoning

    def act(self, action):
        self.action_calls.append(action)
        self.working_memory.current_action = action

    def observe(self, observation):
        self.observation_calls.append(observation)
        self.working_memory.observation = observation

    def step(self):
        self.step_calls += 1
        self.state.cycle += 1


class FakeLiveRuntime:
    def __init__(self, states):
        self.states = list(states)
        self.state_observer = None
        self.calls = 0

    def execute(self, plan, **kwargs):
        self.calls += 1
        history = []
        for item in self.states:
            history.append(item)
            if self.state_observer is not None:
                self.state_observer(item)
        final = history[-1]
        runtime_result = LLMPlanLiveRuntimeExecutionResult(
            report=FakeReport(success=final.success, completed=final.completed, stopped=final.stopped),
            final_state=final,
            state_history=tuple(history),
        )
        return runtime_result


class FakeRejectingState:
    __slots__ = ()


class FakeRejectingAgent(FakeAgent):
    def __init__(self):
        self.state = FakeRejectingState()
        self.working_memory = FakeWorkingMemory()
        self.plan_calls = []
        self.reasoning_calls = []
        self.action_calls = []
        self.observation_calls = []
        self.step_calls = 0


def build_integration(**kwargs):
    runtime = FakeLiveRuntime(
        [
            state(current=None, next_step=1, attempted=(), succeeded=(), progress=0.0),
            state(current=1, next_step=2, attempted=(1,), succeeded=(1,), obs={"key": "held"}, result={"success": True}),
            state(current=2, next_step=None, attempted=(1, 2), succeeded=(1, 2), progress=1.0, status=LLMPlanRuntimeStatus.COMPLETED, completed=True, success=True, obs={"door": "open"}, result={"success": True}),
        ]
    )
    agent = FakeAgent()
    integration = LLMPlanAgentStateIntegrator(agent, runtime, **kwargs)
    return agent, runtime, integration


def test_requires_agent():
    runtime = FakeLiveRuntime([])
    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanAgentStateIntegrator(None, runtime)


def test_requires_state():
    class NoState:
        def set_plan(self, x): pass
        def set_reasoning(self, x): pass
        def act(self, x): pass
        def observe(self, x): pass

    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanAgentStateIntegrator(NoState(), FakeLiveRuntime([]))


def test_requires_agent_methods():
    class Partial:
        state = object()

    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanAgentStateIntegrator(Partial(), FakeLiveRuntime([]))


def test_requires_runtime_execute():
    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanAgentStateIntegrator(FakeAgent(), object())


def test_rejects_blank_state_attribute():
    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanAgentStateIntegrator(FakeAgent(), FakeLiveRuntime([]), state_attribute=" ")


def test_rejects_noncallable_observer():
    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanAgentStateIntegrator(FakeAgent(), FakeLiveRuntime([]), state_observer=123)


def test_agent_state_slot_is_initialized():
    agent, _, integration = build_integration()
    assert integration.last_state is None
    assert agent.llm_plan_runtime_state is None
    assert agent.state.llm_plan_runtime_state is None


def test_attach_is_idempotent():
    agent, runtime, integration = build_integration()
    assert integration.attach() is integration
    assert integration.attach() is integration
    assert integration.attached is True
    integration.detach()
    assert integration.attached is False
    assert runtime.state_observer is None


def test_execute_requires_plan():
    _, _, integration = build_integration()
    with pytest.raises(TypeError):
        integration.execute("not a plan")


def test_execute_projects_complete_state_history():
    agent, runtime, integration = build_integration()
    result = integration.execute(build_plan())
    assert runtime.calls == 1
    assert isinstance(result, LLMPlanAgentStateIntegrationResult)
    assert len(result.projected_state_history) == 3
    assert result.final_state.completed is True
    assert result.final_state.progress == 1.0
    assert integration.last_state == result.final_state
    assert agent.state.llm_plan_runtime_state == result.final_state
    assert agent.llm_plan_runtime_state == result.final_state


def test_execute_syncs_full_plan_to_working_memory():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    assert agent.working_memory.plan == ["take_key", "unlock_door"]
    assert agent.plan_calls
    assert agent.plan_calls[0] == ["take_key", "unlock_door"]


def test_execute_syncs_reasoning_to_working_memory():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    assert agent.working_memory.reasoning == "Use the key to unlock the door."


def test_step_state_syncs_current_action():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    assert agent.action_calls == ["take_key", "unlock_door"]
    assert agent.working_memory.current_action == "unlock_door"


def test_step_state_syncs_observation():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    assert agent.observation_calls == [{"key": "held"}, {"door": "open"}]
    assert agent.working_memory.observation == {"door": "open"}


def test_cycle_is_not_advanced_by_default():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    assert agent.step_calls == 0
    assert agent.state.cycle == 0


def test_cycle_can_be_advanced_explicitly():
    agent, _, integration = build_integration(advance_agent_cycle=True)
    integration.execute(build_plan())
    assert agent.step_calls == 2
    assert agent.state.cycle == 2


def test_custom_state_attribute_is_supported():
    agent, _, integration = build_integration(state_attribute="runtime_bridge_state")
    result = integration.execute(build_plan())
    assert agent.state.runtime_bridge_state == result.final_state
    assert agent.runtime_bridge_state == result.final_state


def test_working_memory_sync_can_be_disabled():
    agent, _, integration = build_integration(sync_working_memory=False)
    integration.execute(build_plan())
    assert agent.plan_calls == []
    assert agent.reasoning_calls == []
    assert agent.action_calls == []
    assert agent.observation_calls == []
    assert agent.state.llm_plan_runtime_state is not None


def test_plan_sync_can_be_disabled():
    agent, _, integration = build_integration(sync_plan=False)
    integration.execute(build_plan())
    assert agent.plan_calls == []
    assert agent.reasoning_calls != []


def test_reasoning_sync_can_be_disabled():
    agent, _, integration = build_integration(sync_reasoning=False)
    integration.execute(build_plan())
    assert agent.reasoning_calls == []
    assert agent.plan_calls != []


def test_action_sync_can_be_disabled():
    agent, _, integration = build_integration(sync_action=False)
    integration.execute(build_plan())
    assert agent.action_calls == []


def test_observation_sync_can_be_disabled():
    agent, _, integration = build_integration(sync_observation=False)
    integration.execute(build_plan())
    assert agent.observation_calls == []


def test_custom_state_observer_receives_projected_states():
    observed = []
    agent, _, integration = build_integration(state_observer=observed.append)
    integration.execute(build_plan())
    assert observed == list(integration.state_history)


def test_custom_state_observer_runs_after_agent_projection():
    checks = []
    holder = {}
    agent, _, integration = build_integration(
        state_observer=lambda current: checks.append(
            agent.state.llm_plan_runtime_state == current
        )
    )
    holder["integration"] = integration
    integration.execute(build_plan())
    assert checks == [True, True, True]


def test_execute_detaches_runtime_observer_after_success():
    _, runtime, integration = build_integration()
    integration.execute(build_plan())
    assert runtime.state_observer is None
    assert integration.attached is False
    assert integration.active is False


def test_last_state_is_preserved_after_execute():
    agent, _, integration = build_integration()
    result = integration.execute(build_plan())
    assert integration.last_state is result.final_state
    assert agent.llm_plan_runtime_state is result.final_state


def test_state_history_is_immutable_view():
    _, _, integration = build_integration()
    integration.execute(build_plan())
    history = integration.state_history
    assert isinstance(history, tuple)
    with pytest.raises(AttributeError):
        history.append(None)


def test_reset_clears_projection():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    integration.reset()
    assert integration.last_state is None
    assert integration.state_history == ()
    assert agent.state.llm_plan_runtime_state is None
    assert agent.llm_plan_runtime_state is None


def test_reset_rejects_active_execution():
    agent, runtime, integration = build_integration()
    integration._active = True
    try:
        with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
            integration.reset()
    finally:
        integration._active = False


def test_reusing_integrator_for_second_run_resets_history():
    agent, _, integration = build_integration()
    plan = build_plan()
    first = integration.execute(plan)
    second = integration.execute(plan)
    assert len(first.projected_state_history) == 3
    assert len(second.projected_state_history) == 3
    assert len(integration.state_history) == 3


def test_runtime_result_properties_are_forwarded():
    _, _, integration = build_integration()
    result = integration.execute(build_plan())
    assert result.success is True
    assert result.completed is True
    assert result.stopped is False
    assert result.progress == 1.0
    assert result.next_step_index is None


def test_failed_final_state_is_projected():
    runtime = FakeLiveRuntime([
        state(current=1, next_step=2, attempted=(1,), failed=(1,), progress=0.5, status=LLMPlanRuntimeStatus.STOPPED, stopped=True, success=False, obs={"error": "bad"}),
    ])
    agent = FakeAgent()
    integration = LLMPlanAgentStateIntegrator(agent, runtime)
    result = integration.execute(build_plan())
    assert result.stopped is True
    assert result.success is False
    assert agent.state.llm_plan_runtime_state == result.final_state
    assert agent.action_calls == ["take_key"]


def test_final_state_is_projected_even_when_runtime_emits_no_states():
    final = state(current=None, next_step=1, attempted=(), succeeded=(), progress=0.0)
    runtime = FakeLiveRuntime([])
    # Replace execute so the fake runtime returns a final state without callbacks.
    runtime.execute = lambda plan, **kwargs: LLMPlanLiveRuntimeExecutionResult(
        report=FakeReport(success=False, completed=False, stopped=False), final_state=final, state_history=()
    )
    agent = FakeAgent()
    integration = LLMPlanAgentStateIntegrator(agent, runtime)
    result = integration.execute(build_plan())
    assert integration.last_state == final
    assert result.projected_state_history == (final,)
    assert agent.state.llm_plan_runtime_state == final


def test_projection_error_is_raised_when_state_attribute_is_rejected():
    agent = FakeRejectingAgent()
    with pytest.raises(LLMPlanAgentStateIntegrationConfigurationError):
        LLMPlanStatefulDummy(agent)  # noqa: F821


class LLMPlanStatefulDummy:
    def __new__(cls, agent):
        return LLMPlanAgentStateIntegrator(agent, FakeLiveRuntime([]))


def test_missing_observation_does_not_call_agent_observe():
    runtime = FakeLiveRuntime([
        state(current=None, next_step=1, attempted=(), succeeded=(), obs=None),
    ])
    agent = FakeAgent()
    integration = LLMPlanAgentStateIntegrator(agent, runtime)
    integration.execute(build_plan())
    assert agent.observation_calls == []


def test_no_duplicate_plan_projection_contract():
    agent, _, integration = build_integration()
    integration.execute(build_plan())
    assert all(call == ["take_key", "unlock_door"] for call in agent.plan_calls)


def test_final_projected_state_matches_runtime_plan_goal():
    _, _, integration = build_integration()
    result = integration.execute(build_plan())
    assert result.final_state.plan_goal == "open door"


def test_integration_does_not_require_agent_decide_and_act():
    agent, _, integration = build_integration()
    assert not hasattr(agent, "decide_and_act")
    integration.execute(build_plan())


def test_integration_does_not_create_new_execution_engine():
    agent, runtime, integration = build_integration()
    assert integration.live_runtime is runtime
    integration.execute(build_plan())
    assert runtime.calls == 1
