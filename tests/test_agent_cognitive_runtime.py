from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent.llm import AgentContext, AgentCognitiveRuntime, AgentCognitiveRuntimeTrace


class FakeResult:
    def __init__(self) -> None:
        self._data = {"ok": True}

    def to_dict(self):
        return self._data.copy()


class FakeOrchestrator:
    def __init__(self) -> None:
        self.build_calls = []
        self.run_calls = []
        self.clear_calls = 0

    def build_context(self, **kwargs):
        self.build_calls.append(kwargs)
        return AgentContextBuilderForTest.build(**kwargs)

    def run(self, observation, context, **kwargs):
        self.run_calls.append((observation, context, kwargs))
        return FakeResult()

    def clear_trace(self):
        self.clear_calls += 1


class AgentContextBuilderForTest:
    @staticmethod
    def build(**kwargs):
        return AgentContext(
            goal=kwargs.get("goal"),
            state=kwargs.get("state"),
            world_model=kwargs.get("world_model"),
            relevant_memory=kwargs.get("relevant_memory"),
            strategy=kwargs.get("strategy"),
            self_awareness=kwargs.get("self_awareness"),
            calibration=kwargs.get("calibration"),
            metacognition=kwargs.get("metacognition"),
            metadata=kwargs.get("metadata"),
        )


@dataclass
class FakeState:
    cycle: int = 7
    last_observation: dict = field(default_factory=lambda: {"door": "open"})
    world_model: dict = field(default_factory=lambda: {"door": "open"})


@dataclass
class FakeGoal:
    description: str = "reach door"


@dataclass
class FakeSelfAwarenessSnapshot:
    confidence: float = 0.8

    def to_dict(self):
        return {"confidence": self.confidence}


@dataclass
class FakeCalibration:
    calibrated_confidence: float = 0.75


@dataclass
class FakeReflection:
    diagnosis: str = "failure_with_partial_progress"


class FakeWorkingMemory:
    def snapshot(self):
        return {"plan": ["open_door"], "reasoning": "door is unlocked"}


class FakeAgent:
    def __init__(self):
        self.name = "agent-1"
        self.state = FakeState()
        self.current_goal = FakeGoal()
        self.active_strategy = {"name": "safe"}
        self.working_memory = FakeWorkingMemory()
        self.last_self_awareness_calibration = FakeCalibration()
        self.last_metacognitive_reflection = FakeReflection()
        self.telemetry_events = []

    def record_capability_telemetry(self, capability, *, success=None, source="runtime"):
        self.telemetry_events.append({
            "capability": capability,
            "success": success,
            "source": source,
        })

    def get_self_awareness_snapshot(self):
        return FakeSelfAwarenessSnapshot()

    def get_metacognitive_reflection(self):
        return self.last_metacognitive_reflection


def build_runtime():
    agent = FakeAgent()
    orchestrator = FakeOrchestrator()
    runtime = AgentCognitiveRuntime(agent, orchestrator)
    return agent, orchestrator, runtime


def test_requires_agent():
    with pytest.raises(ValueError, match="agent is required"):
        AgentCognitiveRuntime(None, FakeOrchestrator())


def test_requires_orchestrator():
    with pytest.raises(ValueError, match="orchestrator is required"):
        AgentCognitiveRuntime(FakeAgent(), None)


def test_build_context_uses_agent_defaults():
    agent, orchestrator, runtime = build_runtime()

    context = runtime.build_context()

    assert context.goal == agent.current_goal
    assert context.state == agent.state
    assert context.world_model == agent.state.world_model
    assert context.relevant_memory == {
        "plan": ["open_door"],
        "reasoning": "door is unlocked",
    }
    assert context.strategy == agent.active_strategy
    assert context.self_awareness.confidence == 0.8
    assert context.calibration == agent.last_self_awareness_calibration
    assert context.metacognition == agent.last_metacognitive_reflection
    assert context.metadata["integration"] == "agent_cognitive_runtime"
    assert context.metadata["agent_name"] == "agent-1"
    assert context.metadata["cycle"] == 7


def test_explicit_values_override_agent_defaults():
    _, _, runtime = build_runtime()
    context = runtime.build_context(
        goal="explicit goal",
        state={"x": 1},
        world_model={"y": 2},
        relevant_memory=["memory"],
        strategy="explicit",
        self_awareness={"confidence": 1.0},
        calibration={"confidence": 0.9},
        metacognition={"diagnosis": "explicit"},
    )

    assert context.goal == "explicit goal"
    assert context.state == {"x": 1}
    assert context.world_model == {"y": 2}
    assert context.relevant_memory == ["memory"]
    assert context.strategy == "explicit"
    assert context.self_awareness == {"confidence": 1.0}
    assert context.calibration == {"confidence": 0.9}
    assert context.metacognition == {"diagnosis": "explicit"}


def test_run_uses_last_observation_by_default():
    agent, orchestrator, runtime = build_runtime()

    result = runtime.run()

    assert result.to_dict() == {"ok": True}
    observation, _, kwargs = orchestrator.run_calls[0]
    assert observation == agent.state.last_observation
    assert kwargs["metadata"]["integration"] == "agent_cognitive_runtime"
    assert kwargs["metadata"]["agent_name"] == "agent-1"
    assert kwargs["metadata"]["cycle"] == 7


def test_run_accepts_explicit_observation():
    _, orchestrator, runtime = build_runtime()
    runtime.run({"new": "observation"})

    observation, _, _ = orchestrator.run_calls[0]
    assert observation == {"new": "observation"}


def test_explicit_state_and_goal_flow_into_run():
    _, orchestrator, runtime = build_runtime()
    runtime.run(
        {"seen": True},
        goal="goal-x",
        state="state-x",
    )

    _, context, _ = orchestrator.run_calls[0]
    assert context.goal == "goal-x"
    assert context.state == "state-x"


def test_latest_trace_is_exposed():
    _, _, runtime = build_runtime()
    result = runtime.run()

    trace = runtime.last_trace

    assert isinstance(trace, AgentCognitiveRuntimeTrace)
    assert trace is not None
    assert trace.result is result
    assert trace.cycle == 7


def test_get_last_result_returns_latest_result():
    _, _, runtime = build_runtime()
    assert runtime.get_last_result() is None

    result = runtime.run()
    assert runtime.get_last_result() is result


def test_clear_trace_clears_runtime_and_orchestrator():
    _, orchestrator, runtime = build_runtime()
    runtime.run()
    runtime.clear_trace()

    assert runtime.last_trace is None
    assert orchestrator.clear_calls == 1


def test_runtime_does_not_mutate_agent_state():
    agent, _, runtime = build_runtime()
    before = {
        "cycle": agent.state.cycle,
        "observation": agent.state.last_observation.copy(),
        "goal": agent.current_goal,
        "strategy": agent.active_strategy,
    }

    runtime.run()

    after = {
        "cycle": agent.state.cycle,
        "observation": agent.state.last_observation.copy(),
        "goal": agent.current_goal,
        "strategy": agent.active_strategy,
    }

    assert after == before


def test_world_model_falls_back_to_state_world_model():
    agent, _, runtime = build_runtime()
    delattr(agent, "active_strategy")

    context = runtime.build_context()

    assert context.world_model == agent.state.world_model


def test_top_level_world_model_takes_precedence():
    agent, _, runtime = build_runtime()
    agent.world_model = {"top": True}

    context = runtime.build_context()

    assert context.world_model == {"top": True}


def test_working_memory_object_is_used_when_snapshot_is_missing():
    agent, _, runtime = build_runtime()
    agent.working_memory = {"plain": "memory"}

    context = runtime.build_context()

    assert context.relevant_memory == {"plain": "memory"}


def test_self_awareness_falls_back_to_facade():
    agent, _, runtime = build_runtime()
    agent.get_self_awareness_snapshot = None
    agent.self_awareness = {"uncertainty": 0.2}

    context = runtime.build_context()

    assert context.self_awareness == {"uncertainty": 0.2}


def test_metacognition_falls_back_to_attribute():
    agent, _, runtime = build_runtime()
    agent.get_metacognitive_reflection = None

    context = runtime.build_context()

    assert context.metacognition == agent.last_metacognitive_reflection


def test_missing_observation_is_rejected():
    agent, _, runtime = build_runtime()
    agent.state.last_observation = None

    with pytest.raises(
        Exception,
        match="observation is required",
    ):
        runtime.run()


def test_metadata_is_merged_without_losing_runtime_fields():
    _, orchestrator, runtime = build_runtime()
    runtime.run(metadata={"request_id": "r1", "cycle": 99})

    _, context, kwargs = orchestrator.run_calls[0]
    assert context.metadata["request_id"] == "r1"
    assert context.metadata["integration"] == "agent_cognitive_runtime"
    assert context.metadata["cycle"] == 99
    assert kwargs["metadata"]["cycle"] == 99


def test_deterministic_reflection_is_forwarded():
    _, orchestrator, runtime = build_runtime()
    reflection = object()
    episode = {"id": "ep-1"}

    runtime.run(
        include_metacognition=True,
        episode=episode,
        deterministic_reflection=reflection,
    )

    _, _, kwargs = orchestrator.run_calls[0]
    assert kwargs["include_metacognition"] is True
    assert kwargs["episode"] == episode
    assert kwargs["deterministic_reflection"] is reflection


def test_last_trace_to_dict_is_serializable_for_simple_values():
    _, _, runtime = build_runtime()
    runtime.run({"obs": "x"})

    payload = runtime.last_trace.to_dict()

    assert payload["cycle"] == 7
    assert payload["observation"] == {"obs": "x"}
    assert payload["context"]["metadata"]["integration"] == (
        "agent_cognitive_runtime"
    )
    assert payload["result"] == {"ok": True}


def test_runtime_is_reusable_for_multiple_cycles():
    agent, orchestrator, runtime = build_runtime()

    runtime.run({"cycle": 1})
    agent.state.cycle = 8
    agent.state.last_observation = {"cycle": 2}
    runtime.run()

    assert len(orchestrator.run_calls) == 2
    assert runtime.last_trace.cycle == 8
    assert runtime.last_trace.observation == {"cycle": 2}


def test_second_run_replaces_first_trace():
    _, _, runtime = build_runtime()

    first = runtime.run({"n": 1})
    first_trace = runtime.last_trace
    second = runtime.run({"n": 2})

    assert first is not second
    assert runtime.last_trace.result is second
    assert runtime.last_trace is not first_trace


def test_context_metadata_can_carry_custom_agent_id():
    _, _, runtime = build_runtime()
    context = runtime.build_context(metadata={"agent_id": "A-01"})

    assert context.metadata["agent_id"] == "A-01"


def test_explicit_relevant_memory_can_be_empty_mapping():
    _, _, runtime = build_runtime()
    context = runtime.build_context(relevant_memory={})

    assert context.relevant_memory == {}


def test_explicit_world_model_can_be_empty_mapping():
    _, _, runtime = build_runtime()
    context = runtime.build_context(world_model={})

    assert context.world_model == {}


def test_trace_to_dict_does_not_expose_agent_object_reference():
    agent, _, runtime = build_runtime()
    runtime.run()

    payload = runtime.last_trace.to_dict()

    assert "agent" not in payload
    assert payload["context"]["metadata"]["agent_name"] == agent.name


def test_run_returns_orchestrator_result_identity():
    _, _, runtime = build_runtime()
    result = runtime.run()

    assert runtime.last_trace.result is result


def test_context_returns_agent_context_instance():
    _, _, runtime = build_runtime()
    context = runtime.build_context()

    assert isinstance(context, AgentContext)



def test_run_records_cognitive_cycle_success_without_execution_claim():
    agent, _, runtime = build_runtime()

    runtime.run()

    assert agent.telemetry_events == [{
        "capability": "cognitive_cycle",
        "success": True,
        "source": "agent_cognitive_runtime",
    }]


def test_run_records_cognitive_cycle_failure_and_reraises_original_error():
    agent, orchestrator, runtime = build_runtime()

    def fail(*args, **kwargs):
        raise RuntimeError("provider failed")

    orchestrator.run = fail

    with pytest.raises(RuntimeError, match="provider failed"):
        runtime.run()

    assert agent.telemetry_events == [{
        "capability": "cognitive_cycle",
        "success": False,
        "source": "agent_cognitive_runtime",
    }]


def test_telemetry_failure_does_not_mask_cognitive_result():
    agent, _, runtime = build_runtime()

    def fail_telemetry(*args, **kwargs):
        raise RuntimeError("telemetry unavailable")

    agent.record_capability_telemetry = fail_telemetry

    result = runtime.run()

    assert result.to_dict() == {"ok": True}


def test_agent_without_telemetry_api_remains_supported():
    agent, _, runtime = build_runtime()
    # The real Agent exposes this method on its class, so deleting the
    # instance attribute raises AttributeError. Shadow it with a non-callable
    # value to model an Agent that does not expose a usable telemetry API.
    agent.record_capability_telemetry = None

    result = runtime.run()

    assert result.to_dict() == {"ok": True}
