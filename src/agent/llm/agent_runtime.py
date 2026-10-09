from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .context import AgentContext
from .orchestrator import CognitiveCycleResult, CognitiveOrchestrator


class AgentCognitiveRuntimeError(RuntimeError):
    """Base error for Agent-to-LLM cognitive runtime integration."""


class AgentCognitiveRuntimeConfigurationError(AgentCognitiveRuntimeError):
    """Raised when the runtime cannot obtain a required integration value."""


@dataclass(frozen=True)
class AgentCognitiveRuntimeTrace:
    """Immutable trace for the latest Agent cognitive pass."""

    cycle: int | None
    observation: Any
    context: AgentContext
    result: CognitiveCycleResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "observation": self.observation,
            "context": self.context.to_dict(),
            "result": self.result.to_dict(),
        }


class AgentCognitiveRuntime:
    """Bridge an existing Agent to the provider-neutral cognitive orchestrator.

    The runtime deliberately does not mutate the Agent. It reads public Agent
    state and produces an immutable trace/result. Execution remains owned by
    the Agent's existing decision/executor path until a later controlled stage.
    """

    def __init__(
        self,
        agent: Any,
        orchestrator: CognitiveOrchestrator,
    ) -> None:
        if agent is None:
            raise ValueError("agent is required")
        if orchestrator is None:
            raise ValueError("orchestrator is required")

        self.agent = agent
        self.orchestrator = orchestrator
        self.last_trace: AgentCognitiveRuntimeTrace | None = None

    def build_context(
        self,
        *,
        goal: Any = None,
        state: Any = None,
        world_model: Any = None,
        relevant_memory: Any = None,
        strategy: Any = None,
        self_awareness: Any = None,
        calibration: Any = None,
        metacognition: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentContext:
        """Build context from explicit values plus safe Agent fallbacks."""

        resolved_goal = self._resolve_goal(goal)
        resolved_state = self._resolve_state(state)
        resolved_world_model = self._resolve_world_model(world_model)
        resolved_memory = self._resolve_relevant_memory(relevant_memory)
        resolved_strategy = self._resolve_strategy(strategy)
        resolved_self_awareness = self._resolve_self_awareness(
            self_awareness
        )
        resolved_calibration = self._resolve_calibration(calibration)
        resolved_metacognition = self._resolve_metacognition(metacognition)

        call_metadata = {
            "integration": "agent_cognitive_runtime",
            "agent_name": getattr(self.agent, "name", None),
            "cycle": self._resolve_cycle(),
            **dict(metadata or {}),
        }

        return self.orchestrator.build_context(
            goal=resolved_goal,
            state=resolved_state,
            world_model=resolved_world_model,
            relevant_memory=resolved_memory,
            strategy=resolved_strategy,
            self_awareness=resolved_self_awareness,
            calibration=resolved_calibration,
            metacognition=resolved_metacognition,
            metadata=call_metadata,
        )

    def run(
        self,
        observation: Any = None,
        *,
        goal: Any = None,
        state: Any = None,
        world_model: Any = None,
        relevant_memory: Any = None,
        strategy: Any = None,
        self_awareness: Any = None,
        calibration: Any = None,
        metacognition: Any = None,
        episode: Any = None,
        deterministic_reflection: Any = None,
        include_metacognition: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CognitiveCycleResult:
        """Run one cognitive pass using Agent state as the default context."""

        resolved_observation = self._resolve_observation(observation)
        context = self.build_context(
            goal=goal,
            state=state,
            world_model=world_model,
            relevant_memory=relevant_memory,
            strategy=strategy,
            self_awareness=self_awareness,
            calibration=calibration,
            metacognition=metacognition,
            metadata=metadata,
        )

        result = self.orchestrator.run(
            resolved_observation,
            context,
            episode=episode,
            deterministic_reflection=deterministic_reflection,
            include_metacognition=include_metacognition,
            metadata={
                "integration": "agent_cognitive_runtime",
                "agent_name": getattr(self.agent, "name", None),
                "cycle": self._resolve_cycle(),
                **dict(metadata or {}),
            },
        )

        self.last_trace = AgentCognitiveRuntimeTrace(
            cycle=self._resolve_cycle(),
            observation=resolved_observation,
            context=context,
            result=result,
        )
        return result

    def get_last_result(self) -> CognitiveCycleResult | None:
        """Return the latest cognitive result without touching the Agent."""

        if self.last_trace is None:
            return None
        return self.last_trace.result

    def clear_trace(self) -> None:
        """Clear runtime and underlying orchestrator traces."""

        self.last_trace = None
        self.orchestrator.clear_trace()

    def _resolve_goal(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit
        return getattr(self.agent, "current_goal", None)

    def _resolve_state(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit
        return getattr(self.agent, "state", None)

    def _resolve_world_model(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit

        value = getattr(self.agent, "world_model", None)
        if value is not None:
            return value

        state = getattr(self.agent, "state", None)
        return getattr(state, "world_model", None)

    # def _resolve_relevant_memory(self, explicit: Any) -> Any:
    #     if explicit is not None:
    #         return explicit

    #     working_memory = getattr(self.agent, "working_memory", None)
    #     if working_memory is None:
    #         return None

    #     snapshot = getattr(working_memory, "snapshot", None)
    #     if callable(snapshot):
    #         return snapshot()

    #     return working_memory
    # def _resolve_relevant_memory(self, explicit: Any) -> Any:
    #     if explicit is not None:
    #         return explicit

    #     working_memory = getattr(self.agent, "working_memory", None)
    #     if working_memory is None:
    #         return []

    #     # Some memory implementations expose snapshot(); others expose
    #     # their current state through attributes. Normalize both forms.
    #     snapshot = getattr(working_memory, "snapshot", None)
    #     if callable(snapshot):
    #         value = snapshot()
    #         if value is None:
    #             return []
    #         if isinstance(value, Mapping):
    #             return [dict(value)]
    #         if isinstance(value, (str, bytes, bytearray)):
    #             return [value]
    #         try:
    #             return list(value)
    #         except TypeError:
    #             return [value]

    #     fields = (
    #         "goal",
    #         "context",
    #         "current_action",
    #         "plan",
    #         "reasoning",
    #         "recent_observation",
    #     )

    #     memory_state = {}
    #     for field in fields:
    #         value = getattr(working_memory, field, None)
    #         if value is not None:
    #             memory_state[field] = value

    #     return [memory_state] if memory_state else []
    def _resolve_relevant_memory(self, explicit: Any) -> Any:
        """
        Resolve relevant memory while preserving the existing data contract.

        - Explicit memory takes precedence.
        - Snapshot results are returned without changing their structure.
        - Mapping-based working memory remains a mapping.
        - WorkingMemory objects without snapshot() are converted to a mapping.
        """
        if explicit is not None:
            return explicit

        working_memory = getattr(self.agent, "working_memory", None)

        if working_memory is None:
            return []

        # Preserve existing behavior: don't wrap snapshot results in a list.
        snapshot = getattr(working_memory, "snapshot", None)
        if callable(snapshot):
            value = snapshot()
            return [] if value is None else value

        # Preserve dictionaries and other mapping-based memory.
        if isinstance(working_memory, Mapping):
            return working_memory

        # WorkingMemory objects may not implement snapshot() or slicing.
        fields = (
            "goal",
            "context",
            "current_action",
            "plan",
            "reasoning",
            "recent_observation",
        )

        memory_state = {}
        for field in fields:
            value = getattr(working_memory, field, None)
            if value is not None:
                memory_state[field] = value

        # return memory_state if memory_state else []
        return [memory_state] if memory_state else []


    def _resolve_strategy(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit
        return getattr(self.agent, "active_strategy", None)

    def _resolve_self_awareness(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit

        getter = getattr(self.agent, "get_self_awareness_snapshot", None)
        if callable(getter):
            return getter()
        return getattr(self.agent, "self_awareness", None)

    def _resolve_calibration(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit
        return getattr(self.agent, "last_self_awareness_calibration", None)

    def _resolve_metacognition(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit

        getter = getattr(self.agent, "get_metacognitive_reflection", None)
        if callable(getter):
            return getter()
        return getattr(self.agent, "last_metacognitive_reflection", None)

    def _resolve_observation(self, explicit: Any) -> Any:
        if explicit is not None:
            return explicit

        state = getattr(self.agent, "state", None)
        observation = getattr(state, "last_observation", None)
        if observation is None:
            raise AgentCognitiveRuntimeConfigurationError(
                "observation is required when Agent.state.last_observation is unavailable"
            )
        return observation

    def _resolve_cycle(self) -> int | None:
        state = getattr(self.agent, "state", None)
        value = getattr(state, "cycle", None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "AgentCognitiveRuntime",
    "AgentCognitiveRuntimeConfigurationError",
    "AgentCognitiveRuntimeError",
    "AgentCognitiveRuntimeTrace",
]
