from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .llm_plan_live_runtime import (
    LLMPlanLiveRuntimeExecutionResult,
    LLMPlanLiveRuntimeExecutor,
)
from .llm_plan_runtime_state import LLMPlanRuntimeState
from .planning import LLMPlan


class LLMPlanAgentStateIntegrationError(RuntimeError):
    """Base error for Agent runtime-state integration."""


class LLMPlanAgentStateIntegrationConfigurationError(
    LLMPlanAgentStateIntegrationError
):
    """Raised when the target Agent cannot be integrated safely."""


class LLMPlanAgentStateIntegrationProjectionError(
    LLMPlanAgentStateIntegrationError
):
    """Raised when a runtime state cannot be projected to the Agent."""


AgentStateObserver = Callable[[LLMPlanRuntimeState], None]


@dataclass(frozen=True)
class LLMPlanAgentStateIntegrationResult:
    """Execution result plus the projected Agent runtime-state history."""

    runtime_result: LLMPlanLiveRuntimeExecutionResult
    projected_state_history: tuple[LLMPlanRuntimeState, ...] = field(default_factory=tuple)

    @property
    def success(self) -> bool:
        return self.runtime_result.success

    @property
    def completed(self) -> bool:
        return self.runtime_result.completed

    @property
    def stopped(self) -> bool:
        return self.runtime_result.stopped

    @property
    def final_state(self) -> LLMPlanRuntimeState:
        return self.runtime_result.final_state

    @property
    def progress(self) -> float:
        return self.runtime_result.progress

    @property
    def next_step_index(self) -> int | None:
        return self.runtime_result.next_step_index


class LLMPlanAgentStateIntegrator:
    """
    Project live LLM-plan runtime state into an Agent safely.

    The integrator does not own execution. It delegates to the existing
    ``LLMPlanLiveRuntimeExecutor`` and only mirrors runtime state into:

      * ``agent.llm_plan_runtime_state`` (public convenience attribute)
      * ``agent.state.llm_plan_runtime_state`` (Agent-state projection)
      * Agent working-memory APIs for plan, reasoning, action, observation

    The integration is intentionally additive and does not call
    ``Agent.decide_and_act()`` or create another execution engine.
    """

    STATE_ATTRIBUTE = "llm_plan_runtime_state"

    def __init__(
        self,
        agent: Any,
        live_runtime: LLMPlanLiveRuntimeExecutor,
        *,
        state_attribute: str = STATE_ATTRIBUTE,
        sync_working_memory: bool = True,
        sync_plan: bool = True,
        sync_reasoning: bool = True,
        sync_action: bool = True,
        sync_observation: bool = True,
        advance_agent_cycle: bool = False,
        state_observer: AgentStateObserver | None = None,
    ) -> None:
        self._validate_agent(agent)
        self._validate_runtime(live_runtime)
        if not isinstance(state_attribute, str) or not state_attribute.strip():
            raise LLMPlanAgentStateIntegrationConfigurationError(
                "state_attribute must be a non-empty string."
            )
        if state_observer is not None and not callable(state_observer):
            raise LLMPlanAgentStateIntegrationConfigurationError(
                "state_observer must be callable."
            )

        self.agent = agent
        self.live_runtime = live_runtime
        self.state_attribute = state_attribute.strip()
        self.sync_working_memory = bool(sync_working_memory)
        self.sync_plan = bool(sync_plan)
        self.sync_reasoning = bool(sync_reasoning)
        self.sync_action = bool(sync_action)
        self.sync_observation = bool(sync_observation)
        self.advance_agent_cycle = bool(advance_agent_cycle)
        self.state_observer = state_observer

        self._lock = RLock()
        self._plan: LLMPlan | None = None
        self._state_history: list[LLMPlanRuntimeState] = []
        self._last_state: LLMPlanRuntimeState | None = None
        self._attached = False
        self._active = False
        self._observer_callback = self._handle_state

        self._ensure_state_slot_available()
        self._publish_to_agent(None)

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        live_runtime: LLMPlanLiveRuntimeExecutor,
        **kwargs: Any,
    ) -> "LLMPlanAgentStateIntegrator":
        return cls(agent, live_runtime, **kwargs)

    @staticmethod
    def _validate_agent(agent: Any) -> None:
        if agent is None:
            raise LLMPlanAgentStateIntegrationConfigurationError(
                "agent cannot be None."
            )
        if not hasattr(agent, "state"):
            raise LLMPlanAgentStateIntegrationConfigurationError(
                "agent must expose a state object."
            )
        for name in ("set_plan", "set_reasoning", "act", "observe"):
            if not callable(getattr(agent, name, None)):
                raise LLMPlanAgentStateIntegrationConfigurationError(
                    f"agent must expose callable {name}(...)."
                )

    @staticmethod
    def _validate_runtime(live_runtime: Any) -> None:
        if live_runtime is None or not callable(getattr(live_runtime, "execute", None)):
            raise LLMPlanAgentStateIntegrationConfigurationError(
                "live_runtime must expose callable execute(...)."
            )

    def _ensure_state_slot_available(self) -> None:
        state = self.agent.state
        try:
            existing = getattr(state, self.state_attribute)
        except AttributeError:
            existing = None

        # A missing slot is supported when the state object allows
        # additive attributes. Probe once and remove only when possible.
        try:
            setattr(state, self.state_attribute, existing)
        except Exception as exc:
            raise LLMPlanAgentStateIntegrationConfigurationError(
                f"agent.state does not allow attribute '{self.state_attribute}'."
            ) from exc

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def active(self) -> bool:
        return self._active

    @property
    def last_state(self) -> LLMPlanRuntimeState | None:
        return self._last_state

    @property
    def state_history(self) -> tuple[LLMPlanRuntimeState, ...]:
        return tuple(self._state_history)

    def attach(self) -> "LLMPlanAgentStateIntegrator":
        with self._lock:
            if self._attached:
                return self
            self.live_runtime.state_observer = self._observer_callback
            self._attached = True
            return self

    def detach(self) -> None:
        with self._lock:
            if not self._attached:
                return
            if getattr(self.live_runtime, "state_observer", None) is self._observer_callback:
                self.live_runtime.state_observer = None
            self._attached = False

    def _write_agent_state(self, state: LLMPlanRuntimeState) -> None:
        try:
            setattr(self.agent.state, self.state_attribute, state)
            setattr(self.agent, self.state_attribute, state)
        except Exception as exc:
            raise LLMPlanAgentStateIntegrationProjectionError(
                f"Unable to publish runtime state to Agent attribute '{self.state_attribute}'."
            ) from exc

    def _project_working_memory(self, state: LLMPlanRuntimeState) -> None:
        if not self.sync_working_memory:
            return

        if self.sync_plan and self._plan is not None and state.current_step_index is None:
            self.agent.set_plan(list(self._plan.actions))
            if self.sync_reasoning and self._plan.rationale:
                self.agent.set_reasoning(self._plan.rationale)

        elif self.sync_plan and self._plan is not None:
            # Keep the complete intended plan available in working memory.
            self.agent.set_plan(list(self._plan.actions))

        if self.sync_action and self._plan is not None and state.current_step_index is not None:
            step = next(
                (item for item in self._plan.steps if item.index == state.current_step_index),
                None,
            )
            if step is not None:
                self.agent.act(step.action)

        if self.sync_observation and state.last_observation is not None:
            self.agent.observe(state.last_observation)

        if self.advance_agent_cycle and state.current_step_index is not None:
            self.agent.step()

    def _publish_to_agent(self, state: LLMPlanRuntimeState | None) -> None:
        if state is None:
            try:
                setattr(self.agent.state, self.state_attribute, None)
                setattr(self.agent, self.state_attribute, None)
            except Exception:
                # Constructor validation will already have caught a state
                # object that rejects the slot.
                pass
            return

        self._write_agent_state(state)
        self._project_working_memory(state)
        self._last_state = state
        self._state_history.append(state)
        if self.state_observer is not None:
            self.state_observer(state)

    def _handle_state(self, state: LLMPlanRuntimeState) -> None:
        try:
            self._publish_to_agent(state)
        except LLMPlanAgentStateIntegrationError:
            raise
        except Exception as exc:
            raise LLMPlanAgentStateIntegrationProjectionError(
                f"Unable to project runtime state: {exc}"
            ) from exc

    def execute(
        self,
        plan: LLMPlan,
        **kwargs: Any,
    ) -> LLMPlanAgentStateIntegrationResult:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")

        with self._lock:
            if self._active:
                raise LLMPlanAgentStateIntegrationConfigurationError(
                    "An Agent runtime-state integration execution is already active."
                )

            self._active = True
            self._plan = plan
            self._state_history = []
            self._last_state = None

            if self.sync_working_memory and self.sync_plan:
                self.agent.set_plan(list(plan.actions))
            if self.sync_working_memory and self.sync_reasoning and plan.rationale:
                self.agent.set_reasoning(plan.rationale)

            self._write_agent_state_placeholder()
            self.attach()
            try:
                runtime_result = self.live_runtime.execute(plan, **kwargs)
                final_state = runtime_result.final_state
                # Protect against runtimes whose observer was not invoked.
                if self._last_state is None:
                    self._publish_to_agent(final_state)
                elif self._last_state != final_state:
                    self._publish_to_agent(final_state)
                return LLMPlanAgentStateIntegrationResult(
                    runtime_result=runtime_result,
                    projected_state_history=tuple(self._state_history),
                )
            finally:
                self.detach()
                self._plan = None
                self._active = False

    def _write_agent_state_placeholder(self) -> None:
        try:
            setattr(self.agent.state, self.state_attribute, None)
            setattr(self.agent, self.state_attribute, None)
        except Exception as exc:
            raise LLMPlanAgentStateIntegrationProjectionError(
                f"Unable to reset Agent runtime-state attribute '{self.state_attribute}'."
            ) from exc

    def reset(self) -> None:
        with self._lock:
            if self._active:
                raise LLMPlanAgentStateIntegrationConfigurationError(
                    "Cannot reset while execution is active."
                )
            self._plan = None
            self._state_history = []
            self._last_state = None
            self._write_agent_state_placeholder()


__all__ = [
    "LLMPlanAgentStateIntegrationError",
    "LLMPlanAgentStateIntegrationConfigurationError",
    "LLMPlanAgentStateIntegrationProjectionError",
    "LLMPlanAgentStateIntegrationResult",
    "LLMPlanAgentStateIntegrator",
]
