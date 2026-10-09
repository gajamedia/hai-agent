from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Mapping

from .llm_plan_live_runtime import LLMPlanLiveRuntimeExecutor
from .llm_plan_runtime_execution import (
    LLMPlanRuntimeEvent,
    LLMPlanRuntimeExecutionError,
    RuntimeEventObserver,
)
from .planning import LLMPlan


class LLMPlanAgentExperienceIntegrationError(RuntimeError):
    """Base error for LLM-plan experience integration."""


class LLMPlanAgentExperienceIntegrationConfigurationError(
    LLMPlanAgentExperienceIntegrationError
):
    """Raised when the integration is configured incorrectly."""


class LLMPlanAgentExperienceRecordingError(
    LLMPlanAgentExperienceIntegrationError
):
    """Raised when an executed step cannot be recorded as experience."""


ExperienceObserver = Callable[[Any, int, str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class LLMPlanAgentExperienceRecord:
    """Trace linking one executed LLM plan step to Agent experience."""

    step_index: int
    action: str
    observation_before: Any
    observation_after: Any
    result: Any
    success: bool
    episode: Any


@dataclass(frozen=True)
class LLMPlanAgentExperienceIntegrationResult:
    """Live runtime result plus the experiences recorded from each step."""

    runtime_result: Any
    experiences: tuple[LLMPlanAgentExperienceRecord, ...] = field(
        default_factory=tuple
    )

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
    def plan_goal(self) -> str:
        return self.runtime_result.final_state.plan_goal

    @property
    def final_state(self) -> Any:
        return self.runtime_result.final_state


class LLMPlanAgentExperienceIntegrator:
    """
    Convert each executed LLM-plan step into a normal Agent experience.

    The runtime executor remains the only execution engine. This integration
    only observes runtime events and delegates episode creation to the Agent's
    existing ``record_experience(...)`` lifecycle.
    """

    def __init__(
        self,
        agent: Any,
        live_runtime: LLMPlanLiveRuntimeExecutor,
        *,
        experience_observer: ExperienceObserver | None = None,
        project_agent_action: bool = True,
        project_agent_observation: bool = True,
        advance_agent_cycle: bool = False,
    ) -> None:
        self._validate_agent(agent)
        self._validate_runtime(live_runtime)
        if experience_observer is not None and not callable(experience_observer):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "experience_observer must be callable."
            )

        self.agent = agent
        self.live_runtime = live_runtime
        self.experience_observer = experience_observer
        self.project_agent_action = bool(project_agent_action)
        self.project_agent_observation = bool(project_agent_observation)
        self.advance_agent_cycle = bool(advance_agent_cycle)

        self._runtime_executor = getattr(live_runtime, "runtime_executor", None)
        self._lock = RLock()
        self._active = False
        self._attached = False
        self._plan: LLMPlan | None = None
        self._previous_observation: Any = None
        self._experiences: list[LLMPlanAgentExperienceRecord] = []
        self._recorded_step_indices: set[int] = set()
        self._last_result: LLMPlanAgentExperienceIntegrationResult | None = None
        self._observer_callback: RuntimeEventObserver = self._handle_event

    @staticmethod
    def _validate_agent(agent: Any) -> None:
        if agent is None:
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "agent cannot be None."
            )
        if not callable(getattr(agent, "record_experience", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "agent must expose callable record_experience(...)."
            )

        state = getattr(agent, "state", None)
        if state is None:
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "agent must expose a state object."
            )

        if not callable(getattr(agent, "act", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "agent must expose callable act(...)."
            )
        if not callable(getattr(agent, "observe", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "agent must expose callable observe(...)."
            )
        if not callable(getattr(agent, "step", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "agent must expose callable step(...)."
            )

    @staticmethod
    def _validate_runtime(live_runtime: Any) -> None:
        if live_runtime is None or not callable(getattr(live_runtime, "execute", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "live_runtime must expose callable execute(...)."
            )
        runtime_executor = getattr(live_runtime, "runtime_executor", None)
        if runtime_executor is None:
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "live_runtime must expose runtime_executor."
            )
        if not callable(getattr(runtime_executor, "add_event_observer", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "live_runtime.runtime_executor must expose add_event_observer(...)."
            )
        if not callable(getattr(runtime_executor, "remove_event_observer", None)):
            raise LLMPlanAgentExperienceIntegrationConfigurationError(
                "live_runtime.runtime_executor must expose remove_event_observer(...)."
            )

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        live_runtime: LLMPlanLiveRuntimeExecutor,
        **kwargs: Any,
    ) -> "LLMPlanAgentExperienceIntegrator":
        return cls(agent, live_runtime, **kwargs)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def last_result(self) -> LLMPlanAgentExperienceIntegrationResult | None:
        return self._last_result

    @property
    def experiences(self) -> tuple[LLMPlanAgentExperienceRecord, ...]:
        return tuple(self._experiences)

    @property
    def recorded_step_indices(self) -> tuple[int, ...]:
        return tuple(record.step_index for record in self._experiences)

    def attach(self) -> "LLMPlanAgentExperienceIntegrator":
        with self._lock:
            if self._attached:
                return self
            self._runtime_executor.add_event_observer(self._observer_callback)
            self._attached = True
            return self

    def detach(self) -> None:
        with self._lock:
            if not self._attached:
                return
            self._runtime_executor.remove_event_observer(self._observer_callback)
            self._attached = False

    def _step_by_index(self, step_index: int) -> Any:
        if self._plan is None:
            raise LLMPlanAgentExperienceRecordingError(
                "No active plan is available for step lookup."
            )
        for step in self._plan.steps:
            if step.index == step_index:
                return step
        raise LLMPlanAgentExperienceRecordingError(
            f"Step index {step_index} is not part of the active plan."
        )

    @staticmethod
    def _result_mapping(result: Any) -> Mapping[str, Any]:
        if isinstance(result, Mapping):
            return dict(result)
        return {"value": result}

    def _record_step_experience(self, event: LLMPlanRuntimeEvent) -> None:
        if event.record is None:
            raise LLMPlanAgentExperienceRecordingError(
                "step_recorded event requires a record."
            )

        record = event.record
        step_index = record.step.index
        if step_index in self._recorded_step_indices:
            return

        step = self._step_by_index(step_index)
        observation_before = self._previous_observation
        observation_after = record.observation
        result = self._result_mapping(record.result)

        try:
            if self.project_agent_action:
                self.agent.act(step.action)
            if self.project_agent_observation:
                self.agent.observe(observation_after)
            if self.advance_agent_cycle:
                self.agent.step()

            episode = self.agent.record_experience(
                action=step.action,
                observation_before=(
                    observation_before
                ),
                observation_after=(
                    observation_after
                ),
                result=result,
            )
        except Exception as exc:
            raise LLMPlanAgentExperienceRecordingError(
                f"Unable to record experience for step {step_index}: {exc}"
            ) from exc

        experience_record = LLMPlanAgentExperienceRecord(
            step_index=step_index,
            action=step.action,
            observation_before=observation_before,
            observation_after=observation_after,
            result=record.result,
            success=record.success,
            episode=episode,
        )
        self._experiences.append(experience_record)
        self._recorded_step_indices.add(step_index)
        self._previous_observation = observation_after

        if self.experience_observer is not None:
            metadata = dict(event.metadata)
            self.experience_observer(
                episode,
                step_index,
                step.action,
                metadata,
            )

    def _handle_event(self, event: LLMPlanRuntimeEvent) -> None:
        if not self._active:
            return

        if event.phase == "started":
            self._previous_observation = event.world_state
            return

        if event.phase == "step_recorded":
            self._record_step_experience(event)
            return

    def execute(
        self,
        plan: LLMPlan,
        **kwargs: Any,
    ) -> LLMPlanAgentExperienceIntegrationResult:
        if not isinstance(plan, LLMPlan):
            raise TypeError("plan must be an LLMPlan.")

        with self._lock:
            if self._active:
                raise LLMPlanAgentExperienceIntegrationConfigurationError(
                    "An Agent experience integration execution is already active."
                )

            self._active = True
            self._plan = plan
            self._previous_observation = None
            self._experiences = []
            self._recorded_step_indices = set()
            self._last_result = None

            self.attach()
            try:
                runtime_result = self.live_runtime.execute(plan, **kwargs)
                result = LLMPlanAgentExperienceIntegrationResult(
                    runtime_result=runtime_result,
                    experiences=tuple(self._experiences),
                )
                self._last_result = result
                return result
            except LLMPlanRuntimeExecutionError:
                raise
            finally:
                self.detach()
                self._plan = None
                self._active = False

    def reset(self) -> None:
        with self._lock:
            if self._active:
                raise LLMPlanAgentExperienceIntegrationConfigurationError(
                    "Cannot reset while execution is active."
                )
            self._plan = None
            self._previous_observation = None
            self._experiences = []
            self._recorded_step_indices = set()
            self._last_result = None


__all__ = [
    "LLMPlanAgentExperienceIntegrationError",
    "LLMPlanAgentExperienceIntegrationConfigurationError",
    "LLMPlanAgentExperienceRecordingError",
    "LLMPlanAgentExperienceRecord",
    "LLMPlanAgentExperienceIntegrationResult",
    "LLMPlanAgentExperienceIntegrator",
]
