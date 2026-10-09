from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .controlled_execution import (
    ControlledExecutionReport,
    ControlledPlanExecutor,
)
from .plan_validation import (
    LLMPlanValidationResult,
    LLMPlanValidator,
)


class AgentExecutionBoundaryError(RuntimeError):
    """Raised when an LLM plan cannot cross the Agent execution boundary."""


@dataclass(frozen=True)
class AgentExecutionRequest:
    """Normalized request passed from an LLM plan step to an Agent boundary."""

    action: str
    step: Any
    position: int
    goal: Any = None
    world_state: Any = None
    experience_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = str(self.action).strip()
        if not action:
            raise ValueError("action cannot be empty.")
        if self.position < 1:
            raise ValueError("position must be >= 1.")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "metadata", dict(self.metadata))


class AgentExecutionBoundary(Protocol):
    """Protocol for an already-existing Agent execution mechanism."""

    def execute(self, request: AgentExecutionRequest) -> Any:
        """Execute exactly one normalized request."""
        ...


class CallbackAgentExecutionBoundary:
    """Adapt a caller-owned Agent execution callback to the boundary protocol."""

    def __init__(
        self,
        callback: Callable[[AgentExecutionRequest], Any],
    ) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable.")
        self._callback = callback

    def execute(self, request: AgentExecutionRequest) -> Any:
        return self._callback(request)


@dataclass(frozen=True)
class AgentPlanExecutionResult:
    """Combined validation and controlled execution result."""

    validation: LLMPlanValidationResult
    report: ControlledExecutionReport | None

    @property
    def accepted(self) -> bool:
        return self.validation.valid

    @property
    def executed(self) -> bool:
        return self.report is not None


class LLMPlanAgentExecutionAdapter:
    """
    Bridge a validated LLMPlan to an existing Agent execution boundary.

    This layer deliberately does not import or call Agent.decide_and_act(),
    UnifiedDecisionExecutor, or any concrete executor implementation. The
    application supplies the established execution boundary explicitly.
    """

    def __init__(
        self,
        *,
        validator: LLMPlanValidator | None = None,
    ) -> None:
        self.validator = validator or LLMPlanValidator()
        self.last_validation: LLMPlanValidationResult | None = None
        self.last_result: AgentPlanExecutionResult | None = None

    def validate(
        self,
        plan: Any,
        *,
        goal: Any = None,
        available_actions: Any = None,
    ) -> LLMPlanValidationResult:
        result = self.validator.validate(
            plan,
            goal=goal,
            available_actions=available_actions,
        )
        self.last_validation = result
        return result

    def execute(
        self,
        plan: Any,
        *,
        boundary: AgentExecutionBoundary,
        goal: Any = None,
        available_actions: Any = None,
        world_state: Any = None,
        experience_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        stop_checker: Callable[[Any, int, Any], bool] | None = None,
    ) -> AgentPlanExecutionResult:
        if not hasattr(boundary, "execute") or not callable(boundary.execute):
            raise TypeError("boundary must provide a callable execute(request) method.")

        validation = self.validate(
            plan,
            goal=goal,
            available_actions=available_actions,
        )

        if not validation.valid:
            result = AgentPlanExecutionResult(
                validation=validation,
                report=None,
            )
            self.last_result = result
            return result

        request_metadata = dict(metadata or {})

        def execute_step(action: str, step: Any, index: int) -> Any:
            request = AgentExecutionRequest(
                action=action,
                step=step,
                position=index,
                goal=goal,
                world_state=world_state,
                experience_id=experience_id,
                metadata=request_metadata,
            )
            return boundary.execute(request)

        report = ControlledPlanExecutor(execute_step).execute(
            plan,
            validation,
            stop_checker=stop_checker,
        )

        result = AgentPlanExecutionResult(
            validation=validation,
            report=report,
        )
        self.last_result = result
        return result
