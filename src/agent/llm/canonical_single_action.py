
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .agent_execution_boundary import AgentExecutionRequest
from .unified_executor_binding import (
    AgentUnifiedExecutionReport,
    AgentUnifiedExecutorBinding,
)


class CanonicalSingleActionExecutionError(RuntimeError):
    """Base error for the canonical one-action execution API."""


class CanonicalSingleActionConfigurationError(
    CanonicalSingleActionExecutionError
):
    """Raised when the canonical executor is configured incorrectly."""


@dataclass(frozen=True)
class CanonicalSingleActionRequest:
    """
    Canonical request for executing exactly one Agent action.
    """

    action: str
    step: Any
    position: int
    goal: Any = None
    world_state: Any = None
    environment: Any = None
    strategies: Any = None
    information_gain: Mapping[str, float] = field(
        default_factory=dict
    )
    action_risk: Mapping[str, float] = field(
        default_factory=dict
    )
    exploration_risk: float = 0.0
    action_cost: Mapping[str, float] = field(
        default_factory=dict
    )
    experience_id: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        action = str(self.action).strip()

        if not action:
            raise ValueError("action cannot be empty.")

        if self.position < 1:
            raise ValueError("position must be >= 1.")

        exploration_risk = float(self.exploration_risk)

        if not 0.0 <= exploration_risk <= 1.0:
            raise ValueError(
                "exploration_risk must be between 0 and 1."
            )

        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "information_gain",
            dict(self.information_gain),
        )
        object.__setattr__(
            self,
            "action_risk",
            dict(self.action_risk),
        )
        object.__setattr__(
            self,
            "action_cost",
            dict(self.action_cost),
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )


@dataclass(frozen=True)
class CanonicalSingleActionResult:
    """Canonical result of exactly one execution-boundary call."""

    request: CanonicalSingleActionRequest
    boundary_request: AgentExecutionRequest
    report: AgentUnifiedExecutionReport

    @property
    def result(self) -> Any:
        return self.report.result

    @property
    def decision(self) -> Any:
        return self.report.decision

    @property
    def success(self) -> bool:
        """
        Resolve success from supported result representations.

        Supported forms:
        1. {"success": True}
        2. An object with a .success attribute
        3. An object wrapping a result in .value
        4. Other legacy truthy/falsy result values
        """
        value = self.report.result

        # Form 1: a result dictionary.
        if isinstance(value, Mapping):
            if "success" in value:
                return bool(value["success"])

        # Form 2: an object exposing an explicit success property.
        result_success = getattr(value, "success", None)

        if result_success is not None:
            return bool(result_success)

        # Form 3: wrapper objects such as FakeExecutorResult(value={...}).
        wrapped_value = getattr(value, "value", None)

        if isinstance(wrapped_value, Mapping):
            if "success" in wrapped_value:
                return bool(wrapped_value["success"])

        if wrapped_value is not None:
            wrapped_success = getattr(
                wrapped_value,
                "success",
                None,
            )

            if wrapped_success is not None:
                return bool(wrapped_success)

        # Form 4: preserve the legacy truthiness behavior.
        return bool(value)


class CanonicalSingleActionBoundary(Protocol):
    """Protocol implemented by the concrete single-action binding."""

    def execute_request(
        self,
        request: AgentExecutionRequest,
        *,
        environment: Any = None,
        strategies: Any = None,
        information_gain: Mapping[str, float] | None = None,
        action_risk: Mapping[str, float] | None = None,
        exploration_risk: float = 0.0,
        action_cost: Mapping[str, float] | None = None,
    ) -> AgentUnifiedExecutionReport:
        ...


class CanonicalSingleActionExecutor:
    """
    Canonical one-action API backed by the existing Agent executor.

    This class:
    - accepts exactly one action request;
    - never calls Agent.decide_and_act();
    - never invokes an LLM;
    - never creates a second executor;
    - does not implement planning or plan validation.
    """

    def __init__(
        self,
        binding: CanonicalSingleActionBoundary,
    ) -> None:
        execute_request = getattr(
            binding,
            "execute_request",
            None,
        )

        if not callable(execute_request):
            raise CanonicalSingleActionConfigurationError(
                "binding must provide callable execute_request(...)."
            )

        self.binding = binding
        self.last_result: CanonicalSingleActionResult | None = None

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        *,
        decision_factory,
    ) -> "CanonicalSingleActionExecutor":
        binding = AgentUnifiedExecutorBinding(
            agent,
            decision_factory=decision_factory,
        )

        return cls(binding)

    @staticmethod
    def normalize_request(
        request: CanonicalSingleActionRequest,
    ) -> tuple[
        AgentExecutionRequest,
        CanonicalSingleActionRequest,
    ]:
        if not isinstance(request, CanonicalSingleActionRequest):
            raise TypeError(
                "request must be CanonicalSingleActionRequest."
            )

        boundary_request = AgentExecutionRequest(
            action=request.action,
            step=request.step,
            position=request.position,
            goal=request.goal,
            world_state=request.world_state,
            experience_id=request.experience_id,
            metadata=request.metadata,
        )

        return boundary_request, request

    def execute(
        self,
        request: CanonicalSingleActionRequest,
    ) -> CanonicalSingleActionResult:
        boundary_request, canonical_request = (
            self.normalize_request(request)
        )

        report = self.binding.execute_request(
            boundary_request,
            environment=canonical_request.environment,
            strategies=canonical_request.strategies,
            information_gain=canonical_request.information_gain,
            action_risk=canonical_request.action_risk,
            exploration_risk=canonical_request.exploration_risk,
            action_cost=canonical_request.action_cost,
        )

        result = CanonicalSingleActionResult(
            request=canonical_request,
            boundary_request=boundary_request,
            report=report,
        )

        self.last_result = result
        return result

    def execute_from_boundary_request(
        self,
        request: AgentExecutionRequest,
        **kwargs: Any,
    ) -> CanonicalSingleActionResult:
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError(
                "request must be AgentExecutionRequest."
            )

        canonical_request = CanonicalSingleActionRequest(
            action=request.action,
            step=request.step,
            position=request.position,
            goal=request.goal,
            world_state=request.world_state,
            experience_id=request.experience_id,
            metadata=request.metadata,
            environment=kwargs.pop("environment", None),
            strategies=kwargs.pop("strategies", None),
            information_gain=kwargs.pop(
                "information_gain",
                {},
            ),
            action_risk=kwargs.pop(
                "action_risk",
                {},
            ),
            exploration_risk=kwargs.pop(
                "exploration_risk",
                0.0,
            ),
            action_cost=kwargs.pop(
                "action_cost",
                {},
            ),
        )

        if kwargs:
            raise TypeError(
                "Unsupported execution arguments: "
                + ", ".join(sorted(kwargs))
            )

        return self.execute(canonical_request)
