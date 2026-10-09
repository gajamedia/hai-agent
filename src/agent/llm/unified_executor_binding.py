
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .agent_execution_boundary import AgentExecutionRequest


class AgentUnifiedExecutorBindingError(RuntimeError):
    """Base error for the concrete UnifiedDecisionExecutor binding."""


class AgentUnifiedExecutorBindingConfigurationError(
    AgentUnifiedExecutorBindingError
):
    """Raised when the Agent execution configuration is invalid."""


class DecisionFactory(Protocol):
    def __call__(self, request: AgentExecutionRequest) -> Any:
        """Build the existing AgentDecision for one validated step."""
        ...


@dataclass(frozen=True)
class AgentUnifiedExecutionReport:
    """Result returned after one UnifiedDecisionExecutor invocation."""

    request: AgentExecutionRequest
    decision: Any
    result: Any


class AgentUnifiedExecutorBinding:
    """
    Bind AgentExecutionBoundary requests to the Agent's existing executor.

    When the executor provides execute_action(), this binding uses the
    exact registered Action requested by the caller. Older executors and
    test doubles continue to use execute().
    """

    def __init__(
        self,
        agent: Any,
        *,
        decision_factory: DecisionFactory,
    ) -> None:
        if agent is None:
            raise ValueError("agent cannot be None.")

        if not callable(decision_factory):
            raise TypeError("decision_factory must be callable.")

        self.agent = agent
        self.decision_factory = decision_factory
        self.last_report: AgentUnifiedExecutionReport | None = None

    def _ensure_executor(self) -> Any:
        executor = getattr(
            self.agent,
            "unified_executor",
            None,
        )

        if executor is None:
            configure_planner = getattr(
                self.agent,
                "configure_planner",
                None,
            )
            if callable(configure_planner):
                configure_planner()
                executor = getattr(
                    self.agent,
                    "unified_executor",
                    None,
                )

        if executor is None:
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Agent must expose a unified_executor."
            )

        if not callable(getattr(executor, "execute", None)):
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Agent unified_executor must expose callable execute(...)."
            )

        return executor

    def _ensure_action(self, action: str) -> Any:
        """Return the registered Action object for the requested name."""
        actions = getattr(self.agent, "actions", None)

        if actions is None:
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Agent must expose an actions registry."
            )

        if isinstance(actions, Mapping):
            if action not in actions:
                raise AgentUnifiedExecutorBindingError(
                    f"Action '{action}' is not registered with the Agent."
                )

            return actions[action]

        try:
            for item in actions:
                name = str(
                    getattr(item, "name", item)
                ).strip()

                if name == action:
                    return item

        except TypeError as exc:
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Agent actions registry must be a mapping or iterable."
            ) from exc

        raise AgentUnifiedExecutorBindingError(
            f"Action '{action}' is not registered with the Agent."
        )

    def _get_actions(self) -> list[Any]:
        actions = getattr(self.agent, "actions", None)

        if isinstance(actions, Mapping):
            return list(actions.values())

        if actions is None:
            return []

        return list(actions)

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
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError(
                "request must be AgentExecutionRequest."
            )

        action_obj = self._ensure_action(request.action)
        executor = self._ensure_executor()

        decision = self.decision_factory(request)
        if decision is None:
            raise AgentUnifiedExecutorBindingError(
                "decision_factory returned None."
            )

        common_kwargs = {
            "decision": decision,
            "agent": self.agent,
            "environment": environment,
            "actions": self._get_actions(),
            "information_gain": dict(information_gain or {}),
            "action_risk": dict(action_risk or {}),
            "exploration_risk": exploration_risk,
            "experience_id": request.experience_id,
            "goal": request.goal,
            "world_state": request.world_state,
            "strategies": strategies,
            "action_cost": dict(action_cost or {}),
        }

        execute_action = getattr(
            executor,
            "execute_action",
            None,
        )

        if callable(execute_action):
            result = execute_action(
                **common_kwargs,
                action=action_obj,
            )
        else:
            # Compatibility path for existing fake and legacy executors.
            result = executor.execute(**common_kwargs)

        report = AgentUnifiedExecutionReport(
            request=request,
            decision=decision,
            result=result,
        )

        self.last_report = report
        return report

    def execute(
        self,
        request: AgentExecutionRequest,
        **kwargs: Any,
    ) -> Any:
        """Boundary-compatible method returning the executor result."""
        return self.execute_request(
            request,
            **kwargs,
        ).result
