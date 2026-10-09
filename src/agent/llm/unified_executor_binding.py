from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .agent_execution_boundary import AgentExecutionRequest


class AgentUnifiedExecutorBindingError(RuntimeError):
    """Base error for concrete UnifiedDecisionExecutor binding."""


class AgentUnifiedExecutorBindingConfigurationError(
    AgentUnifiedExecutorBindingError
):
    """Raised when the Agent does not expose the expected execution primitive."""


class DecisionFactory(Protocol):
    def __call__(self, request: AgentExecutionRequest) -> Any:
        """Build the existing AgentDecision for one validated plan step."""
        ...


@dataclass(frozen=True)
class AgentUnifiedExecutionReport:
    """Result returned after one concrete UnifiedDecisionExecutor invocation."""

    request: AgentExecutionRequest
    decision: Any
    result: Any


class AgentUnifiedExecutorBinding:
    """
    Bind the Stage 24.1 AgentExecutionBoundary to the Agent's existing
    ``UnifiedDecisionExecutor`` primitive.

    This class intentionally does NOT recreate ``Agent.decide_and_act``.
    The caller supplies a decision factory because the current Agent's
    canonical AgentDecision construction is internal to its existing
    meta-learning/self-awareness pipeline.

    The binding therefore guarantees only the execution primitive:

        AgentExecutionRequest
            -> DecisionFactory
            -> Agent.unified_executor.execute(...)

    It never calls an LLM, never bypasses action registration, and never
    synthesizes a second executor.
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
        executor = getattr(self.agent, "unified_executor", None)

        if executor is None:
            configure_planner = getattr(self.agent, "configure_planner", None)
            if callable(configure_planner):
                configure_planner()
                executor = getattr(self.agent, "unified_executor", None)

        if executor is None or not callable(getattr(executor, "execute", None)):
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Agent must expose a callable unified_executor.execute(...) primitive."
            )

        return executor

    def _ensure_action(self, action: str) -> None:
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
            return

        try:
            names = {
                str(getattr(item, "name", item)).strip()
                for item in actions
            }
        except TypeError as exc:
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Agent actions registry must be a mapping or iterable."
            ) from exc

        if action not in names:
            raise AgentUnifiedExecutorBindingError(
                f"Action '{action}' is not registered with the Agent."
            )

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
            raise TypeError("request must be AgentExecutionRequest.")

        self._ensure_action(request.action)
        executor = self._ensure_executor()

        decision = self.decision_factory(request)
        if decision is None:
            raise AgentUnifiedExecutorBindingError(
                "decision_factory returned None."
            )

        result = executor.execute(
            decision=decision,
            agent=self.agent,
            environment=environment,
            actions=(
                list(getattr(self.agent, "actions", {}).values())
                if isinstance(getattr(self.agent, "actions", None), Mapping)
                else list(getattr(self.agent, "actions", []))
            ),
            information_gain=dict(information_gain or {}),
            action_risk=dict(action_risk or {}),
            exploration_risk=exploration_risk,
            experience_id=request.experience_id,
            goal=request.goal,
            world_state=request.world_state,
            strategies=strategies,
            action_cost=dict(action_cost or {}),
        )

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
        """Boundary-compatible convenience method returning only executor output."""
        return self.execute_request(request, **kwargs).result
