from __future__ import annotations

from typing import Any

from .agent_execution_boundary import AgentExecutionRequest
from .canonical_single_action import (
    CanonicalSingleActionExecutor,
    CanonicalSingleActionRequest,
    CanonicalSingleActionResult,
)


class AgentCanonicalSingleActionAPIError(RuntimeError):
    """Base error for the Agent-owned canonical single-action API."""


class AgentCanonicalSingleActionAPIConfigurationError(
    AgentCanonicalSingleActionAPIError
):
    """Raised when the Agent cannot be bound safely."""


class AgentCanonicalSingleActionAPI:
    """
    Agent-owned façade for canonical single-action execution.

    This stage intentionally avoids changing ``agent.py`` directly.
    The façade receives the real Agent instance and a decision factory,
    then delegates execution to ``CanonicalSingleActionExecutor``.

    Guarantees:

    * one canonical request per execution;
    * the Agent's existing ``unified_executor`` is reused;
    * no call to ``Agent.decide_and_act``;
    * no LLM invocation;
    * no plan generation or plan validation;
    * no second execution engine is created;
    * the Agent's existing action registry is enforced by the binding.
    """

    def __init__(
        self,
        agent: Any,
        *,
        decision_factory,
    ) -> None:
        if agent is None:
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "agent cannot be None."
            )

        if not callable(decision_factory):
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "decision_factory must be callable."
            )

        if not hasattr(agent, "actions"):
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "Agent must expose an actions registry."
            )

        if not hasattr(agent, "unified_executor") and not callable(
            getattr(agent, "configure_planner", None)
        ):
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "Agent must expose unified_executor or configure_planner()."
            )

        self.agent = agent
        self.decision_factory = decision_factory
        self.executor = CanonicalSingleActionExecutor.for_agent(
            agent,
            decision_factory=decision_factory,
        )

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        *,
        decision_factory,
    ) -> "AgentCanonicalSingleActionAPI":
        """Build the API façade for an existing Agent instance."""
        return cls(
            agent,
            decision_factory=decision_factory,
        )

    @property
    def last_result(self) -> CanonicalSingleActionResult | None:
        """Return the most recent canonical execution result."""
        return self.executor.last_result

    def execute(
        self,
        request: CanonicalSingleActionRequest,
    ) -> CanonicalSingleActionResult:
        """Execute exactly one canonical Agent action request."""
        return self.executor.execute(request)

    def attach(self) -> Any:
        """Attach the canonical API as explicit capabilities on this Agent instance.

        The Agent class source file is intentionally left unchanged. The method
        is attached only to this specific instance, so existing Agent instances
        and the established ``decide_and_act`` lifecycle are unaffected unless
        the caller explicitly opts into this API.
        """
        existing_api = getattr(
            self.agent,
            "canonical_single_action_api",
            None,
        )
        if existing_api is not None and existing_api is not self:
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "Agent already has a different canonical_single_action_api."
            )

        existing_method = getattr(
            self.agent,
            "execute_single_action",
            None,
        )
        if existing_method is not None and existing_method is not self.execute:
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "Agent already exposes execute_single_action()."
            )

        self.agent.canonical_single_action_api = self
        self.agent.execute_single_action = self.execute
        return self.agent

    def detach(self) -> Any:
        """Remove only the capabilities installed by ``attach()``."""
        if getattr(self.agent, "canonical_single_action_api", None) is self:
            delattr(self.agent, "canonical_single_action_api")

        bound_method = getattr(
            self.agent,
            "execute_single_action",
            None,
        )
        if getattr(bound_method, "__self__", None) is self and getattr(
            bound_method, "__func__", None
        ) is getattr(self.execute, "__func__", None):
            delattr(self.agent, "execute_single_action")

        return self.agent

    def execute_from_boundary_request(
        self,
        request: AgentExecutionRequest,
        **kwargs: Any,
    ) -> CanonicalSingleActionResult:
        """Compatibility bridge from the Stage 24.1 request shape."""
        return self.executor.execute_from_boundary_request(
            request,
            **kwargs,
        )
