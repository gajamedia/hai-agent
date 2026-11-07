from __future__ import annotations

from typing import Any

from .agent_execution_boundary import AgentExecutionRequest
from .execution_security import ExecutionSecurityPolicy
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
    """Agent-owned façade for canonical single-action execution.

    The existing Agent unified executor and action registry are reused. The
    optional trusted execution context is an explicit runtime argument; it is
    never added to request metadata or plan content.
    """

    enforces_execution_security = True

    def __init__(
        self,
        agent: Any,
        *,
        decision_factory,
        security_policy: ExecutionSecurityPolicy | None = None,
        trusted_capability_invocation_port: Any = None,
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
        self.security_policy = (
            security_policy
            or getattr(agent, "execution_security_policy", None)
            or ExecutionSecurityPolicy()
        )
        if not isinstance(self.security_policy, ExecutionSecurityPolicy):
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "security_policy must be an ExecutionSecurityPolicy."
            )
        self.executor = CanonicalSingleActionExecutor.for_agent(
            agent,
            decision_factory=decision_factory,
            security_policy=self.security_policy,
            trusted_capability_invocation_port=trusted_capability_invocation_port,
        )

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        *,
        decision_factory,
        security_policy: ExecutionSecurityPolicy | None = None,
        trusted_capability_invocation_port: Any = None,
    ) -> "AgentCanonicalSingleActionAPI":
        return cls(
            agent,
            decision_factory=decision_factory,
            security_policy=security_policy,
            trusted_capability_invocation_port=trusted_capability_invocation_port,
        )

    @property
    def last_result(self) -> CanonicalSingleActionResult | None:
        return self.executor.last_result

    def execute(
        self,
        request: CanonicalSingleActionRequest,
        *,
        execution_context: Any = None,
    ) -> CanonicalSingleActionResult:
        """Execute one action with optional out-of-band trusted context."""
        return self.executor.execute(request, execution_context=execution_context)

    def attach(self) -> Any:
        """Attach explicit canonical capabilities to this Agent instance."""
        existing_api = getattr(self.agent, "canonical_single_action_api", None)
        if existing_api is not None and existing_api is not self:
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "Agent already has a different canonical_single_action_api."
            )
        existing_method = getattr(self.agent, "execute_single_action", None)
        if existing_method is not None and existing_method is not self.execute:
            raise AgentCanonicalSingleActionAPIConfigurationError(
                "Agent already exposes execute_single_action()."
            )
        self.agent.canonical_single_action_api = self
        self.agent.execute_single_action = self.execute
        return self.agent

    def detach(self) -> Any:
        """Remove only the capabilities installed by attach()."""
        if getattr(self.agent, "canonical_single_action_api", None) is self:
            delattr(self.agent, "canonical_single_action_api")
        bound_method = getattr(self.agent, "execute_single_action", None)
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
        return self.executor.execute_from_boundary_request(request, **kwargs)
