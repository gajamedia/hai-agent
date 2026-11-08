from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .agent_execution_boundary import AgentExecutionRequest
from .execution_security import ExecutionSecurityPolicy


class AgentUnifiedExecutorBindingError(RuntimeError):
    """Base error for the concrete UnifiedDecisionExecutor binding."""


class AgentUnifiedExecutorBindingConfigurationError(AgentUnifiedExecutorBindingError):
    """Raised when the Agent execution configuration is invalid."""


class DecisionFactory(Protocol):
    def __call__(self, request: AgentExecutionRequest) -> Any:
        """Build the existing AgentDecision for one validated step."""
        ...


@dataclass(frozen=True)
class AgentUnifiedExecutionReport:
    """Result returned after one canonical boundary invocation."""

    request: AgentExecutionRequest
    decision: Any
    result: Any


class AgentUnifiedExecutorBinding:
    """Securely bind boundary requests to existing execution mechanisms.

    Trusted capability actions are intercepted only when an explicit port
    mapping exists. They do not need to appear in Agent.actions and never reach
    the simulation executor or placeholder Action handler. All other actions
    retain the historical Agent executor path.
    """

    def __init__(
        self,
        agent: Any,
        *,
        decision_factory: DecisionFactory,
        security_policy: ExecutionSecurityPolicy | None = None,
        trusted_capability_invocation_port: Any = None,
    ) -> None:
        if agent is None:
            raise ValueError("agent cannot be None.")
        if not callable(decision_factory):
            raise TypeError("decision_factory must be callable.")
        self.agent = agent
        self.decision_factory = decision_factory
        self.security_policy = (
            security_policy
            or getattr(agent, "execution_security_policy", None)
            or ExecutionSecurityPolicy()
        )
        if not isinstance(self.security_policy, ExecutionSecurityPolicy):
            raise TypeError("security_policy must be an ExecutionSecurityPolicy.")
        if trusted_capability_invocation_port is not None:
            if not callable(
                getattr(trusted_capability_invocation_port, "supports_action", None)
            ) or not callable(getattr(trusted_capability_invocation_port, "invoke", None)):
                raise TypeError(
                    "trusted_capability_invocation_port must expose "
                    "supports_action() and invoke()."
                )
        self.trusted_capability_invocation_port = trusted_capability_invocation_port
        self.last_report: AgentUnifiedExecutionReport | None = None

    def _ensure_executor(self) -> Any:
        executor = getattr(self.agent, "unified_executor", None)
        if executor is None:
            configure_planner = getattr(self.agent, "configure_planner", None)
            if callable(configure_planner):
                configure_planner()
                executor = getattr(self.agent, "unified_executor", None)
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
                name = str(getattr(item, "name", item)).strip()
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

    def _execute_trusted_capability(
        self,
        request: AgentExecutionRequest,
        execution_context: Any,
    ) -> AgentUnifiedExecutionReport:
        port = self.trusted_capability_invocation_port
        if execution_context is None:
            raise AgentUnifiedExecutorBindingConfigurationError(
                "Trusted capability action requires TrustedExecutionContext."
            )

        # The canonical policy remains an independent gate. The authenticated
        # gateway then performs principal permission, registry trust, and its
        # own capability risk checks before calling an adapter.
        self.security_policy.authorize(request)
        try:
            decision = port.describe_action(request.action)
            result = port.invoke(request, execution_context=execution_context)
        except Exception:
            # A trace/persistence failure may occur after a read has completed.
            # Mark the outcome unknown so callers never infer safe retryability.
            self.security_policy.record_execution_outcome(
                request,
                success=None,
                reason="trusted_capability_outcome_unknown",
            )
            raise

        self.security_policy.record_execution_outcome(
            request,
            success=True,
            reason="trusted_capability_read_completed",
        )
        report = AgentUnifiedExecutionReport(
            request=request,
            decision=decision,
            result=result,
        )
        self.last_report = report
        return report

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
        execution_context: Any = None,
    ) -> AgentUnifiedExecutionReport:
        if not isinstance(request, AgentExecutionRequest):
            raise TypeError("request must be AgentExecutionRequest.")

        port = self.trusted_capability_invocation_port
        if port is not None and port.supports_action(request.action):
            return self._execute_trusted_capability(request, execution_context)

        # A trusted context may still be consumed by a pre-existing explicit
        # trusted executor method for non-port actions (C.2 compatibility).
        action_obj = self._ensure_action(request.action)
        self.security_policy.authorize(request)
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
        execute_action = getattr(executor, "execute_action", None)
        execute_with_trusted_context = getattr(
            executor, "execute_with_trusted_context", None
        )
        try:
            if execution_context is not None:
                if not callable(execute_with_trusted_context):
                    raise AgentUnifiedExecutorBindingConfigurationError(
                        "Trusted execution context requires executor "
                        "execute_with_trusted_context(...); refusing generic execution."
                    )
                result = execute_with_trusted_context(
                    **common_kwargs,
                    action=action_obj,
                    execution_context=execution_context,
                )
            elif callable(execute_action):
                result = execute_action(**common_kwargs, action=action_obj)
            else:
                result = executor.execute(**common_kwargs)
        except Exception:
            self.security_policy.record_execution_outcome(
                request, success=False, reason="executor_exception"
            )
            raise

        success = (
            bool(result.get("success", False))
            if isinstance(result, dict)
            else bool(getattr(result, "success", result))
        )
        self.security_policy.record_execution_outcome(
            request,
            success=success,
            reason="execution_succeeded" if success else "execution_returned_failure",
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
        """Boundary-compatible method returning the executor result."""
        return self.execute_request(request, **kwargs).result
