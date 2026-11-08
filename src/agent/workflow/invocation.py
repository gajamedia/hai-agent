from __future__ import annotations

"""Execute one ready workflow capability task through the existing canonical boundary.

This module is an orchestration adapter only. It does not create a gateway,
database connection, executor, authentication provider, or approval.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping

from api.security.trusted_execution_context import TrustedExecutionContext
from agent.llm.capability_execution_contracts import (
    CapabilityAuthorizationError,
    CapabilityReadResult,
)
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.workflow.contracts import TaskStatus, WorkflowStatus, WorkflowTask


TRUSTED_CAPABILITY_TASK_TYPE = "trusted_capability.invoke"
_ALLOWED_TASK_INPUT_FIELDS = frozenset({"action", "arguments"})


class WorkflowCapabilityInvocationError(RuntimeError):
    """A workflow capability task could not be completed safely."""

    def __init__(
        self,
        message: str,
        *,
        task: WorkflowTask | None = None,
        error_code: str = "workflow_capability_invocation_failed",
    ) -> None:
        super().__init__(message)
        self.task = task
        self.error_code = error_code


@dataclass(frozen=True)
class WorkflowCapabilityInvocationOutcome:
    task: WorkflowTask
    execution_id: str
    action: str
    target: str
    capability_id: str
    result: CapabilityReadResult


class WorkflowCapabilityTaskExecutor:
    """Bridge a persisted workflow task to the existing canonical capability path.

    Task input contract:
      {
        "action": "<server-mapped action name>",
        "arguments": { ... structured adapter arguments ... }
      }

    The action name is only a lookup key. Target and capability_id are never
    accepted from workflow task input; the TrustedCapabilityInvocationPort owns
    those mappings. A task is claimed with the store's version-CAS transition
    before invocation. A task left RUNNING after a process crash is deliberately
    not retried here; recovery belongs to the later C.7.6 stage.
    """

    def __init__(
        self,
        *,
        store: Any,
        agent: Any,
        trusted_capability_invocation_port: Any,
    ) -> None:
        if store is None:
            raise TypeError("store is required")
        for name in ("get_workflow", "get_task", "list_tasks",
                     "get_task_dependencies", "transition_task"):
            if not callable(getattr(store, name, None)):
                raise TypeError(f"store must expose callable {name}(...).")
        if agent is None or not hasattr(agent, "actions"):
            raise TypeError("agent must expose an actions registry")
        if not callable(getattr(trusted_capability_invocation_port, "supports_action", None)):
            raise TypeError("trusted capability port must expose supports_action(...).")
        self.store = store
        self.agent = agent
        self.port = trusted_capability_invocation_port

    @staticmethod
    def _request(task: WorkflowTask):
        from agent.llm.agent_execution_boundary import AgentExecutionRequest

        if task.task_type != TRUSTED_CAPABILITY_TASK_TYPE:
            raise WorkflowCapabilityInvocationError(
                f"Unsupported task_type: {task.task_type}",
                task=task,
                error_code="unsupported_workflow_task_type",
            )
        payload = dict(task.input_data)
        unknown = set(payload) - _ALLOWED_TASK_INPUT_FIELDS
        if unknown:
            raise WorkflowCapabilityInvocationError(
                "Capability task contains unsupported input fields.",
                task=task,
                error_code="invalid_capability_task_input",
            )
        action = payload.get("action")
        arguments = payload.get("arguments", {})
        if not isinstance(action, str) or not action.strip():
            raise WorkflowCapabilityInvocationError(
                "Capability task action must be non-empty text.",
                task=task,
                error_code="invalid_capability_task_input",
            )
        if not isinstance(arguments, Mapping):
            raise WorkflowCapabilityInvocationError(
                "Capability task arguments must be a JSON object.",
                task=task,
                error_code="invalid_capability_task_input",
            )
        return AgentExecutionRequest(
            action=action.strip(),
            step={"arguments": dict(arguments)},
            position=1,
            goal=None,
            world_state=None,
            experience_id=None,
            metadata={},
        )

    def execute_task(
        self,
        workflow_id: str,
        task_id: str,
        *,
        execution_context: TrustedExecutionContext,
    ) -> WorkflowCapabilityInvocationOutcome:
        if type(execution_context) is not TrustedExecutionContext:
            raise TypeError("execution_context must be the exact TrustedExecutionContext type")
        if execution_context.principal.is_authenticated is not True:
            raise TypeError("execution_context must contain an authenticated principal")

        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise WorkflowCapabilityInvocationError(
                "Workflow does not exist.", error_code="workflow_not_found"
            )
        if workflow.status in {
            WorkflowStatus.SUCCEEDED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED
        }:
            raise WorkflowCapabilityInvocationError(
                "Terminal workflows cannot execute tasks.",
                error_code="workflow_terminal",
            )
        task = self.store.get_task(task_id)
        if task is None or task.workflow_id != workflow_id:
            raise WorkflowCapabilityInvocationError(
                "Task does not belong to the requested workflow.",
                error_code="task_not_found",
            )
        if workflow.status is not WorkflowStatus.RUNNING:
            raise WorkflowCapabilityInvocationError(
                "Workflow must be running before a capability task can execute.",
                error_code="workflow_not_running",
            )

        if task.status is not TaskStatus.READY:
            raise WorkflowCapabilityInvocationError(
                "Only a ready workflow task can be invoked.",
                task=task,
                error_code="task_not_ready",
            )

        request = self._request(task)
        action = request.action
        if not self.port.supports_action(action):
            raise WorkflowCapabilityInvocationError(
                "Task action is not present in the server-owned capability mapping.",
                task=task,
                error_code="trusted_action_not_configured",
            )

        # Defense in depth: readiness is not trusted solely because of the
        # persisted status. Re-check all dependency states immediately pre-claim.
        tasks_by_key = {item.task_key: item for item in self.store.list_tasks(workflow_id)}
        dependency_keys = self.store.get_task_dependencies(workflow_id, task.task_key)
        for dependency_key in dependency_keys:
            dependency = tasks_by_key.get(dependency_key)
            if dependency is None or dependency.status is not TaskStatus.SUCCEEDED:
                raise WorkflowCapabilityInvocationError(
                    "Task dependencies are not all succeeded.",
                    task=task,
                    error_code="task_dependencies_incomplete",
                )

        security_policy = getattr(self.agent, "execution_security_policy", None)
        if not isinstance(security_policy, ExecutionSecurityPolicy):
            raise WorkflowCapabilityInvocationError(
                "Agent execution security policy is not configured.",
                task=task,
                error_code="execution_security_policy_unavailable",
            )

        # Reuse the existing Agent canonical API and unified executor binding.
        # The lambda is not an approval: capability actions bypass the ordinary
        # decision factory, while ExecutionSecurityPolicy remains mandatory.
        from agent.llm.agent_canonical_single_action_api import AgentCanonicalSingleActionAPI

        canonical_api = AgentCanonicalSingleActionAPI.for_agent(
            self.agent,
            decision_factory=lambda _request: None,
            security_policy=security_policy,
            trusted_capability_invocation_port=self.port,
        )

        try:
            claimed = self.store.transition_task(
                task.task_id,
                TaskStatus.RUNNING,
                expected_version=task.version,
                metadata={
                    "reason": "canonical_capability_invocation",
                    "action": action,
                },
            )
        except Exception as exc:
            raise WorkflowCapabilityInvocationError(
                "Task could not be claimed; no capability invocation was attempted.",
                task=task,
                error_code="task_claim_failed",
            ) from exc

        try:
            canonical_result = canonical_api.execute_from_boundary_request(
                request,
                execution_context=execution_context,
            )
            read_result = canonical_result.report.result
            if not isinstance(read_result, CapabilityReadResult):
                raise TypeError("Canonical capability path returned an invalid result.")

            result_payload = {
                "execution_id": execution_context.execution_id,
                "action": action,
                "target": read_result.target,
                "capability_id": read_result.capability_id,
                "data": read_result.data,
                "metadata": dict(read_result.metadata),
                "read_only": True,
            }
            # Fail closed if adapter data cannot be durably represented as JSON.
            encoded = json.dumps(
                result_payload, ensure_ascii=False, allow_nan=False,
                sort_keys=True, separators=(",", ":"),
            )
            persisted_payload = json.loads(encoded)
            completed = self.store.transition_task(
                claimed.task_id,
                TaskStatus.SUCCEEDED,
                expected_version=claimed.version,
                result_data=persisted_payload,
                metadata={
                    "reason": "canonical_capability_read_completed",
                    "execution_id": execution_context.execution_id,
                    "action": action,
                },
            )
        except Exception as exc:
            # Security denial is raised before the invocation port is entered.
            # Every other exception after dispatch is treated as unknown because
            # a read may have completed before evidence/result persistence failed.
            from agent.llm.execution_security import ExecutionSecurityError
            known_denial = isinstance(
                exc, (ExecutionSecurityError, CapabilityAuthorizationError)
            )
            unknown = not known_denial
            error_code = (
                "capability_invocation_outcome_unknown"
                if unknown else (
                    "capability_authorization_denied"
                    if isinstance(exc, CapabilityAuthorizationError)
                    else "canonical_execution_denied"
                )
            )
            try:
                blocked = self.store.transition_task(
                    claimed.task_id,
                    TaskStatus.BLOCKED,
                    expected_version=claimed.version,
                    error_code=error_code,
                    metadata={
                        "reason": error_code,
                        "exception_type": type(exc).__name__,
                        "outcome_unknown": unknown,
                        "execution_id": execution_context.execution_id,
                    },
                )
            except Exception as persistence_exc:
                raise WorkflowCapabilityInvocationError(
                    "Invocation failed and the workflow outcome could not be persisted.",
                    task=claimed,
                    error_code="workflow_outcome_persistence_failed",
                ) from persistence_exc
            raise WorkflowCapabilityInvocationError(
                "Capability invocation did not complete as a verified workflow result.",
                task=blocked,
                error_code=error_code,
            ) from exc

        return WorkflowCapabilityInvocationOutcome(
            task=completed,
            execution_id=execution_context.execution_id,
            action=action,
            target=read_result.target,
            capability_id=read_result.capability_id,
            result=read_result,
        )


__all__ = [
    "TRUSTED_CAPABILITY_TASK_TYPE",
    "WorkflowCapabilityInvocationError",
    "WorkflowCapabilityInvocationOutcome",
    "WorkflowCapabilityTaskExecutor",
]
