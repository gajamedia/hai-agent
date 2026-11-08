from __future__ import annotations

import pytest

from api.security.authentication import AuthenticatedPrincipal
from api.security.trusted_execution_context import TrustedExecutionContext
from agent.llm.capability_execution_contracts import (
    CapabilityAuthorizationError,
    CapabilityReadResult,
)
from agent.llm.execution_security import ExecutionSecurityPolicy
from agent.llm.traced_authenticated_capability_execution_gateway import (
    CapabilityTracePersistenceError,
)
from agent.llm.trusted_capability_invocation_port import (
    CapabilityInvocationSpec,
    TrustedCapabilityInvocationPort,
)
from agent.workflow import (
    SQLiteWorkflowStore,
    TaskDependencyPlanner,
    TaskSpec,
    TaskStatus,
    WorkflowStatus,
)
from agent.workflow.invocation import (
    TRUSTED_CAPABILITY_TASK_TYPE,
    WorkflowCapabilityInvocationError,
    WorkflowCapabilityTaskExecutor,
)


def make_context():
    principal = AuthenticatedPrincipal(
        actor_id="workflow-test-user",
        roles=("analyst",),
        permissions=("business:database:read",),
        issuer="workflow-test",
        subject="workflow-test-subject",
        is_authenticated=True,
    )
    return TrustedExecutionContext(principal)


class FakeGateway:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def read(self, principal, request, *, execution_id):
        self.calls.append((principal, request, execution_id))
        if self.error:
            raise self.error
        return CapabilityReadResult(
            target=request.target,
            capability_id=request.capability_id,
            data=[{"id": 1, "name": "Sample"}],
            metadata={"row_count": 1},
        )


class FakeAction:
    def __init__(self, name):
        self.name = name
        self.executed = 0

    def execute(self, **kwargs):
        self.executed += 1
        raise AssertionError("Simulation placeholder action must not execute")


class FakeUnifiedExecutor:
    def execute(self, **kwargs):
        raise AssertionError("Capability task must use the trusted port")


class FakeAgent:
    def __init__(self):
        self.actions = {"simulation_placeholder": FakeAction("simulation_placeholder")}
        self.unified_executor = FakeUnifiedExecutor()
        self.execution_security_policy = ExecutionSecurityPolicy(
            action_risks={"read_business_data": 0.1}
        )


def make_workflow(tmp_path, *, input_data=None, dependencies=()):
    store = SQLiteWorkflowStore(tmp_path / "workflow.sqlite3")
    workflow = store.create_workflow(
        goal="Read business data safely",
        idempotency_key="workflow-capability-test",
    )
    specs = [
        TaskSpec(
            task_key="read",
            task_type=TRUSTED_CAPABILITY_TASK_TYPE,
            description="Read business records",
            input_data=input_data or {
                "action": "read_business_data",
                "arguments": {"table": "customers", "limit": 10},
            },
        )
    ]
    if dependencies:
        specs.insert(0, TaskSpec(
            task_key="prepare",
            task_type="internal.prepare",
            description="Preparation",
        ))
        specs[1] = TaskSpec(
            task_key="read",
            task_type=TRUSTED_CAPABILITY_TASK_TYPE,
            description="Read business records",
            depends_on=tuple(dependencies),
            input_data=input_data or {
                "action": "read_business_data",
                "arguments": {"table": "customers", "limit": 10},
            },
        )
    plan = TaskDependencyPlanner().plan(specs)
    tasks = store.create_tasks_from_plan(workflow.workflow_id, plan)
    if dependencies:
        prepare = next(item for item in tasks if item.task_key == "prepare")
        prepare = store.transition_task(
            prepare.task_id, TaskStatus.READY, expected_version=prepare.version
        )
        prepare = store.transition_task(
            prepare.task_id, TaskStatus.RUNNING, expected_version=prepare.version
        )
        store.transition_task(
            prepare.task_id, TaskStatus.SUCCEEDED, expected_version=prepare.version
        )
    store.refresh_task_readiness(workflow.workflow_id)
    current_workflow = store.get_workflow(workflow.workflow_id)
    current_workflow = store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.PLANNED,
        expected_version=current_workflow.version,
    )
    current_workflow = store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.READY,
        expected_version=current_workflow.version,
    )
    store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.RUNNING,
        expected_version=current_workflow.version,
    )
    read_task = next(item for item in store.list_tasks(workflow.workflow_id)
                     if item.task_key == "read")
    return store, workflow, read_task


def make_executor(store, gateway):
    port = TrustedCapabilityInvocationPort(
        gateway,
        {
            "read_business_data": CapabilityInvocationSpec(
                action_name="read_business_data",
                target="business_database",
                capability_id="business_database.select",
            )
        },
    )
    return WorkflowCapabilityTaskExecutor(
        store=store,
        agent=FakeAgent(),
        trusted_capability_invocation_port=port,
    )


def test_ready_task_invokes_through_canonical_trusted_gateway(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    gateway = FakeGateway()
    executor = make_executor(store, gateway)

    outcome = executor.execute_task(
        workflow.workflow_id, task.task_id, execution_context=make_context()
    )

    assert outcome.task.status is TaskStatus.SUCCEEDED
    assert outcome.task.attempt_count == 1
    assert outcome.task.result_data["read_only"] is True
    assert outcome.task.result_data["target"] == "business_database"
    assert len(gateway.calls) == 1
    principal, request, execution_id = gateway.calls[0]
    assert principal.actor_id == "workflow-test-user"
    assert request.target == "business_database"
    assert request.capability_id == "business_database.select"
    assert execution_id == outcome.execution_id


def test_task_cannot_override_server_owned_target_or_capability(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    gateway = FakeGateway()
    executor = make_executor(store, gateway)

    outcome = executor.execute_task(
        workflow.workflow_id, task.task_id, execution_context=make_context()
    )

    assert outcome.target == "business_database"
    assert outcome.capability_id == "business_database.select"


def test_unmapped_action_is_rejected_before_claim(tmp_path):
    store, workflow, task = make_workflow(
        tmp_path,
        input_data={"action": "attacker.action", "arguments": {}},
    )
    gateway = FakeGateway()
    executor = make_executor(store, gateway)

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(
            workflow.workflow_id, task.task_id, execution_context=make_context()
        )

    assert error.value.error_code == "trusted_action_not_configured"
    assert store.get_task(task.task_id).status is TaskStatus.READY
    assert gateway.calls == []


def test_task_input_rejects_target_and_capability_overrides(tmp_path):
    store, workflow, task = make_workflow(
        tmp_path,
        input_data={
            "action": "read_business_data",
            "arguments": {"table": "customers"},
            "target": "attacker",
            "capability_id": "attacker.capability",
        },
    )
    executor = make_executor(store, FakeGateway())

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(
            workflow.workflow_id, task.task_id, execution_context=make_context()
        )

    assert error.value.error_code == "invalid_capability_task_input"
    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_missing_or_invalid_trusted_context_is_rejected(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    executor = make_executor(store, FakeGateway())

    with pytest.raises(TypeError):
        executor.execute_task(workflow.workflow_id, task.task_id, execution_context=None)

    assert store.get_task(task.task_id).status is TaskStatus.READY


def test_dependency_must_be_succeeded_before_invocation(tmp_path):
    store, workflow, task = make_workflow(tmp_path, dependencies=("prepare",))
    gateway = FakeGateway()
    executor = make_executor(store, gateway)

    # Make the prerequisite non-successful after task readiness was evaluated.
    prepare = next(item for item in store.list_tasks(workflow.workflow_id)
                   if item.task_key == "prepare")
    # A succeeded task is terminal; create a second workflow where the task is not ready.
    store2 = SQLiteWorkflowStore(tmp_path / "blocked.sqlite3")
    wf2 = store2.create_workflow(goal="blocked", idempotency_key="blocked")
    plan2 = TaskDependencyPlanner().plan([
        TaskSpec(task_key="first", task_type="internal", description="first"),
        TaskSpec(
            task_key="second", task_type=TRUSTED_CAPABILITY_TASK_TYPE,
            description="read", depends_on=("first",),
            input_data={"action": "read_business_data", "arguments": {}},
        ),
    ])
    tasks2 = store2.create_tasks_from_plan(wf2.workflow_id, plan2)
    wf2 = store2.transition_workflow(
        wf2.workflow_id, WorkflowStatus.PLANNED, expected_version=wf2.version
    )
    wf2 = store2.transition_workflow(
        wf2.workflow_id, WorkflowStatus.READY, expected_version=wf2.version
    )
    store2.transition_workflow(
        wf2.workflow_id, WorkflowStatus.RUNNING, expected_version=wf2.version
    )
    second = next(item for item in tasks2 if item.task_key == "second")
    executor2 = make_executor(store2, gateway)
    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor2.execute_task(wf2.workflow_id, second.task_id, execution_context=make_context())
    assert error.value.error_code == "task_not_ready"
    assert gateway.calls == []


def test_gateway_authorization_denial_blocks_task_without_success(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    executor = make_executor(store, FakeGateway(CapabilityAuthorizationError("denied")))

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(
            workflow.workflow_id, task.task_id, execution_context=make_context()
        )

    persisted = store.get_task(task.task_id)
    assert error.value.error_code == "capability_authorization_denied"
    assert persisted.status is TaskStatus.BLOCKED
    assert persisted.result_data is None


def test_trace_failure_is_recorded_as_unknown_outcome_and_not_retried(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    gateway = FakeGateway(CapabilityTracePersistenceError("trace unavailable"))
    executor = make_executor(store, gateway)

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(
            workflow.workflow_id, task.task_id, execution_context=make_context()
        )

    persisted = store.get_task(task.task_id)
    assert error.value.error_code == "capability_invocation_outcome_unknown"
    assert persisted.status is TaskStatus.BLOCKED
    assert persisted.error_code == "capability_invocation_outcome_unknown"
    assert persisted.attempt_count == 1
    assert len(gateway.calls) == 1


def test_wrong_workflow_task_pair_is_rejected(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    other = store.create_workflow(goal="other", idempotency_key="other")
    executor = make_executor(store, FakeGateway())

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(other.workflow_id, task.task_id, execution_context=make_context())

    assert error.value.error_code == "task_not_found"


def test_only_ready_task_can_be_claimed(tmp_path):
    store = SQLiteWorkflowStore(tmp_path / "not-ready.sqlite3")
    workflow = store.create_workflow(goal="test", idempotency_key="not-ready")
    task = store.create_task(
        workflow.workflow_id,
        task_key="read",
        task_type=TRUSTED_CAPABILITY_TASK_TYPE,
        description="Read",
        input_data={"action": "read_business_data", "arguments": {}},
    )
    workflow = store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.PLANNED,
        expected_version=workflow.version,
    )
    workflow = store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.READY,
        expected_version=workflow.version,
    )
    store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.RUNNING,
        expected_version=workflow.version,
    )
    executor = make_executor(store, FakeGateway())

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(workflow.workflow_id, task.task_id, execution_context=make_context())

    assert error.value.error_code == "task_not_ready"
    assert store.get_task(task.task_id).status is TaskStatus.PENDING


def test_workflow_terminal_state_blocks_invocation(tmp_path):
    store, workflow, task = make_workflow(tmp_path)
    current = store.get_workflow(workflow.workflow_id)
    store.transition_workflow(
        workflow.workflow_id, WorkflowStatus.CANCELLED,
        expected_version=current.version,
    )
    executor = make_executor(store, FakeGateway())

    with pytest.raises(WorkflowCapabilityInvocationError) as error:
        executor.execute_task(workflow.workflow_id, task.task_id, execution_context=make_context())

    assert error.value.error_code == "workflow_terminal"
