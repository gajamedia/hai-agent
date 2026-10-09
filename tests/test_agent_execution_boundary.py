from dataclasses import dataclass

import pytest

from agent.llm.agent_execution_boundary import (
    AgentExecutionRequest,
    CallbackAgentExecutionBoundary,
    LLMPlanAgentExecutionAdapter,
)
from agent.llm.plan_validation import LLMPlanValidator


@dataclass
class Step:
    action: str
    index: int
    risk: float = 0.1
    success_criteria: str | None = "done"


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    confidence: float = 0.9


def make_plan(actions=("open", "deploy")):
    return Plan(
        goal="ship change",
        steps=[Step(a, i + 1) for i, a in enumerate(actions)],
    )


def make_boundary(calls):
    return CallbackAgentExecutionBoundary(
        lambda request: calls.append(request) or {"success": True}
    )


def test_request_normalizes_action_and_copies_metadata():
    metadata = {"trace": "abc"}
    request = AgentExecutionRequest(
        action="  open ",
        step={"action": "open"},
        position=1,
        metadata=metadata,
    )
    metadata["mutated"] = True
    assert request.action == "open"
    assert request.metadata == {"trace": "abc"}


def test_request_rejects_empty_action():
    with pytest.raises(ValueError, match="action cannot be empty"):
        AgentExecutionRequest(action=" ", step={}, position=1)


def test_request_rejects_invalid_position():
    with pytest.raises(ValueError, match="position must be >= 1"):
        AgentExecutionRequest(action="open", step={}, position=0)


def test_callback_boundary_requires_callable():
    with pytest.raises(TypeError, match="callback must be callable"):
        CallbackAgentExecutionBoundary(None)


def test_callback_boundary_executes_request():
    seen = []
    boundary = CallbackAgentExecutionBoundary(lambda request: seen.append(request) or {"success": True})
    request = AgentExecutionRequest(action="open", step={}, position=1)
    assert boundary.execute(request) == {"success": True}
    assert seen == [request]


def test_valid_plan_is_validated_and_executed():
    calls = []
    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
    )
    assert result.accepted is True
    assert result.executed is True
    assert result.report.completed is True
    assert [request.action for request in calls] == ["open", "deploy"]


def test_request_contains_context_for_each_step():
    calls = []
    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("open", "deploy")),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
        world_state={"status": "ready"},
        experience_id="exp-1",
        metadata={"source": "llm"},
    )
    assert result.report.completed is True
    assert calls[0].goal == "ship change"
    assert calls[0].world_state == {"status": "ready"}
    assert calls[0].experience_id == "exp-1"
    assert calls[0].metadata == {"source": "llm"}
    assert calls[0].position == 1
    assert calls[1].position == 2


def test_invalid_plan_is_not_sent_to_boundary():
    calls = []
    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("missing",)),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert result.accepted is False
    assert result.executed is False
    assert result.report is None
    assert calls == []


def test_missing_boundary_execute_is_rejected():
    adapter = LLMPlanAgentExecutionAdapter()
    with pytest.raises(TypeError, match="boundary must provide"):
        adapter.execute(
            make_plan(actions=("open",)),
            boundary=object(),
            goal="ship change",
            available_actions={"open": object()},
        )


def test_step_failure_stops_boundary_execution():
    calls = []

    def run(request):
        calls.append(request.action)
        return {"success": request.action == "open"}

    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(),
        boundary=CallbackAgentExecutionBoundary(run),
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
    )
    assert result.report.stopped is True
    assert result.report.stop_reason == "step_failed"
    assert calls == ["open", "deploy"]


def test_boundary_exception_is_contained_in_report():
    def run(_request):
        raise RuntimeError("boom")

    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("open",)),
        boundary=CallbackAgentExecutionBoundary(run),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert result.report.stopped is True
    assert result.report.stop_reason == "executor_error:RuntimeError"
    assert result.report.steps[0].success is False


def test_stop_checker_can_stop_after_a_successful_step():
    calls = []
    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
        stop_checker=lambda step, index, outcome: index == 1,
    )
    assert result.report.stop_reason == "stop_condition"
    assert [request.action for request in calls] == ["open"]


def test_repeated_actions_are_preserved_in_execution_order():
    calls = []
    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("open", "open", "deploy")),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
    )
    assert result.report.completed is True
    assert [request.action for request in calls] == ["open", "open", "deploy"]


def test_custom_validator_is_used():
    validator = LLMPlanValidator(min_confidence=0.95)
    adapter = LLMPlanAgentExecutionAdapter(validator=validator)
    calls = []
    result = adapter.execute(
        make_plan(),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
    )
    assert result.accepted is False
    assert any(issue.code == "low_confidence" for issue in result.validation.issues)
    assert calls == []


def test_last_validation_is_retained():
    adapter = LLMPlanAgentExecutionAdapter()
    result = adapter.validate(
        make_plan(actions=("open",)),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert adapter.last_validation == result


def test_last_result_is_retained():
    adapter = LLMPlanAgentExecutionAdapter()
    result = adapter.execute(
        make_plan(actions=("open",)),
        boundary=make_boundary([]),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert adapter.last_result == result


def test_invalid_plan_result_is_retained():
    adapter = LLMPlanAgentExecutionAdapter()
    result = adapter.execute(
        make_plan(actions=("missing",)),
        boundary=make_boundary([]),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert adapter.last_result == result
    assert adapter.last_validation == result.validation


def test_metadata_is_available_without_mutating_caller_mapping():
    metadata = {"source": "llm"}
    calls = []
    LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("open",)),
        boundary=make_boundary(calls),
        goal="ship change",
        available_actions={"open": object()},
        metadata=metadata,
    )
    calls[0].metadata["local"] = True
    assert metadata == {"source": "llm"}


def test_explicit_validator_reuses_same_instance():
    validator = LLMPlanValidator()
    adapter = LLMPlanAgentExecutionAdapter(validator=validator)
    assert adapter.validator is validator


def test_execution_result_reports_acceptance_and_execution_separately():
    invalid = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("missing",)),
        boundary=make_boundary([]),
        goal="ship change",
        available_actions={"open": object()},
    )
    valid = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("open",)),
        boundary=make_boundary([]),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert invalid.accepted is False and invalid.executed is False
    assert valid.accepted is True and valid.executed is True


def test_validation_can_be_used_without_execution():
    adapter = LLMPlanAgentExecutionAdapter()
    result = adapter.validate(
        make_plan(actions=("open",)),
        goal="ship change",
        available_actions={"open": object()},
    )
    assert result.valid is True
    assert adapter.last_result is None


def test_goal_object_with_description_is_supported():
    class Goal:
        description = "ship change"

    calls = []
    result = LLMPlanAgentExecutionAdapter().execute(
        make_plan(actions=("open",)),
        boundary=make_boundary(calls),
        goal=Goal(),
        available_actions={"open": object()},
    )
    assert result.accepted is True
    assert calls[0].goal.description == "ship change"
