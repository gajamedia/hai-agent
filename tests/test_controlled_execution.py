from dataclasses import dataclass
import pytest

from agent.llm.controlled_execution import (
    ControlledExecutionError,
    ControlledPlanExecutor,
)
from agent.llm.plan_validation import (
    LLMPlanValidator,
)


@dataclass
class Step:
    action: str
    index: int
    risk: float = 0.1


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    confidence: float = 0.9


def validated_plan():
    plan = Plan("ship change", [Step("open", 1), Step("deploy", 2)])
    validation = LLMPlanValidator().validate(
        plan,
        goal="ship change",
        available_actions={"open": object(), "deploy": object()},
    )
    assert validation.valid
    return plan, validation


def test_validated_plan_executes_in_order():
    plan, validation = validated_plan()
    calls = []

    executor = ControlledPlanExecutor(
        lambda action, step, index: calls.append((action, index)) or {"success": True}
    )
    report = executor.execute(plan, validation)

    assert report.completed is True
    assert calls == [("open", 1), ("deploy", 2)]


def test_invalid_plan_cannot_execute():
    plan, validation = validated_plan()
    bad_validation = validation.__class__(False, (validation.issues[0],), 2, ()) if validation.issues else validation.__class__(False, (), 2, ())
    executor = ControlledPlanExecutor(lambda *_: {"success": True})
    with pytest.raises(ControlledExecutionError, match="invalid LLM plan"):
        executor.execute(plan, bad_validation)


def test_failed_step_stops_execution():
    plan, validation = validated_plan()
    calls = []

    def run(action, step, index):
        calls.append(action)
        return {"success": action == "open"}

    report = ControlledPlanExecutor(run).execute(plan, validation)
    assert report.stopped is True
    assert report.stop_reason == "step_failed"
    assert calls == ["open", "deploy"]
    assert report.completed is False


def test_executor_exception_stops_safely():
    plan, validation = validated_plan()

    def run(action, step, index):
        raise RuntimeError("boom")

    report = ControlledPlanExecutor(run).execute(plan, validation)
    assert report.stopped is True
    assert report.stop_reason == "executor_error:RuntimeError"
    assert report.steps[0].success is False


def test_stop_checker_stops_after_step():
    plan, validation = validated_plan()
    calls = []

    report = ControlledPlanExecutor(
        lambda action, step, index: calls.append(action) or {"success": True}
    ).execute(plan, validation, stop_checker=lambda step, index, outcome: index == 1)

    assert report.stop_reason == "stop_condition"
    assert calls == ["open"]


def test_successful_plan_completes():
    plan, validation = validated_plan()
    report = ControlledPlanExecutor(lambda *_: {"success": True}).execute(plan, validation)
    assert report.started is True
    assert report.completed is True
    assert report.stopped is False
    assert len(report.steps) == 2


def test_empty_validated_plan_reports_completed():
    class EmptyPlan:
        steps = []
    from agent.llm.plan_validation import LLMPlanValidationResult
    validation = LLMPlanValidationResult(True, (), 0, ())
    report = ControlledPlanExecutor(lambda *_: True).execute(EmptyPlan(), validation)
    assert report.started is False
    assert report.completed is True


def test_dict_plan_is_supported():
    plan = {"goal": "ship change", "steps": [{"action": "open"}]}
    validation = LLMPlanValidator().validate(plan, goal="ship change", available_actions={"open": object()})
    assert validation.valid
    report = ControlledPlanExecutor(lambda *_: {"success": True}).execute(plan, validation)
    assert report.completed is True


def test_non_callable_executor_is_rejected():
    with pytest.raises(TypeError, match="must be callable"):
        ControlledPlanExecutor(None)


def test_report_preserves_raw_outcome():
    plan, validation = validated_plan()
    outcome = {"success": True, "value": 42}
    report = ControlledPlanExecutor(lambda *_: outcome).execute(plan, validation, stop_checker=lambda *_: True)
    assert report.steps[0].result is outcome
