from dataclasses import dataclass
import pytest

from agent.llm.plan_validation import (
    LLMPlanValidationError,
    LLMPlanValidator,
)


@dataclass
class Step:
    action: str
    risk: float = 0.1
    success_criteria: str | None = "done"
    index: int | None = None


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    confidence: float = 0.9


class Goal:
    def __init__(self, description: str):
        self.description = description


def make_plan(goal="ship change", actions=("open", "deploy"), confidence=0.9):
    return Plan(
        goal=goal,
        confidence=confidence,
        steps=[Step(action=a, index=i + 1) for i, a in enumerate(actions)],
    )


def test_valid_plan_passes():
    result = LLMPlanValidator().validate(
        make_plan(), goal=Goal("ship change"), available_actions={"open": object(), "deploy": object()}
    )
    assert result.valid is True
    assert result.executable_step_indices == (1, 2)


def test_empty_plan_is_rejected():
    result = LLMPlanValidator().validate(
        Plan(goal="ship change", steps=[], confidence=0.9), goal="ship change", available_actions={"open": object()}
    )
    assert result.valid is False
    assert any(i.code == "empty_plan" for i in result.issues)


def test_missing_goal_is_rejected_when_required():
    result = LLMPlanValidator().validate(
        {"steps": [{"action": "open"}], "confidence": 0.9},
        goal="ship change",
        available_actions={"open": object()},
    )
    assert result.valid is False
    assert any(i.code == "missing_goal" for i in result.issues)


def test_goal_mismatch_is_rejected():
    result = LLMPlanValidator().validate(
        make_plan(goal="other"), goal="ship change", available_actions={"open": object(), "deploy": object()}
    )
    assert result.valid is False
    assert any(i.code == "goal_mismatch" for i in result.issues)


def test_goal_check_can_be_disabled():
    result = LLMPlanValidator(require_goal_match=False).validate(
        make_plan(goal="other"), goal="ship change", available_actions={"open": object(), "deploy": object()}
    )
    assert result.valid is True


def test_low_confidence_is_rejected():
    result = LLMPlanValidator(min_confidence=0.8).validate(
        make_plan(confidence=0.5), goal="ship change", available_actions={"open": object(), "deploy": object()}
    )
    assert result.valid is False
    assert any(i.code == "low_confidence" for i in result.issues)


def test_invalid_confidence_is_rejected():
    result = LLMPlanValidator().validate(
        {"goal": "ship change", "steps": [{"action": "open"}], "confidence": 2.0},
        goal="ship change", available_actions={"open": object()}
    )
    assert result.valid is False
    assert any(i.code == "invalid_confidence" for i in result.issues)


def test_unknown_action_is_rejected():
    result = LLMPlanValidator().validate(
        make_plan(actions=("missing",)), goal="ship change", available_actions={"open": object()}
    )
    assert result.valid is False
    assert any(i.code == "unknown_action" for i in result.issues)


def test_missing_action_is_rejected():
    result = LLMPlanValidator().validate(
        {"goal": "ship change", "steps": [{"risk": 0.1}], "confidence": 0.9},
        goal="ship change", available_actions={"open": object()}
    )
    assert result.valid is False
    assert any(i.code == "missing_action" for i in result.issues)


def test_duplicate_indices_are_rejected():
    plan = Plan(goal="ship change", steps=[Step("open", index=1), Step("deploy", index=1)])
    result = LLMPlanValidator().validate(plan, goal="ship change", available_actions={"open": object(), "deploy": object()})
    assert result.valid is False
    assert any(i.code == "duplicate_step_index" for i in result.issues)


def test_repeated_actions_are_allowed_when_available():
    result = LLMPlanValidator().validate(
        make_plan(actions=("open", "open")), goal="ship change", available_actions={"open": object()}
    )
    assert result.valid is True
    assert result.executable_step_indices == (1, 2)


def test_step_risk_above_limit_is_rejected():
    plan = Plan(goal="ship change", steps=[Step("deploy", risk=0.8)], confidence=0.9)
    result = LLMPlanValidator(max_step_risk=0.5).validate(
        plan, goal="ship change", available_actions={"deploy": object()}
    )
    assert result.valid is False
    assert any(i.code == "risk_above_limit" for i in result.issues)


def test_invalid_step_risk_is_rejected():
    plan = {"goal": "ship change", "confidence": 0.9, "steps": [{"action": "deploy", "risk": "high"}]}
    result = LLMPlanValidator().validate(plan, goal="ship change", available_actions={"deploy": object()})
    assert result.valid is False
    assert any(i.code == "invalid_step_risk" for i in result.issues)


def test_success_criteria_can_be_required():
    plan = {"goal": "ship change", "confidence": 0.9, "steps": [{"action": "deploy", "risk": 0.1}]}
    result = LLMPlanValidator(require_success_criteria=True).validate(
        plan, goal="ship change", available_actions={"deploy": object()}
    )
    assert result.valid is False
    assert any(i.code == "missing_success_criteria" for i in result.issues)


def test_success_criteria_requirement_can_be_disabled():
    plan = {"goal": "ship change", "confidence": 0.9, "steps": [{"action": "deploy", "risk": 0.1}]}
    result = LLMPlanValidator(require_success_criteria=False).validate(
        plan, goal="ship change", available_actions={"deploy": object()}
    )
    assert result.valid is True


def test_max_step_count_is_enforced():
    result = LLMPlanValidator(max_plan_steps=1).validate(
        make_plan(actions=("open", "deploy")), goal="ship change", available_actions={"open": object(), "deploy": object()}
    )
    assert result.valid is False
    assert any(i.code == "too_many_steps" for i in result.issues)


def test_action_mapping_keys_are_used():
    result = LLMPlanValidator().validate(
        make_plan(actions=("open",)), goal="ship change", available_actions={"open": object()}
    )
    assert result.valid is True


def test_action_objects_are_supported():
    @dataclass
    class Action:
        name: str
    result = LLMPlanValidator().validate(
        make_plan(actions=("open",)), goal="ship change", available_actions=[Action("open")]
    )
    assert result.valid is True


def test_invalid_validator_configuration_is_rejected():
    with pytest.raises(ValueError):
        LLMPlanValidator(max_plan_steps=0)


def test_validation_result_is_deterministic():
    plan = make_plan()
    validator = LLMPlanValidator()
    first = validator.validate(plan, goal="ship change", available_actions={"open": object(), "deploy": object()})
    second = validator.validate(plan, goal="ship change", available_actions={"open": object(), "deploy": object()})
    assert first == second
