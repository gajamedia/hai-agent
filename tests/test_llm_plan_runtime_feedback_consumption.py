from __future__ import annotations

import pytest

from agent.llm.llm_plan_runtime_feedback_consumption import (
    LLMPlanRuntimeFeedbackApplication,
    LLMPlanRuntimeFeedbackAwareBridge,
    LLMPlanRuntimeFeedbackConsumptionConfigurationError,
    LLMPlanRuntimeFeedbackConsumptionError,
    LLMPlanRuntimeFeedbackConsumptionOrderError,
    LLMPlanRuntimeFeedbackConsumer,
    LLMPlanRuntimeFeedbackDecision,
)
from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)
from agent.llm.planning import LLMPlan, LLMPlanStep


def feedback(
    *,
    step_index=0,
    success=True,
    progress_delta=0.2,
    continue_recommended=True,
    exploration_recommended=False,
    risk_level="low",
    change_required=False,
):
    return LLMPlanRuntimeLearningFeedback(
        step_index=step_index,
        action="act",
        success=success,
        progress_delta=progress_delta,
        evidence_quality=1.0,
        confidence=1.0,
        diagnosis="ok",
        lesson="keep going",
        next_step="continue",
        change_required=change_required,
        continue_recommended=continue_recommended,
        exploration_recommended=exploration_recommended,
        risk_level=risk_level,
        risk_score={"low": 0.25, "medium": 0.5, "high": 0.75}[risk_level],
        information_gain=0.2,
    )


def plan():
    return LLMPlan(
        goal="finish",
        rationale="test",
        steps=(
            LLMPlanStep(1, "a", "do a"),
            LLMPlanStep(2, "b", "do b"),
        ),
    )


class FakeBridge:
    def __init__(self):
        self.calls = []
        self.api = object()
        self.last_execution_request = None

    def execute_step(self, plan, step, **kwargs):
        self.calls.append((step, kwargs))
        self.last_execution_request = kwargs
        return type("Result", (), {"success": True, "result": {"success": True}})()


def test_defaults_validate():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    assert consumer.exploration_risk_increment == 0.2
    assert consumer.reduced_action_risk_multiplier == 0.75


def test_negative_exploration_increment_rejected():
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionConfigurationError):
        LLMPlanRuntimeFeedbackConsumer(exploration_risk_increment=-0.1)


def test_invalid_exploration_multiplier_rejected():
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionConfigurationError):
        LLMPlanRuntimeFeedbackConsumer(exploration_information_gain_multiplier=0)


def test_invalid_risk_multiplier_rejected():
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionConfigurationError):
        LLMPlanRuntimeFeedbackConsumer(reduced_action_risk_multiplier=1.1)


def test_consume_stores_feedback():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    result = consumer.consume(feedback(step_index=0))
    assert consumer.latest_feedback.step_index == 0
    assert result.source_step_index == 0
    assert result.target_step_index == 1


def test_success_feedback_recommends_continue():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback())
    decision = consumer.before_step(1)
    assert decision.should_continue is True
    assert decision.should_stop is False
    assert decision.reason == "runtime_feedback_continue"


def test_exploration_feedback_increases_exploration():
    consumer = LLMPlanRuntimeFeedbackConsumer(exploration_risk_increment=0.25)
    consumer.consume(feedback(exploration_recommended=True, risk_level="medium"))
    decision = consumer.before_step(1)
    assert decision.increase_exploration is True
    assert decision.reduce_risk is True
    assert decision.exploration_risk == 0.25
    assert decision.action_risk_multiplier == 0.75


def test_high_risk_feedback_reduces_risk():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(risk_level="high"))
    decision = consumer.before_step(1)
    assert decision.reduce_risk is True
    assert decision.action_risk_multiplier == 0.75


def test_stop_feedback_blocks_continue():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(success=False, continue_recommended=False, exploration_recommended=True, risk_level="high"))
    decision = consumer.before_step(1)
    assert decision.should_stop is True
    assert decision.should_continue is False
    assert decision.reason == "runtime_feedback_stop"


def test_change_required_stops_even_when_successful():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(change_required=True))
    decision = consumer.before_step(1)
    assert decision.should_stop is True


def test_stop_on_feedback_can_be_disabled():
    consumer = LLMPlanRuntimeFeedbackConsumer(stop_on_feedback=False)
    consumer.consume(feedback(success=False, continue_recommended=False, risk_level="high"))
    decision = consumer.before_step(1)
    assert decision.should_continue is True
    assert decision.should_stop is False


def test_neutral_without_feedback():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    decision = consumer.before_step(1)
    assert decision.source_step_index is None
    assert decision.should_continue is True
    assert decision.reason == "runtime_feedback_neutral"


def test_negative_step_rejected():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionOrderError):
        consumer.before_step(-1)


def test_same_step_feedback_cannot_be_consumed():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(step_index=1))
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionOrderError):
        consumer.before_step(1)


def test_non_immediate_feedback_rejected():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(step_index=0))
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionOrderError):
        consumer.before_step(2)


def test_feedback_cannot_be_consumed_twice():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(step_index=0))
    consumer.before_step(1)
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionOrderError):
        consumer.before_step(1)


def test_reset_clears_runtime_feedback_state():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback())
    consumer.before_step(1)
    consumer.reset()
    assert consumer.latest_feedback is None
    assert consumer.last_decision is None
    assert consumer.decision_history == ()
    assert consumer.consumed_steps == ()


def test_apply_to_inputs_neutral_preserves_inputs():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    result = consumer.apply_to_execution_inputs(
        1,
        exploration_risk=0.4,
        action_risk={"a": 0.6},
        information_gain={"a": 0.5},
        metadata={"x": 1},
    )
    assert isinstance(result, LLMPlanRuntimeFeedbackApplication)
    assert result.exploration_risk == 0.4
    assert result.action_risk == {"a": 0.6}
    assert result.information_gain == {"a": 0.5}
    assert result.metadata["x"] == 1


def test_apply_to_inputs_increases_exploration_and_scales_information_gain():
    consumer = LLMPlanRuntimeFeedbackConsumer(exploration_risk_increment=0.25)
    consumer.consume(feedback(exploration_recommended=True, risk_level="low"))
    result = consumer.apply_to_execution_inputs(
        1,
        exploration_risk=0.2,
        action_risk={"a": 0.4},
        information_gain={"a": 0.5},
    )
    assert result.exploration_risk == 0.45
    assert result.information_gain["a"] == 0.55


def test_apply_to_inputs_reduces_action_risk():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(risk_level="medium"))
    result = consumer.apply_to_execution_inputs(
        1,
        exploration_risk=0.1,
        action_risk={"a": 0.8},
        information_gain={"a": 0.2},
    )
    assert result.action_risk["a"] == 0.6


def test_apply_to_inputs_stop_does_not_execute():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(success=False, continue_recommended=False, risk_level="high"))
    result = consumer.apply_to_execution_inputs(
        1,
        exploration_risk=0.4,
        action_risk={"a": 0.8},
        information_gain={"a": 0.2},
    )
    assert result.decision.should_stop is True
    assert result.metadata["runtime_feedback_stop"] is True


def test_consumer_history_is_immutable_tuple():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback())
    assert isinstance(consumer.decision_history, tuple)


def test_consumer_decision_type():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback())
    assert isinstance(consumer.last_decision, LLMPlanRuntimeFeedbackDecision)


def test_multiple_feedback_steps_are_consumable_in_order():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(step_index=0))
    assert consumer.before_step(1).source_step_index == 0
    consumer.consume(feedback(step_index=1))
    assert consumer.before_step(2).source_step_index == 1
    assert consumer.consumed_steps == (1, 2)


def test_feedback_action_is_preserved():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback())
    assert consumer.before_step(1).action == "act"


def test_feedback_metadata_contains_diagnosis():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback())
    assert consumer.before_step(1).metadata["feedback_diagnosis"] == "ok"


def test_bridge_requires_execute_step():
    consumer = LLMPlanRuntimeFeedbackConsumer()
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionConfigurationError):
        LLMPlanRuntimeFeedbackAwareBridge(object(), consumer)


def test_bridge_requires_consumer():
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionConfigurationError):
        LLMPlanRuntimeFeedbackAwareBridge(FakeBridge(), object())


def test_bridge_forwards_neutral_execution():
    base = FakeBridge()
    consumer = LLMPlanRuntimeFeedbackConsumer()
    bridge = LLMPlanRuntimeFeedbackAwareBridge(base, consumer)
    p = plan()
    result = bridge.execute_step(
        p,
        p.steps[0],
        exploration_risk=0.2,
        action_risk={"a": 0.5},
        information_gain={"a": 0.4},
    )
    assert result.success is True
    assert len(base.calls) == 1
    assert base.calls[0][1]["exploration_risk"] == 0.2


def test_bridge_consumes_feedback_before_next_step():
    base = FakeBridge()
    consumer = LLMPlanRuntimeFeedbackConsumer(exploration_risk_increment=0.25)
    consumer.consume(feedback(step_index=1, exploration_recommended=True, risk_level="medium"))
    bridge = LLMPlanRuntimeFeedbackAwareBridge(base, consumer)
    p = plan()
    bridge.execute_step(
        p,
        p.steps[1],
        exploration_risk=0.2,
        action_risk={"b": 0.8},
        information_gain={"b": 0.5},
    )
    kwargs = base.calls[0][1]
    assert kwargs["exploration_risk"] == 0.45
    assert kwargs["action_risk"]["b"] == 0.6
    assert kwargs["information_gain"]["b"] == 0.55


def test_bridge_stop_feedback_returns_blocked_result():
    base = FakeBridge()
    consumer = LLMPlanRuntimeFeedbackConsumer()
    consumer.consume(feedback(step_index=1, success=False, continue_recommended=False, risk_level="high"))
    bridge = LLMPlanRuntimeFeedbackAwareBridge(base, consumer)
    p = plan()
    result = bridge.execute_step(p, p.steps[1])
    assert result.success is False
    assert result.result["blocked_by_runtime_feedback"] is True
    assert base.calls == []


def test_bridge_delegates_unknown_attributes():
    base = FakeBridge()
    consumer = LLMPlanRuntimeFeedbackConsumer()
    bridge = LLMPlanRuntimeFeedbackAwareBridge(base, consumer)
    assert bridge.api is base.api


def test_bridge_execute_step_index_rejects_missing_step():
    base = FakeBridge()
    consumer = LLMPlanRuntimeFeedbackConsumer()
    bridge = LLMPlanRuntimeFeedbackAwareBridge(base, consumer)
    with pytest.raises(LLMPlanRuntimeFeedbackConsumptionOrderError):
        bridge.execute_step_index(plan(), 99)


def test_stage_2411_bridge_can_feed_stage_2412_consumer():
    from agent.llm.llm_plan_runtime_learning_feedback import (
        LLMPlanRuntimeLearningFeedbackBridge,
    )

    consumer = LLMPlanRuntimeFeedbackConsumer(
        exploration_risk_increment=0.25,
    )
    learning_bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: {
            "success": True,
            "progress_delta": 0.0,
            "evidence_quality": 1.0,
            "confidence": 1.0,
            "diagnosis": "partial_progress",
            "lesson": "explore more",
            "next_step": "try an alternative",
            "change_required": False,
            "exploration_recommended": True,
            "continue_recommended": True,
            "risk_level": "medium",
        },
        feedback_observer=consumer.consume,
    )

    learning_bridge.observe_experience(
        {"step_index": 1, "action": "a"},
        1,
        "a",
        {},
    )

    decision = consumer.before_step(2)
    assert decision.source_step_index == 1
    assert decision.increase_exploration is True
    assert decision.reduce_risk is True


def test_end_to_end_feedback_is_applied_to_next_bridge_call():
    from agent.llm.llm_plan_runtime_learning_feedback import (
        LLMPlanRuntimeLearningFeedbackBridge,
    )

    consumer = LLMPlanRuntimeFeedbackConsumer(
        exploration_risk_increment=0.25,
    )
    learning_bridge = LLMPlanRuntimeLearningFeedbackBridge(
        reflection_provider=lambda episode: {
            "success": True,
            "progress_delta": 0.0,
            "evidence_quality": 1.0,
            "confidence": 1.0,
            "diagnosis": "partial_progress",
            "lesson": "explore more",
            "next_step": "try an alternative",
            "change_required": False,
            "exploration_recommended": True,
            "continue_recommended": True,
            "risk_level": "medium",
        },
        feedback_observer=consumer.consume,
    )

    base = FakeBridge()
    bridge = LLMPlanRuntimeFeedbackAwareBridge(base, consumer)
    p = plan()

    base_result = bridge.execute_step(
        p,
        p.steps[0],
        exploration_risk=0.1,
        action_risk={"a": 0.8},
        information_gain={"a": 0.2},
    )
    assert base_result.success is True

    learning_bridge.observe_experience(
        {"step_index": 1, "action": "a"},
        1,
        "a",
        {},
    )

    next_result = bridge.execute_step(
        p,
        p.steps[1],
        exploration_risk=0.1,
        action_risk={"b": 0.8},
        information_gain={"b": 0.2},
    )
    assert next_result.success is True
    kwargs = base.calls[-1][1]
    assert kwargs["exploration_risk"] == 0.35
    assert kwargs["action_risk"]["b"] == 0.6
    assert kwargs["information_gain"]["b"] == 0.22
