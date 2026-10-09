from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.llm.llm_plan_adaptive_replanning import (
    LLMPlanAdaptiveReplanner,
    LLMPlanAdaptiveReplanningConfigurationError,
    LLMPlanAdaptiveReplanningRequestError,
    LLMPlanAdaptiveReplanningSafetyError,
    LLMPlanAdaptiveReplanningValidationError,
    LLMPlanReplanningRequest,
)
from agent.llm.llm_plan_runtime_learning_feedback import (
    LLMPlanRuntimeLearningFeedback,
)
from agent.llm.plan_validation import LLMPlanValidator
from agent.llm.planning import LLMPlan, LLMPlanStep


def step(index, action, purpose, *, risk=0.1):
    return LLMPlanStep(
        index=index,
        action=action,
        purpose=purpose,
        preconditions=("ready",),
        expected_observation=f"observe {action}",
        success_criteria=f"{action} succeeded",
        risk=risk,
    )


def plan(*steps_, goal="reach target", confidence=0.9):
    return LLMPlan(
        goal=goal,
        rationale="test plan",
        steps=tuple(steps_),
        assumptions=("test assumption",),
        constraints=(),
        stop_conditions=(),
        confidence=confidence,
        provider="test",
        model="test-model",
    )


def feedback(
    step_index=1,
    *,
    success=True,
    progress_delta=0.3,
    change_required=False,
    continue_recommended=True,
    exploration_recommended=False,
    risk_level="low",
):
    return LLMPlanRuntimeLearningFeedback(
        step_index=step_index,
        action="act",
        success=success,
        progress_delta=progress_delta,
        evidence_quality=0.9,
        confidence=0.8,
        diagnosis="test",
        lesson="test lesson",
        next_step="continue",
        change_required=change_required,
        continue_recommended=continue_recommended,
        exploration_recommended=exploration_recommended,
        risk_level=risk_level,
        risk_score=0.2,
        information_gain=max(progress_delta, 0.0),
        reflection=None,
        episode=None,
        runtime_evaluation=None,
    )


def make_replanner(provider, **kwargs):
    return LLMPlanAdaptiveReplanner(
        provider,
        validator=LLMPlanValidator(max_step_risk=1.0),
        **kwargs,
    )


def test_replanner_requires_callable():
    with pytest.raises(LLMPlanAdaptiveReplanningConfigurationError):
        LLMPlanAdaptiveReplanner(None)


def test_negative_max_replans_rejected():
    with pytest.raises(LLMPlanAdaptiveReplanningConfigurationError):
        LLMPlanAdaptiveReplanner(lambda request: request.original_plan, max_replans=-1)


def test_invalid_stagnation_threshold_rejected():
    with pytest.raises(LLMPlanAdaptiveReplanningConfigurationError):
        LLMPlanAdaptiveReplanner(lambda request: request.original_plan, min_progress_delta=1.1)


def test_zero_max_replacement_steps_rejected():
    with pytest.raises(LLMPlanAdaptiveReplanningConfigurationError):
        LLMPlanAdaptiveReplanner(lambda request: request.original_plan, max_replacement_steps=0)


def test_trigger_change_required():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(feedback(change_required=True, progress_delta=0.4))
    assert trigger.should_replan
    assert "change_required" in trigger.reason


def test_trigger_continue_not_recommended():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(
        feedback(continue_recommended=False, progress_delta=0.4)
    )
    assert trigger.should_replan
    assert "continue_not_recommended" in trigger.reason


def test_trigger_stagnation():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(feedback(progress_delta=0.0))
    assert trigger.should_replan
    assert "progress_stagnation" in trigger.reason


def test_trigger_negative_progress():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(feedback(progress_delta=-0.2))
    assert trigger.should_replan


def test_trigger_high_risk():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(feedback(progress_delta=0.5, risk_level="high"))
    assert trigger.should_replan
    assert "high_risk" in trigger.reason


def test_medium_risk_alone_does_not_trigger():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(feedback(progress_delta=0.5, risk_level="medium"))
    assert not trigger.should_replan
    assert trigger.reason == "no_replan_trigger"


def test_positive_progress_no_trigger():
    r = make_replanner(lambda request: request.original_plan)
    trigger = r.evaluate_trigger(feedback(progress_delta=0.5))
    assert not trigger.should_replan


def test_trigger_respects_stagnation_toggle():
    r = make_replanner(lambda request: request.original_plan, replan_on_stagnation=False)
    trigger = r.evaluate_trigger(feedback(progress_delta=0.0))
    assert not trigger.should_replan


def test_trigger_respects_high_risk_toggle():
    r = make_replanner(lambda request: request.original_plan, replan_on_high_risk=False)
    trigger = r.evaluate_trigger(feedback(progress_delta=0.5, risk_level="high"))
    assert not trigger.should_replan


def test_trigger_invalid_feedback_rejected():
    r = make_replanner(lambda request: request.original_plan)
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.evaluate_trigger(None)


def test_non_trigger_does_not_call_provider():
    calls = []
    r = make_replanner(lambda request: calls.append(request))
    original = plan(step(1, "a", "first"), step(2, "b", "second"))
    result = r.replan(original, feedback=feedback(progress_delta=0.4))
    assert not result.replanned
    assert calls == []
    assert result.request is None


def test_preserves_successful_steps_in_request():
    captured = []
    replacement = plan(step(1, "c", "new step"))

    def provider(request):
        captured.append(request)
        return replacement

    r = make_replanner(provider)
    original = plan(
        step(1, "a", "first"),
        step(2, "b", "second"),
        step(3, "c", "third"),
    )
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1, 3),
        goal="reach target",
        world_state={"x": 2},
        failed_step_index=2,
        available_actions={"b": object(), "c": object()},
    )
    request = captured[0]
    assert request.preserved_step_indices == (1, 3)
    assert request.remaining_step_indices == (2,)
    assert request.failed_step_index == 2
    assert result.preserved_step_indices == (1, 3)


def test_request_is_frozen_dataclass():
    original = plan(step(1, "a", "first"))
    fb = feedback(change_required=True, progress_delta=0.0)
    req = LLMPlanReplanningRequest(
        original_plan=original,
        goal=original.goal,
        world_state={},
        feedback=fb,
        preserved_steps=(original.steps[0],),
        remaining_steps=(),
        failed_step_index=None,
        next_step_index=2,
        generation=1,
    )
    with pytest.raises(Exception):
        req.next_step_index = 3


def test_successful_indices_must_belong_to_original_plan():
    r = make_replanner(lambda request: plan(step(2, "x", "new")))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            successful_step_indices=(2,),
        )


def test_bool_successful_index_rejected():
    r = make_replanner(lambda request: plan(step(2, "x", "new")))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            successful_step_indices=(True,),
        )


def test_failed_step_must_belong_to_original_plan():
    r = make_replanner(lambda request: plan(step(2, "x", "new")))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            failed_step_index=9,
        )


def test_provider_must_return_plan():
    r = make_replanner(lambda request: None)
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
        )


def test_provider_empty_plan_rejected():
    r = make_replanner(lambda request: LLMPlan(
        goal="reach target", rationale="none", steps=(), confidence=0.9
    ))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
        )


def test_goal_mismatch_rejected():
    r = make_replanner(lambda request: plan(step(1, "x", "new"), goal="wrong goal"))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningValidationError) as exc:
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
        )
    assert any(issue.code == "goal_mismatch" for issue in exc.value.validation.issues)


def test_default_next_step_index_is_first_remaining_step():
    r = make_replanner(lambda request: plan(step(1, "x", "replacement")))
    original = plan(step(1, "a", "first"), step(2, "b", "second"), step(3, "c", "third"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
    )
    assert result.next_step_index == 2
    assert result.replacement_step_indices == (2,)


def test_replacement_is_rebased_to_next_global_step():
    r = make_replanner(lambda request: plan(
        step(99, "x", "replacement one"),
        step(100, "y", "replacement two"),
    ))
    original = plan(step(1, "a", "first"), step(2, "b", "second"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
        next_step_index=2,
    )
    assert result.replacement_step_indices == (2, 3)


def test_replacement_preserves_order_after_rebase():
    r = make_replanner(lambda request: plan(
        step(30, "second", "two"),
        step(10, "first", "one"),
    ))
    original = plan(step(1, "a", "first"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=4,
    )
    assert [s.action for s in result.replacement_plan.steps] == ["first", "second"]
    assert result.replacement_step_indices == (4, 5)


def test_duplicate_action_names_are_allowed():
    r = make_replanner(lambda request: plan(
        step(1, "move", "move closer"),
        step(2, "move", "move around obstacle"),
    ))
    original = plan(step(1, "look", "inspect"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        available_actions={"move": object()},
    )
    assert result.replacement_step_indices == (1, 2)


def test_exact_completed_step_repetition_is_blocked():
    completed = step(1, "open", "open the box")
    r = make_replanner(lambda request: plan(step(1, "open", "open the box")))
    original = plan(completed, step(2, "take", "take key"))
    with pytest.raises(LLMPlanAdaptiveReplanningSafetyError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            successful_step_indices=(1,),
        )


def test_same_action_different_purpose_is_allowed():
    completed = step(1, "move", "move to room A")
    r = make_replanner(lambda request: plan(step(1, "move", "move to room B")))
    original = plan(completed, step(2, "look", "inspect"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
    )
    assert result.replanned


def test_repetition_safety_can_be_disabled():
    completed = step(1, "open", "open the box")
    r = make_replanner(
        lambda request: plan(step(1, "open", "open the box")),
        prevent_exact_completed_step_repetition=False,
    )
    original = plan(completed, step(2, "take", "take key"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
    )
    assert result.replanned


def test_all_successful_steps_are_preserved_and_remaining_is_empty():
    calls = []
    r = make_replanner(lambda request: calls.append(request) or plan(step(1, "noop", "future")))
    original = plan(step(1, "a", "first"), step(2, "b", "second"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1, 2),
        available_actions={"noop": object()},
    )
    assert calls[0].remaining_steps == ()
    assert result.remaining_original_step_indices == ()


def test_validation_rejects_unknown_action():
    r = make_replanner(lambda request: plan(step(1, "unknown", "new")))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningValidationError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            available_actions={"a": object()},
        )


def test_validation_success_is_returned():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"), step(2, "b", "second"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        available_actions={"b": object()},
    )
    assert result.validation is not None
    assert result.validation.valid is True


def test_max_replacement_steps_limit():
    r = make_replanner(
        lambda request: plan(step(1, "b", "new"), step(2, "c", "new2")),
        max_replacement_steps=1,
    )
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
        )


def test_replan_count_increments_only_on_success():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    assert r.replan_count == 0
    r.replan(original, feedback=feedback(progress_delta=0.5))
    assert r.replan_count == 0
    r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
    )
    assert r.replan_count == 1


def test_replan_limit_stops_triggering():
    r = make_replanner(lambda request: plan(step(1, "b", "new")), max_replans=1)
    original = plan(step(1, "a", "first"))
    first = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
    )
    assert first.replanned
    second = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
    )
    assert not second.replanned
    assert second.trigger.reason.startswith("replan_limit_reached:")


def test_can_replan_property():
    r = make_replanner(lambda request: plan(step(1, "b", "new")), max_replans=1)
    assert r.can_replan
    original = plan(step(1, "a", "first"))
    r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
    )
    assert not r.can_replan


def test_reset_restores_replanning_budget():
    r = make_replanner(lambda request: plan(step(1, "b", "new")), max_replans=1)
    original = plan(step(1, "a", "first"))
    r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
    )
    r.reset()
    assert r.replan_count == 0
    assert r.can_replan
    assert r.last_result is None


def test_metadata_is_carried_to_request_and_result():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        metadata={"trace_id": "abc"},
    )
    assert result.metadata["trace_id"] == "abc"
    assert result.request.metadata["trace_id"] == "abc"


def test_world_state_is_forwarded_unchanged():
    world = {"position": [1, 2], "door": "closed"}
    captured = []
    r = make_replanner(lambda request: captured.append(request) or plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    r.replan(original, feedback=feedback(change_required=True, progress_delta=0.0), world_state=world)
    assert captured[0].world_state is world


def test_generation_can_be_explicit():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        generation=7,
        next_step_index=4,
    )
    assert result.generation == 7
    assert result.request.generation == 7


def test_invalid_generation_rejected():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            generation=0,
        )


def test_invalid_next_step_index_rejected():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    with pytest.raises(LLMPlanAdaptiveReplanningRequestError):
        r.replan(
            original,
            feedback=feedback(change_required=True, progress_delta=0.0),
            next_step_index=0,
        )


def test_original_plan_is_not_mutated():
    original = plan(step(1, "a", "first"), step(2, "b", "second"))
    before = original.to_dict()
    r = make_replanner(lambda request: plan(step(99, "c", "replacement")))
    r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
        next_step_index=2,
    )
    assert original.to_dict() == before


def test_replacement_keeps_goal_and_provider_metadata():
    r = make_replanner(lambda request: LLMPlan(
        goal=request.goal,
        rationale="replacement rationale",
        steps=(step(1, "b", "new"),),
        assumptions=("a",),
        constraints=("c",),
        stop_conditions=("stop",),
        confidence=0.7,
        provider="custom",
        model="custom-v1",
    ))
    original = plan(step(1, "a", "first"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        next_step_index=2,
    )
    rp = result.replacement_plan
    assert rp.goal == original.goal
    assert rp.provider == "custom"
    assert rp.model == "custom-v1"
    assert "preserve_completed_successful_steps" in rp.constraints


def test_preserve_successful_steps_toggle():
    captured = []
    r = make_replanner(
        lambda request: captured.append(request) or plan(step(1, "c", "new")),
        preserve_successful_steps=False,
    )
    original = plan(step(1, "a", "first"), step(2, "b", "second"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
    )
    assert result.preserved_step_indices == ()
    assert captured[0].remaining_step_indices == (1, 2)


def test_result_properties():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
        successful_step_indices=(1,),
        next_step_index=2,
        available_actions={"b": object()},
    )
    assert result.replanned
    assert result.completed_successfully == (1,)
    assert result.replacement_step_indices == (2,)


def test_last_result_and_trigger_are_updated():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    original = plan(step(1, "a", "first"))
    result = r.replan(
        original,
        feedback=feedback(change_required=True, progress_delta=0.0),
    )
    assert r.last_result is result
    assert r.last_trigger is result.trigger


def test_non_llm_plan_input_rejected():
    r = make_replanner(lambda request: plan(step(1, "b", "new")))
    with pytest.raises(TypeError):
        r.replan(SimpleNamespace(goal="x", steps=[]), feedback=feedback(change_required=True, progress_delta=0.0))
