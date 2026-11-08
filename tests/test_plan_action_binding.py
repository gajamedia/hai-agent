import pytest

from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.world.environment_capability_discovery import (
    EnvironmentCapabilityCandidate,
    EnvironmentCapabilityDiscoveryReport,
    EnvironmentCapabilityState,
)
from agent.llm.capability_registry import CapabilityTrustLevel
from agent.world.plan_action_binding import (
    PlanActionBinder,
    PlanActionBindingState,
    PlanActionBindingValidationError,
)


def make_plan(actions=("open_gate", "close_gate")):
    return LLMPlan(
        goal="secure the area",
        rationale="Use the available gate controls.",
        steps=tuple(
            LLMPlanStep(
                index=i,
                action=action,
                purpose=f"Perform {action}.",
                expected_observation=f"{action} completed",
                success_criteria=f"{action} completed",
                risk="medium",
            )
            for i, action in enumerate(actions, start=1)
        ),
        assumptions=(),
        constraints=(),
        stop_conditions=(),
        confidence=0.9,
        provider="test",
        model="test-model",
    )


def candidate(action, capability, state=EnvironmentCapabilityState.AVAILABLE):
    return EnvironmentCapabilityCandidate(
        environment_instance_id="sim-1",
        environment_type="demo.sim",
        capability_id=capability,
        action_name=action,
        description=f"Trusted {action}",
        risk_level="medium",
        permissions=("gate:operate",),
        trust_level=CapabilityTrustLevel.APPLICATION,
        registry_version=7,
        state=state,
    )


def discovery(candidates=None):
    return EnvironmentCapabilityDiscoveryReport(
        environment_instance_id="sim-1",
        environment_type="demo.sim",
        registry_version=7,
        candidates=tuple(candidates or (
            candidate("open_gate", "demo.open_gate"),
            candidate("close_gate", "demo.close_gate"),
        )),
        available_capability_ids=tuple(
            item.capability_id for item in (candidates or (
                candidate("open_gate", "demo.open_gate"),
                candidate("close_gate", "demo.close_gate"),
            )) if item.state == EnvironmentCapabilityState.AVAILABLE
        ),
        excluded_capability_ids=(),
    )


def test_binds_each_plan_step_to_matching_available_capability():
    report = PlanActionBinder().bind(make_plan(), discovery())
    assert report.is_complete
    assert report.state == PlanActionBindingState.BOUND
    assert report.capability_ids == ("demo.open_gate", "demo.close_gate")
    assert [step.step_index for step in report.steps] == [1, 2]


def test_binding_preserves_plan_order_and_step_identity():
    plan = make_plan(("close_gate", "open_gate"))
    report = PlanActionBinder().bind(plan, discovery())
    assert [step.action_name for step in report.steps] == ["close_gate", "open_gate"]
    assert [step.capability_id for step in report.steps] == [
        "demo.close_gate", "demo.open_gate"
    ]


def test_missing_capability_fails_closed_by_default():
    plan = make_plan(("open_gate", "delete_database"))
    binder = PlanActionBinder()
    with pytest.raises(PlanActionBindingValidationError, match="delete_database"):
        binder.bind(plan, discovery())
    assert binder.last_report is not None
    assert binder.last_report.unbound_steps == (
        (2, "delete_database", "capability_not_available"),
    )


def test_partial_binding_can_be_inspected_without_execution():
    plan = make_plan(("open_gate", "unknown_action"))
    report = PlanActionBinder().bind(plan, discovery(), require_complete=False)
    assert report.state == PlanActionBindingState.UNBOUND
    assert report.capability_ids == ("demo.open_gate",)
    assert report.unbound_steps[0][1] == "unknown_action"


def test_disabled_or_revoked_candidate_is_not_bindable():
    report = discovery((
        candidate("open_gate", "demo.open_gate", EnvironmentCapabilityState.INACTIVE),
    ))
    with pytest.raises(PlanActionBindingValidationError):
        PlanActionBinder().bind(make_plan(("open_gate",)), report)


def test_missing_candidate_is_not_bindable():
    report = discovery((
        candidate("open_gate", "demo.open_gate", EnvironmentCapabilityState.MISSING),
    ))
    with pytest.raises(PlanActionBindingValidationError):
        PlanActionBinder().bind(make_plan(("open_gate",)), report)


def test_ambiguous_action_name_fails_closed():
    report = discovery((
        candidate("open_gate", "demo.open_gate"),
        candidate("open_gate", "demo.open_gate_alt"),
    ))
    with pytest.raises(PlanActionBindingValidationError, match="ambiguous"):
        PlanActionBinder().bind(make_plan(("open_gate",)), report)


def test_candidate_environment_mismatch_rejected():
    item = candidate("open_gate", "demo.open_gate")
    object.__setattr__(item, "environment_instance_id", "other-instance")
    report = discovery((item,))
    with pytest.raises(PlanActionBindingValidationError, match="identity/version"):
        PlanActionBinder().bind(make_plan(("open_gate",)), report)


def test_candidate_registry_version_mismatch_rejected():
    item = candidate("open_gate", "demo.open_gate")
    object.__setattr__(item, "registry_version", 6)
    report = discovery((item,))
    with pytest.raises(PlanActionBindingValidationError, match="identity/version"):
        PlanActionBinder().bind(make_plan(("open_gate",)), report)


def test_invalid_plan_type_rejected():
    with pytest.raises(PlanActionBindingValidationError):
        PlanActionBinder().bind(object(), discovery())


def test_duplicate_plan_step_indices_rejected():
    plan = make_plan(("open_gate", "close_gate"))
    object.__setattr__(plan.steps[1], "index", 1)
    with pytest.raises(PlanActionBindingValidationError, match="Duplicate"):
        PlanActionBinder().bind(plan, discovery())


def test_does_not_execute_or_mutate_plan():
    plan = make_plan()
    original_actions = tuple(step.action for step in plan.steps)
    binder = PlanActionBinder()
    report = binder.bind(plan, discovery())
    assert tuple(step.action for step in plan.steps) == original_actions
    assert not any(hasattr(step, "handler") for step in report.steps)
    assert not hasattr(report.steps[0], "action_object")
