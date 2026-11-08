from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from agent.llm.agent_runtime import AgentCognitiveRuntimeTrace
from agent.llm.planning import LLMPlan, LLMPlanStep
from agent.llm.cognitive_pass_handoff import (
    CognitivePassHandoff,
    CognitivePassHandoffSerializationError,
    CognitivePassHandoffValidationError,
    DEFAULT_HANDOFF_TTL_SECONDS,
    HANDOFF_CONTRACT_VERSION,
    create_cognitive_pass_handoff,
    validate_cognitive_pass_handoff,
)


FIXED_NOW = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)


def make_trace(*, observation=None, cycle=12, goal="reach the door"):
    plan = LLMPlan(
        goal=goal,
        rationale="The door is already unlocked.",
        steps=(
            LLMPlanStep(
                index=1,
                action="open_door",
                purpose="Open the door.",
                preconditions=("door is unlocked",),
                expected_observation="door is open",
                success_criteria="door state is open",
                risk=None,
            ),
        ),
        assumptions=(),
        constraints=("Do not force the door.",),
        stop_conditions=("Stop if the door is blocked.",),
        confidence=0.9,
        provider="test-provider",
        model="test-model",
    )
    result = SimpleNamespace(plan=plan)
    return AgentCognitiveRuntimeTrace(
        cycle=cycle,
        observation={"door": "unlocked"} if observation is None else observation,
        context=None,
        result=result,
    )


def test_create_handoff_from_cognitive_trace():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    data = handoff.to_dict()

    assert isinstance(handoff, CognitivePassHandoff)
    assert data["contract_version"] == HANDOFF_CONTRACT_VERSION
    assert data["cognitive_cycle"] == 12
    assert data["goal"] == "reach the door"
    assert data["plan"]["goal"] == data["goal"]
    assert len(data["observation_fingerprint"]) == 64
    assert len(data["plan_fingerprint"]) == 64
    assert datetime.fromisoformat(data["expires_at"]) - datetime.fromisoformat(data["created_at"]) == timedelta(seconds=DEFAULT_HANDOFF_TTL_SECONDS)


def test_create_handoff_supports_missing_cognitive_cycle():
    handoff = create_cognitive_pass_handoff(make_trace(cycle=None), now=FIXED_NOW)
    assert handoff.cognitive_cycle is None


def test_plan_property_and_to_dict_return_independent_values():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    first = handoff.plan
    first["steps"][0]["action"] = "tampered"
    second = handoff.to_dict()
    assert second["plan"]["steps"][0]["action"] == "open_door"


def test_round_trip_from_dict_preserves_contract():
    original = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    restored = CognitivePassHandoff.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_validation_accepts_matching_goal_and_observation():
    trace = make_trace()
    handoff = create_cognitive_pass_handoff(trace, now=FIXED_NOW)
    validated = validate_cognitive_pass_handoff(
        handoff,
        expected_goal="reach the door",
        expected_observation={"door": "unlocked"},
        now=FIXED_NOW + timedelta(seconds=1),
    )
    assert validated.handoff_id == handoff.handoff_id


def test_validation_rejects_mismatched_execution_goal():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    with pytest.raises(CognitivePassHandoffValidationError, match="execution goal"):
        validate_cognitive_pass_handoff(
            handoff,
            expected_goal="leave the room",
            now=FIXED_NOW + timedelta(seconds=1),
        )


def test_validation_rejects_mismatched_observation():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    with pytest.raises(CognitivePassHandoffValidationError, match="observation"):
        validate_cognitive_pass_handoff(
            handoff,
            expected_observation={"door": "closed"},
            now=FIXED_NOW + timedelta(seconds=1),
        )


def test_validation_rejects_expired_handoff():
    handoff = create_cognitive_pass_handoff(make_trace(), ttl_seconds=5, now=FIXED_NOW)
    with pytest.raises(CognitivePassHandoffValidationError, match="expired"):
        validate_cognitive_pass_handoff(handoff, now=FIXED_NOW + timedelta(seconds=5))


def test_validation_rejects_handoff_from_future():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    with pytest.raises(CognitivePassHandoffValidationError, match="not yet valid"):
        validate_cognitive_pass_handoff(handoff, now=FIXED_NOW - timedelta(seconds=1))


def test_from_dict_rejects_tampered_plan_fingerprint():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    payload = handoff.to_dict()
    payload["plan"]["steps"][0]["action"] = "delete_everything"
    with pytest.raises(CognitivePassHandoffValidationError, match="plan_fingerprint"):
        CognitivePassHandoff.from_dict(payload)


def test_from_dict_rejects_goal_plan_mismatch_even_with_recomputed_hash():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    payload = handoff.to_dict()
    payload["goal"] = "a different goal"
    # Recreate a self-consistent plan digest to exercise the independent goal check.
    import hashlib
    import json

    encoded = json.dumps(payload["plan"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["plan_fingerprint"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    with pytest.raises(CognitivePassHandoffValidationError, match="goal does not match"):
        CognitivePassHandoff.from_dict(payload)


def test_create_rejects_non_json_observation():
    with pytest.raises(CognitivePassHandoffSerializationError, match="observation"):
        create_cognitive_pass_handoff(make_trace(observation=object()), now=FIXED_NOW)


def test_create_rejects_invalid_ttl():
    with pytest.raises(ValueError, match="ttl_seconds"):
        create_cognitive_pass_handoff(make_trace(), ttl_seconds=0, now=FIXED_NOW)


def test_create_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        create_cognitive_pass_handoff(make_trace(), now=datetime(2026, 11, 1, 10, 0))


def test_from_dict_rejects_unsupported_contract_version():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    payload = handoff.to_dict()
    payload["contract_version"] = 2
    with pytest.raises(CognitivePassHandoffValidationError, match="contract_version"):
        CognitivePassHandoff.from_dict(payload)


def test_from_dict_rejects_lifetime_above_maximum():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    payload = handoff.to_dict()
    payload["expires_at"] = (
        FIXED_NOW + timedelta(seconds=3601)
    ).isoformat()
    with pytest.raises(CognitivePassHandoffValidationError, match="lifetime"):
        CognitivePassHandoff.from_dict(payload)


def test_validate_rechecks_direct_object_contract_fields():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    tampered = replace(handoff, observation_fingerprint="not-a-sha256-digest")
    with pytest.raises(CognitivePassHandoffValidationError, match="observation_fingerprint"):
        validate_cognitive_pass_handoff(tampered, now=FIXED_NOW + timedelta(seconds=1))


def test_validate_rechecks_direct_object_plan_integrity():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    payload = handoff.to_dict()
    payload["plan"]["steps"][0]["action"] = "tampered_action"
    from agent.llm.cognitive_pass_handoff import _canonical_json

    tampered_plan_json = _canonical_json(payload["plan"], field_name="plan")
    tampered = replace(handoff, _plan_json=tampered_plan_json)
    with pytest.raises(CognitivePassHandoffValidationError, match="plan_fingerprint"):
        validate_cognitive_pass_handoff(tampered, now=FIXED_NOW + timedelta(seconds=1))


def test_validate_rejects_direct_object_with_excessive_lifetime():
    handoff = create_cognitive_pass_handoff(make_trace(), now=FIXED_NOW)
    excessive_expiry = (FIXED_NOW + timedelta(seconds=3601)).isoformat()
    tampered = replace(handoff, expires_at=excessive_expiry)
    with pytest.raises(CognitivePassHandoffValidationError, match="lifetime"):
        validate_cognitive_pass_handoff(tampered, now=FIXED_NOW + timedelta(seconds=1))
