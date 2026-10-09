import pytest

from agent.llm.goal_management import GoalLifecycleManager, GoalStatus
from agent.llm.goal_conflict_resolution import (
    GoalConflictCandidate,
    GoalConflictPolicy,
    GoalConflictRelation,
    GoalConflictResolutionApplier,
    GoalConflictResolutionApplicationError,
    GoalConflictResolutionConfigurationError,
    GoalConflictResolutionDecision,
    GoalConflictResolutionInputError,
    GoalConflictResolver,
)


def g(i, p=0.5, progress=0.0, status=GoalStatus.PENDING, seq=1):
    return {
        "goal_id": i,
        "description": i,
        "priority": p,
        "progress": progress,
        "status": status,
        "sequence": seq,
    }


def rel(a, b, severity=1.0, reason="conflict"):
    return GoalConflictRelation(a, b, reason=reason, severity=severity)


def test_default_policy_normalizes_weights():
    p = GoalConflictPolicy()
    assert sum((p.priority_weight, p.arbitration_weight, p.urgency_weight, p.progress_weight)) == pytest.approx(1.0)


def test_policy_rejects_all_zero():
    with pytest.raises(GoalConflictResolutionConfigurationError):
        GoalConflictPolicy(0, 0, 0, 0)


def test_policy_rejects_negative():
    with pytest.raises(GoalConflictResolutionConfigurationError):
        GoalConflictPolicy(priority_weight=-1)


def test_relation_normalizes_ids():
    x = GoalConflictRelation(" a ", " b ", reason="  x ", source=" y ")
    assert x.goal_id == "a"
    assert x.conflicting_goal_id == "b"
    assert x.pair_key == ("a", "b")


def test_relation_rejects_self_conflict():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictRelation("a", "a")


def test_relation_rejects_bad_severity():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictRelation("a", "b", severity=2)


def test_no_relations_returns_empty_result():
    result = GoalConflictResolver().resolve([g("a"), g("b")])
    assert result.conflict_count == 0
    assert result.resolved is True
    assert result.decisions == ()


def test_single_pair_creates_component():
    result = GoalConflictResolver().resolve([g("a", seq=1), g("b", seq=2)], relations=[rel("a", "b")])
    assert result.conflict_count == 1
    assert result.decisions[0].goal_ids == ("a", "b")


def test_unknown_goal_relation_rejected():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictResolver().resolve([g("a")], relations=[rel("a", "b")])


def test_mapping_relation_supported():
    result = GoalConflictResolver().resolve([g("a"), g("b", seq=2)], relations=[{"goal_id":"a", "conflicting_goal_id":"b"}])
    assert result.has_conflicts


def test_terminal_goal_ineligible():
    result = GoalConflictResolver().resolve(
        [g("a", p=.9, status=GoalStatus.COMPLETED), g("b", p=.1, seq=2)],
        relations=[rel("a", "b")],
    )
    d = result.decisions[0]
    assert d.winner_goal_id == "b"
    assert d.deferred_goal_ids == ("a",)


def test_blocked_goal_ineligible():
    result = GoalConflictResolver().resolve(
        [g("a", p=.9), g("b", p=.1, seq=2)],
        relations=[rel("a", "b")],
        blocked_goal_ids=["a"],
    )
    assert result.decisions[0].winner_goal_id == "b"
    assert result.blocked_goal_ids == ("a",)


def test_all_blocked_has_no_winner_and_is_resolved():
    result = GoalConflictResolver().resolve(
        [g("a"), g("b", seq=2)], relations=[rel("a", "b")], blocked_goal_ids=["a", "b"]
    )
    d = result.decisions[0]
    assert d.winner_goal_id is None
    assert d.resolved is False


def test_priority_can_break_conflict():
    result = GoalConflictResolver().resolve(
        [g("a", p=.9), g("b", p=.2, seq=2)], relations=[rel("a", "b")]
    )
    assert result.winner_goal_ids == ("a",)
    assert result.deferred_goal_ids == ("b",)


def test_progress_can_break_equal_priority():
    result = GoalConflictResolver().resolve(
        [g("a", p=.5, progress=.8), g("b", p=.5, progress=.2, seq=2)], relations=[rel("a", "b")]
    )
    assert result.winner_goal_ids == ("a",)


def test_urgency_can_influence_winner():
    result = GoalConflictResolver().resolve(
        [g("a", p=.5), g("b", p=.5, seq=2)], relations=[rel("a", "b")], urgency={"a":.0, "b":1.0}
    )
    assert result.winner_goal_ids == ("b",)


def test_arbitration_score_can_influence_winner():
    candidates = (
        GoalConflictCandidate("a", .5, 0, 0, .1, .1, True, GoalStatus.PENDING, 1),
        GoalConflictCandidate("b", .5, 0, 0, .9, .9, True, GoalStatus.PENDING, 2),
    )
    arb = type("Arb", (), {"candidates": candidates})()
    result = GoalConflictResolver().resolve(
        [g("a", p=.5, seq=1), g("b", p=.5, seq=2)], relations=[rel("a", "b")], arbitration=arb
    )
    assert result.winner_goal_ids == ("b",)


def test_two_components_produce_two_decisions():
    goals = [g("a", seq=1), g("b", seq=2), g("c", seq=3), g("d", seq=4)]
    result = GoalConflictResolver().resolve(goals, relations=[rel("a","b"), rel("c","d")])
    assert result.conflict_count == 2


def test_transitive_relations_form_one_component():
    goals = [g("a", seq=1), g("b", seq=2), g("c", seq=3)]
    result = GoalConflictResolver().resolve(goals, relations=[rel("a","b"), rel("b","c")])
    assert result.conflict_count == 1
    assert result.decisions[0].goal_ids == ("a", "b", "c")


def test_duplicate_relation_deduplicates():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b"), rel("b","a")])
    assert result.metadata["relation_count"] == 1


def test_severity_is_max_component_severity():
    result = GoalConflictResolver().resolve(
        [g("a"), g("b"), g("c")],
        relations=[rel("a","b",.2), rel("b","c",.9)],
    )
    assert result.decisions[0].conflict_severity == .9


def test_candidate_metadata_contains_relation_count():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")])
    ca = next(x for x in result.decisions[0].candidates if x.goal_id == "a")
    assert ca.metadata["relation_count"] == 1


def test_mapping_goal_supported():
    result = GoalConflictResolver().resolve(
        [g("a", p="0.8"), g("b", p=.2, seq=2)], relations=[rel("a","b")]
    )
    assert result.winner_goal_ids == ("a",)


def test_invalid_goal_input_rejected():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictResolver().resolve([object()], relations=[])


def test_invalid_urgency_rejected():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")], urgency={"a": 2})


def test_string_urgency_normalized():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")], urgency={"a":"1"})
    assert next(x for x in result.decisions[0].candidates if x.goal_id == "a").urgency == 1.0


def test_preblocked_unknown_id_is_ignored_from_result():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")], blocked_goal_ids=["x"])
    assert result.blocked_goal_ids == ()


def test_default_arbitration_score_is_zero():
    result = GoalConflictResolver().resolve([g("a"), g("b", seq=2)], relations=[rel("a","b")])
    assert all(c.arbitration_score == 0 for c in result.decisions[0].candidates)


def test_deferred_ids_sorted():
    result = GoalConflictResolver().resolve(
        [g("c", p=.5, seq=3), g("a", p=.8, seq=1), g("b", p=.4, seq=2)],
        relations=[rel("c","a"), rel("a","b")],
    )
    assert result.deferred_goal_ids == ("c", "b")


def test_winner_ids_deduplicated_across_components():
    result = GoalConflictResolver().resolve(
        [g("a", seq=1), g("b", seq=2), g("c", seq=3)],
        relations=[rel("a","b"), rel("a","c")],
    )
    assert result.winner_goal_ids == ("a",)


def test_decision_properties():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")])
    d = result.decisions[0]
    assert d.has_winner
    assert d.should_defer


def test_result_properties():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")])
    assert result.has_conflicts
    assert result.resolved


def test_manager_resolution_uses_snapshot():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A", priority=.9)
    manager.add_goal("b", "B", priority=.1)
    result = GoalConflictResolver().resolve_manager(manager, relations=[rel("a","b")])
    assert result.winner_goal_ids == ("a",)


def test_manager_type_validation():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictResolver().resolve_manager(object(), relations=[])


def test_apply_does_not_mutate_by_default():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A")
    manager.add_goal("b", "B")
    decision = GoalConflictResolver().resolve_manager(manager, relations=[rel("a","b")]).decisions[0]
    out = GoalConflictResolutionApplier().apply(manager, decision)
    assert out is None
    assert manager.current_goal_id is None


def test_apply_can_set_current_winner():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A")
    manager.add_goal("b", "B", priority=.1)
    decision = GoalConflictResolver().resolve_manager(manager, relations=[rel("a","b")]).decisions[0]
    out = GoalConflictResolutionApplier().apply(manager, decision, set_current=True)
    assert out.goal_id == "a"
    assert manager.current_goal_id == "a"


def test_apply_rejects_wrong_manager():
    decision = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")]).decisions[0]
    with pytest.raises(GoalConflictResolutionApplicationError):
        GoalConflictResolutionApplier().apply(object(), decision)


def test_apply_rejects_wrong_decision():
    manager = GoalLifecycleManager()
    with pytest.raises(GoalConflictResolutionApplicationError):
        GoalConflictResolutionApplier().apply(manager, object())


def test_apply_rejects_unresolved():
    manager = GoalLifecycleManager()
    d = GoalConflictResolutionDecision("x", ("a",), None, (), (), False, 1.0, "x", (), (), {})
    with pytest.raises(GoalConflictResolutionApplicationError):
        GoalConflictResolutionApplier().apply(manager, d, set_current=True)


def test_apply_rejects_blocked_winner():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A")
    d = GoalConflictResolutionDecision("x", ("a",), "a", (), ("a",), True, 1.0, "x", (), (), {})
    with pytest.raises(GoalConflictResolutionApplicationError):
        GoalConflictResolutionApplier().apply(manager, d, set_current=True)


def test_apply_none_winner_returns_none():
    manager = GoalLifecycleManager()
    d = GoalConflictResolutionDecision("x", ("a",), None, (), (), True, 1.0, "x", (), (), {})
    assert GoalConflictResolutionApplier().apply(manager, d, set_current=True) is None


def test_order_is_deterministic_for_equal_candidates():
    a = [g("a", p=.5, seq=1), g("b", p=.5, seq=2)]
    r1 = GoalConflictResolver().resolve(a, relations=[rel("a","b")])
    r2 = GoalConflictResolver().resolve(reversed(a), relations=[rel("a","b")])
    assert r1.winner_goal_ids == r2.winner_goal_ids == ("a",)


def test_negative_arbitration_scores_are_clamped():
    candidate = GoalConflictCandidate("a", .1, 0, 0, -2, -2, True, GoalStatus.PENDING, 1)
    arb = type("Arb", (), {"candidates": (candidate,)})()
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")], arbitration=arb)
    ca = next(c for c in result.decisions[0].candidates if c.goal_id == "a")
    assert ca.arbitration_score == 0.0


def test_high_arbitration_score_clamped():
    candidate = GoalConflictCandidate("a", .1, 0, 0, 2, 2, True, GoalStatus.PENDING, 1)
    arb = type("Arb", (), {"candidates": (candidate,)})()
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")], arbitration=arb)
    ca = next(c for c in result.decisions[0].candidates if c.goal_id == "a")
    assert ca.arbitration_score == 1.0


def test_relation_metadata_copied():
    r = GoalConflictRelation("a", "b", metadata={"source_id": 3})
    assert r.metadata["source_id"] == 3


def test_candidate_reasons_terminal():
    result = GoalConflictResolver().resolve(
        [g("a", status=GoalStatus.FAILED), g("b")], relations=[rel("a","b")]
    )
    a = next(c for c in result.decisions[0].candidates if c.goal_id == "a")
    assert a.reasons == ("terminal_goal",)


def test_blocked_goal_reason():
    result = GoalConflictResolver().resolve(
        [g("a"), g("b")], relations=[rel("a","b")], blocked_goal_ids=["b"]
    )
    b = next(c for c in result.decisions[0].candidates if c.goal_id == "b")
    assert b.reasons == ("blocked_goal",)


def test_reason_is_conflict_resolution_score():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")])
    assert result.decisions[0].reason == "higher_conflict_resolution_score"


def test_component_does_not_include_isolated_goal():
    result = GoalConflictResolver().resolve([g("a"), g("b"), g("c")], relations=[rel("a","b")])
    assert "c" not in result.decisions[0].goal_ids


def test_source_and_reason_preserved():
    result = GoalConflictResolver().resolve(
        [g("a"), g("b")], relations=[GoalConflictRelation("a","b", reason="resource", source="planner")]
    )
    d = result.decisions[0]
    assert d.relations[0].reason == "resource"
    assert d.relations[0].source == "planner"


def test_multiple_winners_are_allowed_across_components():
    result = GoalConflictResolver().resolve(
        [g("a", seq=1), g("b", seq=2), g("c", seq=3), g("d", seq=4)],
        relations=[rel("a","b"), rel("c","d")],
    )
    assert set(result.winner_goal_ids) == {"a", "c"}


def test_manager_apply_pending_goal_activates_it():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A")
    manager.add_goal("b", "B", priority=.1)
    d = GoalConflictResolver().resolve_manager(manager, relations=[rel("a","b")]).decisions[0]
    GoalConflictResolutionApplier().apply(manager, d, set_current=True)
    assert manager.get_goal("a").status == GoalStatus.ACTIVE


def test_apply_without_set_current_preserves_existing_current():
    manager = GoalLifecycleManager()
    manager.add_goal("a", "A")
    manager.add_goal("b", "B", priority=.9)
    manager.set_current_goal("a")
    d = GoalConflictResolver().resolve_manager(manager, relations=[rel("a","b")]).decisions[0]
    GoalConflictResolutionApplier().apply(manager, d)
    assert manager.current_goal_id == "a"


def test_resolution_result_metadata_relation_count():
    result = GoalConflictResolver().resolve([g("a"), g("b")], relations=[rel("a","b")])
    assert result.metadata["relation_count"] == 1


def test_empty_goal_input_with_relation_is_unknown():
    with pytest.raises(GoalConflictResolutionInputError):
        GoalConflictResolver().resolve([], relations=[rel("a","b")])
