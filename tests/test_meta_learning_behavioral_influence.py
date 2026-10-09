import pytest

from agent.learning.behavioral_evidence import (
    BehavioralEvidenceModel,
)
from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_behavioral_influence import (
    MetaLearningBehavioralInfluence,
)


def build_evidence(
    profile,
):
    return (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )


def test_empty_profile_produces_neutral_influence():
    profile = MetaLearningProfile()

    evidence = build_evidence(
        profile
    )

    influence = (
        MetaLearningBehavioralInfluence().evaluate(
            evidence
        )
    )

    assert influence.applied is False
    assert influence.is_neutral

    assert (
        influence.explore_bonus
        == 0.0
    )

    assert (
        influence.exploit_bonus
        == 0.0
    )


def test_single_mode_evidence_does_not_influence_decision():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

    evidence = build_evidence(
        profile
    )

    influence = (
        MetaLearningBehavioralInfluence().evaluate(
            evidence
        )
    )

    assert influence.applied is False
    assert influence.is_neutral


def test_balanced_exploration_preference_creates_explore_bonus():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

    profile.record_exploitation(
        success=False,
        event_id="exploit-1",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-2",
    )

    profile.record_exploitation(
        success=True,
        event_id="exploit-3",
    )

    evidence = build_evidence(
        profile
    )

    influence = (
        MetaLearningBehavioralInfluence().evaluate(
            evidence
        )
    )

    assert influence.applied is True

    assert (
        influence.preferred_mode
        == "explore"
    )

    assert (
        influence.explore_bonus
        > 0
    )

    assert (
        influence.exploit_bonus
        == 0.0
    )

    assert (
        influence.should_prefer_explore
    )


def test_balanced_exploitation_preference_creates_exploit_bonus():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=False,
        event_id="explore-1",
    )

    profile.record_exploration(
        success=False,
        event_id="explore-2",
    )

    profile.record_exploration(
        success=True,
        event_id="explore-3",
    )

    for index in range(5):
        profile.record_exploitation(
            success=True,
            event_id=f"exploit-{index}",
        )

    evidence = build_evidence(
        profile
    )

    influence = (
        MetaLearningBehavioralInfluence().evaluate(
            evidence
        )
    )

    assert influence.applied is True

    assert (
        influence.preferred_mode
        == "exploit"
    )

    assert (
        influence.exploit_bonus
        > 0
    )

    assert (
        influence.explore_bonus
        == 0.0
    )

    assert (
        influence.should_prefer_exploit
    )


def test_bonus_is_bounded():
    profile = MetaLearningProfile()

    for index in range(100):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

        profile.record_exploitation(
            success=False,
            event_id=f"exploit-{index}",
        )

    evidence = build_evidence(
        profile
    )

    policy = (
        MetaLearningBehavioralInfluence(
            maximum_bonus=0.20,
        )
    )

    influence = policy.evaluate(
        evidence
    )

    assert (
        influence.explore_bonus
        <= 0.20
    )

    assert (
        influence.exploit_bonus
        <= 0.20
    )


def test_near_equal_behavior_is_neutral():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

        profile.record_exploitation(
            success=True,
            event_id=f"exploit-{index}",
        )

    evidence = build_evidence(
        profile
    )

    influence = (
        MetaLearningBehavioralInfluence().evaluate(
            evidence
        )
    )

    assert influence.applied is False
    assert influence.is_neutral


def test_low_confidence_does_not_apply_influence():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="explore",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit",
    )

    evidence = build_evidence(
        profile
    )

    policy = (
        MetaLearningBehavioralInfluence(
            minimum_confidence=0.90,
        )
    )

    influence = policy.evaluate(
        evidence
    )

    assert influence.applied is False
    assert influence.is_neutral


def test_low_strength_does_not_apply_influence():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

        profile.record_exploitation(
            success=True,
            event_id=f"exploit-{index}",
        )

    profile.record_exploration(
        success=False,
        event_id="explore-failure",
    )

    evidence = build_evidence(
        profile
    )

    policy = (
        MetaLearningBehavioralInfluence(
            minimum_strength=0.90,
        )
    )

    influence = policy.evaluate(
        evidence
    )

    assert influence.applied is False


def test_influence_changes_explore_score_when_explore_is_preferred():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

    profile.record_exploitation(
        success=False,
        event_id="exploit-1",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-2",
    )

    profile.record_exploitation(
        success=True,
        event_id="exploit-3",
    )

    evidence = build_evidence(
        profile
    )

    policy = (
        MetaLearningBehavioralInfluence()
    )

    influence = policy.evaluate(
        evidence
    )

    explore, exploit = (
        policy.apply_to_scores(
            explore_score=0.40,
            exploit_score=0.40,
            influence=influence,
        )
    )

    assert explore > 0.40
    assert exploit == 0.40
    assert explore > exploit


def test_influence_changes_exploit_score_when_exploit_is_preferred():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=False,
        event_id="explore-1",
    )

    profile.record_exploration(
        success=False,
        event_id="explore-2",
    )

    profile.record_exploration(
        success=True,
        event_id="explore-3",
    )

    for index in range(5):
        profile.record_exploitation(
            success=True,
            event_id=f"exploit-{index}",
        )

    evidence = build_evidence(
        profile
    )

    policy = (
        MetaLearningBehavioralInfluence()
    )

    influence = policy.evaluate(
        evidence
    )

    explore, exploit = (
        policy.apply_to_scores(
            explore_score=0.40,
            exploit_score=0.40,
            influence=influence,
        )
    )

    assert explore == 0.40
    assert exploit > 0.40
    assert exploit > explore


def test_select_mode_preserves_exploit_on_tie():
    policy = (
        MetaLearningBehavioralInfluence()
    )

    influence = (
        policy.evaluate(
            build_evidence(
                MetaLearningProfile()
            )
        )
    )

    mode = policy.select_mode(
        explore_score=0.50,
        exploit_score=0.50,
        influence=influence,
    )

    assert mode == "exploit"


def test_select_mode_uses_behavioral_influence():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

    profile.record_exploitation(
        success=False,
        event_id="exploit-1",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-2",
    )

    profile.record_exploitation(
        success=True,
        event_id="exploit-3",
    )

    evidence = build_evidence(
        profile
    )

    policy = (
        MetaLearningBehavioralInfluence()
    )

    influence = policy.evaluate(
        evidence
    )

    mode = policy.select_mode(
        explore_score=0.49,
        exploit_score=0.50,
        influence=influence,
    )

    assert mode == "explore"


def test_none_evidence_is_rejected():
    policy = (
        MetaLearningBehavioralInfluence()
    )

    with pytest.raises(
        ValueError,
        match="evidence cannot be None",
    ):
        policy.evaluate(
            None
        )


def test_none_influence_is_rejected():
    policy = (
        MetaLearningBehavioralInfluence()
    )

    with pytest.raises(
        ValueError,
        match="influence cannot be None",
    ):
        policy.apply_to_scores(
            explore_score=0.5,
            exploit_score=0.5,
            influence=None,
        )


def test_invalid_configuration_is_rejected():
    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            minimum_confidence=-0.1,
        )

    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            minimum_confidence=1.1,
        )

    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            minimum_strength=-0.1,
        )

    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            minimum_strength=1.1,
        )

    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            maximum_bonus=-0.1,
        )

    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            maximum_bonus=1.1,
        )

    with pytest.raises(
        ValueError,
    ):
        MetaLearningBehavioralInfluence(
            minimum_comparative_evidence=0,
        )
