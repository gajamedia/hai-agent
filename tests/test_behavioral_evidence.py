import pytest

from agent.learning.behavioral_evidence import (
    BehavioralEvidenceModel,
    build_behavioral_evidence,
)
from agent.learning.meta_learning import (
    MetaLearningProfile,
)


def test_empty_profile_has_no_behavioral_evidence():
    profile = MetaLearningProfile()

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert not evidence.has_evidence

    assert (
        evidence.preferred_mode
        is None
    )

    assert (
        evidence.total_attempts
        == 0
    )

    assert (
        evidence.total_successes
        == 0
    )

    assert (
        evidence.total_failures
        == 0
    )


def test_exploration_evidence_is_measured():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="explore-1",
    )

    profile.record_exploration(
        success=True,
        event_id="explore-2",
    )

    profile.record_exploration(
        success=False,
        event_id="explore-3",
    )

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    exploration = (
        evidence.exploration
    )

    assert (
        exploration.mode
        == "explore"
    )

    assert (
        exploration.attempts
        == 3
    )

    assert (
        exploration.successes
        == 2
    )

    assert (
        exploration.failures
        == 1
    )

    assert (
        exploration.success_rate
        == pytest.approx(
            2 / 3,
            rel=1e-6,
        )
    )


def test_exploitation_evidence_is_measured():
    profile = MetaLearningProfile()

    profile.record_exploitation(
        success=True,
        event_id="exploit-1",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-2",
    )

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    exploitation = (
        evidence.exploitation
    )

    assert (
        exploitation.mode
        == "exploit"
    )

    assert (
        exploitation.attempts
        == 2
    )

    assert (
        exploitation.successes
        == 1
    )

    assert (
        exploitation.failures
        == 1
    )


def test_preference_is_explore_when_exploration_is_stronger():
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

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert (
        evidence.preferred_mode
        == "explore"
    )

    assert (
        evidence.preference_strength
        > 0
    )


def test_preference_is_exploit_when_exploitation_is_stronger():
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

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert (
        evidence.preferred_mode
        == "exploit"
    )


def test_nearly_equal_modes_have_no_preference():
    profile = MetaLearningProfile()

    for index in range(4):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

        profile.record_exploitation(
            success=True,
            event_id=f"exploit-{index}",
        )

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert (
        evidence.preferred_mode
        is None
    )

    assert (
        evidence.preference_strength
        == 0.0
    )


def test_evidence_strength_increases_with_more_attempts():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="one",
    )

    evidence_one = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    for index in range(1, 10):
        profile.record_exploration(
            success=True,
            event_id=f"event-{index}",
        )

    evidence_many = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert (
        evidence_many.exploration
        .evidence_strength
        >
        evidence_one.exploration
        .evidence_strength
    )


def test_reliability_is_bounded():
    profile = MetaLearningProfile()

    for index in range(20):
        profile.record_exploration(
            success=True,
            event_id=f"success-{index}",
        )

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert 0.0 <= (
        evidence.exploration.reliability
    ) <= 1.0


def test_mixed_behavior_produces_adaptation_signal():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="explore-success",
    )

    profile.record_exploration(
        success=False,
        event_id="explore-failure",
    )

    profile.record_exploitation(
        success=True,
        event_id="exploit-success",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-failure",
    )

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert (
        evidence.adaptation_signal
        > 0
    )


def test_single_mode_does_not_create_adaptation_signal():
    profile = MetaLearningProfile()

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

    evidence = (
        BehavioralEvidenceModel().analyze(
            profile
        )
    )

    assert (
        evidence.adaptation_signal
        == 0.0
    )


def test_snapshot_is_serializable_and_complete():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True,
        event_id="explore-1",
    )

    profile.record_exploitation(
        success=False,
        event_id="exploit-1",
    )

    evidence = (
        build_behavioral_evidence(
            profile
        )
    )

    snapshot = evidence.snapshot()

    assert isinstance(
        snapshot,
        dict,
    )

    assert (
        "exploration"
        in snapshot
    )

    assert (
        "exploitation"
        in snapshot
    )

    assert (
        snapshot["exploration"]["attempts"]
        == 1
    )

    assert (
        snapshot["exploitation"]["attempts"]
        == 1
    )

    assert (
        snapshot["total_attempts"]
        == 2
    )


def test_invalid_configuration_is_rejected():
    with pytest.raises(
        ValueError,
    ):
        BehavioralEvidenceModel(
            minimum_evidence=0,
        )

    with pytest.raises(
        ValueError,
    ):
        BehavioralEvidenceModel(
            preference_threshold=-0.1,
        )

    with pytest.raises(
        ValueError,
    ):
        BehavioralEvidenceModel(
            preference_threshold=1.1,
        )


def test_none_profile_is_rejected():
    model = (
        BehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        model.analyze(
            None
        )


def test_invalid_mode_evidence_is_rejected():
    model = (
        BehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="mode must be",
    ):
        model._build_mode_evidence(
            mode="invalid",
            attempts=1,
            successes=1,
            failures=0,
        )


def test_inconsistent_outcome_counts_are_rejected():
    model = (
        BehavioralEvidenceModel()
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed attempts",
    ):
        model._build_mode_evidence(
            mode="explore",
            attempts=2,
            successes=2,
            failures=1,
        )
