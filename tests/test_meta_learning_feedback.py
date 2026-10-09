import pytest

from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_feedback import (
    MetaLearningFeedback,
    MetaLearningFeedbackEngine,
)


def test_feedback_is_neutral_without_enough_evidence():
    profile = MetaLearningProfile()

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert isinstance(
        feedback,
        MetaLearningFeedback,
    )

    assert feedback.preferred_mode is None
    assert feedback.is_neutral
    assert feedback.exploration_pressure == 0.0
    assert feedback.exploitation_pressure == 0.0
    assert feedback.confidence == 0.0


def test_feedback_detects_strong_exploitation():
    profile = MetaLearningProfile()

    for _ in range(9):
        profile.record_exploitation(
            success=True
        )

    for _ in range(3):
        profile.record_exploration(
            success=False
        )

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert feedback.preferred_mode == "exploit"
    assert feedback.should_exploit
    assert not feedback.should_explore
    assert feedback.exploitation_pressure > 0.0
    assert feedback.confidence > 0.0


def test_feedback_detects_strong_exploration():
    profile = MetaLearningProfile()

    for _ in range(9):
        profile.record_exploration(
            success=True
        )

    for _ in range(3):
        profile.record_exploitation(
            success=False
        )

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert feedback.preferred_mode == "explore"
    assert feedback.should_explore
    assert not feedback.should_exploit
    assert feedback.exploration_pressure > 0.0


def test_feedback_detects_weak_exploration():
    profile = MetaLearningProfile()

    for _ in range(3):
        profile.record_exploration(
            success=False
        )

    for _ in range(3):
        profile.record_exploitation(
            success=True
        )

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert (
        "exploration performance is weak"
        in feedback.recommendations
    )


def test_feedback_detects_weak_exploitation():
    profile = MetaLearningProfile()

    for _ in range(3):
        profile.record_exploration(
            success=True
        )

    for _ in range(3):
        profile.record_exploitation(
            success=False
        )

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert (
        "exploitation performance is weak"
        in feedback.recommendations
    )


def test_feedback_detects_adaptation_pressure():
    profile = MetaLearningProfile()

    profile.record_exploration(
        success=True
    )
    profile.record_exploitation(
        success=True
    )
    profile.record_evolution(
        success=False
    )
    profile.record_recovery(
        success=False
    )

    profile.record_evolution(
        success=False
    )

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert feedback.requires_adaptation
    assert feedback.adaptation_pressure > 0.0


def test_successful_adaptation_has_no_adaptation_pressure():
    profile = MetaLearningProfile()

    profile.record_evolution(
        success=True
    )
    profile.record_evolution(
        success=True
    )
    profile.record_recovery(
        success=True
    )

    profile.record_exploration(
        success=True
    )
    profile.record_exploitation(
        success=True
    )
    profile.record_exploration(
        success=True
    )

    engine = MetaLearningFeedbackEngine()

    feedback = engine.analyze(
        profile
    )

    assert feedback.adaptation_pressure == 0.0


def test_feedback_confidence_increases_with_decisions():
    profile = MetaLearningProfile()

    engine = MetaLearningFeedbackEngine()

    profile.record_exploitation(
        success=True
    )
    profile.record_exploitation(
        success=True
    )
    profile.record_exploitation(
        success=True
    )

    feedback_low = engine.analyze(
        profile
    )

    for _ in range(20):
        profile.record_exploitation(
            success=True
        )

    feedback_high = engine.analyze(
        profile
    )

    assert (
        feedback_high.confidence
        > feedback_low.confidence
    )


def test_feedback_pressure_is_bounded():
    profile = MetaLearningProfile()

    for _ in range(100):
        profile.record_exploration(
            success=True
        )

    for _ in range(100):
        profile.record_exploitation(
            success=False
        )

    engine = MetaLearningFeedbackEngine(
        max_pressure=1.0
    )

    feedback = engine.analyze(
        profile
    )

    assert 0.0 <= (
        feedback.exploration_pressure
    ) <= 1.0

    assert 0.0 <= (
        feedback.exploitation_pressure
    ) <= 1.0

    assert 0.0 <= (
        feedback.adaptation_pressure
    ) <= 1.0


def test_feedback_rejects_none_profile():
    engine = MetaLearningFeedbackEngine()

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        engine.analyze(None)


def test_feedback_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            min_decisions=-1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            min_adaptation_events=-1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            exploration_advantage=-1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            exploitation_advantage=-1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            weak_success_rate=1.1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            strong_success_rate=-0.1
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            weak_success_rate=0.8,
            strong_success_rate=0.7,
        )

    with pytest.raises(ValueError):
        MetaLearningFeedbackEngine(
            max_pressure=0.0
        )


def test_feedback_helpers():
    explore = MetaLearningFeedback(
        preferred_mode="explore"
    )

    assert explore.should_explore
    assert not explore.should_exploit
    assert not explore.is_neutral

    exploit = MetaLearningFeedback(
        preferred_mode="exploit"
    )

    assert exploit.should_exploit
    assert not exploit.should_explore
    assert not exploit.is_neutral

    neutral = MetaLearningFeedback()

    assert neutral.is_neutral
    assert not neutral.should_explore
    assert not neutral.should_exploit
