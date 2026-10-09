from dataclasses import dataclass

from agent.learning.metacognition import (
    MetacognitionEngine,
)


engine = MetacognitionEngine()


def test_success_with_progress_is_preserved():
    reflection = engine.reflect_on_episode(
        {
            "episode_id": "ep-1",
            "goal": "finish task",
            "action": "execute_task",
            "result": {"success": True},
        },
        progress_before=0.2,
        progress_after=0.8,
    )

    assert reflection.success is True
    assert reflection.progress_delta == 0.6
    assert reflection.diagnosis == "successful_outcome_with_progress"
    assert reflection.change_required is False
    assert reflection.confidence == 1.0


def test_failure_without_progress_requires_change():
    reflection = engine.reflect_on_episode(
        {
            "episode_id": "ep-2",
            "result": {"success": False},
        },
        progress_before=0.4,
        progress_after=0.4,
    )

    assert reflection.diagnosis == "failure_without_measured_progress"
    assert reflection.progress_delta == 0.0
    assert reflection.change_required is True


def test_failure_with_partial_progress_preserves_useful_component():
    reflection = engine.reflect_on_episode(
        {
            "result": {"success": False},
        },
        progress_before=0.1,
        progress_after=0.5,
    )

    assert reflection.diagnosis == "failure_with_partial_progress"
    assert reflection.change_required is True
    assert "preserve" in reflection.lesson.lower()


def test_success_with_progress_regression_requires_review():
    reflection = engine.reflect_on_episode(
        {
            "result": {"success": True},
        },
        progress_before=0.8,
        progress_after=0.5,
    )

    assert (
        reflection.diagnosis
        == "successful_outcome_but_progress_regressed"
    )
    assert reflection.change_required is True


def test_success_without_progress_has_lower_evidence_quality():
    reflection = engine.reflect_on_episode(
        {
            "result": {"success": True},
        }
    )

    assert reflection.evidence_quality == 0.6
    assert reflection.confidence == 0.68
    assert (
        reflection.diagnosis
        == "successful_outcome_without_measured_progress"
    )
    assert reflection.change_required is False


def test_missing_outcome_evidence_does_not_force_strategy_change():
    reflection = engine.reflect_on_episode(
        {
            "episode_id": "ep-unknown",
            "goal": "investigate",
        }
    )

    assert reflection.success is None
    assert reflection.diagnosis == "outcome_evidence_missing"
    assert reflection.evidence_quality == 0.0
    assert reflection.confidence == 0.2
    assert reflection.change_required is False


def test_reported_error_is_recorded_as_diagnostic_signal():
    reflection = engine.reflect_on_episode(
        {
            "result": {
                "success": False,
                "error": "timeout",
            }
        }
    )

    assert reflection.diagnosis == "failure_with_reported_error"
    assert reflection.change_required is True


def test_explicit_progress_overrides_episode_progress():
    reflection = engine.reflect_on_episode(
        {
            "progress_before": 0.1,
            "progress_after": 0.2,
            "result": {"success": True},
        },
        progress_before=0.3,
        progress_after=0.9,
    )

    assert reflection.progress_before == 0.3
    assert reflection.progress_after == 0.9
    assert reflection.progress_delta == 0.6


@dataclass
class EpisodeObject:
    episode_id: str
    goal: str
    action: str
    success: bool
    progress_before: float
    progress_after: float



def test_object_episode_is_supported():
    episode = EpisodeObject(
        episode_id="ep-object",
        goal="ship change",
        action="deploy",
        success=True,
        progress_before=0.5,
        progress_after=0.9,
    )

    reflection = engine.reflect_on_episode(episode)

    assert reflection.episode_id == "ep-object"
    assert reflection.goal == "ship change"
    assert reflection.action == "deploy"
    assert reflection.success is True
    assert reflection.progress_delta == 0.4


def test_progress_values_are_clamped():
    reflection = engine.reflect_on_episode(
        {
            "success": True,
            "progress_before": -2.0,
            "progress_after": 4.0,
        }
    )

    assert reflection.progress_before == 0.0
    assert reflection.progress_after == 1.0
    assert reflection.progress_delta == 1.0


def test_reflection_output_is_deterministic():
    episode = {
        "episode_id": "ep-deterministic",
        "goal": "complete job",
        "action": "run",
        "result": {"success": False},
    }

    first = engine.reflect_on_episode(
        episode,
        progress_before=0.2,
        progress_after=0.2,
    )

    second = engine.reflect_on_episode(
        episode,
        progress_before=0.2,
        progress_after=0.2,
    )

    assert first == second
