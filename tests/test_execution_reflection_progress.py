from api.routes.execution import _build_reflection_provider


def test_successful_action_without_target_progress_is_not_reported_as_failure():
    reflection_provider = _build_reflection_provider(
        {"door": "unlocked"}
    )

    episode = {
        "observation_before": {"door": "locked", "box": "closed"},
        "observation_after": {"door": "locked", "box": "open"},
        "result": {"success": True},
    }

    reflection = reflection_provider(episode)

    assert reflection["success"] is True
    assert reflection["progress_delta"] == 0.0
    assert reflection["diagnosis"] == "no_observed_target_progress"
    assert reflection["change_required"] is False
