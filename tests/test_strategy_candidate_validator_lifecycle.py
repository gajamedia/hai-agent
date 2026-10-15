
from agent.learning.strategy import Strategy
from agent.learning.strategy_candidate_validator import (
    StrategyCandidateValidator,
)


def test_validator_rejects_paused_candidate():
    candidate = Strategy(
        name="paused_candidate",
        goal="Open door",
        steps=["open_box", "unlock_door"],
    )
    candidate.pause()

    validator = StrategyCandidateValidator()

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={},
        goal_state={"door": "unlocked"},
    )

    assert result.valid is False
    assert "not active" in result.reason
    assert "paused" in result.reason


def test_validator_rejects_retired_candidate():
    candidate = Strategy(
        name="retired_candidate",
        goal="Open door",
        steps=["open_box", "unlock_door"],
    )
    candidate.retire()

    validator = StrategyCandidateValidator()

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={},
        goal_state={"door": "unlocked"},
    )

    assert result.valid is False
    assert "not active" in result.reason
    assert "retired" in result.reason
