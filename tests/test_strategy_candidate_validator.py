from agent.core.action import Action
from agent.learning.strategy import Strategy
from agent.learning.strategy_candidate_validator import (
    StrategyCandidateValidator,
)


def make_actions():
    return [
        Action(
            name="open_box",
            description="Open box",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "closed",
            },
            effects={
                "box": "open",
            },
        ),
        Action(
            name="take_key",
            description="Take key",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock door",
            handler=lambda: {
                "success": True,
            },
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={
                "door": "unlocked",
            },
        ),
    ]


def test_valid_candidate_reaches_goal():
    source = Strategy(
        name="source_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    # Candidate must differ from source for this test.
    candidate.steps = [
        "open_box",
        "unlock_door",
    ]

    # With no key already acquired, this candidate
    # should fail validation.
    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=source,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.failed_step == "unlock_door"


def test_valid_candidate_with_matching_preconditions():
    source = Strategy(
        name="source_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=source,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is True
    assert result.reason == "candidate is valid"


def test_unknown_action_is_rejected():
    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "unknown_action",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.failed_step == "unknown_action"
    assert result.reason == (
        "action is not registered"
    )


def test_unsatisfied_precondition_is_rejected():
    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[
            "unlock_door",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.failed_step == "unlock_door"
    assert result.reason == (
        "action preconditions are not satisfied"
    )


def test_candidate_that_does_not_reach_goal_is_rejected():
    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[
            "open_box",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.reason == (
        "candidate does not reach goal state"
    )


def test_candidate_identical_to_source_is_rejected():
    source = Strategy(
        name="source_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    candidate = Strategy(
        name="another_name",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=source,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.reason == (
        "candidate is identical "
        "to source strategy"
    )


def test_duplicate_registered_strategy_is_rejected():
    existing = Strategy(
        name="existing_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
        strategies=[
            existing,
        ],
    )

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.reason == (
        "candidate already exists "
        "in strategy registry"
    )


def test_candidate_goal_must_match_source_goal():
    source = Strategy(
        name="source_strategy",
        goal="Open door",
        steps=[
            "open_box",
        ],
    )

    candidate = Strategy(
        name="candidate_strategy",
        goal="Close door",
        steps=[
            "open_box",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=source,
        candidate=candidate,
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.reason == (
        "candidate goal differs "
        "from source strategy"
    )


def test_empty_candidate_steps_are_rejected():
    candidate = Strategy(
        name="candidate_strategy",
        goal="Open door",
        steps=[],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    result = validator.validate(
        source_strategy=None,
        candidate=candidate,
        world_state={
            "door": "locked",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert result.valid is False
    assert result.reason == (
        "candidate has no steps"
    )


def test_none_candidate_is_rejected():
    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    try:
        validator.validate(
            source_strategy=None,
            candidate=None,
            world_state={},
            goal_state={
                "door": "unlocked",
            },
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "candidate cannot be None."
        )


def test_validate_all_returns_results_for_all_candidates():
    candidate_one = Strategy(
        name="candidate_one",
        goal="Open door",
        steps=[
            "open_box",
        ],
    )

    candidate_two = Strategy(
        name="candidate_two",
        goal="Open door",
        steps=[
            "unknown_action",
        ],
    )

    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    results = validator.validate_all(
        source_strategy=None,
        candidates=[
            candidate_one,
            candidate_two,
        ],
        world_state={
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
        },
        goal_state={
            "door": "unlocked",
        },
    )

    assert len(results) == 2
    assert all(
        result.valid is False
        for result in results
    )


def test_validate_all_rejects_none_candidates():
    validator = StrategyCandidateValidator(
        actions=make_actions(),
    )

    try:
        validator.validate_all(
            source_strategy=None,
            candidates=None,
            world_state={},
            goal_state={
                "door": "unlocked",
            },
        )
        assert False
    except ValueError as exc:
        assert str(exc) == (
            "candidates cannot be None."
        )
