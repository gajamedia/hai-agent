from agent.llm.goal_feasibility import assess_goal_feasibility


ACTIONS = [
    {
        "id": "open_box",
        "preconditions": {"box": "closed"},
        "effects": {"box": "open"},
    },
    {
        "id": "take_key",
        "preconditions": {
            "box": "open",
            "key": "inside_box",
        },
        "effects": {"key": "agent"},
    },
    {
        "id": "find_spare_key",
        "preconditions": {"spare_key": "available"},
        "effects": {"key": "agent"},
    },
    {
        "id": "unlock_door",
        "preconditions": {
            "key": "agent",
            "door": "locked",
        },
        "effects": {"door": "unlocked"},
    },
]


def test_goal_reachable_through_box():
    result = assess_goal_feasibility(
        {
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "unavailable",
        },
        {"door": "unlocked"},
        ACTIONS,
    )

    assert result.status == "reachable"
    assert result.action_sequence == (
        "open_box",
        "take_key",
        "unlock_door",
    )


def test_goal_reachable_with_spare_key():
    result = assess_goal_feasibility(
        {
            "door": "locked",
            "box": "closed",
            "key": "missing",
            "spare_key": "available",
        },
        {"door": "unlocked"},
        ACTIONS,
    )

    assert result.status == "reachable"
    assert result.action_sequence == (
        "find_spare_key",
        "unlock_door",
    )


def test_goal_unreachable_when_no_key_is_available():
    result = assess_goal_feasibility(
        {
            "door": "locked",
            "box": "closed",
            "key": "missing",
            "spare_key": "unavailable",
        },
        {"door": "unlocked"},
        ACTIONS,
    )

    assert result.status == "unreachable"
    assert result.action_sequence == ()


def test_goal_already_satisfied():
    result = assess_goal_feasibility(
        {"door": "unlocked"},
        {"door": "unlocked"},
        ACTIONS,
    )

    assert result.status == "reachable"
    assert result.explored_states == 1


def test_state_limit_returns_inconclusive():
    result = assess_goal_feasibility(
        {
            "door": "locked",
            "box": "closed",
            "key": "inside_box",
            "spare_key": "unavailable",
        },
        {"door": "unlocked"},
        ACTIONS,
        max_states=1,
    )

    assert result.status == "inconclusive"


def test_invalid_empty_goal_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        assess_goal_feasibility(
            {"door": "locked"},
            {},
            ACTIONS,
        )
