from agent.cognition.goal_decomposer import (
    GoalDecomposer,
)
from agent.core.goal import Goal


def test_decomposer_can_register_rule():

    decomposer = GoalDecomposer()

    def decompose_open_door(goal):

        return [
            Goal("obtain the key"),
            Goal("unlock the door"),
        ]

    decomposer.register_rule(
        "open the door",
        decompose_open_door,
    )

    goal = Goal(
        "open the door"
    )

    result = decomposer.decompose(
        goal
    )

    assert result is goal

    assert [
        item.description
        for item in result.sub_goals
    ] == [
        "obtain the key",
        "unlock the door",
    ]


def test_decomposer_without_rule_keeps_goal_unchanged():

    decomposer = GoalDecomposer()

    goal = Goal(
        "open the box"
    )

    result = decomposer.decompose(
        goal
    )

    assert result is goal
    assert result.has_sub_goals is False


def test_decomposer_can_build_nested_goal_tree():

    decomposer = GoalDecomposer()

    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal("obtain the key"),
            Goal("unlock the door"),
        ],
    )

    decomposer.register_rule(
        "obtain the key",
        lambda goal: [
            Goal("open the box"),
            Goal("take the key"),
        ],
    )

    root = Goal(
        "open the door"
    )

    decomposer.decompose_recursive(
        root
    )

    assert [
        goal.description
        for goal in root.sub_goals
    ] == [
        "obtain the key",
        "unlock the door",
    ]

    obtain_key = root.get_sub_goal(
        "obtain the key"
    )

    assert obtain_key is not None

    assert [
        goal.description
        for goal in obtain_key.sub_goals
    ] == [
        "open the box",
        "take the key",
    ]


def test_recursive_decomposition_preserves_leaf_goals():

    decomposer = GoalDecomposer()

    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal("obtain the key"),
        ],
    )

    root = Goal(
        "open the door"
    )

    decomposer.decompose_recursive(
        root
    )

    child = root.get_sub_goal(
        "obtain the key"
    )

    assert child is not None

    assert child.has_sub_goals is False


def test_decomposer_does_not_duplicate_existing_sub_goals():

    decomposer = GoalDecomposer()

    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal("obtain the key"),
        ],
    )

    root = Goal(
        "open the door"
    )

    root.add_sub_goal(
        Goal("obtain the key")
    )

    decomposer.decompose(
        root
    )

    assert len(
        root.sub_goals
    ) == 2
