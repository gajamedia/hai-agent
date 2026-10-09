from agent.core.goal import Goal


def test_goal_can_have_sub_goals():

    parent = Goal(
        "open the door"
    )

    child = Goal(
        "obtain the key"
    )

    parent.add_sub_goal(
        child
    )

    assert parent.has_sub_goals is True

    assert parent.get_sub_goal(
        "obtain the key"
    ) is child


def test_goal_can_remove_sub_goal():

    parent = Goal(
        "open the door"
    )

    child = Goal(
        "obtain the key"
    )

    parent.add_sub_goal(
        child
    )

    removed = parent.remove_sub_goal(
        "obtain the key"
    )

    assert removed is True
    assert parent.has_sub_goals is False


def test_goal_status_is_updated_from_children():

    parent = Goal(
        "open the door"
    )

    child_a = Goal(
        "obtain the key"
    )

    child_b = Goal(
        "unlock the door"
    )

    parent.add_sub_goal(
        child_a
    )

    parent.add_sub_goal(
        child_b
    )

    child_a.complete()

    parent.update_status_from_sub_goals()

    assert parent.completed is False

    child_b.complete()

    parent.update_status_from_sub_goals()

    assert parent.completed is True
    assert parent.failed is False


def test_goal_fails_when_all_children_fail():

    parent = Goal(
        "open the door"
    )

    child_a = Goal(
        "obtain the key"
    )

    child_b = Goal(
        "unlock the door"
    )

    parent.add_sub_goal(
        child_a
    )

    parent.add_sub_goal(
        child_b
    )

    child_a.fail()
    child_b.fail()

    parent.update_status_from_sub_goals()

    assert parent.failed is True
    assert parent.completed is False


def test_goal_can_flatten_hierarchy():

    root = Goal(
        "open the door"
    )

    obtain_key = Goal(
        "obtain the key"
    )

    open_box = Goal(
        "open the box"
    )

    take_key = Goal(
        "take the key"
    )

    unlock_door = Goal(
        "unlock the door"
    )

    obtain_key.add_sub_goal(
        open_box
    )

    obtain_key.add_sub_goal(
        take_key
    )

    root.add_sub_goal(
        obtain_key
    )

    root.add_sub_goal(
        unlock_door
    )

    flattened = root.flatten()

    assert [
        goal.description
        for goal in flattened
    ] == [
        "obtain the key",
        "open the box",
        "take the key",
        "unlock the door",
    ]

def test_goal_can_store_target_state():

    goal = Goal(
        description="open the box",
        target_state={
            "box": "open",
        },
    )

    assert goal.target_state == {
        "box": "open",
    }


def test_goal_can_check_target_state():

    goal = Goal(
        description="open the box",
        target_state={
            "box": "open",
        },
    )

    assert goal.is_satisfied(
        {
            "box": "open",
        }
    ) is True

    assert goal.is_satisfied(
        {
            "box": "closed",
        }
    ) is False


def test_goal_with_multiple_target_states():

    goal = Goal(
        description="open the door",
        target_state={
            "door": "unlocked",
            "key": "agent",
        },
    )

    assert goal.is_satisfied(
        {
            "door": "unlocked",
            "key": "agent",
        }
    ) is True

    assert goal.is_satisfied(
        {
            "door": "unlocked",
            "key": "inside_box",
        }
    ) is False
