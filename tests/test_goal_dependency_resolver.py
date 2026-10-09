from agent.cognition.goal_dependency_resolver import (
    GoalDependencyResolver,
)
from agent.core.goal import Goal


def test_resolver_returns_dependency_order():

    root = Goal(
        "open the door"
    )

    obtain_key = Goal(
        "obtain the key"
    )

    unlock_door = Goal(
        "unlock the door"
    )

    root.add_sub_goal(
        unlock_door
    )

    root.add_sub_goal(
        obtain_key
    )

    unlock_door.add_dependency(
        obtain_key
    )

    resolver = GoalDependencyResolver()

    ordered = resolver.resolve(
        root
    )

    assert ordered == [
        obtain_key,
        unlock_door,
    ]


def test_resolver_rejects_dependency_cycle():

    root = Goal(
        "root"
    )

    goal_a = Goal("A")
    goal_b = Goal("B")

    root.add_sub_goal(
        goal_a
    )

    root.add_sub_goal(
        goal_b
    )

    goal_a.add_dependency(
        goal_b
    )

    goal_b.add_dependency(
        goal_a
    )

    resolver = GoalDependencyResolver()

    try:
        resolver.resolve(root)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected dependency cycle "
            "to be rejected."
        )


def test_resolver_handles_goal_without_dependencies():

    root = Goal(
        "simple task"
    )

    resolver = GoalDependencyResolver()

    assert resolver.resolve(
        root
    ) == []
