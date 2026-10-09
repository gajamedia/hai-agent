from agent.core.goal import Goal


def test_goal_can_have_dependency():

    obtain_key = Goal(
        "obtain the key"
    )

    unlock_door = Goal(
        "unlock the door"
    )

    unlock_door.add_dependency(
        obtain_key
    )

    assert (
        unlock_door.has_dependencies
        is True
    )

    assert unlock_door.dependencies == [
        obtain_key
    ]


def test_goal_dependencies_must_be_completed():

    obtain_key = Goal(
        "obtain the key"
    )

    unlock_door = Goal(
        "unlock the door"
    )

    unlock_door.add_dependency(
        obtain_key
    )

    assert (
        unlock_door.dependencies_satisfied()
        is False
    )

    assert (
        unlock_door.can_execute()
        is False
    )

    obtain_key.complete()

    assert (
        unlock_door.dependencies_satisfied()
        is True
    )

    assert (
        unlock_door.can_execute()
        is True
    )


def test_goal_cannot_depend_on_itself():

    goal = Goal(
        "open the door"
    )

    try:
        goal.add_dependency(goal)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError."
        )


def test_goal_dependency_cycle_is_detected():

    goal_a = Goal("A")
    goal_b = Goal("B")
    goal_c = Goal("C")

    goal_a.add_dependency(
        goal_b
    )

    goal_b.add_dependency(
        goal_c
    )

    goal_c.add_dependency(
        goal_a
    )

    assert (
        goal_a.has_dependency_cycle()
        is True
    )

    try:
        goal_a.validate_dependencies()
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected dependency cycle "
            "validation to fail."
        )


def test_goal_dependency_graph_is_built():

    root = Goal("unlock the door")

    obtain_key = Goal(
        "obtain the key"
    )

    open_box = Goal(
        "open the box"
    )

    root.add_dependency(
        obtain_key
    )

    obtain_key.add_dependency(
        open_box
    )

    graph = (
        root.dependency_graph()
    )

    assert id(root) in graph

    assert graph[id(root)] == [
        obtain_key
    ]

    assert graph[id(obtain_key)] == [
        open_box
    ]

    assert graph[id(open_box)] == []


def test_duplicate_dependency_is_not_added():

    dependency = Goal(
        "obtain the key"
    )

    goal = Goal(
        "unlock the door"
    )

    goal.add_dependency(
        dependency
    )

    goal.add_dependency(
        dependency
    )

    assert len(
        goal.dependencies
    ) == 1


def test_dependency_can_be_removed():

    dependency = Goal(
        "obtain the key"
    )

    goal = Goal(
        "unlock the door"
    )

    goal.add_dependency(
        dependency
    )

    removed = (
        goal.remove_dependency(
            dependency
        )
    )

    assert removed is True
    assert goal.has_dependencies is False

def test_goal_topological_order_respects_dependencies():

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

    ordered = root.topological_order()

    assert ordered == [
        obtain_key,
        unlock_door,
    ]


def test_goal_topological_order_handles_multiple_dependencies():

    root = Goal(
        "complete task"
    )

    prepare = Goal(
        "prepare"
    )

    authenticate = Goal(
        "authenticate"
    )

    execute = Goal(
        "execute"
    )

    root.add_sub_goal(
        execute
    )

    root.add_sub_goal(
        authenticate
    )

    root.add_sub_goal(
        prepare
    )

    authenticate.add_dependency(
        prepare
    )

    execute.add_dependency(
        authenticate
    )

    ordered = root.topological_order()

    assert ordered == [
        prepare,
        authenticate,
        execute,
    ]
