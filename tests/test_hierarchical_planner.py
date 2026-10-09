from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.goal_decomposer import (
    GoalDecomposer,
)
from agent.cognition.hierarchical_planner import (
    HierarchicalPlanner,
)
from agent.cognition.planner import Planner


def create_actions():

    return [
        Action(
            name="open_box",
            description="Open the box",
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
            description="Take the key",
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
            description="Unlock the door",
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


def create_decomposer():

    decomposer = GoalDecomposer()

    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal(
                "obtain the key"
            ),
            Goal(
                "unlock the door",
                target_state={
                    "door": "unlocked",
                },
            ),
        ],
    )

    decomposer.register_rule(
        "obtain the key",
        lambda goal: [
            Goal(
                "open the box",
                target_state={
                    "box": "open",
                },
            ),
            Goal(
                "take the key",
                target_state={
                    "key": "agent",
                },
            ),
        ],
    )

    return decomposer

def create_dependency_decomposer():

    decomposer = GoalDecomposer()

    def decompose_open_door(goal):

        obtain_key = Goal(
            "obtain the key",
        )

        unlock_door = Goal(
            "unlock the door",
            target_state={
                "door": "unlocked",
            },
        )

        unlock_door.add_dependency(
            obtain_key
        )

        return [
            unlock_door,
            obtain_key,
        ]

    decomposer.register_rule(
        "open the door",
        decompose_open_door,
    )

    decomposer.register_rule(
        "obtain the key",
        lambda goal: [
            Goal(
                "take the key",
                target_state={
                    "key": "agent",
                },
            ),
        ],
    )

    return decomposer


def test_hierarchical_planner_creates_action_sequence():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = create_decomposer()

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    root = Goal(
        "open the door"
    )

    world = {
        "door": "locked",
        "box": "closed",
        "key": "inside_box",
    }

    plan = hierarchical_planner.plan(
        goal=root,
        world_state=world,
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
        "take_key",
        "unlock_door",
    ]


def test_hierarchical_planner_handles_leaf_goal():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    plan = hierarchical_planner.plan(
        goal=goal,
        world_state={
            "box": "closed",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
    ]


def test_hierarchical_planner_returns_empty_when_subgoal_is_unreachable():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal("unknown task"),
        ],
    )

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    root = Goal(
        "open the door"
    )

    plan = hierarchical_planner.plan(
        goal=root,
        world_state={
            "door": "locked",
        },
    )

    assert plan == []


def test_hierarchical_planner_updates_simulated_state_between_subgoals():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = create_decomposer()

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    root = Goal(
        "open the door"
    )

    world = {
        "door": "locked",
        "box": "closed",
        "key": "inside_box",
    }

    plan = hierarchical_planner.plan(
        goal=root,
        world_state=world,
    )

    assert len(plan) == 3

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
        "take_key",
        "unlock_door",
    ]

def test_hierarchical_planner_uses_goal_target_state():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    goal = Goal(
        description="open the box",
        target_state={
            "box": "open",
        },
    )

    plan = hierarchical_planner.plan(
        goal=goal,
        world_state={
            "box": "closed",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
    ]


def test_hierarchical_planner_does_not_depend_on_goal_text():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    goal = Goal(
        description="ANY DESCRIPTION",
        target_state={
            "box": "open",
        },
    )

    plan = hierarchical_planner.plan(
        goal=goal,
        world_state={
            "box": "closed",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
    ]


def test_hierarchical_planner_handles_multiple_target_states():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    goal = Goal(
        description="unlock",
        target_state={
            "door": "unlocked",
        },
    )

    world = {
        "door": "locked",
        "key": "agent",
    }

    plan = hierarchical_planner.plan(
        goal=goal,
        world_state=world,
    )

    assert [
        action.name
        for action in plan
    ] == [
        "unlock_door",
    ]

def test_hierarchical_goal_dependencies_are_explicit():

    obtain_key = Goal(
        "obtain the key"
    )

    unlock_door = Goal(
        "unlock the door",
        target_state={
            "door": "unlocked",
        },
    )

    unlock_door.add_dependency(
        obtain_key
    )

    root = Goal(
        "open the door"
    )

    root.add_sub_goal(
        obtain_key
    )

    root.add_sub_goal(
        unlock_door
    )

    assert (
        unlock_door.dependencies
        == [obtain_key]
    )

    assert (
        unlock_door.can_execute()
        is False
    )

    obtain_key.complete()

    assert (
        unlock_door.can_execute()
        is True
    )

def test_hierarchical_planner_respects_dependency_order():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    decomposer.register_rule(
        "open the door",
        lambda goal: [
            Goal(
                "unlock the door",
                target_state={
                    "door": "unlocked",
                },
            ),
            Goal(
                "obtain the key",
            ),
        ],
    )

    decomposer.register_rule(
        "obtain the key",
        lambda goal: [
            Goal(
                "take the key",
                target_state={
                    "key": "agent",
                },
            ),
        ],
    )

    root = Goal(
        "open the door"
    )

    decomposed = (
        decomposer.decompose_recursive(
            root
        )
    )

    unlock_door = (
        decomposed.get_sub_goal(
            "unlock the door"
        )
    )

    obtain_key = (
        decomposed.get_sub_goal(
            "obtain the key"
        )
    )

    assert unlock_door is not None
    assert obtain_key is not None

    unlock_door.add_dependency(
        obtain_key
    )

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=GoalDecomposer(),
        )
    )

    # Use already-decomposed tree directly by
    # providing a decomposer that leaves it unchanged.
    plan = hierarchical_planner.plan(
        goal=decomposed,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]

def test_hierarchical_planner_respects_dependency_order():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = (
        create_dependency_decomposer()
    )

    hierarchical_planner = (
        HierarchicalPlanner(
            planner=planner,
            decomposer=decomposer,
        )
    )

    root = Goal(
        "open the door"
    )

    plan = hierarchical_planner.plan(
        goal=root,
        world_state={
            "door": "locked",
            "box": "open",
            "key": "inside_box",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]
