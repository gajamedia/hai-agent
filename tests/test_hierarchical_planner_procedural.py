from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.memory.procedural_store import ProceduralMemoryStore


def create_actions():
    return [
        Action(
            name="open_box",
            description="Open the box",
            handler=lambda **kwargs: {
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
            description="Take the key from the box",
            handler=lambda **kwargs: {
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
            name="find_spare_key",
            description="Find the spare key",
            handler=lambda **kwargs: {
                "success": True,
            },
            preconditions={
                "spare_key": "available",
            },
            effects={
                "spare_key": "found",
                "key": "agent",
            },
        ),
        Action(
            name="unlock_door",
            description="Unlock the door",
            handler=lambda **kwargs: {
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


def test_hierarchical_planner_uses_procedural_memory():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    procedural_store = ProceduralMemoryStore()

    memory = procedural_store.get_or_create(
        "open_the_box",
        [
            "open_box",
        ],
    )

    memory.record_success(
        [
            "experience-1",
        ]
    )

    hierarchical_planner = HierarchicalPlanner(
        planner=planner,
        decomposer=decomposer,
        procedural_store=procedural_store,
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


def test_hierarchical_planner_falls_back_when_procedure_is_invalid():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    procedural_store = ProceduralMemoryStore()

    memory = procedural_store.get_or_create(
        "open_the_box",
        [
            "unlock_door",
        ],
    )

    memory.record_success(
        [
            "experience-1",
        ]
    )

    hierarchical_planner = HierarchicalPlanner(
        planner=planner,
        decomposer=decomposer,
        procedural_store=procedural_store,
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
            "door": "locked",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "open_box",
    ]


def test_hierarchical_planner_does_not_trust_failed_procedure():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    procedural_store = ProceduralMemoryStore()

    memory = procedural_store.get_or_create(
        "open_the_box",
        [
            "unlock_door",
        ],
    )

    memory.record_success(
        [
            "experience-1",
        ]
    )

    memory.record_failure(
        [
            "experience-2",
        ]
    )

    memory.record_failure(
        [
            "experience-3",
        ]
    )

    hierarchical_planner = HierarchicalPlanner(
        planner=planner,
        decomposer=decomposer,
        procedural_store=procedural_store,
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
