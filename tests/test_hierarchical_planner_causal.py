from agent.core.action import Action
from agent.core.goal import Goal
from agent.cognition.planner import Planner
from agent.cognition.goal_decomposer import GoalDecomposer
from agent.cognition.hierarchical_planner import HierarchicalPlanner
from agent.memory.procedural_store import ProceduralMemoryStore
from agent.memory.causal import CausalMemory
from agent.memory.causal_store import CausalMemoryStore


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


def test_hierarchical_planner_accepts_procedure_with_known_causal_relation():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    procedural_store = ProceduralMemoryStore()
    causal_store = CausalMemoryStore()

    procedural = procedural_store.get_or_create(
        "open_the_door",
        [
            "take_key",
            "unlock_door",
        ],
    )

    procedural.record_success(
        [
            "experience-1",
        ]
    )

    causal = CausalMemory(
        content="Taking the key enables unlocking the door.",
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )

    causal.reinforce(
        [
            "experience-1",
        ]
    )

    causal_store.add(
        causal
    )

    hierarchical_planner = HierarchicalPlanner(
        planner=planner,
        decomposer=decomposer,
        procedural_store=procedural_store,
        causal_store=causal_store,
    )

    goal = Goal(
        "open the door",
        target_state={
            "door": "unlocked",
        },
    )

    plan = hierarchical_planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "key": "inside_box",
            "box": "open",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]


def test_hierarchical_planner_does_not_require_causal_memory():

    actions = create_actions()

    planner = Planner(
        actions
    )

    decomposer = GoalDecomposer()

    procedural_store = ProceduralMemoryStore()

    procedural = procedural_store.get_or_create(
        "open_the_door",
        [
            "take_key",
            "unlock_door",
        ],
    )

    procedural.record_success(
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
        "open the door",
        target_state={
            "door": "unlocked",
        },
    )

    plan = hierarchical_planner.plan(
        goal=goal,
        world_state={
            "door": "locked",
            "key": "inside_box",
            "box": "open",
        },
    )

    assert [
        action.name
        for action in plan
    ] == [
        "take_key",
        "unlock_door",
    ]
