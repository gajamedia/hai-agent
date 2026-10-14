from agent.core.action import Action
from agent.cognition.planner import Planner
from agent.cognition.adaptive_planner import AdaptivePlanner
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)
from agent.learning.strategy import Strategy
from agent.learning.strategy_selector import StrategySelector


def create_actions():
    return [
        Action(
            name="take_key",
            description="Take key",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "box": "open",
                "key": "inside_box",
            },
            effects={"key": "agent"},
        ),
        Action(
            name="find_spare_key",
            description="Find spare key",
            handler=lambda **kwargs: {"success": True},
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
            description="Unlock door",
            handler=lambda **kwargs: {"success": True},
            preconditions={
                "key": "agent",
                "door": "locked",
            },
            effects={"door": "unlocked"},
        ),
    ]


def create_strategy(name, steps):
    return Strategy(
        name=name,
        goal="open door",
        steps=steps,
    )


def test_new_failure_changes_strategy_score_for_next_planning_cycle():
    actions = create_actions()
    planner = Planner(actions)
    selector = RiskAwareStrategySelector()

    reliable = create_strategy(
        "reliable",
        ["take_key", "unlock_door"],
    )
    alternative = create_strategy(
        "alternative",
        ["find_spare_key", "unlock_door"],
    )

    for index in range(4):
        reliable.record_success(
            experience_id=f"reliable-success-{index}",
            action_count=2,
            progress_delta=1.0,
        )

    alternative.record_success(
        experience_id="alternative-success",
        action_count=2,
        progress_delta=1.0,
    )

    adaptive_planner = AdaptivePlanner(
        planner=planner,
        strategy_selector=StrategySelector(),
        risk_strategy_selector=selector,
    )

    world_state = {
        "door": "locked",
        "box": "open",
        "key": "inside_box",
        "spare_key": "available",
    }
    goal_state = {"door": "unlocked"}

    before_score = selector.score(reliable)

    reliable.record_failure(
        experience_id="new-failed-experience",
        action_count=2,
        progress_delta=0.0,
    )

    after_score = selector.score(reliable)

    assert after_score < before_score

    plan = adaptive_planner.plan(
        world_state=world_state,
        goal_state=goal_state,
        goal="open door",
        strategies=[reliable, alternative],
    )

    assert plan
    assert plan[-1].name == "unlock_door"
    assert all(
        action.name in {
            "take_key",
            "find_spare_key",
            "unlock_door",
        }
        for action in plan
    )