from agent.learning.strategy import Strategy
from agent.cognition.risk_strategy_selector import (
    RiskAwareStrategySelector,
)


def test_strategy_records_positive_progress():

    strategy = Strategy(
        name="good_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    strategy.record_progress(
        experience_id="experience-1",
        progress_delta=0.5,
    )

    assert (
        strategy.progress_count
        == 1
    )

    assert (
        strategy.total_progress
        == 0.5
    )

    assert (
        strategy.average_progress
        == 0.5
    )

    assert (
        strategy.positive_progress_count
        == 1
    )


def test_strategy_records_negative_progress():

    strategy = Strategy(
        name="bad_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    strategy.record_progress(
        experience_id="experience-1",
        progress_delta=-0.5,
    )

    assert (
        strategy.average_progress
        == -0.5
    )

    assert (
        strategy.negative_progress_count
        == 1
    )


def test_strategy_averages_progress():

    strategy = Strategy(
        name="strategy",
        goal="prepare room",
    )

    strategy.record_progress(
        experience_id="experience-1",
        progress_delta=0.5,
    )

    strategy.record_progress(
        experience_id="experience-2",
        progress_delta=0.25,
    )

    assert (
        strategy.average_progress
        == 0.375
    )


def test_progress_changes_strategy_score():

    selector = RiskAwareStrategySelector(
        progress_weight=1.0,
    )

    good = Strategy(
        name="good",
        goal="open the box",
        success_count=5,
    )

    bad = Strategy(
        name="bad",
        goal="open the box",
        success_count=5,
    )

    good.record_progress(
        experience_id="good-1",
        progress_delta=0.8,
    )

    bad.record_progress(
        experience_id="bad-1",
        progress_delta=0.0,
    )

    assert (
        selector.score(good)
        > selector.score(bad)
    )


def test_negative_progress_penalizes_strategy():

    selector = RiskAwareStrategySelector(
        progress_weight=1.0,
    )

    neutral = Strategy(
        name="neutral",
        goal="open the box",
        success_count=5,
    )

    regression = Strategy(
        name="regression",
        goal="open the box",
        success_count=5,
    )

    regression.record_progress(
        experience_id="regression-1",
        progress_delta=-0.5,
    )

    assert (
        selector.score(regression)
        < selector.score(neutral)
    )


def test_selector_prefers_strategy_with_better_progress():

    selector = RiskAwareStrategySelector(
        progress_weight=1.0,
    )

    slow = Strategy(
        name="slow",
        goal="open the box",
        success_count=5,
    )

    fast = Strategy(
        name="fast",
        goal="open the box",
        success_count=5,
    )

    slow.record_progress(
        experience_id="slow-1",
        progress_delta=0.1,
    )

    fast.record_progress(
        experience_id="fast-1",
        progress_delta=0.8,
    )

    selected = selector.select(
        [
            slow,
            fast,
        ]
    )

    assert selected is fast

def test_agent_records_progress_into_active_strategy():

    from agent.core.agent import Agent
    from agent.core.action import Action
    from agent.core.goal import Goal
    from agent.world.room import RoomEnvironment

    agent = Agent(
        "strategy-progress-agent"
    )

    action = Action(
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
    )

    agent.register_action(
        action
    )

    strategy = Strategy(
        name="open_box_strategy",
        goal="open the box",
        steps=[
            "open_box",
        ],
    )

    agent.set_active_strategy(
        strategy
    )

    goal = Goal(
        "open the box",
        target_state={
            "box": "open",
        },
    )

    result = agent.decide_and_act(
        goal=goal,
        environment=RoomEnvironment(),
        information_gain={
            "open_box": 1.0,
        },
        action_risk={
            "open_box": 0.1,
        },
        risk=0.2,
        experience_id="strategy-progress-1",
    )

    assert result is not None

    assert (
        strategy.progress_count
        == 1
    )

    assert (
        strategy.average_progress
        == 1.0
    )
