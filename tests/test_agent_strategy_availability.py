from agent.core.agent import Agent
from agent.core.goal import Goal
from agent.learning.strategy import Strategy


class CapturingExecutor:
    def __init__(self):
        self.strategies = None

    def execute(self, **kwargs):
        self.strategies = kwargs["strategies"]
        return {
            "success": True,
        }


def test_decide_and_act_defaults_to_active_strategies(monkeypatch):
    agent = Agent(name="test-agent")

    active = Strategy(
        name="active_strategy",
        goal="Open door",
        steps=["open_box"],
    )

    paused = Strategy(
        name="paused_strategy",
        goal="Open door",
        steps=["take_key"],
    )

    retired = Strategy(
        name="retired_strategy",
        goal="Open door",
        steps=["unlock_door"],
    )

    paused.pause()
    retired.retire()

    agent.register_strategies(
        [
            active,
            paused,
            retired,
        ]
    )

    executor = CapturingExecutor()

    agent.unified_executor = executor

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    result = agent.decide_and_act(
        goal=goal,
        world_state={
            "door": "locked",
        },
    )

    assert result["success"] is True
    assert executor.strategies == [
        active
    ]


def test_decide_and_act_filters_explicit_strategies(monkeypatch):
    agent = Agent(name="test-agent")

    active = Strategy(
        name="active_strategy",
        goal="Open door",
        steps=["open_box"],
    )

    paused = Strategy(
        name="paused_strategy",
        goal="Open door",
        steps=["take_key"],
    )

    retired = Strategy(
        name="retired_strategy",
        goal="Open door",
        steps=["unlock_door"],
    )

    paused.pause()
    retired.retire()

    executor = CapturingExecutor()

    agent.unified_executor = executor

    goal = Goal(
        description="Open door",
        target_state={
            "door": "unlocked",
        },
    )

    agent.decide_and_act(
        goal=goal,
        world_state={
            "door": "locked",
        },
        strategies=[
            active,
            paused,
            retired,
        ],
    )

    assert executor.strategies == [
        active
    ]
