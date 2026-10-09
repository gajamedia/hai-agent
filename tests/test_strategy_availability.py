from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def test_get_available_strategies_returns_only_active():
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

    available = agent.get_available_strategies()

    assert available == [active]
