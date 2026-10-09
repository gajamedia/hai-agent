from agent.core.agent import Agent
from agent.core.goal import Goal


def test_agent_can_have_goal():

    agent = Agent("Agent-001")

    goal = Goal(
        description="Open the door",
        priority=10
    )

    agent.set_goal(goal)

    assert agent.current_goal is goal
    assert agent.current_goal.description == "Open the door"
    assert agent.current_goal.priority == 10
