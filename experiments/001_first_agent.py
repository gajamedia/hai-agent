from agent.core.agent import Agent
from agent.core.goal import Goal

def main():

    agent = Agent("Agent-001")

    goal = Goal(
        description="Open the door",
        priority=10
    )

    agent.set_goal(goal)

    print("Agent:", agent.name)
    print("Goal:", agent.current_goal.description)
    print("Priority:", agent.current_goal.priority)

    agent.observe({
        "door": "locked"
    })

    print("Observation:", agent.state.last_observation)

    agent.act("search_for_key")

    print("Action:", agent.state.previous_action)

    agent.step()

    print("Cycle:", agent.state.cycle)


if __name__ == "__main__":
    main()
