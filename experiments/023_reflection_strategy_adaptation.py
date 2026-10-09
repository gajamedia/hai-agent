from agent.core.action import Action
from agent.core.agent import Agent
from agent.learning.strategy import Strategy


def main():

    print("=" * 50)
    print("REFLECTION → STRATEGY ADAPTATION")
    print("=" * 50)

    agent = Agent("hAI")

    action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {
            "success": False,
        },
        preconditions={
            "box": "open",
        },
        effects={
            "key": "agent",
        },
    )

    agent.register_action(action)

    strategy_a = Strategy(
        name="strategy_a",
        goal="open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
        success_count=3,
        failure_count=0,
        total_actions=6,
    )

    agent.set_active_strategy(
        strategy_a
    )

    print()
    print("BEFORE FAILURE")
    print("-" * 50)

    print(
        "Strategy:",
        strategy_a.name,
    )

    print(
        "Reflection penalty:",
        strategy_a.reflection_penalty,
    )

    print(
        "Score:",
        strategy_a.success_rate
        - strategy_a.average_actions * 0.01
        - strategy_a.reflection_penalty,
    )

    agent.record_experience(
        action="take_key",
        observation_before={
            "box": "closed",
        },
        observation_after={
            "box": "closed",
        },
        result={
            "success": False,
            "message": "The box is closed.",
        },
    )

    reflection = (
        agent.reflection_memory.latest()
    )

    print()
    print("AFTER FAILURE")
    print("-" * 50)

    print(
        "Reflection created:",
        reflection is not None,
    )

    print(
        "Reflection applied:",
        reflection.applied
        if reflection
        else None,
    )

    print(
        "Strategy:",
        strategy_a.name,
    )

    print(
        "Reflection penalty:",
        strategy_a.reflection_penalty,
    )

    print(
        "Score:",
        strategy_a.success_rate
        - strategy_a.average_actions * 0.01
        - strategy_a.reflection_penalty,
    )

    print()
    print("RESULT")
    print("-" * 50)

    if (
        reflection is not None
        and reflection.applied is True
        and strategy_a.reflection_penalty
        > 0.0
    ):
        print(
            "SUCCESS:"
        )

        print(
            "Agent converted failure into "
            "a strategy penalty."
        )

    else:
        print(
            "FAILURE:"
        )

        print(
            "Strategy adaptation did not occur."
        )


if __name__ == "__main__":
    main()
