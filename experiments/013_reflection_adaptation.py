from agent.core.action import Action

from agent.learning.reflection import (
    ReflectionEngine,
)

from agent.learning.reflection_strategy_updater import (
    ReflectionStrategyUpdater,
)

from agent.learning.strategy import Strategy

from agent.learning.strategy_selector import (
    StrategySelector,
)

from agent.memory.episodic import EpisodicMemory

from agent.memory.reflection_store import (
    ReflectionMemoryStore,
)


def main():

    print("================================")
    print("REFLECTION-DRIVEN ADAPTATION")
    print("================================")
    print()

    selector = StrategySelector()

    reflection_store = (
        ReflectionMemoryStore()
    )

    reflection_engine = ReflectionEngine(
        reflection_store
    )

    updater = ReflectionStrategyUpdater()

    # ==================================================
    # STRATEGY A
    #
    # This is initially preferred.
    # ==================================================

    strategy_a = Strategy(
        name="strategy_a",
        goal="Open the door",
        steps=[
            "take_key",
            "unlock_door",
        ],
        success_count=8,
        failure_count=1,
        total_actions=18,
    )

    # ==================================================
    # STRATEGY B
    #
    # Initially less reliable.
    # ==================================================

    strategy_b = Strategy(
        name="strategy_b",
        goal="Open the door",
        steps=[
            "find_spare_key",
            "unlock_door",
        ],
        success_count=5,
        failure_count=4,
        total_actions=18,
    )

    strategies = [
        strategy_a,
        strategy_b,
    ]

    # ==================================================
    # INITIAL DECISION
    # ==================================================

    print("================================")
    print("INITIAL DECISION")
    print("================================")
    print()

    selected = selector.select(
        strategies,
        goal="Open the door",
    )

    print(
        "Selected strategy:",
        selected.name,
    )

    print(
        "Steps:",
        selected.steps,
    )

    print(
        "Score:",
        selector._score(
            selected
        ),
    )

    print()

    # ==================================================
    # SIMULATE FAILURE OF SELECTED STRATEGY
    #
    # IMPORTANT:
    # The failed action is part of Strategy A.
    # ==================================================

    take_key_action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {},
        preconditions={
            "key": "inside_box",
        },
    )

    failed_episode = EpisodicMemory(
        content="Failed to take key.",
        goal="Open the door",
        action="take_key",
        observation_before={
            "key": "missing",
        },
        observation_after={
            "key": "missing",
        },
        result={
            "success": False,
            "message": (
                "The key is not inside "
                "the box."
            ),
        },
        cycle=1,
    )

    print("================================")
    print("FAILURE")
    print("================================")
    print()

    reflection = reflection_engine.reflect(
        episode=failed_episode,
        action=take_key_action,
    )

    if reflection is None:

        print(
            "No reflection generated."
        )

        return

    print(
        "Failed strategy:",
        strategy_a.name,
    )

    print(
        "Failed action:",
        reflection.action,
    )

    print(
        "Problem:",
        reflection.problem,
    )

    print(
        "Cause:",
        reflection.cause,
    )

    print(
        "Condition:",
        reflection.condition,
    )

    print(
        "Lesson:",
        reflection.lesson,
    )

    print(
        "Confidence:",
        reflection.confidence,
    )

    print()

    # ==================================================
    # APPLY REFLECTION TO THE ACTUAL FAILED STRATEGY
    # ==================================================

    print("================================")
    print("STRATEGY UPDATE")
    print("================================")
    print()

    updater.apply(
        strategy=strategy_a,
        reflection=reflection,
        penalty=0.4,
    )

    print(
        "Reflection applied:",
        reflection.applied,
    )

    print(
        "Strategy A penalty:",
        strategy_a.reflection_penalty,
    )

    print(
        "Strategy A new score:",
        selector._score(
            strategy_a
        ),
    )

    print(
        "Strategy B score:",
        selector._score(
            strategy_b
        ),
    )

    print()

    # ==================================================
    # SECOND DECISION
    # ==================================================

    print("================================")
    print("SECOND DECISION")
    print("================================")
    print()

    selected = selector.select(
        strategies,
        goal="Open the door",
    )

    print(
        "Selected strategy:",
        selected.name,
    )

    print(
        "Steps:",
        selected.steps,
    )

    print(
        "Score:",
        selector._score(
            selected
        ),
    )

    print()

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("================================")
    print("FINAL SUMMARY")
    print("================================")
    print()

    print(
        "Initial strategy:",
        "strategy_a",
    )

    print(
        "Failed action:",
        reflection.action,
    )

    print(
        "Reflection memories:",
        reflection_store.count(),
    )

    print(
        "Strategy A penalty:",
        strategy_a.reflection_penalty,
    )

    print(
        "Final selected strategy:",
        selected.name,
    )


if __name__ == "__main__":
    main()
