from agent.core.experience import Experience
from agent.core.experience_store import (
    ExperienceStore,
)
from agent.learning.strategy_learning import (
    StrategyLearning,
)
from agent.learning.strategy_selector import (
    StrategySelector,
)
from agent.learning.strategy_store import (
    StrategyStore,
)
from agent.memory.episodic import EpisodicMemory


def create_experience(
    experience_id,
    goal,
    actions,
    completed,
):

    experience = Experience(
        experience_id=experience_id,
        goal=goal,
    )

    for cycle, action in enumerate(
        actions,
        start=1,
    ):

        experience.add_episode(
            EpisodicMemory(
                content=(
                    f"Executed {action}."
                ),
                action=action,
                cycle=cycle,
            )
        )

    if completed:
        experience.complete()
    else:
        experience.fail()

    return experience


def main():

    goal = "Open the door"

    experience_store = ExperienceStore()

    # Strategy A
    experience_store.add(
        create_experience(
            experience_id="exp-1",
            goal=goal,
            actions=[
                "open_box",
                "take_key",
                "unlock_door",
            ],
            completed=True,
        )
    )

    experience_store.add(
        create_experience(
            experience_id="exp-2",
            goal=goal,
            actions=[
                "open_box",
                "take_key",
                "unlock_door",
            ],
            completed=True,
        )
    )

    experience_store.add(
        create_experience(
            experience_id="exp-3",
            goal=goal,
            actions=[
                "open_box",
                "take_key",
                "unlock_door",
            ],
            completed=False,
        )
    )

    # Strategy B
    experience_store.add(
        create_experience(
            experience_id="exp-4",
            goal=goal,
            actions=[
                "find_spare_key",
                "unlock_door",
            ],
            completed=True,
        )
    )

    experience_store.add(
        create_experience(
            experience_id="exp-5",
            goal=goal,
            actions=[
                "find_spare_key",
                "unlock_door",
            ],
            completed=True,
        )
    )

    experience_store.add(
        create_experience(
            experience_id="exp-6",
            goal=goal,
            actions=[
                "find_spare_key",
                "unlock_door",
            ],
            completed=True,
        )
    )

    strategy_store = StrategyStore()

    learning = StrategyLearning(
        strategy_store
    )

    learning.learn_from_experiences(
        experience_store.all()
    )

    selector = StrategySelector()

    strategies = (
        strategy_store.for_goal(
            goal
        )
    )

    selected = selector.select(
        strategies
    )

    print("================================")
    print("STRATEGY LEARNING")
    print("================================")
    print()

    for index, strategy in enumerate(
        strategies,
        start=1,
    ):

        print(
            f"Strategy #{index}"
        )

        print(
            "  Name:",
            strategy.name,
        )

        print(
            "  Steps:",
            strategy.steps,
        )

        print(
            "  Attempts:",
            strategy.attempts,
        )

        print(
            "  Success:",
            strategy.success_count,
        )

        print(
            "  Failure:",
            strategy.failure_count,
        )

        print(
            "  Success rate:",
            strategy.success_rate,
        )

        print(
            "  Average actions:",
            strategy.average_actions,
        )

        print()

    print("================================")
    print("SELECTED STRATEGY")
    print("================================")
    print()

    if selected is not None:

        print(
            "Strategy:",
            selected.name,
        )

        print(
            "Steps:",
            selected.steps,
        )

        print(
            "Score:",
            selector._score(selected),
        )


if __name__ == "__main__":
    main()
