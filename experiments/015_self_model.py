from agent.core.experience import Experience
from agent.learning.strategy import Strategy
from agent.memory.episodic import EpisodicMemory
from agent.self.learning import SelfLearning
from agent.self.model import SelfModel


def build_successful_experience():

    experience = Experience(
        experience_id="experience-success",
        goal="Open the door",
    )

    experience.add_episode(
        EpisodicMemory(
            content="Opened box.",
            action="open_box",
            observation_before={
                "box": "closed",
                "key": "inside_box",
                "door": "locked",
            },
            observation_after={
                "box": "open",
                "key": "inside_box",
                "door": "locked",
            },
            result={
                "success": True,
            },
            cycle=1,
        )
    )

    experience.add_episode(
        EpisodicMemory(
            content="Took key.",
            action="take_key",
            observation_before={
                "box": "open",
                "key": "inside_box",
                "door": "locked",
            },
            observation_after={
                "box": "open",
                "key": "agent",
                "door": "locked",
            },
            result={
                "success": True,
            },
            cycle=2,
        )
    )

    experience.add_episode(
        EpisodicMemory(
            content="Unlocked door.",
            action="unlock_door",
            observation_before={
                "box": "open",
                "key": "agent",
                "door": "locked",
            },
            observation_after={
                "box": "open",
                "key": "agent",
                "door": "unlocked",
            },
            result={
                "success": True,
            },
            cycle=3,
        )
    )

    experience.complete()

    return experience


def build_failed_experience():

    experience = Experience(
        experience_id="experience-failure",
        goal="Open the door",
    )

    experience.add_episode(
        EpisodicMemory(
            content="Failed to find spare key.",
            action="find_spare_key",
            observation_before={
                "spare_key": "unavailable",
                "door": "locked",
            },
            observation_after={
                "spare_key": "unavailable",
                "door": "locked",
            },
            result={
                "success": False,
                "message": (
                    "The spare key is not available."
                ),
            },
            cycle=4,
        )
    )

    experience.fail()

    return experience


def main():

    print("================================")
    print("SELF MODEL")
    print("================================")
    print()

    model = SelfModel()

    learning = SelfLearning(
        self_model=model
    )

    # ----------------------------------------------
    # Successful experience
    # ----------------------------------------------

    success_experience = (
        build_successful_experience()
    )

    learning.learn_from_experience(
        success_experience
    )

    # ----------------------------------------------
    # Failed experience
    # ----------------------------------------------

    failed_experience = (
        build_failed_experience()
    )

    learning.learn_from_experience(
        failed_experience
    )

    # ----------------------------------------------
    # Learned strategy
    # ----------------------------------------------

    strategy = Strategy(
        name="box_strategy",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=1,
        failure_count=0,
        total_actions=3,
    )

    learning.learn_from_strategy(
        strategy
    )

    print("================================")
    print("CAPABILITIES")
    print("================================")
    print()

    for name, capability in (
        model.capabilities.items()
    ):

        print(
            f"Capability: {name}"
        )

        print(
            "  Success:",
            capability.success_count,
        )

        print(
            "  Failure:",
            capability.failure_count,
        )

        print(
            "  Success rate:",
            capability.success_rate,
        )

        print(
            "  Confidence:",
            capability.confidence,
        )

        print()

    print("================================")
    print("KNOWN SUCCESSES")
    print("================================")
    print()

    for success in model.known_successes:
        print(
            "  ",
            success,
        )

    print()

    print("================================")
    print("KNOWN FAILURES")
    print("================================")
    print()

    for failure in model.known_failures:
        print(
            "  ",
            failure,
        )

    print()

    print("================================")
    print("LEARNED STRATEGIES")
    print("================================")
    print()

    for name, learned_strategy in (
        model.learned_strategies.items()
    ):

        print(
            "Strategy:",
            name,
        )

        print(
            "  Goal:",
            learned_strategy.goal,
        )

        print(
            "  Steps:",
            learned_strategy.steps,
        )

        print()

    print("================================")
    print("CURRENT CONDITION")
    print("================================")
    print()

    for key, value in (
        model.current_condition.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()

    print("================================")
    print("SUMMARY")
    print("================================")
    print()

    print(
        "Capabilities:",
        len(model.capabilities),
    )

    print(
        "Known successes:",
        len(model.known_successes),
    )

    print(
        "Known failures:",
        len(model.known_failures),
    )

    print(
        "Learned strategies:",
        len(model.learned_strategies),
    )

    print(
        "Current confidence:",
        model.confidence,
    )


if __name__ == "__main__":
    main()
