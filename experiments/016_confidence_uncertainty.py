from agent.learning.strategy import Strategy
from agent.self.assessment import (
    SelfAssessmentEngine,
)
from agent.self.capability import Capability
from agent.self.model import SelfModel


def main():

    print("================================")
    print("CONFIDENCE & UNCERTAINTY")
    print("================================")
    print()

    model = SelfModel()

    engine = SelfAssessmentEngine()

    # ----------------------------------------------
    # Reliable capability
    # ----------------------------------------------

    open_box = Capability(
        name="open_box",
        confidence=0.9,
        success_count=9,
        failure_count=1,
    )

    model.register_capability(
        "open_box",
        open_box,
    )

    # ----------------------------------------------
    # Unreliable capability
    # ----------------------------------------------

    find_spare_key = Capability(
        name="find_spare_key",
        confidence=0.4,
        success_count=2,
        failure_count=3,
    )

    model.register_capability(
        "find_spare_key",
        find_spare_key,
    )

    # ----------------------------------------------
    # Assess capabilities
    # ----------------------------------------------

    print("================================")
    print("CAPABILITY ASSESSMENT")
    print("================================")
    print()

    for (
        name,
        capability,
    ) in model.capabilities.items():

        assessment = (
            engine.assess_capability(
                capability
            )
        )

        model.register_assessment(
            f"capability:{name}",
            assessment,
        )

        print(
            f"Capability: {name}"
        )

        print(
            "  Confidence:",
            assessment.confidence,
        )

        print(
            "  Uncertainty:",
            assessment.uncertainty,
        )

        print(
            "  Risk:",
            assessment.risk,
        )

        print()

    # ----------------------------------------------
    # Strategy assessment
    # ----------------------------------------------

    strategy = Strategy(
        name="box_strategy",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
        success_count=8,
        failure_count=2,
        total_actions=30,
    )

    model.register_strategy(
        strategy
    )

    strategy_assessment = (
        engine.assess_strategy(
            strategy
        )
    )

    model.register_assessment(
        "strategy:box_strategy",
        strategy_assessment,
    )

    print("================================")
    print("STRATEGY ASSESSMENT")
    print("================================")
    print()

    print(
        "Strategy:",
        strategy.name,
    )

    print(
        "  Confidence:",
        strategy_assessment.confidence,
    )

    print(
        "  Uncertainty:",
        strategy_assessment.uncertainty,
    )

    print(
        "  Risk:",
        strategy_assessment.risk,
    )

    print()

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("================================")
    print("SUMMARY")
    print("================================")
    print()

    print(
        "Capabilities:",
        len(model.capabilities),
    )

    print(
        "Assessments:",
        len(model.assessments),
    )

    print(
        "Learned strategies:",
        len(model.learned_strategies),
    )


if __name__ == "__main__":
    main()
