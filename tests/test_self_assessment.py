from agent.learning.strategy import Strategy
from agent.self.assessment import (
    SelfAssessmentEngine,
)
from agent.self.capability import Capability


def test_capability_assessment():

    capability = Capability(
        name="open_box",
        confidence=0.8,
        success_count=8,
        failure_count=2,
    )

    engine = SelfAssessmentEngine()

    assessment = (
        engine.assess_capability(
            capability
        )
    )

    assert assessment.confidence == 0.8
    assert assessment.uncertainty == 0.2
    assert assessment.risk == 0.2


def test_strategy_assessment():

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

    engine = SelfAssessmentEngine()

    assessment = (
        engine.assess_strategy(
            strategy
        )
    )

    assert assessment.confidence == 0.8
    assert assessment.uncertainty == 0.2
    assert assessment.risk == 0.2


def test_untried_strategy_has_high_risk():

    strategy = Strategy(
        name="unknown_strategy",
        goal="Open the door",
        steps=[
            "unknown_action",
        ],
    )

    engine = SelfAssessmentEngine()

    assessment = (
        engine.assess_strategy(
            strategy
        )
    )

    assert assessment.confidence == 0.0
    assert assessment.uncertainty == 1.0
    assert assessment.risk == 1.0
