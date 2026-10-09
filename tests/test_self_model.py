from agent.learning.strategy import Strategy
from agent.self.capability import Capability
from agent.self.model import SelfModel
from agent.self.assessment import (
    SelfAssessmentEngine,
)


def test_self_model_capability():

    model = SelfModel()

    capability = Capability(
        name="open_box"
    )

    model.register_capability(
        name="open_box",
        capability=capability,
    )

    assert (
        model.get_capability(
            "open_box"
        )
        is capability
    )


def test_self_model_limitation():

    model = SelfModel()

    model.add_limitation(
        "find_spare_key requires an available spare key."
    )

    model.add_limitation(
        "find_spare_key requires an available spare key."
    )

    assert len(model.limitations) == 1


def test_self_model_records_outcomes():

    model = SelfModel()

    model.record_success(
        "Achieved goal: Open the door"
    )

    model.record_failure(
        "Failed goal: Open the door"
    )

    assert len(
        model.known_successes
    ) == 1

    assert len(
        model.known_failures
    ) == 1


def test_self_model_registers_strategy():

    model = SelfModel()

    strategy = Strategy(
        name="box_strategy",
        goal="Open the door",
        steps=[
            "open_box",
            "take_key",
            "unlock_door",
        ],
    )

    model.register_strategy(
        strategy
    )

    assert (
        model.learned_strategies[
            "box_strategy"
        ]
        is strategy
    )

def test_self_model_assessment():

    model = SelfModel()

    capability = Capability(
        name="open_box",
        confidence=0.8,
        success_count=8,
        failure_count=2,
    )

    model.register_capability(
        "open_box",
        capability,
    )

    engine = SelfAssessmentEngine()

    assessment = (
        engine.assess_capability(
            capability
        )
    )

    model.register_assessment(
        "capability:open_box",
        assessment,
    )

    stored = model.get_assessment(
        "capability:open_box"
    )

    assert stored is assessment
    assert stored.confidence == 0.8
    assert stored.uncertainty == 0.2
