from agent.core.action import Action
from agent.learning.failure_analyzer import (
    FailureAnalyzer,
)


def test_failure_analyzer_detects_failed_precondition():

    action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {},
        preconditions={
            "box": "open",
            "key": "inside_box",
        },
    )

    analyzer = FailureAnalyzer()

    analysis = analyzer.analyze(
        action=action,
        world_state={
            "box": "closed",
            "key": "inside_box",
        },
        result={
            "success": False,
            "message": "The box is closed.",
        },
    )

    assert analysis is not None

    assert analysis.action == "take_key"

    assert (
        "box"
        in analysis.cause
    )

    assert (
        "verify"
        in analysis.lesson
    )


def test_failure_analyzer_returns_none_for_success():

    action = Action(
        name="take_key",
        description="Take key",
        handler=lambda: {},
    )

    analyzer = FailureAnalyzer()

    analysis = analyzer.analyze(
        action=action,
        world_state={},
        result={
            "success": True,
        },
    )

    assert analysis is None
