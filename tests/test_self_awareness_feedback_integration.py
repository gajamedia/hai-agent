from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent


class DummyGoal:
    def __init__(
        self,
        description="feedback loop",
        target_state=None,
    ):
        self.description = description
        self.target_state = (
            target_state
            if target_state is not None
            else {}
        )
        self.completed = False

    def is_satisfied(
        self,
        world_state,
    ):
        if not self.target_state:
            return False

        return all(
            world_state.get(key) == value
            for key, value in self.target_state.items()
        )

    def complete(self):
        self.completed = True


def build_agent():
    agent = Agent(
        name="self-awareness-feedback-test"
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None
        )
    )

    return agent


def test_agent_initializes_feedback_trace():
    agent = build_agent()

    assert (
        agent.self_awareness_feedback
        is not None
    )

    snapshot = (
        agent.get_self_awareness_feedback_snapshot()
    )

    assert snapshot.count == 0


def test_feedback_trace_is_recorded_after_decision():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    snapshot = (
        agent.get_self_awareness_feedback_snapshot()
    )

    assert snapshot.count == 1
    assert snapshot.has_feedback
    assert snapshot.latest is not None


def test_feedback_contains_self_model_state():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    record = (
        agent.self_awareness_feedback.latest()
    )

    assert record is not None

    assert (
        0.0
        <= record.uncertainty
        <= 1.0
    )

    assert (
        0.0
        <= record.confidence
        <= 1.0
    )

    assert (
        record.evidence_count
        > 0
    )


def test_multiple_decisions_create_feedback_history():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    snapshot = (
        agent.get_self_awareness_feedback_snapshot()
    )

    assert snapshot.count == 3


def test_successive_successes_reduce_self_uncertainty():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    for _ in range(4):
        agent.decide_and_act(
            goal=goal,
            world_state={
                "done": True,
            },
        )

    snapshot = (
        agent.get_self_awareness_feedback_snapshot()
    )

    assert snapshot.count == 4

    assert (
        snapshot.latest.uncertainty
        < snapshot.first.uncertainty
    )

    assert (
        snapshot.latest.confidence
        > snapshot.first.confidence
    )


def test_successive_failures_keep_self_uncertainty_high():
    agent = build_agent()

    goal = DummyGoal(
        target_state={
            "done": True,
        }
    )

    for _ in range(4):
        agent.decide_and_act(
            goal=goal,
            world_state={
                "done": False,
            },
        )

    snapshot = (
        agent.get_self_awareness_feedback_snapshot()
    )

    assert snapshot.count == 4

    # Repeated failures must not create false high confidence.
    # Current calibration still allows uncertainty to decrease
    # somewhat as evidence accumulates, but it must remain high.
    assert (
        snapshot.latest.uncertainty
        >= 0.70
    )

    assert (
        snapshot.latest.confidence
        <= 0.30
    )


def test_feedback_evidence_grows_across_cycles():
    agent = build_agent()

    goal = DummyGoal()

    for _ in range(3):
        agent.decide_and_act(
            goal=goal,
            world_state={},
        )

    snapshot = (
        agent.get_self_awareness_feedback_snapshot()
    )

    assert snapshot.count == 3

    assert (
        snapshot.records[0].evidence_count
        <= snapshot.records[-1].evidence_count
    )


def test_feedback_records_selected_mode_after_self_awareness_exists():
    agent = build_agent()

    goal = DummyGoal()

    # First cycle establishes self-awareness from outcome.
    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    first = (
        agent.self_awareness_feedback.latest()
    )

    assert first is not None

    # Causal behavior:
    # first cycle has no pre-existing self-awareness,
    # therefore no selected_mode is expected.
    assert (
        first.selected_mode
        is None
    )

    # Second cycle can now use self-awareness created by
    # the first cycle.
    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    second = (
        agent.self_awareness_feedback.latest()
    )

    assert second is not None

    assert (
        second.selected_mode
        is not None
    )


def test_feedback_trace_matches_latest_arbitration():
    agent = build_agent()

    goal = DummyGoal()

    # Cycle 1 creates the self-awareness state.
    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert (
        agent.get_self_aware_decision_result()
        is None
    )

    # Cycle 2 uses the self-awareness state from cycle 1.
    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    record = (
        agent.self_awareness_feedback.latest()
    )

    arbitration = (
        agent.get_self_aware_decision_result()
    )

    assert record is not None
    assert arbitration is not None

    selected_mode = (
        arbitration.selected_mode
    )

    if hasattr(
        selected_mode,
        "value",
    ):
        selected_mode = (
            selected_mode.value
        )

    assert (
        record.selected_mode
        == str(selected_mode)
    )

    assert (
        record.mode_changed
        == arbitration.mode_changed
    )


def test_feedback_trace_can_be_reset():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert (
        agent.get_self_awareness_feedback_snapshot()
        .count
        == 1
    )

    agent.reset_self_awareness_feedback()

    assert (
        agent.get_self_awareness_feedback_snapshot()
        .count
        == 0
    )


def test_feedback_trace_is_independent_from_telemetry():
    agent = build_agent()

    goal = DummyGoal()

    agent.decide_and_act(
        goal=goal,
        world_state={},
    )

    feedback_count = (
        agent.get_self_awareness_feedback_snapshot()
        .count
    )

    telemetry_count = (
        agent.get_capability_telemetry_snapshot()
        .event_count
    )

    assert feedback_count == 1
    assert telemetry_count > feedback_count


def test_feedback_trace_is_observational_only():
    agent = build_agent()

    before = (
        agent.get_self_awareness_snapshot()
    )

    agent.record_self_awareness_feedback()

    after = (
        agent.get_self_awareness_snapshot()
    )

    assert before == after
