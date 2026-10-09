from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agent.core.agent import Agent


class DummyGoal:
    def __init__(
        self,
        description="calibrated persistence integration",
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


def build_agent(
    tmp_path: Path,
    name: str = "calibrated-persistent-agent",
):
    database_path = tmp_path / "agent.sqlite3"

    agent = Agent(
        name=name,
        memory_path=database_path,
    )

    agent.unified_executor = Mock()

    agent.unified_executor.execute.return_value = (
        SimpleNamespace(
            exploration_result=None,
        )
    )

    return agent


def prepare_calibrated_agent(agent):
    # ---------------------------------------------------------
    # Provide explicit capability telemetry first.
    #
    # calibrate_self_awareness() builds unified evidence from:
    #
    #     Explicit Telemetry
    #             +
    #     Experience Evidence
    #
    # Without evidence, calibration legitimately returns None.
    # ---------------------------------------------------------

    for _ in range(4):
        agent.record_capability_telemetry(
            "execution",
            success=True,
            source="test",
        )

    agent.record_capability_telemetry(
        "execution",
        success=False,
        source="test",
    )

    for _ in range(4):
        agent.record_capability_telemetry(
            "decision_making",
            success=True,
            source="test",
        )

    agent.record_capability_telemetry(
        "decision_making",
        success=False,
        source="test",
    )

    # ---------------------------------------------------------
    # Evaluate self-awareness from the same evidence profile.
    # ---------------------------------------------------------

    snapshot = agent.evaluate_self_awareness(
        {
            "execution": {
                "attempts": 5,
                "successes": 4,
                "failures": 1,
            },
            "decision_making": {
                "attempts": 5,
                "successes": 4,
                "failures": 1,
            },
        },
        persist=True,
    )

    assert snapshot is not None

    # ---------------------------------------------------------
    # Calibration now has unified evidence available.
    # ---------------------------------------------------------

    calibration = agent.calibrate_self_awareness()

    assert calibration is not None

    influence = (
        agent.get_calibrated_self_aware_influence()
    )

    assert influence is not None

    result = (
        agent.get_calibrated_self_aware_result()
    )

    assert result is not None

    return calibration, influence, result


def test_persistent_agent_creates_calibration_store(
    tmp_path,
):
    agent = build_agent(tmp_path)

    assert agent.is_persistent

    assert (
        agent.calibrated_self_awareness_store
        is not None
    )


def test_calibration_is_persisted(
    tmp_path,
):
    agent = build_agent(tmp_path)

    _, influence, result = (
        prepare_calibrated_agent(agent)
    )

    persisted = (
        agent.get_calibrated_self_awareness_persistence()
    )

    assert persisted is not None

    assert persisted.influence == influence
    assert persisted.calibration_result == result


def test_calibrated_influence_survives_new_agent_instance(
    tmp_path,
):
    agent1 = build_agent(tmp_path)

    _, influence1, result1 = (
        prepare_calibrated_agent(agent1)
    )

    agent2 = build_agent(tmp_path)

    assert (
        agent2.get_calibrated_self_aware_influence()
        is None
    )

    restored = (
        agent2.restore_calibrated_self_awareness()
    )

    assert restored is not None

    assert restored.influence == influence1

    assert (
        restored.calibration_result
        == result1
    )

    assert (
        agent2.get_calibrated_self_aware_influence()
        == restored.influence
    )


def test_restore_exposes_calibration_result(
    tmp_path,
):
    agent1 = build_agent(tmp_path)

    _, influence, result = (
        prepare_calibrated_agent(agent1)
    )

    agent2 = build_agent(tmp_path)

    restored = (
        agent2.restore_calibrated_self_awareness()
    )

    assert restored is not None

    exposed_influence = (
        agent2.get_calibrated_self_aware_influence()
    )

    exposed_result = (
        agent2.get_calibrated_self_aware_result()
    )

    assert exposed_influence == influence
    assert exposed_result == result

    assert (
        restored.calibration_result
        is exposed_result
    )


def test_explicit_calibration_save_updates_persisted_state(
    tmp_path,
):
    agent = build_agent(tmp_path)

    prepare_calibrated_agent(agent)

    first_snapshot = (
        agent.get_calibrated_self_awareness_persistence()
    )

    assert first_snapshot is not None

    first_version = first_snapshot.version

    agent.calibrate_self_awareness()

    second_snapshot = (
        agent.get_calibrated_self_awareness_persistence()
    )

    assert second_snapshot is not None

    assert (
        second_snapshot.version
        > first_version
    )


def test_persistence_is_independent_from_existing_self_model_store(
    tmp_path,
):
    agent = build_agent(tmp_path)

    assert agent.persistent_memory is not None
    assert (
        agent.persistent_memory.self_model
        is not None
    )

    assert (
        agent.calibrated_self_awareness_store
        is not agent.persistent_memory.self_model
    )


def test_restored_calibrated_influence_can_be_used_by_arbiter(
    tmp_path,
):
    agent1 = build_agent(tmp_path)

    _, influence, _ = (
        prepare_calibrated_agent(agent1)
    )

    agent2 = build_agent(tmp_path)

    restored = (
        agent2.restore_calibrated_self_awareness()
    )

    assert restored is not None

    restored_influence = (
        agent2.get_calibrated_self_aware_influence()
    )

    assert restored_influence is not None
    assert restored_influence == influence

    captured = {}

    original_evaluate = (
        agent2.self_aware_decision_arbiter
        .evaluate
    )

    def capture_evaluate(
        decision,
        influence_value,
    ):
        captured["influence"] = (
            influence_value
        )

        return original_evaluate(
            decision,
            influence_value,
        )

    agent2.self_aware_decision_arbiter.evaluate = (
        capture_evaluate
    )

    goal = DummyGoal()

    agent2.decide_and_act(
        goal=goal,
        world_state={},
    )

    assert "influence" in captured

    assert (
        captured["influence"]
        == restored_influence
    )
