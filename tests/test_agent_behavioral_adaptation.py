import pytest

from agent.core.agent import Agent
from agent.learning.agent_behavioral_adaptation import (
    AgentBehavioralAdaptation,
)
from agent.learning.meta_learning import (
    MetaLearningProfile,
)


def populate_explore_success(
    agent,
    count=5,
):
    for index in range(count):
        agent._record_meta_exploration(
            success=True
        )


def populate_exploit_success(
    agent,
    count=5,
):
    for index in range(count):
        agent._record_meta_exploitation(
            success=True
        )


def test_agent_creates_behavioral_adaptation():
    agent = Agent(
        name="agent-behavioral"
    )

    assert (
        agent.behavioral_adaptation
        is not None
    )

    assert isinstance(
        agent.behavioral_adaptation,
        AgentBehavioralAdaptation,
    )


def test_empty_profile_has_neutral_behavior():
    agent = Agent(
        name="agent-neutral"
    )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.preferred_mode
        is None
    )

    assert (
        result.net_bias
        == 0.0
    )


def test_exploration_history_produces_positive_exploration_bias():
    agent = Agent(
        name="agent-explore-bias"
    )

    populate_explore_success(
        agent,
        count=8,
    )

    # Add comparison evidence.
    for index in range(8):
        agent._record_meta_exploitation(
            success=False
        )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.exploration.bias
        > 0.0
    )


def test_exploitation_history_produces_positive_exploitation_bias():
    agent = Agent(
        name="agent-exploit-bias"
    )

    for index in range(8):
        agent._record_meta_exploration(
            success=False
        )

    populate_exploit_success(
        agent,
        count=8,
    )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.exploitation.bias
        > 0.0
    )


def test_exploration_failure_history_produces_negative_bias():
    agent = Agent(
        name="agent-explore-failure"
    )

    for index in range(8):
        agent._record_meta_exploration(
            success=False
        )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.exploration.bias
        < 0.0
    )


def test_exploitation_failure_history_produces_negative_bias():
    agent = Agent(
        name="agent-exploit-failure"
    )

    for index in range(8):
        agent._record_meta_exploitation(
            success=False
        )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.exploitation.bias
        < 0.0
    )


def test_behavioral_adaptation_combines_both_modes():
    agent = Agent(
        name="agent-combined"
    )

    populate_explore_success(
        agent,
        count=8,
    )

    for index in range(8):
        agent._record_meta_exploitation(
            success=False
        )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.exploration_bias
        > result.exploitation_bias
    )

    assert (
        result.net_bias
        > 0.0
    )

    assert (
        result.preferred_mode
        == "explore"
    )


def test_exploitation_can_be_behaviorally_preferred():
    agent = Agent(
        name="agent-exploitation-preferred"
    )

    for index in range(8):
        agent._record_meta_exploration(
            success=False
        )

    populate_exploit_success(
        agent,
        count=8,
    )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.net_bias
        < 0.0
    )

    assert (
        result.preferred_mode
        == "exploit"
    )


def test_behavioral_adaptation_is_stateful():
    agent = Agent(
        name="agent-stateful"
    )

    populate_explore_success(
        agent,
        count=8,
    )

    for index in range(8):
        agent._record_meta_exploitation(
            success=False
        )

    first = (
        agent.evaluate_behavioral_adaptation()
    )

    second = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        second.exploration_bias
        == first.exploration_bias
    )

    assert (
        second.exploitation_bias
        == first.exploitation_bias
    )


def test_behavioral_adaptation_snapshot():
    agent = Agent(
        name="agent-snapshot"
    )

    populate_explore_success(
        agent,
        count=8,
    )

    for index in range(8):
        agent._record_meta_exploitation(
            success=False
        )

    agent.evaluate_behavioral_adaptation()

    snapshot = (
        agent.get_behavioral_adaptation_snapshot()
    )

    assert isinstance(
        snapshot,
        dict,
    )

    assert (
        "previous_exploration_bias"
        in snapshot
    )

    assert (
        "previous_exploitation_bias"
        in snapshot
    )

    assert (
        "net_bias"
        in snapshot
    )


def test_reset_behavioral_adaptation_resets_bias_state():
    agent = Agent(
        name="agent-reset"
    )

    populate_explore_success(
        agent,
        count=8,
    )

    for index in range(8):
        agent._record_meta_exploitation(
            success=False
        )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        result.net_bias
        != 0.0
    )

    agent.reset_behavioral_adaptation()

    snapshot = (
        agent.get_behavioral_adaptation_snapshot()
    )

    assert (
        snapshot["previous_exploration_bias"]
        == 0.0
    )

    assert (
        snapshot["previous_exploitation_bias"]
        == 0.0
    )

    assert (
        snapshot["net_bias"]
        == 0.0
    )


def test_reset_does_not_delete_meta_learning_history():
    agent = Agent(
        name="agent-reset-history"
    )

    populate_explore_success(
        agent,
        count=5,
    )

    before = (
        agent.meta_learning_profile
        .exploration_count
    )

    agent.evaluate_behavioral_adaptation()

    agent.reset_behavioral_adaptation()

    after = (
        agent.meta_learning_profile
        .exploration_count
    )

    assert before == after
    assert after == 5


def test_behavioral_adaptation_does_not_record_new_learning_event():
    agent = Agent(
        name="agent-no-side-effect"
    )

    populate_explore_success(
        agent,
        count=5,
    )

    before = (
        agent.meta_learning_profile
        .total_learning_events
    )

    agent.evaluate_behavioral_adaptation()

    after = (
        agent.meta_learning_profile
        .total_learning_events
    )

    assert before == after


def test_profile_can_be_supplied_directly():
    profile = (
        MetaLearningProfile()
    )

    for index in range(5):
        profile.record_exploration(
            success=True,
            event_id=f"explore-{index}",
        )

    for index in range(5):
        profile.record_exploitation(
            success=False,
            event_id=f"exploit-{index}",
        )

    subsystem = (
        AgentBehavioralAdaptation()
    )

    result = subsystem.evaluate(
        profile
    )

    assert (
        result.preferred_mode
        == "explore"
    )


def test_none_profile_is_rejected():
    subsystem = (
        AgentBehavioralAdaptation()
    )

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        subsystem.evaluate(
            None
        )


def test_first_evaluation_is_limited_by_stability():
    agent = Agent(
        name="agent-stability"
    )

    populate_explore_success(
        agent,
        count=8,
    )

    first = (
        agent.evaluate_behavioral_adaptation()
    )

    second = (
        agent.evaluate_behavioral_adaptation()
    )

    assert (
        abs(
            second.exploration_bias
            - first.exploration_bias
        )
        <= 0.10
    )


def test_behavioral_result_snapshot_contains_all_components():
    agent = Agent(
        name="agent-full-snapshot"
    )

    populate_explore_success(
        agent,
        count=6,
    )

    for index in range(6):
        agent._record_meta_exploitation(
            success=False
        )

    result = (
        agent.evaluate_behavioral_adaptation()
    )

    snapshot = result.snapshot()

    assert "evidence" in snapshot
    assert "exploration_bias" in snapshot
    assert "exploitation_bias" in snapshot
    assert "net_bias" in snapshot
    assert "preferred_mode" in snapshot
    assert "adapted" in snapshot
    assert "stable" in snapshot
