import pytest

from agent.core.agent import Agent
from agent.learning.end_to_end_behavioral_decision import (
    EndToEndBehavioralDecisionEvaluator,
)


def make_explore_dominant_history(
    agent,
    count=8,
):
    for index in range(count):
        agent._record_meta_exploration(
            success=True
        )

        agent._record_meta_exploitation(
            success=False
        )


def make_exploit_dominant_history(
    agent,
    count=8,
):
    for index in range(count):
        agent._record_meta_exploration(
            success=False
        )

        agent._record_meta_exploitation(
            success=True
        )


def test_empty_history_uses_base_decision():
    agent = Agent(
        name="agent-empty-history"
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    assert result is not None

    assert (
        result.behavioral_explore_bias
        == 0.0
    )

    assert (
        result.behavioral_exploit_bias
        == 0.0
    )


def test_explore_dominant_history_produces_positive_explore_bias():
    agent = Agent(
        name="agent-explore-dominant"
    )

    make_explore_dominant_history(
        agent
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.8,
            information_gain=0.5,
            risk=0.5,
        )
    )

    assert (
        result.behavioral_explore_bias
        > result.behavioral_exploit_bias
    )


def test_exploit_dominant_history_produces_positive_exploit_bias():
    agent = Agent(
        name="agent-exploit-dominant"
    )

    make_exploit_dominant_history(
        agent
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.2,
            information_gain=0.5,
            risk=0.5,
        )
    )

    assert (
        result.behavioral_exploit_bias
        > result.behavioral_explore_bias
    )

def test_same_current_signals_can_produce_different_behavior():
    explore_agent = Agent(
        name="agent-history-explore"
    )

    exploit_agent = Agent(
        name="agent-history-exploit"
    )

    make_explore_dominant_history(
        explore_agent
    )

    make_exploit_dominant_history(
        exploit_agent
    )

    # The current signals are identical for both agents.
    #
    # They are intentionally chosen so that the base
    # decision is close enough for learned behavioral
    # evidence to influence the final mode.
    explore_result = (
        explore_agent.evaluate_end_to_end_behavior(
            uncertainty=0.8,
            information_gain=0.9,
            risk=0.2,
        )
    )

    exploit_result = (
        exploit_agent.evaluate_end_to_end_behavior(
            uncertainty=0.8,
            information_gain=0.9,
            risk=0.2,
        )
    )

    assert (
        explore_result.selected_mode
        != exploit_result.selected_mode
    )

    assert (
        explore_result.selected_mode
        == "explore"
    )

    assert (
        exploit_result.selected_mode
        == "exploit"
    )

def test_behavioral_history_can_overcome_small_base_score_difference():
    agent = Agent(
        name="agent-behavior-overcome"
    )

    make_explore_dominant_history(
        agent
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.9,
            information_gain=0.9,
            risk=0.1,
        )
    )

    assert (
        result.selected_mode
        == "explore"
    )


def test_exploitation_history_can_reduce_exploration_preference():
    agent = Agent(
        name="agent-reduce-explore"
    )

    make_exploit_dominant_history(
        agent
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.1,
            information_gain=0.4,
            risk=0.2,
        )
    )

    assert (
        result.behavioral_explore_bias
        <= 0.0
    )


def test_behavioral_bias_remains_bounded():
    agent = Agent(
        name="agent-bounded"
    )

    make_explore_dominant_history(
        agent,
        count=100,
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.5,
        )
    )

    assert (
        -0.25
        <= result.behavioral_explore_bias
        <= 0.25
    )

    assert (
        -0.25
        <= result.behavioral_exploit_bias
        <= 0.25
    )


def test_behavioral_result_contains_final_scores():
    agent = Agent(
        name="agent-final-score"
    )

    make_explore_dominant_history(
        agent
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    assert (
        result.final_explore_score
        ==
        round(
            result.base_explore_score
            + result.behavioral_explore_bias,
            6,
        )
    )

    assert (
        result.final_exploit_score
        ==
        round(
            result.base_exploit_score
            + result.behavioral_exploit_bias,
            6,
        )
    )


def test_snapshot_contains_behavioral_decision():
    agent = Agent(
        name="agent-snapshot"
    )

    make_explore_dominant_history(
        agent
    )

    result = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    snapshot = result.snapshot()

    assert (
        "base_explore_score"
        in snapshot
    )

    assert (
        "base_exploit_score"
        in snapshot
    )

    assert (
        "behavioral_explore_bias"
        in snapshot
    )

    assert (
        "behavioral_exploit_bias"
        in snapshot
    )

    assert (
        "final_explore_score"
        in snapshot
    )

    assert (
        "final_exploit_score"
        in snapshot
    )

    assert (
        "selected_mode"
        in snapshot
    )


def test_evaluator_can_be_used_standalone():
    agent = Agent(
        name="agent-standalone"
    )

    make_explore_dominant_history(
        agent
    )

    evaluator = (
        EndToEndBehavioralDecisionEvaluator()
    )

    result = evaluator.evaluate(
        profile=(
            agent.meta_learning_profile
        ),
        uncertainty=0.5,
        information_gain=0.5,
        risk=0.2,
    )

    assert result is not None

    assert (
        result.selected_mode
        in {
            "explore",
            "exploit",
        }
    )


def test_tie_prefers_exploit():
    evaluator = (
        EndToEndBehavioralDecisionEvaluator()
    )

    from agent.learning.meta_learning import (
        MetaLearningProfile,
    )

    result = evaluator.evaluate(
        profile=MetaLearningProfile(),
        uncertainty=0.0,
        information_gain=0.5,
        risk=0.0,
    )

    # Base scores:
    #
    # explore = 0.25
    # exploit = 1.0
    #
    # Therefore exploit wins.
    assert (
        result.selected_mode
        == "exploit"
    )


def test_invalid_uncertainty_is_rejected():
    evaluator = (
        EndToEndBehavioralDecisionEvaluator()
    )

    from agent.learning.meta_learning import (
        MetaLearningProfile,
    )

    with pytest.raises(
        ValueError,
        match="uncertainty",
    ):
        evaluator.evaluate(
            profile=MetaLearningProfile(),
            uncertainty=-0.1,
            information_gain=0.5,
            risk=0.2,
        )

    with pytest.raises(
        ValueError,
        match="uncertainty",
    ):
        evaluator.evaluate(
            profile=MetaLearningProfile(),
            uncertainty=1.1,
            information_gain=0.5,
            risk=0.2,
        )


def test_invalid_information_gain_is_rejected():
    evaluator = (
        EndToEndBehavioralDecisionEvaluator()
    )

    from agent.learning.meta_learning import (
        MetaLearningProfile,
    )

    with pytest.raises(
        ValueError,
        match="information_gain",
    ):
        evaluator.evaluate(
            profile=MetaLearningProfile(),
            uncertainty=0.5,
            information_gain=-0.1,
            risk=0.2,
        )

    with pytest.raises(
        ValueError,
        match="information_gain",
    ):
        evaluator.evaluate(
            profile=MetaLearningProfile(),
            uncertainty=0.5,
            information_gain=1.1,
            risk=0.2,
        )


def test_invalid_risk_is_rejected():
    evaluator = (
        EndToEndBehavioralDecisionEvaluator()
    )

    from agent.learning.meta_learning import (
        MetaLearningProfile,
    )

    with pytest.raises(
        ValueError,
        match="risk",
    ):
        evaluator.evaluate(
            profile=MetaLearningProfile(),
            uncertainty=0.5,
            information_gain=0.5,
            risk=-0.1,
        )

    with pytest.raises(
        ValueError,
        match="risk",
    ):
        evaluator.evaluate(
            profile=MetaLearningProfile(),
            uncertainty=0.5,
            information_gain=0.5,
            risk=1.1,
        )


def test_none_behavioral_profile_is_rejected():
    evaluator = (
        EndToEndBehavioralDecisionEvaluator()
    )

    with pytest.raises(
        ValueError,
        match="profile cannot be None",
    ):
        evaluator.evaluate(
            profile=None,
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )


def test_behavioral_adaptation_is_stable_across_repeated_evaluations():
    agent = Agent(
        name="agent-repeat"
    )

    make_explore_dominant_history(
        agent
    )

    first = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    second = (
        agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    assert (
        first.selected_mode
        == second.selected_mode
    )

    assert (
        first.final_explore_score
        == second.final_explore_score
    )

    assert (
        first.final_exploit_score
        == second.final_exploit_score
    )


def test_different_history_changes_score_difference():
    explore_agent = Agent(
        name="agent-score-explore"
    )

    exploit_agent = Agent(
        name="agent-score-exploit"
    )

    make_explore_dominant_history(
        explore_agent
    )

    make_exploit_dominant_history(
        exploit_agent
    )

    explore_result = (
        explore_agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    exploit_result = (
        exploit_agent.evaluate_end_to_end_behavior(
            uncertainty=0.5,
            information_gain=0.5,
            risk=0.2,
        )
    )

    assert (
        explore_result.score_difference
        >
        exploit_result.score_difference
    )
