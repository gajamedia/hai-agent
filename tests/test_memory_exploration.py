from agent.cognition.memory_exploration import (
    MemoryAwareExploration,
)
from agent.memory.procedural import (
    ProceduralMemory,
)
from agent.memory.causal import (
    CausalMemory,
)


def create_procedural_memory():

    return ProceduralMemory(
        content="Open door procedure.",
        skill="open_the_door",
        steps=[
            "take_key",
            "unlock_door",
        ],
    )


def create_causal_memory():

    return CausalMemory(
        content=(
            "Taking the key enables "
            "unlocking the door."
        ),
        cause="take_key",
        relation="enables",
        effect="unlock_door",
        condition={},
    )


def test_empty_memory_is_exploration_candidate():

    explorer = MemoryAwareExploration()

    decision = explorer.decide(
        risk=0.2
    )

    assert decision.explore is True


def test_strong_procedural_memory_prefers_exploitation():

    memory = (
        create_procedural_memory()
    )

    for index in range(20):
        memory.record_success(
            [f"experience-{index}"]
        )

    explorer = MemoryAwareExploration()

    assessment = explorer.assess(
        procedural_memory=memory,
        risk=0.1,
    )

    assert assessment.uncertainty < 0.6

    assert (
        assessment.expected_information_gain
        < 0.6
    )

    assert assessment.decision.explore is False


def test_weak_procedural_memory_can_trigger_exploration():

    memory = (
        create_procedural_memory()
    )

    explorer = MemoryAwareExploration()

    assessment = explorer.assess(
        procedural_memory=memory,
        risk=0.2,
    )

    assert assessment.evidence == 0

    assert assessment.uncertainty == 1.0

    assert assessment.decision.explore is True


def test_causal_contradiction_increases_exploration_pressure():

    memory = (
        create_causal_memory()
    )

    memory.reinforce(
        ["episode-1"]
    )

    explorer = MemoryAwareExploration()

    stable = explorer.assess(
        causal_memory=memory,
        risk=0.2,
    )

    for index in range(5):
        memory.contradict()

    unstable = explorer.assess(
        causal_memory=memory,
        risk=0.2,
    )

    assert (
        unstable.uncertainty
        > stable.uncertainty
    )

    assert (
        unstable.expected_information_gain
        > stable.expected_information_gain
    )


def test_highest_memory_uncertainty_is_used():

    procedural = (
        create_procedural_memory()
    )

    for index in range(20):
        procedural.record_success(
            [f"procedure-{index}"]
        )

    causal = (
        create_causal_memory()
    )

    causal.reinforce(
        ["episode-1"]
    )

    explorer = MemoryAwareExploration()

    assessment = explorer.assess(
        procedural_memory=procedural,
        causal_memory=causal,
        risk=0.2,
    )

    causal_only = explorer.assess(
        causal_memory=causal,
        risk=0.2,
    )

    assert (
        assessment.uncertainty
        == causal_only.uncertainty
    )


def test_high_risk_blocks_memory_driven_exploration():

    explorer = MemoryAwareExploration()

    decision = explorer.decide(
        risk=0.95
    )

    assert decision.explore is False

    assert (
        decision.reason
        == "Exploration risk is too high."
    )
