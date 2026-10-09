from agent.self.confidence import ConfidenceScore


def test_confidence_uncertainty():

    score = ConfidenceScore(
        confidence=0.7,
        evidence_count=3,
    )

    assert score.confidence == 0.7
    assert score.uncertainty == 0.3


def test_confidence_reinforcement():

    score = ConfidenceScore(
        confidence=0.5,
    )

    score.reinforce(
        amount=0.2
    )

    assert score.confidence == 0.7
    assert score.evidence_count == 1


def test_confidence_weakening():

    score = ConfidenceScore(
        confidence=0.8,
    )

    score.weaken(
        amount=0.3
    )

    assert score.confidence == 0.5
    assert score.evidence_count == 1
