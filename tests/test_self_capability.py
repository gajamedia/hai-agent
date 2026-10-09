from agent.self.capability import Capability


def test_capability_success():

    capability = Capability(
        name="open_box",
        confidence=0.5,
    )

    capability.record_success(
        source_id="episode-1"
    )

    assert capability.success_count == 1
    assert capability.failure_count == 0
    assert capability.attempts == 1
    assert capability.success_rate == 1.0
    assert capability.confidence == 0.6


def test_capability_failure():

    capability = Capability(
        name="take_key",
        confidence=0.8,
    )

    capability.record_failure(
        source_id="episode-2"
    )

    assert capability.success_count == 0
    assert capability.failure_count == 1
    assert capability.attempts == 1
    assert capability.success_rate == 0.0
    assert capability.confidence == 0.7


def test_capability_does_not_duplicate_evidence():

    capability = Capability(
        name="open_box",
        confidence=0.5,
    )

    capability.record_success(
        source_id="episode-1"
    )

    capability.record_success(
        source_id="episode-1"
    )

    assert capability.success_count == 2
    assert capability.evidence == [
        "episode-1"
    ]
    assert capability.confidence == 0.7
