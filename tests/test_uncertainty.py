from agent.self.uncertainty import Uncertainty


def test_uncertainty_certainty():

    uncertainty = Uncertainty(
        value=0.3
    )

    assert uncertainty.value == 0.3
    assert uncertainty.certainty == 0.7


def test_uncertainty_increase():

    uncertainty = Uncertainty(
        value=0.2
    )

    uncertainty.increase(
        amount=0.3
    )

    assert uncertainty.value == 0.5


def test_uncertainty_decrease():

    uncertainty = Uncertainty(
        value=0.8
    )

    uncertainty.decrease(
        amount=0.3
    )

    assert uncertainty.value == 0.5
