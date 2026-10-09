from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SelfAwarenessCalibrationResult:
    """
    Result of calibrating self-confidence against observed
    capability outcomes.
    """

    raw_confidence: float
    observed_success_rate: float
    calibrated_confidence: float
    calibration_gap: float

    evidence_strength: float
    total_attempts: int
    total_successes: int
    total_failures: int

    overconfidence: float
    underconfidence: float

    capability_count: int
    minimum_evidence: int

    @property
    def calibrated_uncertainty(self) -> float:
        return round(
            1.0 - self.calibrated_confidence,
            6,
        )

    @property
    def confidence_shift(self) -> float:
        return round(
            self.calibrated_confidence
            - self.raw_confidence,
            6,
        )

    @property
    def is_overconfident(self) -> bool:
        return self.overconfidence > 0.10

    @property
    def is_underconfident(self) -> bool:
        return self.underconfidence > 0.10

    @property
    def is_calibrated(self) -> bool:
        return (
            not self.is_overconfident
            and not self.is_underconfident
        )

    @property
    def has_evidence(self) -> bool:
        return self.total_attempts > 0

    @property
    def confidence_consistent(self) -> bool:
        return (
            abs(
                self.calibration_gap
            )
            <= 0.10
        )

    @property
    def snapshot(self) -> dict[str, Any]:
        return {
            "raw_confidence": self.raw_confidence,
            "observed_success_rate": (
                self.observed_success_rate
            ),
            "calibrated_confidence": (
                self.calibrated_confidence
            ),
            "calibrated_uncertainty": (
                self.calibrated_uncertainty
            ),
            "calibration_gap": (
                self.calibration_gap
            ),
            "confidence_shift": (
                self.confidence_shift
            ),
            "evidence_strength": (
                self.evidence_strength
            ),
            "total_attempts": (
                self.total_attempts
            ),
            "total_successes": (
                self.total_successes
            ),
            "total_failures": (
                self.total_failures
            ),
            "overconfidence": (
                self.overconfidence
            ),
            "underconfidence": (
                self.underconfidence
            ),
            "capability_count": (
                self.capability_count
            ),
            "minimum_evidence": (
                self.minimum_evidence
            ),
        }


class SelfAwarenessCalibrator:
    """
    Calibrate self-confidence against observed capability
    outcomes.

    The calibrator deliberately separates:

        evidence volume
        from
        evidence quality

    A large number of attempts does not automatically imply
    high confidence.

    Calibration uses a smoothed evidence weight:

        evidence_strength =
            attempts / (attempts + minimum_evidence)

    Then:

        calibrated_confidence =
            raw_confidence * (1 - strength)
            +
            observed_success_rate * strength

    Therefore small evidence sets preserve the original
    self-model more strongly, while larger evidence sets
    increasingly anchor confidence to actual outcomes.
    """

    def __init__(
        self,
        *,
        minimum_evidence: int = 5,
    ) -> None:

        if not isinstance(
            minimum_evidence,
            int,
        ):
            raise TypeError(
                "minimum_evidence must be an integer."
            )

        if minimum_evidence <= 0:
            raise ValueError(
                "minimum_evidence must be greater than zero."
            )

        self.minimum_evidence = (
            minimum_evidence
        )

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_probability(
        value: Any,
        name: str,
    ) -> float:

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                f"{name} must be numeric."
            )

        normalized = float(value)

        if not 0.0 <= normalized <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        return normalized

    @staticmethod
    def _normalize_evidence(
        evidence: Mapping[
            Any,
            Mapping[str, Any],
        ] | None,
    ) -> dict[
        str,
        dict[str, int],
    ]:

        if evidence is None:
            return {}

        if not isinstance(
            evidence,
            Mapping,
        ):
            raise TypeError(
                "evidence must be a mapping."
            )

        result: dict[
            str,
            dict[str, int],
        ] = {}

        for capability, record in (
            evidence.items()
        ):

            if not isinstance(
                record,
                Mapping,
            ):
                raise TypeError(
                    "capability evidence records "
                    "must be mappings."
                )

            attempts = record.get(
                "attempts",
                0,
            )

            successes = record.get(
                "successes",
                0,
            )

            failures = record.get(
                "failures",
                0,
            )

            values = {
                "attempts": attempts,
                "successes": successes,
                "failures": failures,
            }

            for key, value in values.items():

                if not isinstance(
                    value,
                    int,
                ):
                    raise TypeError(
                        f"{key} must be an integer."
                    )

                if value < 0:
                    raise ValueError(
                        f"{key} cannot be negative."
                    )

            if successes + failures > attempts:
                raise ValueError(
                    f"successes + failures cannot "
                    f"exceed attempts for {capability!r}."
                )

            normalized_name = (
                str(
                    capability
                )
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )

            while "__" in normalized_name:
                normalized_name = (
                    normalized_name.replace(
                        "__",
                        "_",
                    )
                )

            if not normalized_name:
                raise ValueError(
                    "capability cannot be empty."
                )

            result[
                normalized_name
            ] = values

        return result

    # =====================================================
    # Aggregate evidence
    # =====================================================

    @staticmethod
    def aggregate_evidence(
        evidence: Mapping[
            Any,
            Mapping[str, Any],
        ],
    ) -> tuple[
        int,
        int,
        int,
    ]:
        """
        Return:

            total_attempts
            total_successes
            total_failures
        """

        total_attempts = 0
        total_successes = 0
        total_failures = 0

        for record in evidence.values():
            total_attempts += (
                record["attempts"]
            )
            total_successes += (
                record["successes"]
            )
            total_failures += (
                record["failures"]
            )

        return (
            total_attempts,
            total_successes,
            total_failures,
        )

    # =====================================================
    # Calibration
    # =====================================================

    def calibrate(
        self,
        *,
        raw_confidence: float,
        evidence: Mapping[
            Any,
            Mapping[str, Any],
        ] | None = None,
    ) -> SelfAwarenessCalibrationResult:

        normalized_confidence = (
            self._validate_probability(
                raw_confidence,
                "raw_confidence",
            )
        )

        normalized_evidence = (
            self._normalize_evidence(
                evidence
            )
        )

        (
            total_attempts,
            total_successes,
            total_failures,
        ) = self.aggregate_evidence(
            normalized_evidence
        )

        if total_attempts <= 0:

            return (
                SelfAwarenessCalibrationResult(
                    raw_confidence=(
                        normalized_confidence
                    ),
                    observed_success_rate=(
                        normalized_confidence
                    ),
                    calibrated_confidence=(
                        normalized_confidence
                    ),
                    calibration_gap=0.0,
                    evidence_strength=0.0,
                    total_attempts=0,
                    total_successes=0,
                    total_failures=0,
                    overconfidence=0.0,
                    underconfidence=0.0,
                    capability_count=(
                        len(
                            normalized_evidence
                        )
                    ),
                    minimum_evidence=(
                        self.minimum_evidence
                    ),
                )
            )

        observed_success_rate = (
            total_successes
            / total_attempts
        )

        evidence_strength = (
            total_attempts
            / (
                total_attempts
                + self.minimum_evidence
            )
        )

        calibrated_confidence = (
            (
                normalized_confidence
                * (1.0 - evidence_strength)
            )
            +
            (
                observed_success_rate
                * evidence_strength
            )
        )

        calibrated_confidence = max(
            0.0,
            min(
                1.0,
                calibrated_confidence,
            ),
        )

        calibration_gap = (
            observed_success_rate
            - normalized_confidence
        )

        overconfidence = max(
            0.0,
            -calibration_gap,
        ) * evidence_strength

        underconfidence = max(
            0.0,
            calibration_gap,
        ) * evidence_strength

        return (
            SelfAwarenessCalibrationResult(
                raw_confidence=(
                    round(
                        normalized_confidence,
                        6,
                    )
                ),
                observed_success_rate=(
                    round(
                        observed_success_rate,
                        6,
                    )
                ),
                calibrated_confidence=(
                    round(
                        calibrated_confidence,
                        6,
                    )
                ),
                calibration_gap=(
                    round(
                        calibration_gap,
                        6,
                    )
                ),
                evidence_strength=(
                    round(
                        evidence_strength,
                        6,
                    )
                ),
                total_attempts=(
                    total_attempts
                ),
                total_successes=(
                    total_successes
                ),
                total_failures=(
                    total_failures
                ),
                overconfidence=(
                    round(
                        overconfidence,
                        6,
                    )
                ),
                underconfidence=(
                    round(
                        underconfidence,
                        6,
                    )
                ),
                capability_count=(
                    len(
                        normalized_evidence
                    )
                ),
                minimum_evidence=(
                    self.minimum_evidence
                ),
            )
        )
