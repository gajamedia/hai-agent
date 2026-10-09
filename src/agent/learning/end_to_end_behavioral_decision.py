"""
End-to-end behavioral decision evaluation.

Combines:

    base decision signals
        +
    AgentBehavioralAdaptation
        +
    stabilized behavioral biases

to produce an adaptive EXPLORE / EXPLOIT mode.

This layer evaluates behavior.
It does not execute actions.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.learning.agent_behavioral_adaptation import (
    AgentBehavioralAdaptation,
    BehavioralAdaptationResult,
)


@dataclass(frozen=True)
class BehavioralDecisionResult:
    """
    Final behavioral decision evaluation.
    """

    base_explore_score: float
    base_exploit_score: float

    behavioral_explore_bias: float
    behavioral_exploit_bias: float

    final_explore_score: float
    final_exploit_score: float

    selected_mode: str

    adaptation: BehavioralAdaptationResult

    @property
    def mode_changed(self) -> bool:
        """
        Whether behavioral adaptation selected a mode
        different from the stronger base score.
        """

        base_mode = (
            "explore"
            if self.base_explore_score
            > self.base_exploit_score
            else "exploit"
        )

        return (
            self.selected_mode
            != base_mode
        )

    @property
    def exploration_selected(self) -> bool:
        return self.selected_mode == "explore"

    @property
    def exploitation_selected(self) -> bool:
        return self.selected_mode == "exploit"

    @property
    def score_difference(self) -> float:
        return round(
            self.final_explore_score
            - self.final_exploit_score,
            6,
        )

    def snapshot(self) -> dict:
        return {
            "base_explore_score": (
                self.base_explore_score
            ),
            "base_exploit_score": (
                self.base_exploit_score
            ),
            "behavioral_explore_bias": (
                self.behavioral_explore_bias
            ),
            "behavioral_exploit_bias": (
                self.behavioral_exploit_bias
            ),
            "final_explore_score": (
                self.final_explore_score
            ),
            "final_exploit_score": (
                self.final_exploit_score
            ),
            "selected_mode": (
                self.selected_mode
            ),
            "mode_changed": (
                self.mode_changed
            ),
            "score_difference": (
                self.score_difference
            ),
            "adaptation": (
                self.adaptation.snapshot()
            ),
        }


class EndToEndBehavioralDecisionEvaluator:
    """
    Evaluates the agent's final behavioral preference.

    Base score formula intentionally matches the existing
    AdaptiveDecisionPolicy convention:

        exploration =
            0.50 * information_gain
            + 0.30 * uncertainty
            - 0.20 * risk

        exploitation =
            0.50 * (1 - uncertainty)
            + 0.50 * (1 - risk)

    Behavioral adaptation then adds stabilized
    exploration/exploitation biases.

    Tie -> exploit.
    """

    def __init__(
        self,
        behavioral_adaptation: (
            AgentBehavioralAdaptation | None
        ) = None,
    ) -> None:

        self.behavioral_adaptation = (
            behavioral_adaptation
            if behavioral_adaptation is not None
            else AgentBehavioralAdaptation()
        )

    def evaluate(
        self,
        profile,
        uncertainty: float,
        information_gain: float,
        risk: float,
    ) -> BehavioralDecisionResult:

        self._validate_signal(
            uncertainty,
            "uncertainty",
        )

        self._validate_signal(
            information_gain,
            "information_gain",
        )

        self._validate_signal(
            risk,
            "risk",
        )

        adaptation = (
            self.behavioral_adaptation.evaluate(
                profile
            )
        )

        base_explore_score = (
            0.50
            * information_gain
            + 0.30
            * uncertainty
            - 0.20
            * risk
        )

        base_exploit_score = (
            0.50
            * (1.0 - uncertainty)
            + 0.50
            * (1.0 - risk)
        )

        final_explore_score = round(
            base_explore_score
            + adaptation.exploration_bias,
            6,
        )

        final_exploit_score = round(
            base_exploit_score
            + adaptation.exploitation_bias,
            6,
        )

        if (
            final_explore_score
            > final_exploit_score
        ):
            selected_mode = "explore"
        else:
            selected_mode = "exploit"

        return BehavioralDecisionResult(
            base_explore_score=round(
                base_explore_score,
                6,
            ),
            base_exploit_score=round(
                base_exploit_score,
                6,
            ),
            behavioral_explore_bias=round(
                adaptation.exploration_bias,
                6,
            ),
            behavioral_exploit_bias=round(
                adaptation.exploitation_bias,
                6,
            ),
            final_explore_score=(
                final_explore_score
            ),
            final_exploit_score=(
                final_exploit_score
            ),
            selected_mode=selected_mode,
            adaptation=adaptation,
        )

    @staticmethod
    def _validate_signal(
        value: float,
        name: str,
    ) -> None:

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )
