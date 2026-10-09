"""
Agent behavioral adaptation integration.

Coordinates:

    MetaLearningProfile
          ↓
    BehavioralEvidence
          ↓
    ExplorationBias
          ↓
    ExploitationBias
          ↓
    AdaptationStabilityController
          ↓
    stabilized behavioral state

This module does not execute actions and does not directly
replace the agent's decision maker.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent.learning.behavioral_evidence import (
    BehavioralEvidence,
    BehavioralEvidenceModel,
)
from agent.learning.exploration_bias import (
    ExplorationBias,
    ExplorationBiasAdapter,
)
from agent.learning.exploitation_bias import (
    ExploitationBias,
    ExploitationBiasAdapter,
)
from agent.learning.adaptation_stability import (
    AdaptationStabilityController,
    StabilityResult,
)


@dataclass(frozen=True)
class BehavioralAdaptationResult:
    """
    Combined behavioral adaptation result.
    """

    evidence: BehavioralEvidence

    exploration: ExplorationBias
    exploitation: ExploitationBias

    exploration_stability: StabilityResult
    exploitation_stability: StabilityResult

    exploration_bias: float
    exploitation_bias: float

    net_bias: float

    @property
    def adapted(self) -> bool:
        return (
            self.exploration_stability.adapted
            or self.exploitation_stability.adapted
        )

    @property
    def stable(self) -> bool:
        return not self.adapted

    @property
    def preferred_mode(self) -> str | None:
        if self.net_bias > 0.0:
            return "explore"

        if self.net_bias < 0.0:
            return "exploit"

        return None

    @property
    def exploration_favored(self) -> bool:
        return self.net_bias > 0.0

    @property
    def exploitation_favored(self) -> bool:
        return self.net_bias < 0.0

    def snapshot(self) -> dict:
        return {
            "evidence": self.evidence.snapshot(),
            "exploration_bias": {
                "raw": self.exploration.bias,
                "stabilized": (
                    self.exploration_bias
                ),
            },
            "exploitation_bias": {
                "raw": self.exploitation.bias,
                "stabilized": (
                    self.exploitation_bias
                ),
            },
            "net_bias": self.net_bias,
            "preferred_mode": self.preferred_mode,
            "adapted": self.adapted,
            "stable": self.stable,
        }


class AgentBehavioralAdaptation:
    """
    Behavioral adaptation subsystem owned by an Agent.

    The subsystem maintains the previous stabilized bias
    for both decision modes.

    It is intentionally stateful because hysteresis and
    step-limiting require knowledge of the previous state.
    """

    def __init__(
        self,
        evidence_model: BehavioralEvidenceModel | None = None,
        exploration_adapter: ExplorationBiasAdapter | None = None,
        exploitation_adapter: ExploitationBiasAdapter | None = None,
        stability_controller: (
            AdaptationStabilityController | None
        ) = None,
    ) -> None:

        self.evidence_model = (
            evidence_model
            if evidence_model is not None
            else BehavioralEvidenceModel()
        )

        self.exploration_adapter = (
            exploration_adapter
            if exploration_adapter is not None
            else ExplorationBiasAdapter()
        )

        self.exploitation_adapter = (
            exploitation_adapter
            if exploitation_adapter is not None
            else ExploitationBiasAdapter()
        )

        self.stability_controller = (
            stability_controller
            if stability_controller is not None
            else AdaptationStabilityController()
        )

        self.previous_exploration_bias = 0.0
        self.previous_exploitation_bias = 0.0

    def evaluate(
        self,
        profile,
    ) -> BehavioralAdaptationResult:

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        evidence = (
            self.evidence_model.analyze(
                profile
            )
        )

        exploration = (
            self.exploration_adapter.evaluate_profile(
                profile
            )
        )

        exploitation = (
            self.exploitation_adapter.evaluate_profile(
                profile
            )
        )

        exploration_stability = (
            self.stability_controller.update(
                previous_bias=(
                    self.previous_exploration_bias
                ),
                proposed_bias=(
                    exploration.bias
                ),
            )
        )

        exploitation_stability = (
            self.stability_controller.update(
                previous_bias=(
                    self.previous_exploitation_bias
                ),
                proposed_bias=(
                    exploitation.bias
                ),
            )
        )

        self.previous_exploration_bias = (
            exploration_stability.applied_bias
        )

        self.previous_exploitation_bias = (
            exploitation_stability.applied_bias
        )

        net_bias = round(
            (
                self.previous_exploration_bias
                - self.previous_exploitation_bias
            ),
            6,
        )

        return BehavioralAdaptationResult(
            evidence=evidence,
            exploration=exploration,
            exploitation=exploitation,
            exploration_stability=(
                exploration_stability
            ),
            exploitation_stability=(
                exploitation_stability
            ),
            exploration_bias=(
                self.previous_exploration_bias
            ),
            exploitation_bias=(
                self.previous_exploitation_bias
            ),
            net_bias=net_bias,
        )

    def reset(self) -> None:
        """
        Reset only behavioral adaptation state.

        Meta-learning history is not deleted.
        """

        self.previous_exploration_bias = 0.0
        self.previous_exploitation_bias = 0.0

    def snapshot(self) -> dict:
        return {
            "previous_exploration_bias": (
                self.previous_exploration_bias
            ),
            "previous_exploitation_bias": (
                self.previous_exploitation_bias
            ),
            "net_bias": round(
                (
                    self.previous_exploration_bias
                    - self.previous_exploitation_bias
                ),
                6,
            ),
        }
