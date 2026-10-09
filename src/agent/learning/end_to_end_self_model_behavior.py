from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from agent.learning.agent_self_awareness import (
    AgentSelfAwareness,
    AgentSelfAwarenessSnapshot,
)

from agent.learning.self_model_decision_influence import (
    SELF_MODEL_MODE_EXPLOIT,
    SELF_MODEL_MODE_EXPLORE,
    SELF_MODEL_MODE_NEUTRAL,
)


# ============================================================
# End-to-End Result
# ============================================================


@dataclass(frozen=True)
class EndToEndSelfModelBehaviorResult:
    """
    Result of an end-to-end self-model behavior evaluation.

    The result compares two self-awareness states:

        initial state
              ↓
        new capability evidence
              ↓
        final state

    It can also contain a restored state loaded from
    persistence.
    """

    initial_snapshot: AgentSelfAwarenessSnapshot
    final_snapshot: AgentSelfAwarenessSnapshot
    restored_snapshot: AgentSelfAwarenessSnapshot | None

    uncertainty_delta: float
    confidence_delta: float

    mode_changed: bool
    self_knowledge_changed: bool

    persistence_continuity: bool

    @property
    def initial_uncertainty(self) -> float:
        return self.initial_snapshot.uncertainty

    @property
    def final_uncertainty(self) -> float:
        return self.final_snapshot.uncertainty

    @property
    def initial_confidence(self) -> float:
        return self.initial_snapshot.confidence

    @property
    def final_confidence(self) -> float:
        return self.final_snapshot.confidence

    @property
    def initial_mode(self) -> str | None:
        return self.initial_snapshot.preferred_mode

    @property
    def final_mode(self) -> str | None:
        return self.final_snapshot.preferred_mode

    @property
    def improved_self_knowledge(self) -> bool:
        return (
            self.final_snapshot.confidence
            > self.initial_snapshot.confidence
            and self.final_snapshot.uncertainty
            < self.initial_snapshot.uncertainty
        )

    @property
    def degraded_self_knowledge(self) -> bool:
        return (
            self.final_snapshot.confidence
            < self.initial_snapshot.confidence
            and self.final_snapshot.uncertainty
            > self.initial_snapshot.uncertainty
        )

    @property
    def moved_to_exploration(self) -> bool:
        return (
            self.final_mode
            == SELF_MODEL_MODE_EXPLORE
        )

    @property
    def moved_to_exploitation(self) -> bool:
        return (
            self.final_mode
            == SELF_MODEL_MODE_EXPLOIT
        )

    @property
    def is_final_mode_neutral(self) -> bool:
        return (
            self.final_mode
            == SELF_MODEL_MODE_NEUTRAL
        )

    @property
    def restore_matches_final(self) -> bool:
        if self.restored_snapshot is None:
            return False

        return (
            self.restored_snapshot.self_uncertainty
            == self.final_snapshot.self_uncertainty
            and self.restored_snapshot.decision_influence
            == self.final_snapshot.decision_influence
            and self.restored_snapshot.persisted_version
            == self.final_snapshot.persisted_version
        )

    @property
    def end_to_end_success(self) -> bool:
        """
        True when the full pipeline produced a final state
        and, when restoration was requested, persistence
        continuity was preserved.
        """

        basic_state_valid = (
            self.final_snapshot.self_uncertainty
            is not None
            and self.final_snapshot.decision_influence
            is not None
        )

        if not basic_state_valid:
            return False

        if self.restored_snapshot is None:
            return True

        return self.persistence_continuity

    def snapshot(self) -> dict[str, object]:
        return {
            "initial_snapshot": (
                self.initial_snapshot.snapshot()
            ),
            "final_snapshot": (
                self.final_snapshot.snapshot()
            ),
            "restored_snapshot": (
                self.restored_snapshot.snapshot()
                if self.restored_snapshot is not None
                else None
            ),
            "uncertainty_delta": (
                self.uncertainty_delta
            ),
            "confidence_delta": (
                self.confidence_delta
            ),
            "mode_changed": self.mode_changed,
            "self_knowledge_changed": (
                self.self_knowledge_changed
            ),
            "persistence_continuity": (
                self.persistence_continuity
            ),
            "initial_uncertainty": (
                self.initial_uncertainty
            ),
            "final_uncertainty": (
                self.final_uncertainty
            ),
            "initial_confidence": (
                self.initial_confidence
            ),
            "final_confidence": (
                self.final_confidence
            ),
            "initial_mode": (
                self.initial_mode
            ),
            "final_mode": (
                self.final_mode
            ),
            "improved_self_knowledge": (
                self.improved_self_knowledge
            ),
            "degraded_self_knowledge": (
                self.degraded_self_knowledge
            ),
            "moved_to_exploration": (
                self.moved_to_exploration
            ),
            "moved_to_exploitation": (
                self.moved_to_exploitation
            ),
            "is_final_mode_neutral": (
                self.is_final_mode_neutral
            ),
            "restore_matches_final": (
                self.restore_matches_final
            ),
            "end_to_end_success": (
                self.end_to_end_success
            ),
        }


# ============================================================
# Evaluator
# ============================================================


class EndToEndSelfModelBehaviorEvaluator:
    """
    Evaluates the complete self-model pipeline.

    Pipeline:

        capabilities
            ↓
        AgentSelfAwareness
            ↓
        behavioral evidence
            ↓
        capability confidence
            ↓
        self uncertainty
            ↓
        decision influence
            ↓
        persistence
            ↓
        restoration
    """

    def __init__(
        self,
        *,
        self_awareness: AgentSelfAwareness,
    ) -> None:

        if self_awareness is None:
            raise ValueError(
                "self_awareness cannot be None."
            )

        if not isinstance(
            self_awareness,
            AgentSelfAwareness,
        ):
            raise ValueError(
                "self_awareness must be "
                "AgentSelfAwareness."
            )

        self.self_awareness = (
            self_awareness
        )

    # ========================================================
    # Public API
    # ========================================================

    def evaluate(
        self,
        initial_capabilities: Mapping[str, Any],
        final_capabilities: Mapping[str, Any],
        *,
        persist: bool = True,
        restore: bool = False,
    ) -> EndToEndSelfModelBehaviorResult:

        self._validate_capabilities(
            initial_capabilities,
            "initial_capabilities",
        )

        self._validate_capabilities(
            final_capabilities,
            "final_capabilities",
        )

        # ----------------------------------------------------
        # Initial self-model state
        # ----------------------------------------------------

        initial_snapshot = (
            self.self_awareness.evaluate(
                initial_capabilities,
                persist=persist,
            )
        )

        # ----------------------------------------------------
        # Updated self-model state
        # ----------------------------------------------------

        final_snapshot = (
            self.self_awareness.evaluate(
                final_capabilities,
                persist=persist,
            )
        )

        # ----------------------------------------------------
        # Optional persistence restoration
        # ----------------------------------------------------

        restored_snapshot = None

        if restore:
            restored_snapshot = (
                self.self_awareness.restore()
            )

        # ----------------------------------------------------
        # State deltas
        # ----------------------------------------------------

        uncertainty_delta = round(
            final_snapshot.uncertainty
            - initial_snapshot.uncertainty,
            6,
        )

        confidence_delta = round(
            final_snapshot.confidence
            - initial_snapshot.confidence,
            6,
        )

        mode_changed = (
            initial_snapshot.preferred_mode
            != final_snapshot.preferred_mode
        )

        self_knowledge_changed = (
            initial_snapshot.has_self_knowledge
            != final_snapshot.has_self_knowledge
        )

        # ----------------------------------------------------
        # Persistence continuity
        # ----------------------------------------------------

        persistence_continuity = False

        if restored_snapshot is not None:
            persistence_continuity = (
                restored_snapshot.self_uncertainty
                == final_snapshot.self_uncertainty
                and restored_snapshot.decision_influence
                == final_snapshot.decision_influence
                and restored_snapshot.persisted_version
                == final_snapshot.persisted_version
            )

        return EndToEndSelfModelBehaviorResult(
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            restored_snapshot=restored_snapshot,
            uncertainty_delta=(
                uncertainty_delta
            ),
            confidence_delta=(
                confidence_delta
            ),
            mode_changed=mode_changed,
            self_knowledge_changed=(
                self_knowledge_changed
            ),
            persistence_continuity=(
                persistence_continuity
            ),
        )

    # ========================================================
    # Convenience API
    # ========================================================

    def evaluate_and_restore(
        self,
        initial_capabilities: Mapping[str, Any],
        final_capabilities: Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> EndToEndSelfModelBehaviorResult:

        return self.evaluate(
            initial_capabilities,
            final_capabilities,
            persist=persist,
            restore=True,
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _validate_capabilities(
        capabilities: Mapping[str, Any],
        name: str,
    ) -> None:

        if capabilities is None:
            raise ValueError(
                f"{name} cannot be None."
            )

        if not isinstance(
            capabilities,
            Mapping,
        ):
            raise ValueError(
                f"{name} must be a mapping."
            )


# ============================================================
# Convenience Builder
# ============================================================


def build_end_to_end_self_model_behavior(
    *,
    self_awareness: AgentSelfAwareness,
) -> EndToEndSelfModelBehaviorEvaluator:

    return EndToEndSelfModelBehaviorEvaluator(
        self_awareness=self_awareness,
    )
