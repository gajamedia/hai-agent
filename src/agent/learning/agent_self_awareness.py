from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from agent.learning.capability_confidence import (
    CapabilityConfidenceIntegrator,
    CapabilityConfidenceSummary,
)

from agent.learning.self_model_behavioral_evidence import (
    SelfModelBehavioralEvidence,
    SelfModelBehavioralEvidenceModel,
)

from agent.learning.self_model_decision_influence import (
    SelfModelDecisionInfluence,
    SelfModelDecisionInfluenceIntegrator,
)

from agent.learning.self_model_persistence import (
    PersistentSelfModelStore,
    SelfModelContinuitySnapshot,
)

from agent.learning.self_uncertainty import (
    SelfUncertaintyIntegrator,
    SelfUncertaintyResult,
)


# ============================================================
# Agent Self-Awareness Snapshot
# ============================================================


@dataclass(frozen=True)
class AgentSelfAwarenessSnapshot:
    """
    Complete snapshot of the Agent's current self-awareness.

    Pipeline:

        Behavioral Evidence
            ↓
        Capability Confidence
            ↓
        Self Uncertainty
            ↓
        Decision Influence
            ↓
        Persistence
    """

    agent_name: str

    capability_summary: (
        CapabilityConfidenceSummary | None
    )

    self_uncertainty: (
        SelfUncertaintyResult | None
    )

    decision_influence: (
        SelfModelDecisionInfluence | None
    )

    persisted_version: int | None

    restored: bool

    # --------------------------------------------------------
    # Self knowledge
    # --------------------------------------------------------

    @property
    def has_self_knowledge(self) -> bool:
        return (
            self.self_uncertainty is not None
            and self.self_uncertainty.has_evidence
        )

    # --------------------------------------------------------
    # Aggregate self state
    # --------------------------------------------------------

    @property
    def uncertainty(self) -> float:
        if self.self_uncertainty is None:
            return 1.0

        return (
            self.self_uncertainty.uncertainty
        )

    @property
    def confidence(self) -> float:
        if self.self_uncertainty is None:
            return 0.0

        return (
            self.self_uncertainty.confidence
        )

    # --------------------------------------------------------
    # Decision influence
    # --------------------------------------------------------

    @property
    def preferred_mode(self) -> str | None:
        if self.decision_influence is None:
            return None

        return (
            self.decision_influence.preferred_mode
        )

    @property
    def should_explore(self) -> bool:
        if self.decision_influence is None:
            return False

        return (
            self.decision_influence.should_explore
        )

    @property
    def should_exploit(self) -> bool:
        if self.decision_influence is None:
            return False

        return (
            self.decision_influence.should_exploit
        )

    # --------------------------------------------------------
    # Capability information
    # --------------------------------------------------------

    @property
    def strongest_capability(self) -> str | None:
        if self.self_uncertainty is None:
            return None

        return (
            self.self_uncertainty.strongest_capability
        )

    @property
    def weakest_capability(self) -> str | None:
        if self.self_uncertainty is None:
            return None

        return (
            self.self_uncertainty.weakest_capability
        )

    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        return {
            "agent_name": self.agent_name,
            "capability_summary": (
                self.capability_summary.snapshot()
                if self.capability_summary is not None
                else None
            ),
            "self_uncertainty": (
                self.self_uncertainty.snapshot()
                if self.self_uncertainty is not None
                else None
            ),
            "decision_influence": (
                self.decision_influence.snapshot()
                if self.decision_influence is not None
                else None
            ),
            "persisted_version": (
                self.persisted_version
            ),
            "restored": self.restored,
            "has_self_knowledge": (
                self.has_self_knowledge
            ),
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "preferred_mode": (
                self.preferred_mode
            ),
            "should_explore": (
                self.should_explore
            ),
            "should_exploit": (
                self.should_exploit
            ),
            "strongest_capability": (
                self.strongest_capability
            ),
            "weakest_capability": (
                self.weakest_capability
            ),
        }


# ============================================================
# Agent Self-Awareness
# ============================================================


class AgentSelfAwareness:
    """
    High-level self-awareness facade.

    Input:

        capability mapping

    Pipeline:

        capability mapping
            ↓
        SelfModelBehavioralEvidenceModel
            ↓
        CapabilityBehavioralEvidence
            ↓
        CapabilityConfidenceIntegrator
            ↓
        SelfUncertaintyIntegrator
            ↓
        SelfModelDecisionInfluenceIntegrator
            ↓
        optional persistence
    """

    PERSISTENCE_FILENAME = (
        "self_model_continuity.sqlite3"
    )

    def __init__(
        self,
        *,
        agent_name: str,
        memory_path: str | Path | None = None,
    ) -> None:

        self._validate_agent_name(
            agent_name
        )

        self.agent_name = agent_name

        # ----------------------------------------------------
        # Behavioral evidence
        # ----------------------------------------------------

        self.behavioral_evidence_model = (
            SelfModelBehavioralEvidenceModel()
        )

        # ----------------------------------------------------
        # Capability confidence
        # ----------------------------------------------------

        self.capability_confidence_integrator = (
            CapabilityConfidenceIntegrator()
        )

        # ----------------------------------------------------
        # Self uncertainty
        # ----------------------------------------------------

        self.self_uncertainty_integrator = (
            SelfUncertaintyIntegrator()
        )

        # ----------------------------------------------------
        # Decision influence
        # ----------------------------------------------------

        self.decision_influence_integrator = (
            SelfModelDecisionInfluenceIntegrator()
        )

        # ----------------------------------------------------
        # Persistence
        # ----------------------------------------------------

        self.persistence_path = (
            self._resolve_persistence_path(
                memory_path
            )
        )

        self.store = None

        if self.persistence_path is not None:
            self.store = (
                PersistentSelfModelStore(
                    self.persistence_path
                )
            )

        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.behavioral_evidence = None

        self.capability_summary = None

        self.self_uncertainty = None

        self.decision_influence = None

        self.persisted_version = None

        self.restored = False

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        capabilities: Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> AgentSelfAwarenessSnapshot:
        """
        Evaluate current self-awareness.

        The public API accepts a mapping such as:

            {
                "planning": {
                    "attempts": 20,
                    "successes": 18,
                    "failures": 2,
                },
                "reasoning": {
                    "attempts": 10,
                    "successes": 7,
                    "failures": 3,
                },
            }

        The mapping is first converted into behavioral
        evidence objects before being passed into the
        capability-confidence layer.
        """

        if capabilities is None:
            raise ValueError(
                "capabilities cannot be None."
            )

        if not isinstance(
            capabilities,
            Mapping,
        ):
            raise ValueError(
                "capabilities must be a mapping."
            )

        # ----------------------------------------------------
        # Step 1:
        # Build behavioral evidence from raw capability data.
        #
        # IMPORTANT:
        #
        # CapabilityConfidenceIntegrator.assess_capabilities()
        # expects capability evidence objects, not a mapping.
        # ----------------------------------------------------

        behavioral_evidence = (
            self.behavioral_evidence_model.analyze(
                capabilities
            )
        )

        if not isinstance(
            behavioral_evidence,
            SelfModelBehavioralEvidence,
        ):
            raise RuntimeError(
                "Behavioral evidence model did not "
                "return SelfModelBehavioralEvidence."
            )

        self.behavioral_evidence = (
            behavioral_evidence
        )

        # ----------------------------------------------------
        # Step 2:
        # Capability confidence.
        #
        # Pass the evidence objects, NOT the dictionary.
        # ----------------------------------------------------

        summary = (
            self.capability_confidence_integrator
            .assess_capabilities(
                behavioral_evidence.capabilities
            )
        )

        # ----------------------------------------------------
        # Step 3:
        # Self uncertainty.
        # ----------------------------------------------------

        uncertainty = (
            self.self_uncertainty_integrator.assess(
                summary
            )
        )

        # ----------------------------------------------------
        # Step 4:
        # Decision influence.
        # ----------------------------------------------------

        influence = (
            self.decision_influence_integrator.assess(
                uncertainty
            )
        )

        # ----------------------------------------------------
        # Step 5:
        # Update runtime state.
        # ----------------------------------------------------

        self.capability_summary = (
            summary
        )

        self.self_uncertainty = (
            uncertainty
        )

        self.decision_influence = (
            influence
        )

        self.restored = False

        # ----------------------------------------------------
        # Step 6:
        # Optional persistence.
        # ----------------------------------------------------

        if (
            persist
            and self.store is not None
        ):
            persisted = self.store.save(
                agent_name=self.agent_name,
                self_uncertainty=uncertainty,
                decision_influence=influence,
            )

            self.persisted_version = (
                persisted.version
            )

        else:
            self.persisted_version = None

        return self.snapshot()

    # ========================================================
    # Restore
    # ========================================================

    def restore(
        self,
    ) -> AgentSelfAwarenessSnapshot | None:
        """
        Restore persisted self-awareness state.

        The capability-level evidence/summary is not restored
        by PersistentSelfModelStore in this phase.

        The persisted aggregate self-uncertainty and decision
        influence are restored.
        """

        if self.store is None:
            return None

        persisted = self.store.load(
            self.agent_name
        )

        if persisted is None:
            return None

        self.behavioral_evidence = None

        self.capability_summary = None

        self.self_uncertainty = (
            persisted.self_uncertainty
        )

        self.decision_influence = (
            persisted.decision_influence
        )

        self.persisted_version = (
            persisted.version
        )

        self.restored = True

        return self.snapshot()

    # ========================================================
    # Save
    # ========================================================

    def save(
        self,
    ) -> SelfModelContinuitySnapshot:
        """
        Persist current self-awareness state.
        """

        if self.store is None:
            raise ValueError(
                "Self-awareness persistence is not configured."
            )

        if self.self_uncertainty is None:
            raise ValueError(
                "No self-awareness state is available."
            )

        persisted = self.store.save(
            agent_name=self.agent_name,
            self_uncertainty=(
                self.self_uncertainty
            ),
            decision_influence=(
                self.decision_influence
            ),
        )

        self.persisted_version = (
            persisted.version
        )

        return persisted

    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        *,
        clear_persistence: bool = False,
    ) -> None:
        """
        Reset current in-memory self-awareness state.

        When clear_persistence=True, also remove persisted
        self-awareness state for this agent.
        """

        self.behavioral_evidence = None

        self.capability_summary = None

        self.self_uncertainty = None

        self.decision_influence = None

        self.persisted_version = None

        self.restored = False

        if (
            clear_persistence
            and self.store is not None
        ):
            self.store.delete(
                self.agent_name
            )

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(
        self,
    ) -> AgentSelfAwarenessSnapshot:

        return AgentSelfAwarenessSnapshot(
            agent_name=self.agent_name,
            capability_summary=(
                self.capability_summary
            ),
            self_uncertainty=(
                self.self_uncertainty
            ),
            decision_influence=(
                self.decision_influence
            ),
            persisted_version=(
                self.persisted_version
            ),
            restored=self.restored,
        )

    # ========================================================
    # Convenience properties
    # ========================================================

    @property
    def has_self_knowledge(self) -> bool:
        return (
            self.self_uncertainty is not None
            and self.self_uncertainty.has_evidence
        )

    @property
    def uncertainty(self) -> float:
        if self.self_uncertainty is None:
            return 1.0

        return (
            self.self_uncertainty.uncertainty
        )

    @property
    def confidence(self) -> float:
        if self.self_uncertainty is None:
            return 0.0

        return (
            self.self_uncertainty.confidence
        )

    @property
    def preferred_mode(self) -> str | None:
        if self.decision_influence is None:
            return None

        return (
            self.decision_influence.preferred_mode
        )

    @property
    def should_explore(self) -> bool:
        if self.decision_influence is None:
            return False

        return (
            self.decision_influence.should_explore
        )

    @property
    def should_exploit(self) -> bool:
        if self.decision_influence is None:
            return False

        return (
            self.decision_influence.should_exploit
        )

    @property
    def strongest_capability(self) -> str | None:
        if self.self_uncertainty is None:
            return None

        return (
            self.self_uncertainty.strongest_capability
        )

    @property
    def weakest_capability(self) -> str | None:
        if self.self_uncertainty is None:
            return None

        return (
            self.self_uncertainty.weakest_capability
        )

    # ========================================================
    # Internal helpers
    # ========================================================

    @staticmethod
    def _validate_agent_name(
        agent_name: str,
    ) -> None:

        if not isinstance(
            agent_name,
            str,
        ):
            raise ValueError(
                "agent_name must be a string."
            )

        if not agent_name.strip():
            raise ValueError(
                "agent_name cannot be empty."
            )

    @classmethod
    def _resolve_persistence_path(
        cls,
        memory_path: str | Path | None,
    ) -> Path | None:

        if memory_path is None:
            return None

        path = Path(
            memory_path
        )

        suffix = (
            path.suffix.lower()
        )

        if suffix in {
            ".sqlite",
            ".sqlite3",
            ".db",
        }:
            return (
                path.parent
                / cls.PERSISTENCE_FILENAME
            )

        return (
            path
            / cls.PERSISTENCE_FILENAME
        )


# ============================================================
# Convenience Builder
# ============================================================


def build_agent_self_awareness(
    *,
    agent_name: str,
    memory_path: str | Path | None = None,
) -> AgentSelfAwareness:

    return AgentSelfAwareness(
        agent_name=agent_name,
        memory_path=memory_path,
    )
