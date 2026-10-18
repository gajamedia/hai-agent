"""
Meta-learning model.

Tracks how the agent learns from its own decision history.

This first version focuses on recording learning outcomes
without changing the existing strategy, planner, or
decision-making behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetaLearningProfile:
    """
    Aggregated evidence about the agent's learning behavior.

    The profile describes learning outcomes rather than
    individual strategies.
    """

    exploration_count: int = 0
    exploration_success_count: int = 0
    exploration_failure_count: int = 0

    exploitation_count: int = 0
    exploitation_success_count: int = 0
    exploitation_failure_count: int = 0

    evolution_count: int = 0
    evolution_success_count: int = 0
    evolution_failure_count: int = 0

    recovery_count: int = 0
    recovery_success_count: int = 0
    recovery_failure_count: int = 0

    total_progress: float = 0.0
    progress_observation_count: int = 0

    learning_events: List[str] = field(
        default_factory=list
    )

    # =====================================================
    # Exploration
    # =====================================================

    def record_exploration(
        self,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:

        if not self._record_event(event_id):
            return

        self.exploration_count += 1

        if success:
            self.exploration_success_count += 1
        else:
            self.exploration_failure_count += 1


    # =====================================================
    # Exploitation
    # =====================================================

    def record_exploitation(
        self,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:

        if not self._record_event(event_id):
            return

        self.exploitation_count += 1

        if success:
            self.exploitation_success_count += 1
        else:
            self.exploitation_failure_count += 1


    # =====================================================
    # Evolution
    # =====================================================

    def record_evolution(
        self,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:

        if not self._record_event(event_id):
            return

        self.evolution_count += 1

        if success:
            self.evolution_success_count += 1
        else:
            self.evolution_failure_count += 1


    # =====================================================
    # Recovery
    # =====================================================

    def record_recovery(
        self,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:

        if not self._record_event(event_id):
            return

        self.recovery_count += 1

        if success:
            self.recovery_success_count += 1
        else:
            self.recovery_failure_count += 1


    # =====================================================
    # Progress
    # =====================================================

    def record_progress(
        self,
        progress_delta: float,
        event_id: Optional[str] = None,
    ) -> None:

        if not self._record_event(event_id):
            return

        self.total_progress += progress_delta
        self.progress_observation_count += 1


    # =====================================================
    # Internal event tracking
    # =====================================================

    def _record_event(
        self,
        event_id: Optional[str],
    ) -> bool:
        """
        Accept a learning event exactly once when it has an ID.

        Returns False for a duplicate ID so counters and progress are
        not updated twice when runtime callbacks are retried. Calls
        without an event ID preserve the historical non-deduplicated
        behavior.
        """

        if event_id is None:
            return True

        if event_id in self.learning_events:
            return False

        self.learning_events.append(event_id)
        return True

    # =====================================================
    # Rates
    # =====================================================

    @staticmethod
    def _rate(
        success_count: int,
        total_count: int,
    ) -> float:

        if total_count <= 0:
            return 0.0

        return round(
            success_count / total_count,
            6,
        )

    @property
    def exploration_success_rate(self) -> float:
        return self._rate(
            self.exploration_success_count,
            self.exploration_count,
        )

    @property
    def exploitation_success_rate(self) -> float:
        return self._rate(
            self.exploitation_success_count,
            self.exploitation_count,
        )

    @property
    def evolution_success_rate(self) -> float:
        return self._rate(
            self.evolution_success_count,
            self.evolution_count,
        )

    @property
    def recovery_success_rate(self) -> float:
        return self._rate(
            self.recovery_success_count,
            self.recovery_count,
        )

    @property
    def average_progress(self) -> float:
        if self.progress_observation_count <= 0:
            return 0.0

        return round(
            self.total_progress
            / self.progress_observation_count,
            6,
        )

    # =====================================================
    # Aggregate learning metrics
    # =====================================================

    @property
    def total_learning_events(self) -> int:
        return len(
            self.learning_events
        )

    @property
    def total_decisions(self) -> int:
        return (
            self.exploration_count
            + self.exploitation_count
        )

    @property
    def total_adaptation_events(self) -> int:
        return (
            self.evolution_count
            + self.recovery_count
        )

    @property
    def overall_adaptation_success_rate(self) -> float:
        total = (
            self.evolution_count
            + self.recovery_count
        )

        successes = (
            self.evolution_success_count
            + self.recovery_success_count
        )

        return self._rate(
            successes,
            total,
        )

    # =====================================================
    # Snapshot
    # =====================================================

    def snapshot(self) -> Dict[str, float | int]:
        """
        Return a compact metrics snapshot.
        """

        return {
            "exploration_count": (
                self.exploration_count
            ),
            "exploration_success_rate": (
                self.exploration_success_rate
            ),
            "exploitation_count": (
                self.exploitation_count
            ),
            "exploitation_success_rate": (
                self.exploitation_success_rate
            ),
            "evolution_count": (
                self.evolution_count
            ),
            "evolution_success_rate": (
                self.evolution_success_rate
            ),
            "recovery_count": (
                self.recovery_count
            ),
            "recovery_success_rate": (
                self.recovery_success_rate
            ),
            "average_progress": (
                self.average_progress
            ),
            "total_decisions": (
                self.total_decisions
            ),
            "total_adaptation_events": (
                self.total_adaptation_events
            ),
            "total_learning_events": (
                self.total_learning_events
            ),
            "overall_adaptation_success_rate": (
                self.overall_adaptation_success_rate
            ),
        }


class MetaLearningStore:
    """
    In-memory storage for meta-learning profiles.

    A profile is indexed by agent name.
    """

    def __init__(self) -> None:
        self._profiles: Dict[
            str,
            MetaLearningProfile,
        ] = {}

    # =====================================================
    # CRUD
    # =====================================================

    def add(
        self,
        agent_name: str,
        profile: MetaLearningProfile,
    ) -> MetaLearningProfile:

        if not agent_name:
            raise ValueError(
                "agent_name cannot be empty."
            )

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        existing = self._profiles.get(
            agent_name
        )

        if existing is not None:
            return existing

        self._profiles[
            agent_name
        ] = profile

        return profile

    def get(
        self,
        agent_name: str,
    ) -> Optional[MetaLearningProfile]:

        return self._profiles.get(
            agent_name
        )

    def get_or_create(
        self,
        agent_name: str,
    ) -> MetaLearningProfile:

        existing = self.get(
            agent_name
        )

        if existing is not None:
            return existing

        profile = MetaLearningProfile()

        self._profiles[
            agent_name
        ] = profile

        return profile

    def all(
        self,
    ) -> List[MetaLearningProfile]:

        return list(
            self._profiles.values()
        )

    def count(self) -> int:
        return len(
            self._profiles
        )

    def remove(
        self,
        agent_name: str,
    ) -> Optional[MetaLearningProfile]:

        return self._profiles.pop(
            agent_name,
            None,
        )

    def clear(self) -> None:
        self._profiles.clear()
