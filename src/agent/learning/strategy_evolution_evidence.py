# """
# Strategy evolution evidence.

# Tracks the origin and experimental evidence of strategies
# created through strategy evolution.

# This module intentionally keeps evolution evidence separate
# from the Strategy model. Persistence integration can be added
# later without changing the existing Strategy schema.
# """

# from __future__ import annotations

# from dataclasses import dataclass, field
# from typing import List, Optional


# @dataclass
# class StrategyEvolutionEvidence:
#     """
#     Evidence describing how a strategy was evolved.
#     """

#     strategy_name: str
#     source_strategy_name: str
#     source_experience_id: str

#     generation: int = 1

#     trial_count: int = 0
#     success_count: int = 0
#     failure_count: int = 0

#     total_progress: float = 0.0

#     source_experience_ids: List[str] = field(
#         default_factory=list
#     )

#     status: str = "experimental"

#     # =====================================================
#     # Evidence recording
#     # =====================================================

#     def record_trial(
#         self,
#         success: bool,
#         progress_delta: float = 0.0,
#         experience_id: Optional[str] = None,
#     ) -> None:
#         """
#         Record one real-world trial of the evolved strategy.
#         """

#         self.trial_count += 1

#         if success:
#             self.success_count += 1
#         else:
#             self.failure_count += 1

#         self.total_progress += progress_delta

#         if (
#             experience_id is not None
#             and experience_id
#             not in self.source_experience_ids
#         ):
#             self.source_experience_ids.append(
#                 experience_id
#             )

#     # =====================================================
#     # Evidence properties
#     # =====================================================

#     @property
#     def success_rate(self) -> float:
#         if self.trial_count <= 0:
#             return 0.0

#         return round(
#             self.success_count
#             / self.trial_count,
#             6,
#         )

#     @property
#     def failure_rate(self) -> float:
#         if self.trial_count <= 0:
#             return 0.0

#         return round(
#             self.failure_count
#             / self.trial_count,
#             6,
#         )

#     @property
#     def average_progress(self) -> float:
#         if self.trial_count <= 0:
#             return 0.0

#         return round(
#             self.total_progress
#             / self.trial_count,
#             6,
#         )

#     @property
#     def is_experimental(self) -> bool:
#         return self.status == "experimental"

#     @property
#     def is_promoted(self) -> bool:
#         return self.status == "promoted"

#     @property
#     def is_rejected(self) -> bool:
#         return self.status == "rejected"

#     # =====================================================
#     # Lifecycle
#     # =====================================================

#     def promote(self) -> None:
#         """
#         Mark an evolved strategy as promoted.
#         """

#         if self.status == "rejected":
#             return

#         self.status = "promoted"

#     def reject(self) -> None:
#         """
#         Mark an evolved strategy as rejected.
#         """

#         self.status = "rejected"

#     def reset_experimental(self) -> None:
#         """
#         Return evidence to experimental status.

#         Rejected evidence is not automatically promoted.
#         """

#         if self.status == "rejected":
#             return

#         self.status = "experimental"


# class StrategyEvolutionEvidenceStore:
#     """
#     In-memory evidence registry for evolved strategies.
#     """

#     def __init__(self) -> None:
#         self._evidence: dict[
#             str,
#             StrategyEvolutionEvidence,
#         ] = {}

#     # =====================================================
#     # CRUD
#     # =====================================================

#     def add(
#         self,
#         evidence: StrategyEvolutionEvidence,
#     ) -> StrategyEvolutionEvidence:

#         if evidence is None:
#             raise ValueError(
#                 "evidence cannot be None."
#             )

#         existing = self._evidence.get(
#             evidence.strategy_name
#         )

#         if existing is not None:
#             return existing

#         self._evidence[
#             evidence.strategy_name
#         ] = evidence

#         return evidence

#     def get(
#         self,
#         strategy_name: str,
#     ) -> Optional[
#         StrategyEvolutionEvidence
#     ]:

#         return self._evidence.get(
#             strategy_name
#         )

#     def all(
#         self,
#     ) -> list[StrategyEvolutionEvidence]:

#         return list(
#             self._evidence.values()
#         )

#     def count(self) -> int:
#         return len(
#             self._evidence
#         )

#     def remove(
#         self,
#         strategy_name: str,
#     ) -> Optional[
#         StrategyEvolutionEvidence
#     ]:

#         return self._evidence.pop(
#             strategy_name,
#             None,
#         )

#     def clear(self) -> None:
#         self._evidence.clear()

#     # =====================================================
#     # Convenience
#     # =====================================================

#     def get_or_create(
#         self,
#         strategy_name: str,
#         source_strategy_name: str,
#         source_experience_id: str,
#         generation: int = 1,
#     ) -> StrategyEvolutionEvidence:

#         existing = self.get(
#             strategy_name
#         )

#         if existing is not None:
#             return existing

#         evidence = StrategyEvolutionEvidence(
#             strategy_name=strategy_name,
#             source_strategy_name=(
#                 source_strategy_name
#             ),
#             source_experience_id=(
#                 source_experience_id
#             ),
#             generation=generation,
#         )

#         self.add(
#             evidence
#         )

#         return evidence


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory.strategy_evolution_evidence_repository import (
        StrategyEvolutionEvidenceRepository,
    )


@dataclass
class StrategyEvolutionEvidence:
    """Evidence describing how a strategy was evolved."""

    strategy_name: str
    source_strategy_name: str
    source_experience_id: str

    generation: int = 1
    trial_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_progress: float = 0.0
    source_experience_ids: list[str] = field(default_factory=list)
    status: str = "experimental"

    def record_trial(
        self,
        success: bool,
        progress_delta: float = 0.0,
        experience_id: Optional[str] = None,
    ) -> None:
        self.trial_count += 1

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.total_progress += progress_delta

        if (
            experience_id is not None
            and experience_id not in self.source_experience_ids
        ):
            self.source_experience_ids.append(experience_id)

    @property
    def success_rate(self) -> float:
        if self.trial_count <= 0:
            return 0.0
        return round(self.success_count / self.trial_count, 6)

    @property
    def failure_rate(self) -> float:
        if self.trial_count <= 0:
            return 0.0
        return round(self.failure_count / self.trial_count, 6)

    @property
    def average_progress(self) -> float:
        if self.trial_count <= 0:
            return 0.0
        return round(self.total_progress / self.trial_count, 6)

    @property
    def is_experimental(self) -> bool:
        return self.status == "experimental"

    @property
    def is_promoted(self) -> bool:
        return self.status == "promoted"

    @property
    def is_rejected(self) -> bool:
        return self.status == "rejected"

    def promote(self) -> None:
        if self.status != "rejected":
            self.status = "promoted"

    def reject(self) -> None:
        self.status = "rejected"

    def reset_experimental(self) -> None:
        if self.status != "rejected":
            self.status = "experimental"


class StrategyEvolutionEvidenceStore:
    """Evidence registry with optional SQLite persistence."""

    def __init__(
        self,
        repository: Optional[
            StrategyEvolutionEvidenceRepository
        ] = None,
    ) -> None:
        self.repository = repository
        self._evidence: dict[str, StrategyEvolutionEvidence] = {}
        self.load()

    def add(
        self,
        evidence: StrategyEvolutionEvidence,
    ) -> StrategyEvolutionEvidence:
        if evidence is None:
            raise ValueError("evidence cannot be None.")

        existing = self._evidence.get(evidence.strategy_name)
        if existing is not None:
            return existing

        self._evidence[evidence.strategy_name] = evidence
        self.save(evidence)
        return evidence

    def get(
        self,
        strategy_name: str,
    ) -> Optional[StrategyEvolutionEvidence]:
        return self._evidence.get(strategy_name)

    def all(self) -> list[StrategyEvolutionEvidence]:
        return list(self._evidence.values())

    def count(self) -> int:
        return len(self._evidence)

    def save(
        self,
        evidence: StrategyEvolutionEvidence,
    ) -> None:
        if self.repository is not None:
            self.repository.save(evidence)

    def remove(
        self,
        strategy_name: str,
    ) -> Optional[StrategyEvolutionEvidence]:
        evidence = self._evidence.pop(strategy_name, None)

        if self.repository is not None:
            self.repository.delete(strategy_name)

        return evidence

    def clear(self) -> None:
        self._evidence.clear()

        if self.repository is not None:
            self.repository.clear()

    def load(self) -> None:
        if self.repository is None:
            return

        self._evidence = {
            item.strategy_name: item
            for item in self.repository.load_all()
        }

    def get_or_create(
        self,
        strategy_name: str,
        source_strategy_name: str,
        source_experience_id: str,
        generation: int = 1,
    ) -> StrategyEvolutionEvidence:
        existing = self.get(strategy_name)
        if existing is not None:
            return existing

        evidence = StrategyEvolutionEvidence(
            strategy_name=strategy_name,
            source_strategy_name=source_strategy_name,
            source_experience_id=source_experience_id,
            generation=generation,
        )
        return self.add(evidence)
