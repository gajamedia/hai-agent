"""
Persistent meta-learning store.

Provides an in-memory cache backed by MetaLearningRepository.

The store preserves the same conceptual responsibility as
MetaLearningStore while adding persistence.

Flow:

    get_or_create(agent)
        -> memory cache
        -> repository
        -> create new profile

    save(agent, profile)
        -> update memory cache
        -> persist SQLite
"""

from __future__ import annotations

from typing import Optional

from agent.learning.meta_learning import (
    MetaLearningProfile,
)
from agent.learning.meta_learning_repository import (
    MetaLearningRepository,
)


class PersistentMetaLearningStore:
    """
    Meta-learning store with SQLite persistence.

    The in-memory dictionary remains the active working set.
    SQLite is the durable source used for recovery across
    process restarts.
    """

    def __init__(
        self,
        repository: MetaLearningRepository,
    ) -> None:

        if repository is None:
            raise ValueError(
                "repository cannot be None."
            )

        self.repository = repository

        self._profiles: dict[
            str,
            MetaLearningProfile,
        ] = {}

    # =====================================================
    # Get
    # =====================================================

    def get(
        self,
        agent_name: str,
    ) -> Optional[MetaLearningProfile]:

        self._validate_agent_name(
            agent_name
        )

        if agent_name in self._profiles:
            return self._profiles[
                agent_name
            ]

        profile = self.repository.get(
            agent_name
        )

        if profile is not None:
            self._profiles[
                agent_name
            ] = profile

        return profile

    # =====================================================
    # Get or create
    # =====================================================

    def get_or_create(
        self,
        agent_name: str,
    ) -> MetaLearningProfile:

        self._validate_agent_name(
            agent_name
        )

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

    # =====================================================
    # Add
    # =====================================================

    def add(
        self,
        agent_name: str,
        profile: MetaLearningProfile,
    ) -> MetaLearningProfile:

        self._validate_agent_name(
            agent_name
        )

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        self._profiles[
            agent_name
        ] = profile

        return profile

    # =====================================================
    # Save
    # =====================================================

    def save(
        self,
        agent_name: str,
        profile: Optional[
            MetaLearningProfile
        ] = None,
    ) -> MetaLearningProfile:

        self._validate_agent_name(
            agent_name
        )

        if profile is None:
            profile = self.get(
                agent_name
            )

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        self._profiles[
            agent_name
        ] = profile

        self.repository.save(
            agent_name=agent_name,
            profile=profile,
        )

        return profile

    # =====================================================
    # Save all
    # =====================================================

    def save_all(self) -> None:

        for (
            agent_name,
            profile,
        ) in self._profiles.items():

            self.repository.save(
                agent_name=agent_name,
                profile=profile,
            )

    # =====================================================
    # Remove from memory
    # =====================================================

    def remove(
        self,
        agent_name: str,
    ) -> bool:

        self._validate_agent_name(
            agent_name
        )

        return (
            self._profiles.pop(
                agent_name,
                None,
            )
            is not None
        )

    # =====================================================
    # Delete persistent profile
    # =====================================================

    def delete(
        self,
        agent_name: str,
    ) -> bool:

        self._validate_agent_name(
            agent_name
        )

        self._profiles.pop(
            agent_name,
            None,
        )

        return self.repository.delete(
            agent_name
        )

    # =====================================================
    # Count loaded profiles
    # =====================================================

    def count(
        self,
    ) -> int:

        return len(
            self._profiles
        )

    # =====================================================
    # Count persistent profiles
    # =====================================================

    def persistent_count(
        self,
    ) -> int:

        return self.repository.count()

    # =====================================================
    # Clear memory cache
    # =====================================================

    def clear_cache(
        self,
    ) -> None:

        self._profiles.clear()

    # =====================================================
    # Clear persistent storage
    # =====================================================

    def clear(
        self,
    ) -> None:

        self._profiles.clear()

        self.repository.clear()

    # =====================================================
    # Validation
    # =====================================================

    @staticmethod
    def _validate_agent_name(
        agent_name: str,
    ) -> None:

        if not agent_name:
            raise ValueError(
                "agent_name cannot be empty."
            )
