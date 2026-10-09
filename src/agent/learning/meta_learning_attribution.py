"""
Meta-learning event attribution.

Translates agent-level learning events into updates on a
MetaLearningProfile.

This module does not decide whether an action should be taken.
It only records the learning outcome of decisions and
adaptation events.
"""

from __future__ import annotations

from typing import Optional

from agent.learning.meta_learning import (
    MetaLearningProfile,
)


class MetaLearningEventAttributor:
    """
    Attribute learning events to a MetaLearningProfile.
    """

    # =====================================================
    # Decision events
    # =====================================================

    def record_exploration(
        self,
        profile: MetaLearningProfile,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Record an exploration outcome.
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        profile.record_exploration(
            success=success,
            event_id=event_id,
        )

    def record_exploitation(
        self,
        profile: MetaLearningProfile,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Record an exploitation outcome.
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        profile.record_exploitation(
            success=success,
            event_id=event_id,
        )

    # =====================================================
    # Adaptation events
    # =====================================================

    def record_evolution(
        self,
        profile: MetaLearningProfile,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Record a strategy evolution outcome.
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        profile.record_evolution(
            success=success,
            event_id=event_id,
        )

    def record_recovery(
        self,
        profile: MetaLearningProfile,
        success: bool,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Record a strategy recovery outcome.
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        profile.record_recovery(
            success=success,
            event_id=event_id,
        )

    # =====================================================
    # Progress
    # =====================================================

    def record_progress(
        self,
        profile: MetaLearningProfile,
        progress_delta: float,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Record a progress observation.
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        profile.record_progress(
            progress_delta=progress_delta,
            event_id=event_id,
        )

    # =====================================================
    # Generic event
    # =====================================================

    def record_event(
        self,
        profile: MetaLearningProfile,
        event_type: str,
        success: Optional[bool] = None,
        progress_delta: Optional[float] = None,
        event_id: Optional[str] = None,
    ) -> None:
        """
        Record a typed meta-learning event.

        Supported event types:

            exploration
            exploitation
            evolution
            recovery
            progress
        """

        if profile is None:
            raise ValueError(
                "profile cannot be None."
            )

        if not event_type:
            raise ValueError(
                "event_type cannot be empty."
            )

        normalized = (
            event_type.strip().lower()
        )

        if normalized == "exploration":

            if success is None:
                raise ValueError(
                    "success is required for exploration."
                )

            self.record_exploration(
                profile=profile,
                success=success,
                event_id=event_id,
            )

            return

        if normalized == "exploitation":

            if success is None:
                raise ValueError(
                    "success is required for exploitation."
                )

            self.record_exploitation(
                profile=profile,
                success=success,
                event_id=event_id,
            )

            return

        if normalized == "evolution":

            if success is None:
                raise ValueError(
                    "success is required for evolution."
                )

            self.record_evolution(
                profile=profile,
                success=success,
                event_id=event_id,
            )

            return

        if normalized == "recovery":

            if success is None:
                raise ValueError(
                    "success is required for recovery."
                )

            self.record_recovery(
                profile=profile,
                success=success,
                event_id=event_id,
            )

            return

        if normalized == "progress":

            if progress_delta is None:
                raise ValueError(
                    "progress_delta is required for progress."
                )

            self.record_progress(
                profile=profile,
                progress_delta=progress_delta,
                event_id=event_id,
            )

            return

        raise ValueError(
            f"unsupported event type: {event_type}."
        )
