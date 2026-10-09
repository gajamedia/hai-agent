from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Strategy:
    name: str
    goal: str
    steps: list[str] = field(default_factory=list)

    success_count: int = 0
    failure_count: int = 0
    total_actions: int = 0

    reflection_penalty: float = 0.0

    source_experiences: list[str] = field(
        default_factory=list
    )

    total_progress: float = 0.0
    progress_count: int = 0

    positive_progress_count: int = 0
    negative_progress_count: int = 0

    # =================================================
    # Lifecycle
    # =================================================

    status: str = "active"

    # =================================================
    # Outcome recording
    # =================================================

    def record_success(
        self,
        experience_id: str,
        action_count: int,
        progress_delta: float = 0.0,
    ) -> None:

        self.success_count += 1

        self.total_actions += (
            action_count
        )

        self._record_experience(
            experience_id
        )

        self.record_progress(
            experience_id=experience_id,
            progress_delta=progress_delta,
            record_experience=False,
        )

    def record_failure(
        self,
        experience_id: str,
        action_count: int,
        progress_delta: float = 0.0,
    ) -> None:

        self.failure_count += 1

        self.total_actions += (
            action_count
        )

        self._record_experience(
            experience_id
        )

        self.record_progress(
            experience_id=experience_id,
            progress_delta=progress_delta,
            record_experience=False,
        )

    # =================================================
    # Progress
    # =================================================

    def record_progress(
        self,
        experience_id: str,
        progress_delta: float,
        record_experience: bool = True,
    ) -> None:

        self.total_progress += (
            progress_delta
        )

        self.progress_count += 1

        if progress_delta > 0:
            self.positive_progress_count += 1

        elif progress_delta < 0:
            self.negative_progress_count += 1

        if record_experience:
            self._record_experience(
                experience_id
            )

    # =================================================
    # Internal
    # =================================================

    def _record_experience(
        self,
        experience_id: str,
    ) -> None:

        if (
            experience_id
            and experience_id
            not in self.source_experiences
        ):
            self.source_experiences.append(
                experience_id
            )

    # =================================================
    # Metrics
    # =================================================

    @property
    def attempts(self) -> int:
        return (
            self.success_count
            + self.failure_count
        )

    @property
    def success_rate(self) -> float:

        if self.attempts == 0:
            return 0.0

        return (
            self.success_count
            / self.attempts
        )

    @property
    def average_actions(self) -> float:

        if self.attempts == 0:
            return 0.0

        return (
            self.total_actions
            / self.attempts
        )

    @property
    def average_progress(self) -> float:

        if self.progress_count == 0:
            return 0.0

        return round(
            self.total_progress
            / self.progress_count,
            6,
        )

    @property
    def progress_success_rate(self) -> float:

        if self.progress_count == 0:
            return 0.0

        return (
            self.positive_progress_count
            / self.progress_count
        )

    @property
    def progress_failure_rate(self) -> float:

        if self.progress_count == 0:
            return 0.0

        return (
            self.negative_progress_count
            / self.progress_count
        )

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_paused(self) -> bool:
        return self.status == "paused"

    @property
    def is_retired(self) -> bool:
        return self.status == "retired"

    # =================================================
    # Lifecycle operations
    # =================================================

    def pause(self) -> None:
        if self.status == "retired":
            return

        self.status = "paused"

    def activate(self) -> None:
        if self.status == "retired":
            return

        self.status = "active"

    def retire(self) -> None:
        self.status = "retired"

    # =================================================
    # Reflection
    # =================================================

    def apply_reflection_penalty(
        self,
        amount: float = 0.1,
    ) -> None:

        self.reflection_penalty = min(
            1.0,
            round(
                self.reflection_penalty
                + amount,
                6,
            ),
        )
