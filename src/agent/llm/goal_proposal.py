from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional
import re


class GoalProposalError(Exception):
    """Base error for goal proposal."""


class GoalProposalConfigurationError(GoalProposalError):
    """Raised for invalid proposal-engine configuration."""


class GoalProposalInputError(GoalProposalError):
    """Raised when a goal signal cannot be normalized."""


class GoalOrigin(str, Enum):
    EXTERNAL = "external"
    WORLD_GAP = "world_gap"
    MEMORY = "memory"
    INTRINSIC = "intrinsic"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"


@dataclass(frozen=True)
class GoalSignal:
    """Provider-neutral signal that may produce a goal candidate."""

    description: str
    origin: GoalOrigin = GoalOrigin.WORLD_GAP
    urgency: float = 0.5
    expected_value: float = 0.5
    feasibility: float = 0.5
    novelty: float = 0.5
    evidence_quality: float = 0.5
    parent_goal_id: Optional[str] = None
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: Optional[str] = None

    def __post_init__(self) -> None:
        description = str(self.description).strip()
        if not description:
            raise ValueError("description must not be empty.")
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "origin", _coerce_origin(self.origin))
        for field_name in (
            "urgency",
            "expected_value",
            "feasibility",
            "novelty",
            "evidence_quality",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_number(getattr(self, field_name), field_name),
            )
        if self.parent_goal_id is not None:
            parent = str(self.parent_goal_id).strip()
            object.__setattr__(self, "parent_goal_id", parent or None)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.candidate_id is not None:
            candidate_id = str(self.candidate_id).strip()
            object.__setattr__(self, "candidate_id", candidate_id or None)


@dataclass(frozen=True)
class GoalCandidate:
    """Immutable goal candidate; not yet accepted into GoalLifecycleManager."""

    candidate_id: str
    description: str
    origin: GoalOrigin
    priority: float
    proposal_score: float
    urgency: float
    expected_value: float
    feasibility: float
    novelty: float
    evidence_quality: float
    parent_goal_id: Optional[str] = None
    payload: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("candidate_id must not be empty.")
        if not str(self.description).strip():
            raise ValueError("description must not be empty.")
        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "origin", _coerce_origin(self.origin))
        for field_name in (
            "priority",
            "proposal_score",
            "urgency",
            "expected_value",
            "feasibility",
            "novelty",
            "evidence_quality",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_number(getattr(self, field_name), field_name),
            )
        if self.parent_goal_id is not None:
            parent = str(self.parent_goal_id).strip()
            object.__setattr__(self, "parent_goal_id", parent or None)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def manager_kwargs(self) -> dict[str, Any]:
        """Return safe arguments for GoalLifecycleManager.add_goal()."""
        return {
            "goal_id": self.candidate_id,
            "description": self.description,
            "priority": self.priority,
            "parent_id": self.parent_goal_id,
            "payload": self.payload,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GoalProposalBatch:
    """Deterministic proposal result."""

    candidates: tuple[GoalCandidate, ...]
    rejected_descriptions: tuple[str, ...] = ()
    duplicate_descriptions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def best_candidate(self) -> Optional[GoalCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def count(self) -> int:
        return len(self.candidates)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejected_descriptions", tuple(self.rejected_descriptions))
        object.__setattr__(self, "duplicate_descriptions", tuple(self.duplicate_descriptions))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


class GoalProposalEngine:
    """Builds goal candidates from provider-neutral signals.

    The engine is side-effect free. It does not insert goals into the lifecycle
    manager, mutate Agent state, call an LLM, or execute actions.
    """

    _DEFAULT_WEIGHTS = {
        "urgency": 0.30,
        "expected_value": 0.25,
        "feasibility": 0.20,
        "novelty": 0.10,
        "evidence_quality": 0.15,
    }

    def __init__(
        self,
        *,
        weights: Optional[Mapping[str, float]] = None,
        minimum_score: float = 0.35,
        max_candidates: int = 10,
    ) -> None:
        raw_weights = dict(weights or self._DEFAULT_WEIGHTS)
        required = set(self._DEFAULT_WEIGHTS)
        if set(raw_weights) != required:
            raise GoalProposalConfigurationError(
                f"weights must contain exactly {sorted(required)}."
            )
        numeric_weights = {
            key: float(value) for key, value in raw_weights.items()
        }
        if any(value < 0.0 for value in numeric_weights.values()):
            raise GoalProposalConfigurationError("weights must be non-negative.")
        total = sum(numeric_weights.values())
        if total <= 0.0:
            raise GoalProposalConfigurationError("weights must have positive total.")
        self.weights = {
            key: value / total for key, value in numeric_weights.items()
        }
        self.minimum_score = _bounded_number(minimum_score, "minimum_score")
        if int(max_candidates) != max_candidates or max_candidates <= 0:
            raise GoalProposalConfigurationError("max_candidates must be a positive integer.")
        self.max_candidates = int(max_candidates)

    def propose(
        self,
        signals: Iterable[GoalSignal | Mapping[str, Any] | Any],
        *,
        existing_descriptions: Iterable[str] = (),
        existing_candidate_ids: Iterable[str] = (),
        max_candidates: Optional[int] = None,
    ) -> GoalProposalBatch:
        limit = self.max_candidates if max_candidates is None else int(max_candidates)
        if limit <= 0:
            raise GoalProposalConfigurationError("max_candidates must be a positive integer.")

        raw_signals = tuple(signals)

        known_descriptions = {
            _normalize_text(value) for value in existing_descriptions if str(value).strip()
        }
        known_ids = {str(value).strip() for value in existing_candidate_ids if str(value).strip()}
        seen_descriptions: set[str] = set()
        candidates: list[GoalCandidate] = []
        rejected: list[str] = []
        duplicates: list[str] = []

        for raw_signal in raw_signals:
            signal = self._normalize_signal(raw_signal)
            key = _normalize_text(signal.description)
            if key in known_descriptions or key in seen_descriptions:
                duplicates.append(signal.description)
                continue
            seen_descriptions.add(key)

            proposal_score = self.score_signal(signal)
            if proposal_score < self.minimum_score:
                rejected.append(signal.description)
                continue

            candidate_id = self._candidate_id(signal, key)
            if candidate_id in known_ids or any(c.candidate_id == candidate_id for c in candidates):
                duplicates.append(signal.description)
                continue

            priority = round(
                _bounded_number(
                    proposal_score,
                    "priority",
                ),
                12,
            )
            metadata = dict(signal.metadata)
            metadata.setdefault("goal_origin", signal.origin.value)
            metadata.setdefault("proposal_score", proposal_score)
            candidate = GoalCandidate(
                candidate_id=candidate_id,
                description=signal.description,
                origin=signal.origin,
                priority=priority,
                proposal_score=proposal_score,
                urgency=signal.urgency,
                expected_value=signal.expected_value,
                feasibility=signal.feasibility,
                novelty=signal.novelty,
                evidence_quality=signal.evidence_quality,
                parent_goal_id=signal.parent_goal_id,
                payload=signal.payload,
                metadata=metadata,
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                -item.proposal_score,
                -item.urgency,
                -item.expected_value,
                item.candidate_id,
            )
        )
        selected = tuple(candidates[:limit])
        overflow = candidates[limit:]
        rejected.extend(item.description for item in overflow)
        return GoalProposalBatch(
            candidates=selected,
            rejected_descriptions=tuple(rejected),
            duplicate_descriptions=tuple(duplicates),
            metadata={
                "input_count": len(raw_signals),
                "candidate_count": len(selected),
                "rejected_count": len(rejected),
                "duplicate_count": len(duplicates),
                "score_threshold": self.minimum_score,
            },
        )

    def score_signal(self, signal: GoalSignal | Mapping[str, Any] | Any) -> float:
        normalized = self._normalize_signal(signal)
        score = sum(
            getattr(normalized, field_name) * weight
            for field_name, weight in self.weights.items()
        )
        return round(_bounded_number(score, "proposal_score"), 12)

    def _normalize_signal(self, raw: GoalSignal | Mapping[str, Any] | Any) -> GoalSignal:
        if isinstance(raw, GoalSignal):
            return raw
        if isinstance(raw, Mapping):
            try:
                return GoalSignal(**dict(raw))
            except TypeError as exc:
                raise GoalProposalInputError(
                    f"Invalid goal signal mapping: {exc}"
                ) from exc

        description = getattr(raw, "description", None)
        if description is None:
            raise GoalProposalInputError("Goal signal must provide description.")
        values = {
            "description": description,
            "origin": getattr(raw, "origin", GoalOrigin.WORLD_GAP),
            "urgency": getattr(raw, "urgency", 0.5),
            "expected_value": getattr(raw, "expected_value", 0.5),
            "feasibility": getattr(raw, "feasibility", 0.5),
            "novelty": getattr(raw, "novelty", 0.5),
            "evidence_quality": getattr(raw, "evidence_quality", 0.5),
            "parent_goal_id": getattr(raw, "parent_goal_id", None),
            "payload": getattr(raw, "payload", None),
            "metadata": getattr(raw, "metadata", {}),
            "candidate_id": getattr(raw, "candidate_id", None),
        }
        try:
            return GoalSignal(**values)
        except (TypeError, ValueError) as exc:
            raise GoalProposalInputError(str(exc)) from exc

    @staticmethod
    def _candidate_id(signal: GoalSignal, normalized_description: str) -> str:
        if signal.candidate_id:
            return signal.candidate_id
        slug = re.sub(r"[^a-z0-9]+", "-", normalized_description).strip("-")
        if not slug:
            slug = "goal"
        return f"candidate-{slug[:72]}"


def _coerce_origin(value: GoalOrigin | str) -> GoalOrigin:
    if isinstance(value, GoalOrigin):
        return value
    try:
        return GoalOrigin(str(value).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported goal origin: {value!r}.") from exc


def _normalize_text(value: Any) -> str:
    text = " ".join(str(value).strip().lower().split())
    return text


def _bounded_number(value: float, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return number


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0
