from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .agent_runtime import AgentCognitiveRuntimeTrace
from .planning import LLMPlan, LLMPlanParser, LLMPlanningParseError, LLMPlanningSchemaError


HANDOFF_CONTRACT_VERSION = 1
DEFAULT_HANDOFF_TTL_SECONDS = 300
MAX_HANDOFF_TTL_SECONDS = 3600


class CognitivePassHandoffError(ValueError):
    """Base error for invalid Cognitive Pass handoff data."""


class CognitivePassHandoffSerializationError(CognitivePassHandoffError):
    """Raised when a handoff value cannot be represented as canonical JSON."""


class CognitivePassHandoffValidationError(CognitivePassHandoffError):
    """Raised when a handoff fails integrity, context, or expiry validation."""


def _canonical_json(value: Any, *, field_name: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CognitivePassHandoffSerializationError(
            f"{field_name} must contain JSON-compatible values."
        ) from exc


def _sha256_json(value: Any, *, field_name: str) -> str:
    encoded = _canonical_json(value, field_name=field_name).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_datetime(value: datetime | None, *, field_name: str) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if not isinstance(resolved, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return resolved.astimezone(timezone.utc)


def _parse_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise CognitivePassHandoffValidationError(
            f"{field_name} must be a non-empty ISO-8601 timestamp."
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CognitivePassHandoffValidationError(
            f"{field_name} must be a valid ISO-8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CognitivePassHandoffValidationError(
            f"{field_name} must include a timezone."
        )
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class CognitivePassHandoff:
    """Immutable, side-effect-free envelope for a Cognitive Pass plan.

    The plan is retained internally as canonical JSON so callers cannot mutate
    nested plan values after construction. ``to_dict()`` returns a fresh copy.
    This contract does not authorize execution or provide durable single-use
    semantics; those require a later persistence/approval stage.
    """

    handoff_id: str
    contract_version: int
    created_at: str
    expires_at: str
    cognitive_cycle: int | None
    goal: str
    observation_fingerprint: str
    plan_fingerprint: str
    _plan_json: str

    @property
    def plan(self) -> dict[str, Any]:
        return json.loads(self._plan_json)

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "contract_version": self.contract_version,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "cognitive_cycle": self.cognitive_cycle,
            "goal": self.goal,
            "observation_fingerprint": self.observation_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "plan": self.plan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CognitivePassHandoff":
        if not isinstance(payload, Mapping):
            raise CognitivePassHandoffValidationError("handoff must be an object.")
        required = {
            "handoff_id", "contract_version", "created_at", "expires_at",
            "cognitive_cycle", "goal", "observation_fingerprint",
            "plan_fingerprint", "plan",
        }
        missing = required.difference(payload.keys())
        if missing:
            raise CognitivePassHandoffValidationError(
                "handoff is missing required fields: " + ", ".join(sorted(missing))
            )
        handoff_id = payload["handoff_id"]
        try:
            parsed_id = uuid.UUID(str(handoff_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise CognitivePassHandoffValidationError("handoff_id must be a UUID.") from exc
        if str(parsed_id) != str(handoff_id).lower():
            raise CognitivePassHandoffValidationError("handoff_id must use canonical UUID format.")
        version = payload["contract_version"]
        if isinstance(version, bool) or version != HANDOFF_CONTRACT_VERSION:
            raise CognitivePassHandoffValidationError("unsupported handoff contract_version.")
        cycle = payload["cognitive_cycle"]
        if cycle is not None and (isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0):
            raise CognitivePassHandoffValidationError("cognitive_cycle must be a non-negative integer or null.")
        goal = payload["goal"]
        if not isinstance(goal, str) or not goal.strip():
            raise CognitivePassHandoffValidationError("goal must be a non-empty string.")
        observation_fingerprint = payload["observation_fingerprint"]
        plan_fingerprint = payload["plan_fingerprint"]
        for name, fingerprint in (
            ("observation_fingerprint", observation_fingerprint),
            ("plan_fingerprint", plan_fingerprint),
        ):
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
                char not in "0123456789abcdef" for char in fingerprint
            ):
                raise CognitivePassHandoffValidationError(f"{name} must be a lowercase SHA-256 hex digest.")
        created = _parse_timestamp(payload["created_at"], field_name="created_at")
        expires = _parse_timestamp(payload["expires_at"], field_name="expires_at")
        if expires <= created:
            raise CognitivePassHandoffValidationError("expires_at must be later than created_at.")
        ttl_seconds = (expires - created).total_seconds()
        if ttl_seconds > MAX_HANDOFF_TTL_SECONDS:
            raise CognitivePassHandoffValidationError(
                f"handoff lifetime cannot exceed {MAX_HANDOFF_TTL_SECONDS} seconds."
            )
        plan = payload["plan"]
        if not isinstance(plan, Mapping):
            raise CognitivePassHandoffValidationError("plan must be an object.")
        plan_json = _canonical_json(dict(plan), field_name="plan")
        if hashlib.sha256(plan_json.encode("utf-8")).hexdigest() != plan_fingerprint:
            raise CognitivePassHandoffValidationError("plan_fingerprint does not match plan.")
        try:
            parsed_plan = LLMPlanParser.parse(plan_json)
        except (LLMPlanningParseError, LLMPlanningSchemaError) as exc:
            raise CognitivePassHandoffValidationError("plan does not satisfy the LLMPlan contract.") from exc
        if parsed_plan.goal != goal:
            raise CognitivePassHandoffValidationError("handoff goal does not match plan.goal.")
        return cls(
            handoff_id=str(parsed_id),
            contract_version=HANDOFF_CONTRACT_VERSION,
            created_at=created.astimezone(timezone.utc).isoformat(),
            expires_at=expires.astimezone(timezone.utc).isoformat(),
            cognitive_cycle=cycle,
            goal=goal,
            observation_fingerprint=observation_fingerprint,
            plan_fingerprint=plan_fingerprint,
            _plan_json=plan_json,
        )


def create_cognitive_pass_handoff(
    trace: AgentCognitiveRuntimeTrace,
    *,
    ttl_seconds: int = DEFAULT_HANDOFF_TTL_SECONDS,
    now: datetime | None = None,
) -> CognitivePassHandoff:
    """Create a handoff from the immutable trace of a completed Cognitive Pass."""
    if not isinstance(trace, AgentCognitiveRuntimeTrace):
        raise TypeError("trace must be an AgentCognitiveRuntimeTrace")
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
        raise TypeError("ttl_seconds must be an integer")
    if not 1 <= ttl_seconds <= MAX_HANDOFF_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must be between 1 and {MAX_HANDOFF_TTL_SECONDS}")
    if not isinstance(trace.result.plan, LLMPlan):
        raise TypeError("trace.result.plan must be an LLMPlan")

    created = _utc_datetime(now, field_name="now")
    expires = created + timedelta(seconds=ttl_seconds)
    plan_data = trace.result.plan.to_dict()
    plan_json = _canonical_json(plan_data, field_name="plan")
    observation_fingerprint = _sha256_json(trace.observation, field_name="observation")
    plan_fingerprint = hashlib.sha256(plan_json.encode("utf-8")).hexdigest()
    goal = trace.result.plan.goal
    if not isinstance(goal, str) or not goal.strip():
        raise CognitivePassHandoffValidationError("plan.goal must be a non-empty string.")
    cycle = trace.cycle
    if cycle is not None and (isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0):
        raise CognitivePassHandoffValidationError("trace.cycle must be a non-negative integer or null.")

    return CognitivePassHandoff(
        handoff_id=str(uuid.uuid4()),
        contract_version=HANDOFF_CONTRACT_VERSION,
        created_at=created.isoformat(),
        expires_at=expires.isoformat(),
        cognitive_cycle=cycle,
        goal=goal,
        observation_fingerprint=observation_fingerprint,
        plan_fingerprint=plan_fingerprint,
        _plan_json=plan_json,
    )


def validate_cognitive_pass_handoff(
    handoff: CognitivePassHandoff | Mapping[str, Any],
    *,
    expected_goal: str | None = None,
    expected_observation: Any = None,
    now: datetime | None = None,
) -> CognitivePassHandoff:
    """Validate integrity, expiry, and optional execution-context binding.

    This is an integrity/context check only. It does not validate action
    authorization, current world-state preconditions, approval, or replay state.
    """
    if isinstance(handoff, Mapping):
        resolved = CognitivePassHandoff.from_dict(handoff)
    elif isinstance(handoff, CognitivePassHandoff):
        # Frozen dataclasses prevent ordinary mutation, but callers can still
        # construct one directly or use low-level mutation techniques. Rebuild
        # through the canonical parser so both input forms share all checks.
        resolved = CognitivePassHandoff.from_dict(handoff.to_dict())
    else:
        raise TypeError("handoff must be CognitivePassHandoff or a mapping")

    created = _parse_timestamp(resolved.created_at, field_name="created_at")
    expires = _parse_timestamp(resolved.expires_at, field_name="expires_at")
    current = _utc_datetime(now, field_name="now")
    if current < created:
        raise CognitivePassHandoffValidationError("handoff is not yet valid.")
    if current >= expires:
        raise CognitivePassHandoffValidationError("handoff has expired.")

    plan = resolved.plan
    if hashlib.sha256(_canonical_json(plan, field_name="plan").encode("utf-8")).hexdigest() != resolved.plan_fingerprint:
        raise CognitivePassHandoffValidationError("plan_fingerprint does not match plan.")
    if plan.get("goal") != resolved.goal:
        raise CognitivePassHandoffValidationError("handoff goal does not match plan.goal.")
    if expected_goal is not None:
        if not isinstance(expected_goal, str) or expected_goal != resolved.goal:
            raise CognitivePassHandoffValidationError("handoff goal does not match execution goal.")
    if expected_observation is not None:
        expected_fingerprint = _sha256_json(expected_observation, field_name="expected_observation")
        if expected_fingerprint != resolved.observation_fingerprint:
            raise CognitivePassHandoffValidationError("observation does not match Cognitive Pass context.")
    return resolved


__all__ = [
    "CognitivePassHandoff",
    "CognitivePassHandoffError",
    "CognitivePassHandoffSerializationError",
    "CognitivePassHandoffValidationError",
    "HANDOFF_CONTRACT_VERSION",
    "DEFAULT_HANDOFF_TTL_SECONDS",
    "MAX_HANDOFF_TTL_SECONDS",
    "create_cognitive_pass_handoff",
    "validate_cognitive_pass_handoff",
]
