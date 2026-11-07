"""Validated, data-only contracts for autonomous testing. No execution/API imports."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Mapping, Optional

ROOM_STATE_SCHEMA = {
    "door": frozenset({"locked", "unlocked"}),
    "box": frozenset({"closed", "open"}),
    "key": frozenset({"inside_box", "agent", "missing"}),
    "spare_key": frozenset({"available", "found", "unavailable"}),
    "agent_location": frozenset({"room"}),
}

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class BatchStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"

class FaultType(str, Enum):
    ACTION_FAILURE = "action_failure"
    TEMPORARY_UNAVAILABILITY = "temporary_unavailability"
    OBSERVATION_INTERRUPTION = "observation_interruption"

class CognitiveRuntimeMode(str, Enum):
    ISOLATED = "isolated"
    SHARED = "shared"

def _plain(v):
    """Convert nested contracts and immutable containers into JSON primitives."""
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, datetime):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware")
        return v.astimezone(timezone.utc).isoformat()
    if is_dataclass(v) and not isinstance(v, type):
        return {item.name: _plain(getattr(v, item.name)) for item in fields(v)}
    if isinstance(v, Mapping):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_plain(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True))
    return deepcopy(v)

def _freeze(v):
    if isinstance(v, Mapping): return MappingProxyType({str(k): _freeze(x) for k, x in v.items()})
    if isinstance(v, (list, tuple)): return tuple(_freeze(x) for x in v)
    if isinstance(v, set): return frozenset(_freeze(x) for x in v)
    return deepcopy(v)

def _mapping(v, name):
    if not isinstance(v, Mapping): raise TypeError(f"{name} must be a mapping")
    try: json.dumps(_plain(v), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as e: raise ValueError(f"{name} must be JSON serializable") from e
    return _freeze(v)

def _text(v, name, limit=500):
    if not isinstance(v, str) or not v.strip(): raise ValueError(f"{name} must be non-empty text")
    if len(v) > limit: raise ValueError(f"{name} exceeds {limit} characters")
    return v.strip()

def _state(v, name, partial):
    if not isinstance(v, Mapping): raise TypeError(f"{name} must be a mapping")
    keys = set(v)
    if partial:
        if not keys or not keys.issubset(ROOM_STATE_SCHEMA): raise ValueError(f"{name} must use known state keys")
    elif keys != set(ROOM_STATE_SCHEMA):
        raise ValueError(f"{name} must contain exactly {sorted(ROOM_STATE_SCHEMA)}")
    for k, val in v.items():
        if val not in ROOM_STATE_SCHEMA[k]: raise ValueError(f"invalid {name}[{k!r}] value {val!r}")

@dataclass(frozen=True)
class FaultInjectionPolicy:
    enabled: bool = False
    fault_type: Optional[FaultType] = None
    target_action: Optional[str] = None
    trigger_step: Optional[int] = None
    max_occurrences: int = 1
    def __post_init__(self):
        if self.fault_type is not None and not isinstance(self.fault_type, FaultType):
            object.__setattr__(self, "fault_type", FaultType(self.fault_type))
        if self.enabled and self.fault_type is None: raise ValueError("fault_type required when enabled")
        if self.target_action is not None: object.__setattr__(self, "target_action", _text(self.target_action, "target_action", 100))
        if self.trigger_step is not None and (not isinstance(self.trigger_step, int) or self.trigger_step < 1): raise ValueError("trigger_step must be positive")
        if not isinstance(self.max_occurrences, int) or self.max_occurrences < 1: raise ValueError("max_occurrences must be positive")
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class RecoveryPolicy:
    enabled: bool = True
    max_attempts: int = 2
    retryable_fault_types: tuple[FaultType, ...] = (FaultType.ACTION_FAILURE, FaultType.TEMPORARY_UNAVAILABILITY)
    stop_on_unknown_outcome: bool = True
    def __post_init__(self):
        if not isinstance(self.max_attempts, int) or self.max_attempts < 0: raise ValueError("max_attempts must be non-negative")
        vals = tuple(FaultType(x) for x in self.retryable_fault_types)
        if len(set(vals)) != len(vals): raise ValueError("retryable fault types must be unique")
        object.__setattr__(self, "retryable_fault_types", vals)
        if not self.enabled: object.__setattr__(self, "max_attempts", 0)
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class TestEnvironmentConfig:
    __test__ = False
    cognitive_runtime_mode: CognitiveRuntimeMode = CognitiveRuntimeMode.ISOLATED
    sequential_only: bool = True
    simulation_memory_mode: str = "in_memory"
    def __post_init__(self):
        if not isinstance(self.cognitive_runtime_mode, CognitiveRuntimeMode):
            object.__setattr__(self, "cognitive_runtime_mode", CognitiveRuntimeMode(self.cognitive_runtime_mode))
        if self.simulation_memory_mode not in {"in_memory", "temporary_file"}: raise ValueError("invalid simulation_memory_mode")
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    difficulty: Difficulty
    seed: int
    initial_state: Mapping[str, str]
    goal: Mapping[str, str]
    goal_description: str
    fault_policy: Optional[FaultInjectionPolicy] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        for n in ("scenario_id", "name", "goal_description"): object.__setattr__(self, n, _text(getattr(self, n), n))
        if not isinstance(self.difficulty, Difficulty): object.__setattr__(self, "difficulty", Difficulty(self.difficulty))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0: raise ValueError("seed must be non-negative")
        _state(self.initial_state, "initial_state", False); _state(self.goal, "goal", True)
        object.__setattr__(self, "initial_state", _mapping(self.initial_state, "initial_state"))
        object.__setattr__(self, "goal", _mapping(self.goal, "goal"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.fault_policy is not None and not isinstance(self.fault_policy, FaultInjectionPolicy): raise TypeError("invalid fault_policy")
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class AutonomousTestCase:
    case_id: str
    scenario: ScenarioDefinition
    status: CaseStatus = CaseStatus.PENDING
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id", 100))
        if not isinstance(self.scenario, ScenarioDefinition): raise TypeError("scenario must be ScenarioDefinition")
        if not isinstance(self.status, CaseStatus): object.__setattr__(self, "status", CaseStatus(self.status))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class AutonomousTestBatch:
    batch_id: str
    seed: int
    requested_cases: int
    difficulty: Difficulty
    cases: tuple[AutonomousTestCase, ...]
    recovery_policy: RecoveryPolicy = field(default_factory=RecoveryPolicy)
    environment_config: TestEnvironmentConfig = field(default_factory=TestEnvironmentConfig)
    status: BatchStatus = BatchStatus.PENDING
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self, "batch_id", _text(self.batch_id, "batch_id", 100))
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0: raise ValueError("seed must be non-negative")
        if not isinstance(self.requested_cases, int) or self.requested_cases < 1: raise ValueError("requested_cases must be positive")
        if not isinstance(self.difficulty, Difficulty): object.__setattr__(self, "difficulty", Difficulty(self.difficulty))
        cases = tuple(self.cases)
        if len(cases) != self.requested_cases: raise ValueError("cases length must equal requested_cases")
        if any(not isinstance(x, AutonomousTestCase) for x in cases): raise TypeError("invalid case")
        ids = [x.case_id for x in cases]
        if len(ids) != len(set(ids)): raise ValueError("case IDs must be unique")
        object.__setattr__(self, "cases", cases)
        if not isinstance(self.recovery_policy, RecoveryPolicy): raise TypeError("invalid recovery_policy")
        if not isinstance(self.environment_config, TestEnvironmentConfig): raise TypeError("invalid environment_config")
        if not isinstance(self.status, BatchStatus): object.__setattr__(self, "status", BatchStatus(self.status))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class GoalEvaluation:
    goal: Mapping[str, str]
    initial_state: Mapping[str, str]
    final_state: Mapping[str, str]
    goal_satisfied: Optional[bool]
    evidence: tuple[str, ...] = ()
    reason: str = ""
    def __post_init__(self):
        _state(self.goal, "goal", True); _state(self.initial_state, "initial_state", False)
        if not isinstance(self.final_state, Mapping): raise TypeError("final_state must be a mapping")
        if self.final_state: _state(self.final_state, "final_state", True)
        if self.goal_satisfied not in (True, False, None): raise TypeError("goal_satisfied must be bool or None")
        mismatch = any(k in self.final_state and self.final_state[k] != v for k, v in self.goal.items())
        complete = set(self.goal).issubset(self.final_state)
        if self.goal_satisfied is True and (not complete or mismatch): raise ValueError("true requires complete matching final state")
        if self.goal_satisfied is False and not mismatch: raise ValueError("false requires observed mismatch")
        if self.goal_satisfied is None and mismatch: raise ValueError("unknown is invalid when mismatch proves failure")
        object.__setattr__(self, "goal", _mapping(self.goal, "goal"))
        object.__setattr__(self, "initial_state", _mapping(self.initial_state, "initial_state"))
        object.__setattr__(self, "final_state", _mapping(self.final_state, "final_state"))
        object.__setattr__(self, "evidence", tuple(_text(x, "evidence", 1000) for x in self.evidence))
        object.__setattr__(self, "reason", self.reason.strip() if isinstance(self.reason, str) else "")
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class TestCaseResult:
    __test__ = False
    case_id: str
    status: CaseStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    evaluation: Optional[GoalEvaluation] = None
    recovery_attempts: int = 0
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id", 100))
        if not isinstance(self.status, CaseStatus): object.__setattr__(self, "status", CaseStatus(self.status))
        for n in ("started_at", "finished_at"):
            v = getattr(self, n)
            if v is not None and (not isinstance(v, datetime) or v.tzinfo is None or v.utcoffset() is None): raise ValueError(f"{n} must be timezone-aware")
        if self.started_at and self.finished_at and self.finished_at < self.started_at: raise ValueError("finished_at precedes started_at")
        if not isinstance(self.recovery_attempts, int) or self.recovery_attempts < 0: raise ValueError("recovery_attempts must be non-negative")
        if self.status == CaseStatus.PASSED and (self.evaluation is None or self.evaluation.goal_satisfied is not True): raise ValueError("passed requires verified goal success")
        if self.status == CaseStatus.INCONCLUSIVE and self.evaluation is not None and self.evaluation.goal_satisfied is not None: raise ValueError("inconclusive requires unknown evaluation")
        if self.error is not None: object.__setattr__(self, "error", _text(self.error, "error", 2000))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
    def to_dict(self): return _plain(self)

@dataclass(frozen=True)
class TestBatchResult:
    __test__ = False
    batch_id: str
    status: BatchStatus
    results: tuple[TestCaseResult, ...]
    seed: int
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        object.__setattr__(self, "batch_id", _text(self.batch_id, "batch_id", 100))
        if not isinstance(self.status, BatchStatus): object.__setattr__(self, "status", BatchStatus(self.status))
        results = tuple(self.results)
        if any(not isinstance(x, TestCaseResult) for x in results): raise TypeError("invalid result")
        ids = [x.case_id for x in results]
        if len(ids) != len(set(ids)): raise ValueError("result case IDs must be unique")
        object.__setattr__(self, "results", results)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0: raise ValueError("seed must be non-negative")
        for n in ("started_at", "finished_at"):
            v = getattr(self, n)
            if v is not None and (not isinstance(v, datetime) or v.tzinfo is None or v.utcoffset() is None): raise ValueError(f"{n} must be timezone-aware")
        if self.started_at and self.finished_at and self.finished_at < self.started_at: raise ValueError("finished_at precedes started_at")
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
    @property
    def total(self): return len(self.results)
    @property
    def counts(self): return {s.value: sum(x.status == s for x in self.results) for s in CaseStatus}
    def to_dict(self):
        d = _plain(self); d["total"] = self.total; d["counts"] = self.counts; return d

def canonical_json(value: Any) -> str:
    if hasattr(value, "to_dict"): value = value.to_dict()
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

__all__ = [
    "ROOM_STATE_SCHEMA", "Difficulty", "BatchStatus", "CaseStatus", "FaultType",
    "CognitiveRuntimeMode", "FaultInjectionPolicy", "RecoveryPolicy",
    "TestEnvironmentConfig", "ScenarioDefinition", "AutonomousTestCase",
    "AutonomousTestBatch", "GoalEvaluation", "TestCaseResult", "TestBatchResult",
    "canonical_json",
]
