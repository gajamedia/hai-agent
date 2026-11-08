"""Contracts for workload analysis and database recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class WorkloadType(str, Enum):
    INTERNAL_STATE = "internal_state"
    TRANSACTIONAL_READ = "transactional_read"
    ANALYTICS = "analytics"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class WorkloadRequest:
    description: str
    workload_type: Optional[WorkloadType] = None
    read_only: bool = True
    concurrent_users: int = 1
    estimated_dataset_mb: Optional[int] = None
    requires_sql: bool = True
    requires_persistence: bool = True


@dataclass(frozen=True)
class WorkloadAnalysis:
    workload_type: WorkloadType
    confidence: float
    signals: Tuple[str, ...]
    assumptions: Tuple[str, ...]


@dataclass(frozen=True)
class DatabaseRecommendation:
    workload_type: WorkloadType
    recommended_database: Optional[str]
    eligible: bool
    requires_user_approval: bool
    reasons: Tuple[str, ...]
    blockers: Tuple[str, ...]
    detected_options: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
