"""Transparent heuristic workload classification; no execution or DB access."""

from __future__ import annotations

from typing import List

from .contracts import WorkloadAnalysis, WorkloadRequest, WorkloadType


class WorkloadAnalyzer:
    """Classify a task using explicit input first, then conservative keywords."""

    _INTERNAL = (
        "agent memory", "agent state", "internal metadata", "episodic memory",
        "semantic memory", "task state", "checkpoint", "session state",
    )
    _TRANSACTIONAL = (
        "transaction", "order", "invoice", "customer", "inventory",
        "sales record", "business database", "business data",
    )
    _ANALYTICS = (
        "analytics", "analytical", "report", "dashboard", "aggregation",
        "trend", "historical data", "sales analysis", "forecast",
    )
    _MIXED = (
        "join multiple databases", "multiple data sources",
        "combine databases", "cross-database",
    )

    def analyze(self, request: WorkloadRequest) -> WorkloadAnalysis:
        if request.workload_type is not None:
            return WorkloadAnalysis(
                workload_type=request.workload_type,
                confidence=1.0,
                signals=("explicit workload_type supplied by caller",),
                assumptions=(),
            )

        description = request.description.casefold()
        matches = {
            WorkloadType.INTERNAL_STATE: self._matches(description, self._INTERNAL),
            WorkloadType.TRANSACTIONAL_READ: self._matches(description, self._TRANSACTIONAL),
            WorkloadType.ANALYTICS: self._matches(description, self._ANALYTICS),
            WorkloadType.MIXED: self._matches(description, self._MIXED),
        }

        nonempty = [(kind, signals) for kind, signals in matches.items() if signals]
        if not nonempty:
            return WorkloadAnalysis(
                workload_type=WorkloadType.UNKNOWN,
                confidence=0.0,
                signals=(),
                assumptions=("No unambiguous workload keywords were found.",),
            )

        # Multiple categories indicate a combined task; don't silently privilege one.
        if len(nonempty) > 1:
            signals = tuple(
                f"{kind.value}: {signal}"
                for kind, items in nonempty
                for signal in items
            )
            return WorkloadAnalysis(
                workload_type=WorkloadType.MIXED,
                confidence=0.55,
                signals=signals,
                assumptions=("Multiple workload categories matched; explicit clarification may be needed.",),
            )

        kind, signals = nonempty[0]
        return WorkloadAnalysis(
            workload_type=kind,
            confidence=0.8,
            signals=tuple(signals),
            assumptions=("Classification is heuristic and should be confirmed for consequential tasks.",),
        )

    @staticmethod
    def _matches(description: str, vocabulary: tuple[str, ...]) -> List[str]:
        return [term for term in vocabulary if term in description]
