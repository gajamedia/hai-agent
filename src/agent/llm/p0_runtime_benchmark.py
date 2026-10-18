"""Adapter-driven runner for real Step 39 P0 benchmark executions.

The runner deliberately ships without a synthetic/default runtime adapter. A project
adapter must execute the real runtime and derive evidence from observed state,
security/approval decisions, audit records, and trace verification.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from agent.llm.p0_acceptance_gate import (
    P0AcceptanceReport,
    P0AcceptanceThresholds,
    P0RunEvidence,
    evaluate_p0_acceptance,
)


@dataclass(frozen=True)
class P0BenchmarkTask:
    """A controlled task specification supplied by the project adapter."""

    task_id: str
    description: str
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dictionary")


class P0BenchmarkAdapter(Protocol):
    """Contract implemented by a project-specific real-runtime adapter."""

    def tasks(self) -> Iterable[P0BenchmarkTask]:
        """Return the controlled task matrix; do not synthesize evidence here."""
        ...

    def execute(self, task: P0BenchmarkTask, run_index: int) -> P0RunEvidence:
        """Execute one task on the real runtime and return measured evidence."""
        ...


class P0BenchmarkError(RuntimeError):
    """Raised when the adapter or benchmark contract is invalid."""


def load_adapter(factory_path: str) -> P0BenchmarkAdapter:
    """Load ``package.module:factory`` and call the factory to create an adapter."""
    if not isinstance(factory_path, str) or ":" not in factory_path:
        raise P0BenchmarkError("Adapter must be specified as 'module.path:factory_name'.")
    module_name, factory_name = factory_path.split(":", 1)
    if not module_name or not factory_name:
        raise P0BenchmarkError("Adapter must be specified as 'module.path:factory_name'.")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise P0BenchmarkError(f"Could not load adapter factory '{factory_path}': {exc}") from exc
    if not callable(factory):
        raise P0BenchmarkError(f"Adapter factory '{factory_path}' is not callable.")
    adapter = factory()
    if not callable(getattr(adapter, "tasks", None)) or not callable(getattr(adapter, "execute", None)):
        raise P0BenchmarkError("Adapter must expose callable tasks() and execute(task, run_index).")
    return adapter


def _validate_task_matrix(
    tasks: Iterable[P0BenchmarkTask], thresholds: P0AcceptanceThresholds
) -> tuple[P0BenchmarkTask, ...]:
    try:
        rows = tuple(tasks)
    except Exception as exc:
        raise P0BenchmarkError(f"Adapter tasks() failed: {exc}") from exc
    if any(not isinstance(task, P0BenchmarkTask) for task in rows):
        raise P0BenchmarkError("Every task must be a P0BenchmarkTask instance.")
    expected = tuple(f"task-{i:02d}" for i in range(1, thresholds.expected_tasks + 1))
    actual = tuple(task.task_id for task in rows)
    if len(actual) != len(set(actual)):
        raise P0BenchmarkError("Task matrix contains duplicate task IDs.")
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise P0BenchmarkError(
            f"Task matrix must contain exactly {len(expected)} canonical tasks; "
            f"missing={missing}, unknown={unknown}."
        )
    return tuple(sorted(rows, key=lambda task: task.task_id))


def run_p0_benchmark(
    adapter: P0BenchmarkAdapter,
    *,
    evidence_path: str | Path,
    report_path: str | Path,
    thresholds: P0AcceptanceThresholds = P0AcceptanceThresholds(),
) -> P0AcceptanceReport:
    """Run the configured adapter for every task/repetition and persist measured evidence.

    Any adapter exception or evidence/task identity mismatch aborts the run. The output
    files are written only after all expected runs returned valid evidence.
    """
    if not callable(getattr(adapter, "tasks", None)) or not callable(getattr(adapter, "execute", None)):
        raise P0BenchmarkError("Adapter must expose callable tasks() and execute(task, run_index).")
    try:
        supplied_tasks = adapter.tasks()
    except Exception as exc:
        raise P0BenchmarkError(f"Adapter tasks() failed: {exc}") from exc
    tasks = _validate_task_matrix(supplied_tasks, thresholds)
    evidence: list[P0RunEvidence] = []
    expected_count = thresholds.expected_tasks * thresholds.runs_per_task
    for task in tasks:
        for run_index in range(1, thresholds.runs_per_task + 1):
            try:
                row = adapter.execute(task, run_index)
            except Exception as exc:
                raise P0BenchmarkError(
                    f"Runtime adapter failed for {task.task_id} run {run_index}: {exc}"
                ) from exc
            if not isinstance(row, P0RunEvidence):
                raise P0BenchmarkError(
                    f"{task.task_id} run {run_index}: adapter must return P0RunEvidence."
                )
            if row.task_id != task.task_id or row.run_index != run_index:
                raise P0BenchmarkError(
                    f"Evidence identity mismatch for {task.task_id} run {run_index}; "
                    f"received {row.task_id} run {row.run_index}."
                )
            evidence.append(row)

    if len(evidence) != expected_count:
        raise P0BenchmarkError(f"Expected {expected_count} runs but collected {len(evidence)}.")

    report = evaluate_p0_acceptance(evidence, thresholds=thresholds)
    evidence_output = Path(evidence_path)
    report_output = Path(report_path)
    evidence_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    # JSONL is emitted only from completed, validated adapter observations.
    evidence_output.write_text(
        "".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False) + "\n" for row in evidence),
        encoding="utf-8",
    )
    report_output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "P0BenchmarkAdapter",
    "P0BenchmarkError",
    "P0BenchmarkTask",
    "load_adapter",
    "run_p0_benchmark",
]
