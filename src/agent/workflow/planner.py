from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class WorkflowPlanError(ValueError):
    """Invalid workflow task graph."""


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    task_type: str
    description: str
    depends_on: tuple[str, ...] = ()
    input_data: Mapping[str, Any] = field(default_factory=dict)
    max_attempts: int = 3

    def __post_init__(self) -> None:
        for name in ("task_key", "task_type", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise WorkflowPlanError(f"{name} must be non-empty text")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts < 1:
            raise WorkflowPlanError("max_attempts must be a positive integer")
        if isinstance(self.depends_on, (str, bytes)):
            raise WorkflowPlanError("depends_on must be a sequence of task keys")
        deps = tuple(self.depends_on)
        if any(not isinstance(item, str) or not item.strip() for item in deps):
            raise WorkflowPlanError("dependency keys must be non-empty text")
        if len(set(deps)) != len(deps):
            raise WorkflowPlanError(f"duplicate dependency in task {self.task_key!r}")
        object.__setattr__(self, "task_key", self.task_key.strip())
        object.__setattr__(self, "task_type", self.task_type.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "depends_on", tuple(item.strip() for item in deps))
        if not isinstance(self.input_data, Mapping):
            raise WorkflowPlanError("input_data must be a mapping")
        object.__setattr__(self, "input_data", dict(self.input_data))


@dataclass(frozen=True)
class WorkflowPlan:
    tasks: tuple[TaskSpec, ...]
    execution_order: tuple[str, ...]
    dependency_map: Mapping[str, tuple[str, ...]]
    initial_ready: tuple[str, ...]


class TaskDependencyPlanner:
    """Pure deterministic DAG validator and topological scheduler."""

    def plan(self, tasks: Iterable[TaskSpec | Mapping[str, Any]]) -> WorkflowPlan:
        specs = tuple(item if isinstance(item, TaskSpec) else TaskSpec(**dict(item)) for item in tasks)
        if not specs:
            raise WorkflowPlanError("workflow plan must contain at least one task")
        by_key: dict[str, TaskSpec] = {}
        for spec in specs:
            if spec.task_key in by_key:
                raise WorkflowPlanError(f"duplicate task_key: {spec.task_key}")
            by_key[spec.task_key] = spec
        for spec in specs:
            for dep in spec.depends_on:
                if dep not in by_key:
                    raise WorkflowPlanError(f"unknown dependency {dep!r} referenced by {spec.task_key!r}")
                if dep == spec.task_key:
                    raise WorkflowPlanError(f"task {spec.task_key!r} cannot depend on itself")

        # Kahn's algorithm. Original declaration order breaks ties deterministically.
        index = {spec.task_key: i for i, spec in enumerate(specs)}
        indegree = {key: len(spec.depends_on) for key, spec in by_key.items()}
        dependents: dict[str, list[str]] = {key: [] for key in by_key}
        for spec in specs:
            for dep in spec.depends_on:
                dependents[dep].append(spec.task_key)
        ready = sorted((key for key, degree in indegree.items() if degree == 0), key=index.__getitem__)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(dependents[current], key=index.__getitem__):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort(key=index.__getitem__)
        if len(order) != len(specs):
            cyclic = [key for key, degree in indegree.items() if degree > 0]
            raise WorkflowPlanError("dependency cycle detected involving: " + ", ".join(cyclic))
        deps = {spec.task_key: spec.depends_on for spec in specs}
        return WorkflowPlan(
            tasks=specs,
            execution_order=tuple(order),
            dependency_map=deps,
            initial_ready=tuple(key for key in order if not deps[key]),
        )
