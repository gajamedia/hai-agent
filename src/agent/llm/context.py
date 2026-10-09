from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any


_DEFAULT_MAX_MEMORY_ITEMS = 20
_DEFAULT_MAX_STRING_LENGTH = 4000
_DEFAULT_MAX_DEPTH = 8


@dataclass(frozen=True)
class AgentContext:
    """Provider-neutral, LLM-ready snapshot of the Agent's current context."""

    goal: Any = None
    state: Any = None
    world_model: Any = None
    relevant_memory: tuple[Any, ...] = ()
    strategy: Any = None
    self_awareness: Any = None
    calibration: Any = None
    metacognition: Any = None
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the complete context as plain JSON-compatible Python data."""
        return {
            "goal": _normalize(self.goal),
            "state": _normalize(self.state),
            "world_model": _normalize(self.world_model),
            "relevant_memory": [
                _normalize(item) for item in self.relevant_memory
            ],
            "strategy": _normalize(self.strategy),
            "self_awareness": _normalize(self.self_awareness),
            "calibration": _normalize(self.calibration),
            "metacognition": _normalize(self.metacognition),
            "metadata": _normalize(self.metadata),
        }

    def to_prompt(self) -> str:
        """Render the context in a stable, human-readable prompt block."""
        sections = (
            ("GOAL", self.goal),
            ("CURRENT STATE", self.state),
            ("WORLD MODEL", self.world_model),
            ("RELEVANT MEMORY", list(self.relevant_memory)),
            ("ACTIVE STRATEGY", self.strategy),
            ("SELF-AWARENESS", self.self_awareness),
            ("CALIBRATION", self.calibration),
            ("METACOGNITION", self.metacognition),
        )

        lines: list[str] = []
        for title, value in sections:
            lines.append(f"## {title}")
            lines.append(_render(value))
            lines.append("")

        return "\n".join(lines).rstrip()


class AgentContextBuilder:
    """
    Build an LLM-ready AgentContext without coupling the builder to concrete
    Agent, Memory, WorldModel, or Learning implementations.

    The builder deliberately accepts mappings, dataclasses, ordinary objects,
    and scalar values. This makes it safe to introduce into the existing Agent
    incrementally while the internal models continue evolving.
    """

    def __init__(
        self,
        *,
        max_memory_items: int = _DEFAULT_MAX_MEMORY_ITEMS,
        max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
        max_depth: int = _DEFAULT_MAX_DEPTH,
    ) -> None:
        if max_memory_items < 0:
            raise ValueError("max_memory_items must be >= 0")
        if max_string_length <= 0:
            raise ValueError("max_string_length must be > 0")
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")

        self.max_memory_items = max_memory_items
        self.max_string_length = max_string_length
        self.max_depth = max_depth

    def build(
        self,
        *,
        goal: Any = None,
        state: Any = None,
        world_model: Any = None,
        relevant_memory: Sequence[Any] | None = None,
        strategy: Any = None,
        self_awareness: Any = None,
        calibration: Any = None,
        metacognition: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentContext:
        """Create a normalized immutable context snapshot."""
        if relevant_memory is None:
            memory_items: tuple[Any, ...] = ()
        elif isinstance(relevant_memory, (str, bytes, bytearray)):
            raise TypeError("relevant_memory must be a sequence of memory items")
        else:
            memory_items = tuple(relevant_memory[: self.max_memory_items])

        normalized_metadata = _normalize(
            metadata,
            max_string_length=self.max_string_length,
            max_depth=self.max_depth,
        )

        if normalized_metadata is not None and not isinstance(
            normalized_metadata,
            Mapping,
        ):
            raise TypeError("metadata must normalize to a mapping")

        return AgentContext(
            goal=_normalize(
                goal,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            state=_normalize(
                state,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            world_model=_normalize(
                world_model,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            relevant_memory=tuple(
                _normalize(
                    item,
                    max_string_length=self.max_string_length,
                    max_depth=self.max_depth,
                )
                for item in memory_items
            ),
            strategy=_normalize(
                strategy,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            self_awareness=_normalize(
                self_awareness,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            calibration=_normalize(
                calibration,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            metacognition=_normalize(
                metacognition,
                max_string_length=self.max_string_length,
                max_depth=self.max_depth,
            ),
            metadata=normalized_metadata,
        )


def _normalize(
    value: Any,
    *,
    max_string_length: int = _DEFAULT_MAX_STRING_LENGTH,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    _depth: int = 0,
) -> Any:
    """Normalize common Python objects into deterministic JSON-compatible data."""
    if _depth > max_depth:
        return "<max-depth-reached>"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return value[:max_string_length] + "…"

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:max_string_length]

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize(
                getattr(value, field.name),
                max_string_length=max_string_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized[str(key)] = _normalize(
                item,
                max_string_length=max_string_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        return normalized

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _normalize(
                item,
                max_string_length=max_string_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    if isinstance(value, set):
        return [
            _normalize(
                item,
                max_string_length=max_string_length,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in sorted(value, key=repr)
        ]

    if hasattr(value, "__dict__"):
        return _normalize(
            vars(value),
            max_string_length=max_string_length,
            max_depth=max_depth,
            _depth=_depth + 1,
        )

    return str(value)


def _render(value: Any) -> str:
    normalized = _normalize(value)
    if isinstance(normalized, str):
        return normalized

    # Local import keeps the core builder dependency-free at import time.
    import json

    return json.dumps(
        normalized,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


__all__ = [
    "AgentContext",
    "AgentContextBuilder",
]
