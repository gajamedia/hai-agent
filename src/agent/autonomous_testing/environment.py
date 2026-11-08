"""Per-case isolated simulation environment for autonomous testing.

This module owns runtime lifecycle only. Actions are always dispatched through
ExecutionRuntime.run(), which preserves the existing simulation executor path.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from .contracts import ROOM_STATE_SCHEMA, TestEnvironmentConfig


class IsolatedEnvironmentError(RuntimeError):
    """Raised when an isolated test environment cannot be safely created."""


class IsolatedTestEnvironment:
    """One isolated ExecutionRuntime and one private memory directory per case."""

    def __init__(
        self,
        *,
        initial_state: dict[str, str],
        config: Optional[TestEnvironmentConfig] = None,
        runtime_factory: Optional[Callable[..., Any]] = None,
        temp_root: Optional[str | Path] = None,
    ) -> None:
        self.config = config or TestEnvironmentConfig()
        if not self.config.sequential_only:
            raise IsolatedEnvironmentError(
                "Parallel execution is disabled until isolation is explicitly validated."
            )

        self._closed = False
        self._runtime = None
        self._runtime_close_attempted = False
        self._temp_dir: Optional[Path] = None
        self._runtime_factory = runtime_factory

        self._validate_state(initial_state)

        try:
            self._temp_dir = Path(
                tempfile.mkdtemp(prefix="hai-at-case-", dir=str(temp_root) if temp_root else None)
            )
            memory_path = self._temp_dir / "simulation_memory.db"

            factory = runtime_factory or self._default_runtime_factory
            # Never pass None: the production factory would fall back to the
            # shared HAI_SIMULATION_MEMORY_PATH / data/simulation_memory.db.
            self._runtime = factory(memory_path=memory_path)
            observed = self._runtime.set_state(dict(initial_state))
            self._validate_state(observed)
            if observed != initial_state:
                raise IsolatedEnvironmentError(
                    "Runtime did not retain the requested initial state."
                )
        except Exception as exc:
            cleanup_errors = self._cleanup_resources()
            if isinstance(exc, IsolatedEnvironmentError) and not cleanup_errors:
                raise
            message = (
                str(exc)
                if isinstance(exc, IsolatedEnvironmentError)
                else f"Could not create isolated execution runtime: {exc}"
            )
            if cleanup_errors:
                message += "; cleanup also failed: " + "; ".join(
                    self._format_cleanup_error(error) for error in cleanup_errors
                )
            raise IsolatedEnvironmentError(message) from exc

    @staticmethod
    def _default_runtime_factory(*, memory_path: Path):
        # Lazy import prevents importing the simulation runtime at package
        # import time and keeps contracts/scenario generation data-only.
        from api.execution_runtime import create_execution_runtime
        return create_execution_runtime(memory_path=memory_path)

    @staticmethod
    def _validate_state(state: dict[str, str]) -> None:
        if not isinstance(state, dict):
            raise ValueError("state must be a dictionary")
        if set(state) != set(ROOM_STATE_SCHEMA):
            raise ValueError(
                f"state must contain exactly {sorted(ROOM_STATE_SCHEMA)}"
            )
        for key, allowed in ROOM_STATE_SCHEMA.items():
            if not isinstance(state[key], str) or state[key] not in allowed:
                raise ValueError(f"invalid value for state field {key!r}")

    @property
    def runtime(self):
        if self._closed or self._runtime is None:
            raise IsolatedEnvironmentError("Environment is closed.")
        return self._runtime

    @property
    def memory_path(self) -> Path:
        if self._temp_dir is None:
            raise IsolatedEnvironmentError("Environment has no temporary directory.")
        return self._temp_dir / "simulation_memory.db"

    @property
    def storage_directory(self) -> Path:
        if self._temp_dir is None:
            raise IsolatedEnvironmentError("Environment has no temporary directory.")
        return self._temp_dir

    @property
    def is_closed(self) -> bool:
        return self._closed

    def observe(self) -> dict[str, Any]:
        return dict(self.runtime.get_state())

    def run(
        self,
        *,
        actions: list[str],
        goal_description: str,
        target_state: dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Dispatch via the existing ExecutionRuntime; never call handlers here."""
        return self.runtime.run(
            actions=actions,
            goal_description=goal_description,
            target_state=target_state,
            idempotency_key=idempotency_key,
        )

    @staticmethod
    def _format_cleanup_error(error: Exception) -> str:
        return f"{type(error).__name__}: {error}"

    def _cleanup_resources(self) -> list[Exception]:
        """Attempt runtime and private-directory cleanup independently.

        Runtime close is attempted at most once because a failed close may have
        partially released internal resources and is not assumed retry-safe.
        Temporary-directory removal remains retryable: its reference is kept
        until shutil.rmtree succeeds (or the directory is already absent).
        """
        errors: list[Exception] = []

        runtime = self._runtime
        if runtime is not None and not self._runtime_close_attempted:
            self._runtime_close_attempted = True
            # Detach before invoking user/runtime code so re-entrant cleanup
            # cannot invoke the same close hook a second time.
            self._runtime = None
            close = getattr(runtime, "close", None)
            if not callable(close):
                close = getattr(getattr(runtime, "agent", None), "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)

        temp_dir = self._temp_dir
        if temp_dir is not None:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                self._temp_dir = None
            except Exception as exc:
                # Keep the exact private path so a later close() can retry.
                errors.append(exc)

        self._closed = self._runtime is None and self._temp_dir is None
        return errors

    def _close_partial(self) -> None:
        """Compatibility helper for partial-construction cleanup."""
        errors = self._cleanup_resources()
        if errors:
            detail = "; ".join(self._format_cleanup_error(error) for error in errors)
            raise IsolatedEnvironmentError(
                f"Isolated environment cleanup failed: {detail}"
            ) from errors[0]

    def close(self) -> None:
        if self._closed:
            return
        errors = self._cleanup_resources()
        if errors:
            detail = "; ".join(self._format_cleanup_error(error) for error in errors)
            raise IsolatedEnvironmentError(
                f"Isolated environment cleanup failed: {detail}"
            ) from errors[0]

    def __enter__(self) -> "IsolatedTestEnvironment":
        if self._closed:
            raise IsolatedEnvironmentError("Cannot re-enter a closed environment.")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


__all__ = ["IsolatedEnvironmentError", "IsolatedTestEnvironment"]
