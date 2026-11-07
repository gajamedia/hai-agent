"""Sequential batch orchestration for autonomous testing.

The manager owns batch lifecycle only. Individual cases are delegated to the
existing SequentialTestRunner; no action executor is implemented here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Optional
from uuid import uuid4

from .contracts import (
    AutonomousTestBatch,
    BatchStatus,
    CaseStatus,
    Difficulty,
    RecoveryPolicy,
    TestBatchResult,
    TestCaseResult,
    TestEnvironmentConfig,
)
from .result_repository import SQLiteTestResultRepository
from .runner import SequentialTestRunner
from .scenario_generator import AutonomousScenarioGenerator


class BatchManagerError(RuntimeError):
    """Raised when a batch cannot be created or safely orchestrated."""


class BatchShutdownRequested(BatchManagerError):
    """Raised when a non-persisted batch stops at a shutdown boundary."""


class AutonomousTestBatchManager:
    """Create and run deterministic batches one case at a time."""

    def __init__(
        self,
        *,
        runner: Optional[SequentialTestRunner] = None,
        repository: Optional[SQLiteTestResultRepository] = None,
        generator_factory=AutonomousScenarioGenerator,
        heartbeat_interval_seconds: float = 5.0,
    ) -> None:
        self.runner = runner or SequentialTestRunner()
        self.repository = repository
        self.generator_factory = generator_factory
        if (
            not isinstance(heartbeat_interval_seconds, (int, float))
            or isinstance(heartbeat_interval_seconds, bool)
            or heartbeat_interval_seconds <= 0
        ):
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self._lock = Lock()
        self._running = False
        # Sticky, cooperative shutdown signal. It is never cleared: once the
        # application begins shutting down, this manager must not start another
        # case even if a caller attempts to reuse the instance.
        self._shutdown_requested = Event()

    @property
    def shutdown_requested(self) -> bool:
        """Whether a sticky cooperative shutdown has been requested."""
        return self._shutdown_requested.is_set()

    def request_shutdown(self) -> None:
        """Request a cooperative stop at the next case boundary.

        An active case is allowed to finish and its result is persisted before
        the manager observes this signal. The current durable batch remains
        RUNNING for explicit operator reconciliation; it is never replayed.
        """
        self._shutdown_requested.set()

    def create_batch(
        self,
        *,
        count: int = 100,
        seed: int = 0,
        difficulty: Difficulty | str = Difficulty.MEDIUM,
        batch_id: Optional[str] = None,
        case_id_prefix: str = "case",
        inject_fault_every: int = 0,
        recovery_policy: Optional[RecoveryPolicy] = None,
        environment_config: Optional[TestEnvironmentConfig] = None,
        metadata: Optional[dict] = None,
    ) -> AutonomousTestBatch:
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        difficulty = Difficulty(difficulty)
        policy = recovery_policy or RecoveryPolicy()
        config = environment_config or TestEnvironmentConfig()
        if not isinstance(policy, RecoveryPolicy):
            raise TypeError("recovery_policy must be RecoveryPolicy")
        if not isinstance(config, TestEnvironmentConfig):
            raise TypeError("environment_config must be TestEnvironmentConfig")
        if not config.sequential_only:
            raise ValueError("AT-02.8 supports sequential-only execution")

        resolved_batch_id = batch_id or f"batch-{uuid4().hex}"
        generator = self.generator_factory(seed=seed)
        cases = generator.generate_cases(
            count,
            difficulty,
            case_id_prefix=case_id_prefix,
            inject_fault_every=inject_fault_every,
        )
        return AutonomousTestBatch(
            batch_id=resolved_batch_id,
            seed=seed,
            requested_cases=count,
            difficulty=difficulty,
            cases=cases,
            recovery_policy=policy,
            environment_config=config,
            status=BatchStatus.PENDING,
            metadata={
                **(metadata or {}),
                "manager": "AutonomousTestBatchManager",
                "manager_version": 1,
                "execution_mode": "sequential",
                "learning_integration": False,
            },
        )

    def run_batch(
        self,
        batch: AutonomousTestBatch,
        *,
        persist: bool = True,
        already_persisted: bool = False,
        already_claimed: bool = False,
        execution_owner_id: Optional[str] = None,
    ) -> TestBatchResult:
        if not isinstance(batch, AutonomousTestBatch):
            raise TypeError("batch must be AutonomousTestBatch")
        if not isinstance(persist, bool):
            raise TypeError("persist must be bool")
        if not isinstance(already_persisted, bool):
            raise TypeError("already_persisted must be bool")
        if not isinstance(already_claimed, bool):
            raise TypeError("already_claimed must be bool")
        if already_persisted and not persist:
            raise ValueError("already_persisted requires persist=True")
        if already_claimed and not already_persisted:
            raise ValueError("already_claimed requires already_persisted=True")
        if already_claimed and not persist:
            raise ValueError("already_claimed requires persist=True")
        if already_claimed and (
            not isinstance(execution_owner_id, str) or not execution_owner_id.strip()
        ):
            raise ValueError("already_claimed requires execution_owner_id")
        if not batch.environment_config.sequential_only:
            raise BatchManagerError("Parallel batch execution is not supported")
        if not self._lock.acquire(blocking=False):
            raise BatchManagerError("Batch manager is already running")

        self._running = True
        started_at = datetime.now(timezone.utc)
        results: list[TestCaseResult] = []
        claimed_by_this_run = already_claimed
        owner_id = execution_owner_id or uuid4().hex
        heartbeat_stop = Event()
        heartbeat_failed = Event()
        heartbeat_thread: Optional[Thread] = None
        shutdown_stopped = False

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(self.heartbeat_interval_seconds):
                try:
                    refreshed = self.repository.heartbeat_batch(
                        batch.batch_id,
                        owner_id,
                        datetime.now(timezone.utc).isoformat(),
                    )
                    if not refreshed:
                        heartbeat_failed.set()
                        return
                except Exception:
                    heartbeat_failed.set()
                    return

        try:
            if persist:
                if self.repository is None:
                    raise BatchManagerError(
                        "persist=True requires a SQLiteTestResultRepository"
                    )
                if not already_persisted:
                    self.repository.create_batch(batch)
                if not already_claimed:
                    claimed_by_this_run = self.repository.claim_batch_running(
                        batch.batch_id, started_at.isoformat(), owner_id
                    )
                    if not claimed_by_this_run:
                        raise BatchManagerError(
                            "Batch is not pending or has already been claimed"
                        )
                if not self.repository.heartbeat_batch(
                    batch.batch_id, owner_id, datetime.now(timezone.utc).isoformat()
                ):
                    heartbeat_failed.set()
                    raise BatchManagerError("Batch execution ownership was lost")
                heartbeat_thread = Thread(
                    target=heartbeat_loop,
                    name=f"autonomous-test-heartbeat-{batch.batch_id}",
                    daemon=True,
                )
                heartbeat_thread.start()

            for test_case in batch.cases:
                if heartbeat_failed.is_set():
                    raise BatchManagerError(
                        "Batch heartbeat was lost; execution stopped for reconciliation"
                    )
                # Never interrupt an action in progress. The signal is checked
                # only between cases, after the preceding result was persisted.
                if self._shutdown_requested.is_set():
                    shutdown_stopped = True
                    break
                try:
                    result = self.runner.run_case(
                        test_case,
                        recovery_policy=batch.recovery_policy,
                        environment_config=batch.environment_config,
                    )
                    if not isinstance(result, TestCaseResult):
                        raise TypeError(
                            "SequentialTestRunner.run_case must return TestCaseResult"
                        )
                except Exception as exc:
                    # A runner-level exception is isolated to this case. It is
                    # recorded as BLOCKED, not silently treated as a failure.
                    result = TestCaseResult(
                        case_id=test_case.case_id,
                        status=CaseStatus.BLOCKED,
                        started_at=datetime.now(timezone.utc),
                        finished_at=datetime.now(timezone.utc),
                        error=f"Runner error: {type(exc).__name__}: {exc}",
                        metadata={
                            "manager": "AutonomousTestBatchManager",
                            "decision_basis": "runner_exception",
                        },
                    )
                results.append(result)
                if persist:
                    self.repository.save_case_result(batch.batch_id, result)
                    if not self.repository.heartbeat_batch(
                        batch.batch_id, owner_id, datetime.now(timezone.utc).isoformat()
                    ):
                        heartbeat_failed.set()
                        raise BatchManagerError(
                            "Batch execution ownership was lost after case completion"
                        )

            if heartbeat_failed.is_set():
                raise BatchManagerError(
                    "Batch heartbeat was lost; completion cannot be confirmed"
                )
            # Also check after the final case: shutdown may have been requested
            # while that case was running. In that situation it remains a
            # durable RUNNING batch for reconciliation, not COMPLETED.
            if self._shutdown_requested.is_set():
                shutdown_stopped = True
            if shutdown_stopped and not persist:
                # Without durable persistence there is no RUNNING record to
                # reconcile. Preserve the legacy explicit-stop contract rather
                # than returning a partial result that looks durable.
                raise BatchShutdownRequested(
                    "Batch shutdown requested; execution stopped between cases"
                )
            if shutdown_stopped:
                return TestBatchResult(
                    batch_id=batch.batch_id,
                    status=BatchStatus.RUNNING,
                    results=tuple(results),
                    seed=batch.seed,
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    metadata={
                        "manager": "AutonomousTestBatchManager",
                        "manager_version": 1,
                        "execution_mode": "sequential",
                        "learning_integration": False,
                        "requested_cases": batch.requested_cases,
                        "completed_cases": len(results),
                        "shutdown_stopped": True,
                        "requires_reconciliation": True,
                    },
                )
            finished_at = datetime.now(timezone.utc)
            summary = TestBatchResult(
                batch_id=batch.batch_id,
                status=BatchStatus.COMPLETED,
                results=tuple(results),
                seed=batch.seed,
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "manager": "AutonomousTestBatchManager",
                    "manager_version": 1,
                    "execution_mode": "sequential",
                    "learning_integration": False,
                    "requested_cases": batch.requested_cases,
                    "completed_cases": len(results),
                },
            )
            if persist:
                self.repository.complete_batch(summary, worker_id=owner_id)
            return summary
        except Exception as exc:
            if (
                persist
                and self.repository is not None
                and claimed_by_this_run
                and not heartbeat_failed.is_set()
                and not shutdown_stopped
            ):
                try:
                    self.repository.mark_batch_failed(
                        batch.batch_id,
                        datetime.now(timezone.utc).isoformat(),
                        f"{type(exc).__name__}: {exc}",
                        worker_id=owner_id,
                    )
                except Exception:
                    # Preserve the original storage/orchestration exception.
                    pass
            raise
        finally:
            heartbeat_stop.set()
            try:
                if heartbeat_thread is not None:
                    # Do not release the manager lock or return control to the
                    # application lifespan while a heartbeat can still touch
                    # the repository. SQLite busy_timeout bounds ordinary DB
                    # contention; join is intentionally not best-effort.
                    heartbeat_thread.join()
            finally:
                self._running = False
                self._lock.release()


__all__ = [
    "AutonomousTestBatchManager",
    "BatchManagerError",
    "BatchShutdownRequested",
]
