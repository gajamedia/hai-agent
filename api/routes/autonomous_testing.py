"""HTTP API for autonomous-testing batches.

The API coordinates batch lifecycle only. Execution remains delegated to the
AT-02 SequentialTestRunner and its per-case isolated environment.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from threading import Lock
from uuid import uuid4
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request, status

from agent.autonomous_testing.batch_manager import (
    AutonomousTestBatchManager,
    BatchManagerError,
)
from agent.autonomous_testing.contracts import (
    AutonomousTestBatch,
    AutonomousTestCase,
    BatchStatus,
    CaseStatus,
    CognitiveRuntimeMode,
    Difficulty,
    FaultInjectionPolicy,
    FaultType,
    RecoveryPolicy,
    ScenarioDefinition,
    TestEnvironmentConfig,
)
from agent.autonomous_testing.result_repository import (
    DuplicateBatchError,
    ResultNotFoundError,
    ResultRepositoryError,
    SQLiteTestResultRepository,
)

logger = logging.getLogger(__name__)

# A running batch is considered stale only after this durable heartbeat window.
# Staleness is diagnostic: it never authorizes automatic replay.
BATCH_HEARTBEAT_TIMEOUT_SECONDS = 60
router = APIRouter(prefix="/autonomous-testing", tags=["autonomous-testing"])


def _services(request: Request) -> tuple[AutonomousTestBatchManager, SQLiteTestResultRepository]:
    manager = getattr(request.app.state, "autonomous_test_batch_manager", None)
    repository = getattr(request.app.state, "autonomous_test_result_repository", None)
    if manager is None or repository is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "autonomous_testing_unavailable",
                "message": "Autonomous Testing services are not initialized.",
            },
        )
    return manager, repository


def _deserialize_batch(payload: dict[str, Any]) -> AutonomousTestBatch:
    """Rebuild validated contracts from a repository snapshot after restart."""
    try:
        cases = []
        for item in payload["cases"]:
            scenario_data = item["scenario"]
            fault_data = scenario_data.get("fault_policy")
            fault_policy = (
                FaultInjectionPolicy(
                    enabled=fault_data.get("enabled", False),
                    fault_type=(FaultType(fault_data["fault_type"]) if fault_data.get("fault_type") else None),
                    target_action=fault_data.get("target_action"),
                    trigger_step=fault_data.get("trigger_step"),
                    max_occurrences=fault_data.get("max_occurrences", 1),
                )
                if fault_data is not None else None
            )
            scenario = ScenarioDefinition(
                scenario_id=scenario_data["scenario_id"],
                name=scenario_data["name"],
                difficulty=Difficulty(scenario_data["difficulty"]),
                seed=scenario_data["seed"],
                initial_state=scenario_data["initial_state"],
                goal=scenario_data["goal"],
                goal_description=scenario_data["goal_description"],
                fault_policy=fault_policy,
                metadata=scenario_data.get("metadata", {}),
            )
            cases.append(AutonomousTestCase(
                case_id=item["case_id"],
                scenario=scenario,
                status=CaseStatus(item.get("status", "pending")),
                metadata=item.get("metadata", {}),
            ))

        recovery_data = payload.get("recovery_policy", {})
        recovery = RecoveryPolicy(
            enabled=recovery_data.get("enabled", True),
            max_attempts=recovery_data.get("max_attempts", 2),
            retryable_fault_types=tuple(
                FaultType(value)
                for value in recovery_data.get(
                    "retryable_fault_types",
                    [FaultType.ACTION_FAILURE.value, FaultType.TEMPORARY_UNAVAILABILITY.value],
                )
            ),
            stop_on_unknown_outcome=recovery_data.get("stop_on_unknown_outcome", True),
        )
        environment_data = payload.get("environment_config", {})
        environment = TestEnvironmentConfig(
            cognitive_runtime_mode=CognitiveRuntimeMode(
                environment_data.get("cognitive_runtime_mode", "isolated")
            ),
            sequential_only=environment_data.get("sequential_only", True),
            simulation_memory_mode=environment_data.get("simulation_memory_mode", "in_memory"),
        )
        return AutonomousTestBatch(
            batch_id=payload["batch_id"],
            seed=payload["seed"],
            requested_cases=payload["requested_cases"],
            difficulty=Difficulty(payload["difficulty"]),
            cases=tuple(cases),
            recovery_policy=recovery,
            environment_config=environment,
            status=BatchStatus(payload.get("status", "pending")),
            metadata=payload.get("metadata", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "stored_batch_invalid",
                "message": "The stored batch definition cannot be reconstructed safely.",
            },
        ) from exc


def _run_background(
    app, batch: AutonomousTestBatch, execution_owner_id: str
) -> None:
    batch_id = batch.batch_id
    try:
        manager: AutonomousTestBatchManager = app.state.autonomous_test_batch_manager
        manager.run_batch(
            batch,
            persist=True,
            already_persisted=True,
            already_claimed=True,
            execution_owner_id=execution_owner_id,
        )
    except Exception:
        logger.exception("Autonomous test batch %s failed.", batch_id)
    finally:
        lock = app.state.autonomous_test_active_lock
        with lock:
            app.state.autonomous_test_active_batches.discard(batch_id)
        registry = getattr(app.state, "autonomous_test_background_tasks", None)
        if registry is not None:
            registry.complete(batch_id)


@router.post("/batches", status_code=status.HTTP_201_CREATED)
def create_batch(
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    manager, repository = _services(request)
    allowed = {
        "count", "seed", "difficulty", "batch_id", "case_id_prefix",
        "inject_fault_every", "recovery_policy", "environment_config", "metadata",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise HTTPException(status_code=422, detail={"error": "unknown_fields", "fields": sorted(unknown)})

    try:
        recovery_data = payload.get("recovery_policy") or {}
        environment_data = payload.get("environment_config") or {}
        if not isinstance(recovery_data, dict) or not isinstance(environment_data, dict):
            raise ValueError("recovery_policy and environment_config must be objects")
        recovery = RecoveryPolicy(
            enabled=recovery_data.get("enabled", True),
            max_attempts=recovery_data.get("max_attempts", 2),
            retryable_fault_types=tuple(
                FaultType(value) for value in recovery_data.get(
                    "retryable_fault_types",
                    [FaultType.ACTION_FAILURE.value, FaultType.TEMPORARY_UNAVAILABILITY.value],
                )
            ),
            stop_on_unknown_outcome=recovery_data.get("stop_on_unknown_outcome", True),
        )
        environment = TestEnvironmentConfig(
            cognitive_runtime_mode=CognitiveRuntimeMode(
                environment_data.get("cognitive_runtime_mode", "isolated")
            ),
            sequential_only=environment_data.get("sequential_only", True),
            simulation_memory_mode=environment_data.get("simulation_memory_mode", "in_memory"),
        )
        count = payload.get("count", 100)
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 1000:
            raise ValueError("count must be an integer between 1 and 1000")
        batch = manager.create_batch(
            count=count,
            seed=payload.get("seed", 0),
            difficulty=payload.get("difficulty", "medium"),
            batch_id=payload.get("batch_id"),
            case_id_prefix=payload.get("case_id_prefix", "case"),
            inject_fault_every=payload.get("inject_fault_every", 0),
            recovery_policy=recovery,
            environment_config=environment,
            metadata=payload.get("metadata"),
        )
        repository.create_batch(batch)
    except DuplicateBatchError as exc:
        raise HTTPException(status_code=409, detail={"error": "duplicate_batch_id", "message": str(exc)}) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_batch_request", "message": str(exc)}) from exc
    except (BatchManagerError, ResultRepositoryError) as exc:
        raise HTTPException(status_code=503, detail={"error": "batch_creation_failed", "message": str(exc)}) from exc

    return {
        "status": "created",
        "batch_id": batch.batch_id,
        "requested_cases": batch.requested_cases,
        "seed": batch.seed,
        "difficulty": batch.difficulty.value,
        "execution_mode": "sequential",
        "learning_integration": False,
    }


@router.post("/batches/{batch_id}/run", status_code=status.HTTP_202_ACCEPTED)
def run_batch(
    batch_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    manager, repository = _services(request)
    try:
        stored = repository.get_batch(batch_id)
    except ResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "batch_not_found"}) from exc
    except ResultRepositoryError as exc:
        raise HTTPException(status_code=503, detail={"error": "result_store_unavailable", "message": str(exc)}) from exc

    current_status = stored["status"]
    if current_status != BatchStatus.PENDING.value:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "batch_not_runnable",
                "status": current_status,
                "message": "Only a pending batch can be started. Running or completed batches are never automatically retried.",
            },
        )

    try:
        batch = _deserialize_batch(stored["batch"])
    except HTTPException:
        raise

    # Keep the in-process set for observability, but do not rely on it for
    # correctness: the database conditional UPDATE is the cross-worker claim.
    lock = request.app.state.autonomous_test_active_lock
    with lock:
        active = request.app.state.autonomous_test_active_batches
        if batch_id in active:
            raise HTTPException(status_code=409, detail={"error": "batch_already_scheduled"})
        active.add(batch_id)

    registry = getattr(request.app.state, "autonomous_test_background_tasks", None)
    if registry is not None:
        try:
            registry.register(batch_id)
        except RuntimeError as exc:
            with lock:
                request.app.state.autonomous_test_active_batches.discard(batch_id)
            raise HTTPException(
                status_code=503,
                detail={"error": "application_shutting_down", "message": str(exc)},
            ) from exc

    execution_owner_id = uuid4().hex
    try:
        claimed = repository.claim_batch_running(
            batch_id,
            datetime.now(timezone.utc).isoformat(),
            execution_owner_id,
        )
    except ResultRepositoryError as exc:
        with lock:
            active.discard(batch_id)
        if registry is not None:
            registry.complete(batch_id)
        raise HTTPException(
            status_code=503,
            detail={"error": "batch_claim_failed", "message": str(exc)},
        ) from exc
    except Exception:
        with lock:
            active.discard(batch_id)
        if registry is not None:
            registry.complete(batch_id)
        raise

    if not claimed:
        with lock:
            active.discard(batch_id)
        if registry is not None:
            registry.complete(batch_id)
        # Re-read only to return a useful status; the conditional claim above
        # remains the authoritative concurrency decision.
        try:
            latest = repository.get_batch(batch_id)
        except ResultNotFoundError as exc:
            raise HTTPException(status_code=404, detail={"error": "batch_not_found"}) from exc
        raise HTTPException(
            status_code=409,
            detail={
                "error": "batch_not_runnable",
                "status": latest["status"],
                "message": "Only a pending batch can be claimed. Running or completed batches are never automatically retried.",
            },
        )

    try:
        background_tasks.add_task(
            _run_background, request.app, batch, execution_owner_id
        )
    except Exception:
        # The durable running state is deliberately not reset to pending. A
        # caller must reconcile an interrupted/unknown scheduling outcome.
        with lock:
            active.discard(batch_id)
        if registry is not None:
            registry.complete(batch_id)
        raise
    return {
        "status": "accepted",
        "batch_id": batch_id,
        "execution_mode": "sequential",
        "message": "Batch dijadwalkan untuk berjalan di background. Pantau status melalui endpoint detail batch.",
    }


@router.get("/batches")
def list_batches(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    _, repository = _services(request)
    try:
        items = repository.list_batches(limit=limit)
    except ResultRepositoryError as exc:
        raise HTTPException(status_code=503, detail={"error": "result_store_unavailable", "message": str(exc)}) from exc
    return {"items": items, "count": len(items), "limit": limit}


@router.get("/batches/{batch_id}")
def get_batch(batch_id: str, request: Request) -> dict[str, Any]:
    _, repository = _services(request)
    try:
        data = repository.get_batch(batch_id)
    except ResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "batch_not_found"}) from exc
    except ResultRepositoryError as exc:
        raise HTTPException(status_code=503, detail={"error": "result_store_unavailable", "message": str(exc)}) from exc

    heartbeat_value = data.get("heartbeat_at")
    heartbeat_fresh = False
    heartbeat_age_seconds = None
    if data["status"] == BatchStatus.RUNNING.value and heartbeat_value:
        try:
            heartbeat = datetime.fromisoformat(heartbeat_value)
            if heartbeat.tzinfo is None or heartbeat.utcoffset() is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            heartbeat_age_seconds = max(
                0.0,
                (datetime.now(timezone.utc) - heartbeat.astimezone(timezone.utc)).total_seconds(),
            )
            heartbeat_fresh = (
                heartbeat_age_seconds <= BATCH_HEARTBEAT_TIMEOUT_SECONDS
            )
        except (TypeError, ValueError):
            heartbeat_fresh = False
    data["worker_active"] = (
        data["status"] == BatchStatus.RUNNING.value and heartbeat_fresh
    )
    # Owner tokens are internal lease credentials and must never be exposed.
    data.pop("worker_id", None)
    data["heartbeat_age_seconds"] = heartbeat_age_seconds
    data["heartbeat_timeout_seconds"] = BATCH_HEARTBEAT_TIMEOUT_SECONDS
    data["interrupted_requires_reconciliation"] = (
        data["status"] == BatchStatus.RUNNING.value and not heartbeat_fresh
    )
    # Retain this legacy field for clients, but make clear it is process-local
    # observability only and never the source of cross-worker truth.
    lock = request.app.state.autonomous_test_active_lock
    with lock:
        data["active_in_process"] = (
            batch_id in request.app.state.autonomous_test_active_batches
        )
    return data


@router.post("/batches/{batch_id}/reconcile")
def reconcile_interrupted_batch(
    batch_id: str,
    request: Request,
    payload: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    """Operator-confirmed closure of a stale batch; never retries execution."""
    _, repository = _services(request)
    unknown = set(payload) - {"reason", "confirm_no_replay"}
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"error": "unknown_fields", "fields": sorted(unknown)},
        )
    reason = payload.get("reason")
    confirmation = payload.get("confirm_no_replay")
    if not isinstance(reason, str) or not 10 <= len(reason.strip()) <= 1000:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_reconciliation_reason", "message": "reason must contain 10 to 1000 characters."},
        )
    if confirmation is not True:
        raise HTTPException(
            status_code=422,
            detail={"error": "explicit_no_replay_confirmation_required"},
        )

    principal = getattr(request.state, "authenticated_principal", None)
    actor_id = getattr(principal, "actor_id", None) or "local-development"
    now = datetime.now(timezone.utc)
    stale_before = (now - timedelta(seconds=BATCH_HEARTBEAT_TIMEOUT_SECONDS)).isoformat()
    try:
        result = repository.reconcile_stale_batch(
            batch_id,
            stale_before=stale_before,
            reconciled_at=now.isoformat(),
            reconciled_by=actor_id,
            reason=reason,
        )
        events = repository.list_reconciliation_events(batch_id)
    except ResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "batch_not_found"}) from exc
    except ResultRepositoryError as exc:
        message = str(exc)
        code = (
            "batch_not_stale_or_changed"
            if "fresh" in message.lower() or "changed" in message.lower() or "not running" in message.lower()
            else "batch_reconciliation_failed"
        )
        raise HTTPException(
            status_code=409 if code == "batch_not_stale_or_changed" else 503,
            detail={"error": code, "message": message},
        ) from exc
    return {
        "status": "reconciled",
        "batch": result,
        "reconciliation_events": events,
        "replay_performed": False,
    }


@router.get("/batches/{batch_id}/reconciliation")
def get_reconciliation_history(batch_id: str, request: Request) -> dict[str, Any]:
    _, repository = _services(request)
    try:
        events = repository.list_reconciliation_events(batch_id)
    except ResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "batch_not_found"}) from exc
    except ResultRepositoryError as exc:
        raise HTTPException(status_code=503, detail={"error": "result_store_unavailable", "message": str(exc)}) from exc
    return {"batch_id": batch_id, "items": events, "count": len(events)}


@router.get("/batches/{batch_id}/results")
def get_batch_results(batch_id: str, request: Request) -> dict[str, Any]:
    _, repository = _services(request)
    try:
        items = repository.list_case_results(batch_id)
    except ResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "batch_not_found"}) from exc
    except ResultRepositoryError as exc:
        raise HTTPException(status_code=503, detail={"error": "result_store_unavailable", "message": str(exc)}) from exc
    return {"batch_id": batch_id, "items": items, "count": len(items)}


__all__ = ["router"]
