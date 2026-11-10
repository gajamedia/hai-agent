"""Builds the shared platform pipeline for the running application, once, on first use.

The task store lives next to the Database Agent's other working files
(``<work_dir>/platform.sqlite3``) so plans survive a restart. Without a database registry
(some tests, or a deployment that only uses the single configured report service) it is
in-memory. Tasks interrupted by a restart are marked, never re-run.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from agent.platform import Feature, OperationRegistry, PlatformPipeline, SQLiteTaskStore
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.data_change_settings import settings as change_settings
from api.features.database_changes import build_change_operation, build_entry_operation
from api.features.database_creation import build_create_operation
from api.features.database_reports import OutcomeCache, build_report_operation
from api.security.authorization import AuthorizationPolicy
from agent.workflow.table_report_service import TableReportService

_lock = threading.Lock()


def _resolver(app: Any):
    def resolve(database: Optional[str]):
        registry = getattr(app.state, "database_runtime_registry", None)
        if isinstance(registry, DatabaseRuntimeRegistry):
            name = registry.resolve_name(database)
            return registry.service(name), name, registry.service_token(name)
        service = getattr(app.state, "table_report_service", None)
        if not isinstance(service, TableReportService):
            raise RuntimeError("Table reporting is not configured.")
        return service, None, None
    return resolve


def get_pipeline(app: Any) -> PlatformPipeline:
    pipeline = getattr(app.state, "platform_pipeline", None)
    if pipeline is not None:
        return pipeline
    with _lock:
        pipeline = getattr(app.state, "platform_pipeline", None)
        if pipeline is not None:
            return pipeline

        def authorize(principal: Any, permission: str) -> None:
            policy = getattr(app.state, "authorization_policy", None)
            if not isinstance(policy, AuthorizationPolicy):
                raise PermissionError("authorization policy is not configured")
            policy.require(principal, permission)

        import time
        registry = getattr(app.state, "database_runtime_registry", None)
        path = (registry.work_dir / "platform.sqlite3"
                if isinstance(registry, DatabaseRuntimeRegistry) else ":memory:")
        store = SQLiteTaskStore(path, clock=time.time)
        operations = OperationRegistry()
        cache = OutcomeCache()
        database_operations = [build_report_operation(_resolver(app), cache)]
        if isinstance(registry, DatabaseRuntimeRegistry):
            database_operations.append(build_change_operation(lambda: app.state.database_runtime_registry))
            database_operations.append(build_entry_operation(lambda: app.state.database_runtime_registry))
            database_operations.append(build_create_operation(lambda: app.state.database_runtime_registry))
        operations.register(Feature("database", "Database Agent", tuple(database_operations)))
        pipeline = PlatformPipeline(
            registry=operations, store=store, authorize=authorize, clock=time.time, ttl_seconds=900,
            allow_self_approval=lambda: change_settings()["self_approval"],
            step_up_max_age_seconds=lambda: change_settings()["step_up_max_age_seconds"])
        pipeline.recover_interrupted()
        operations.report_outcomes = cache  # shared by every pipeline built over these operations
        app.state.platform_pipeline = pipeline
        return pipeline


def report_outcomes(app: Any) -> OutcomeCache:
    return get_pipeline(app).registry.report_outcomes


__all__ = ["get_pipeline", "report_outcomes"]
