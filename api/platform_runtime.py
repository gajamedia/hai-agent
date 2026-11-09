"""Builds the shared platform pipeline for the running application, once, on first use.

The task store lives next to the Database Agent's other working files
(``<work_dir>/platform.sqlite3``) so plans survive a restart. Without a database registry
(some tests, or a deployment that only uses the single configured report service) it is
in-memory. Tasks interrupted by a restart are marked, never re-run.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

from agent.platform import OperationRegistry, PlatformPipeline, SQLiteTaskStore
from api.database_runtime_registry import DatabaseRuntimeRegistry
from api.features.database_reports import OutcomeCache, build_report_feature
from api.security.authorization import AuthorizationPolicy
from agent.workflow.table_report_service import TableReportService

_lock = threading.Lock()
MAX_PENDING_REPORT_PLANS = 200


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
        operations.register(build_report_feature(_resolver(app), cache))
        pipeline = PlatformPipeline(
            registry=operations, store=store, authorize=authorize, clock=time.time,
            ttl_seconds=900, max_active_per_operation=MAX_PENDING_REPORT_PLANS)
        pipeline.recover_interrupted()
        operations.report_outcomes = cache  # shared by every pipeline built over these operations
        app.state.platform_pipeline = pipeline
        return pipeline


def report_outcomes(app: Any) -> OutcomeCache:
    return get_pipeline(app).registry.report_outcomes


__all__ = ["get_pipeline", "report_outcomes"]
