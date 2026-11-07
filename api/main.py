from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from threading import Lock

from api.resource_lifecycle import (
    BackgroundTaskRegistry,
    CapabilityExecutionResourceBundle,
    close_owned_resources,
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.llm.orchestrator import LLMOrchestrationStageError
from api.dependencies import create_runtime
from api.execution_runtime import create_execution_runtime
from agent.llm.cognitive_capability_runtime import compose_cognitive_capability_runtime
from agent.world.environment_lifecycle import EnvironmentLifecycleManager
from agent.world.environment_capability_discovery import EnvironmentCapabilityDiscovery

logger = logging.getLogger(__name__)


def _create_capability_proposal_store():
    """Create the durable proposal store independently from execution ledgers."""
    from agent.llm.capability_proposal_validator import CapabilityProposalValidator
    from agent.llm.sqlite_capability_proposal_store import SQLiteCapabilityProposalStore

    configured_path = os.getenv("HAI_CAPABILITY_PROPOSAL_DB")
    database_path = (
        Path(configured_path).expanduser()
        if configured_path and configured_path.strip()
        else Path(__file__).resolve().parents[1] / "data" / "capability_proposals.sqlite3"
    )
    return SQLiteCapabilityProposalStore(
        database_path,
        validator=CapabilityProposalValidator(),
    )


def _create_composed_cognitive_capability_runtime(app, cognitive_runtime):
    """Compose capability infrastructure only when the cognitive Agent supports it.

    Some API consumers inject a lightweight cognitive-runtime stub because
    their route tests exercise the independent RoomEnvironment runtime. In
    that case capability composition is unavailable, but must not prevent the
    unrelated execution API from starting. Capability endpoints remain
    fail-closed because no capability runtime/manager is published.
    """
    agent = getattr(cognitive_runtime, "agent", None)
    actions = getattr(agent, "actions", None)
    set_sync_hook = getattr(agent, "set_planner_sync_hook", None)

    if not isinstance(actions, MutableMapping) or not callable(set_sync_hook):
        logger.warning(
            "Cognitive capability runtime not composed: injected cognitive "
            "Agent does not expose the required actions registry and planner "
            "synchronization hook. Capability operations remain unavailable."
        )
        return None

    return compose_cognitive_capability_runtime(agent)



def _require_shared_capability_components(
    manager,
    capability_runtime,
    *,
    label,
    components=("registry", "adapter", "action_builder"),
):
    """Reject managers that do not use the application's shared components.

    The default contract is intentionally strict for approval managers. A
    lifecycle manager owns only the registry, adapter, and Agent; it does not
    expose or own the action builder.
    """
    if manager is None:
        return
    if capability_runtime is None:
        raise RuntimeError(
            f"{label} cannot be configured because the application's "
            "cognitive capability runtime is unavailable. Refusing to start "
            "with a partially wired capability manager."
        )
    expected = {
        "registry": capability_runtime.registry,
        "adapter": capability_runtime.adapter,
        "action_builder": capability_runtime.action_builder,
    }
    for attribute in components:
        component = expected[attribute]
        if getattr(manager, attribute, None) is not component:
            raise RuntimeError(
                f"{label} must use the application's shared capability {attribute}."
            )
    manager_agent = getattr(manager, "agent", capability_runtime.agent)
    if manager_agent is not capability_runtime.agent:
        raise RuntimeError(
            f"{label} must use the cognitive capability runtime Agent."
        )



async def _await_background_drain(registry) -> bool:
    """Wait for app-owned synchronous background work without losing cleanup to cancellation."""
    from api.resource_lifecycle import _await_completion_despite_cancellation

    return await _await_completion_despite_cancellation(
        asyncio.to_thread(registry.wait_for_idle)
    )

def create_app(
    runtime_factory: Callable = create_runtime,
    execution_runtime_factory: Callable = create_execution_runtime,
    capability_store_factory: Callable | None = None,
    capability_approval_manager_factory: Callable | None = None,
    capability_lifecycle_manager_factory: Callable | None = None,
    capability_lifecycle_authorizer: Callable | None = None,
    cognitive_capability_runtime_factory: Callable | None = None,
    authentication_provider_factory: Callable | None = None,
    authorization_policy_factory: Callable | None = None,
    autonomous_testing_repository_factory: Callable | None = None,
    autonomous_testing_manager_factory: Callable | None = None,
    capability_execution_gateway_factory: Callable | None = None,
    trusted_capability_invocation_specs_factory: Callable | None = None,
    table_report_service_factory: Callable | None = None,
    database_runtime_registry_factory: Callable | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Resources are closed only when ownership is explicitly held by
        # this application. The factory remains responsible for resources
        # it creates but fails to return in a resource bundle.
        owned_resources: list[object] = []
        app.state.autonomous_test_background_tasks = BackgroundTaskRegistry()
        shutdown_cancelled = False
        try:
            # Authentication can remain disabled for local development, but a
            # deployment may require it explicitly. This guard prevents a
            # production deployment from silently starting without an identity
            # provider when HAI_AUTH_REQUIRED=true.
            auth_enabled_raw = os.getenv("HAI_AUTH_ENABLED", "false").strip().lower()
            deployment_env = os.getenv(
                "HAI_ENV", os.getenv("ENVIRONMENT", "development")
            ).strip().lower()
            auth_required_default = (
                "true" if deployment_env in {"production", "prod", "staging"} else "false"
            )
            auth_required_raw = os.getenv(
                "HAI_AUTH_REQUIRED", auth_required_default
            ).strip().lower()
            if auth_enabled_raw not in {"true", "false"}:
                raise RuntimeError("HAI_AUTH_ENABLED must be exactly 'true' or 'false'.")
            if auth_required_raw not in {"true", "false"}:
                raise RuntimeError("HAI_AUTH_REQUIRED must be exactly 'true' or 'false'.")
            auth_enabled = auth_enabled_raw == "true"
            auth_required = (
                auth_required_raw == "true"
                or deployment_env in {"production", "prod", "staging"}
            )
            if auth_required and not auth_enabled:
                raise RuntimeError(
                    "HAI_AUTH_REQUIRED=true requires HAI_AUTH_ENABLED=true. "
                    "Refusing to start without an authenticated API boundary."
                )
            app.state.auth_enabled = auth_enabled

            app.state.cognitive_runtime = runtime_factory()
            app.state.execution_runtime = execution_runtime_factory()

            # P2.1: compose the trusted capability registry, workspace adapter,
            # and lifecycle manager around the cognitive Agent. The separate
            # RoomEnvironment execution runtime is deliberately not passed here.
            capability_runtime_factory = (
                cognitive_capability_runtime_factory
                or _create_composed_cognitive_capability_runtime
            )
            app.state.cognitive_capability_runtime = capability_runtime_factory(
                app, app.state.cognitive_runtime
            )

            # P3.5.5A: app-owned environment lifecycle and metadata-only discovery.
            # The registry is shared with the cognitive capability runtime; no
            # second registry and no model-controlled provider registration.
            app.state.environment_lifecycle_manager = EnvironmentLifecycleManager()
            app.state.environment_capability_discovery = None
            environment_runtime = app.state.cognitive_capability_runtime
            if environment_runtime is not None:
                raw_bindings = os.getenv("HAI_ENVIRONMENT_CAPABILITY_BINDINGS", "").strip()
                if raw_bindings:
                    import json
                    try:
                        parsed_bindings = json.loads(raw_bindings)
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError(
                            "HAI_ENVIRONMENT_CAPABILITY_BINDINGS must be valid JSON."
                        ) from exc
                    if not isinstance(parsed_bindings, dict):
                        raise RuntimeError(
                            "HAI_ENVIRONMENT_CAPABILITY_BINDINGS must be a JSON object."
                        )
                    app.state.environment_capability_discovery = EnvironmentCapabilityDiscovery(
                        environments=app.state.environment_lifecycle_manager,
                        registry=environment_runtime.registry,
                        bindings=parsed_bindings,
                    )

            # Environment-backed authentication is disabled unless explicitly enabled.
            provider_factory = authentication_provider_factory
            if provider_factory is None:
                from api.security.authentication_config import (
                    create_environment_authentication_provider,
                )
                provider_factory = create_environment_authentication_provider

            provider_result = provider_factory(app)
            if hasattr(provider_result, "__await__"):
                provider_result = await provider_result
            app.state.authentication_provider = provider_result
            if provider_result is not None:
                owned_resources.append(provider_result)
            if auth_enabled and provider_result is None:
                raise RuntimeError(
                    "HAI_AUTH_ENABLED=true but no authentication provider was created."
                )

            # Stage 2.8B.9: use an application-owned PBAC policy. An injected factory
            # takes precedence; otherwise the environment factory is used. When auth
            # is disabled the policy remains None, preserving fail-closed behavior.
            policy_factory = authorization_policy_factory
            if policy_factory is None:
                from api.security.authorization_config import (
                    create_environment_authorization_policy,
                )
                policy_factory = create_environment_authorization_policy

            policy_result = policy_factory(app)
            if hasattr(policy_result, "__await__"):
                policy_result = await policy_result
            app.state.authorization_policy = policy_result
            if auth_enabled and policy_result is None:
                raise RuntimeError(
                    "HAI_AUTH_ENABLED=true but no authorization policy was created."
                )

            # P2.5: capability execution is opt-in. The application must supply
            # trusted adapters, registry, security policy, and the existing trace store.
            app.state.capability_execution_gateway = None
            app.state.trusted_capability_invocation_port = None
            app.state.capability_execution_registry = None
            if capability_execution_gateway_factory is not None:
                gateway_result = capability_execution_gateway_factory(app)
                if hasattr(gateway_result, "__await__"):
                    gateway_result = await gateway_result

                if isinstance(gateway_result, CapabilityExecutionResourceBundle):
                    # Register transferred ownership before validating the gateway.
                    # If validation or later startup fails, these resources are
                    # still cleaned up by this lifespan's finally block.
                    owned_resources.extend(gateway_result.owned_resources)
                    gateway = gateway_result.gateway
                else:
                    # Backward compatibility: a bare gateway is borrowed. The app
                    # must not infer ownership from its attributes or close method.
                    gateway = gateway_result

                if gateway is not None and not callable(getattr(gateway, "read", None)):
                    raise RuntimeError(
                        "Capability execution gateway factory must return an object "
                        "with read(principal, request, execution_id=...), or a "
                        "CapabilityExecutionResourceBundle containing that gateway."
                    )
                if isinstance(gateway_result, CapabilityExecutionResourceBundle) and gateway is None:
                    raise RuntimeError(
                        "Capability execution resource bundle must contain a gateway."
                    )
                app.state.capability_execution_gateway = gateway


            # C.4: construct the port after gateway initialization.
            # A specs factory without a gateway is a startup configuration error.
            if trusted_capability_invocation_specs_factory is not None:
                existing_gateway = app.state.capability_execution_gateway
                if existing_gateway is None:
                    raise RuntimeError(
                        "Trusted capability invocation specs require an existing "
                        "capability_execution_gateway. Configure the gateway factory first."
                    )

                specs_result = trusted_capability_invocation_specs_factory(app)
                if hasattr(specs_result, "__await__"):
                    specs_result = await specs_result

                from api.trusted_capability_invocation_port_factory import (
                    build_trusted_capability_invocation_port,
                )

                app.state.trusted_capability_invocation_port = (
                    build_trusted_capability_invocation_port(
                        existing_gateway,
                        specs_result,
                    )
                )
            # Dynamic, user-approved database connections are opt-in as well.
            app.state.database_runtime_registry = None
            registry_factory = database_runtime_registry_factory
            if registry_factory is None and os.getenv("HAI_DATABASE_ROOT", "").strip():
                from api.database_runtime_registry import registry_factory_from_environment
                registry_factory = registry_factory_from_environment
            if registry_factory is not None:
                app.state.database_runtime_registry = registry_factory(app)
            # Table reporting is opt-in: without a factory the endpoint answers 503.
            app.state.table_report_service = (
                table_report_service_factory(app)
                if table_report_service_factory is not None else None
            )
            store_factory = capability_store_factory or _create_capability_proposal_store
            app.state.capability_proposal_store = store_factory()

            # Proposal approval remains opt-in and fail-closed.
            app.state.capability_approval_manager = (
                capability_approval_manager_factory(app, app.state.capability_proposal_store)
                if capability_approval_manager_factory is not None
                else None
            )
            _require_shared_capability_components(
                app.state.capability_approval_manager,
                app.state.cognitive_capability_runtime,
                label="Capability approval manager",
            )
            if (
                app.state.capability_approval_manager is not None
                and getattr(app.state.capability_approval_manager, "store", None)
                is not app.state.capability_proposal_store
            ):
                raise RuntimeError(
                    "Capability approval manager must use the application's "
                    "shared capability proposal store."
                )

            # Lifecycle manager is part of the composed cognitive capability
            # runtime by default. An explicit factory may replace it for tests or
            # deployments, but must use the same cognitive registry/adapter.
            app.state.capability_lifecycle_manager = (
                capability_lifecycle_manager_factory(app, app.state.cognitive_runtime)
                if capability_lifecycle_manager_factory is not None
                else getattr(
                    app.state.cognitive_capability_runtime,
                    "lifecycle_manager",
                    None,
                )
            )
            _require_shared_capability_components(
                app.state.capability_lifecycle_manager,
                app.state.cognitive_capability_runtime,
                label="Capability lifecycle manager",
                components=("registry", "adapter"),
            )
            app.state.capability_lifecycle_authorizer = capability_lifecycle_authorizer

            # AT-02.9: result persistence is independent from agent memory, execution
            # traces, handoff storage, and the application's shared simulation runtime.
            from agent.autonomous_testing.batch_manager import AutonomousTestBatchManager
            from agent.autonomous_testing.result_repository import SQLiteTestResultRepository

            configured_results_path = os.getenv("HAI_AUTONOMOUS_TEST_RESULTS_DB")
            results_path = (
                Path(configured_results_path).expanduser()
                if configured_results_path and configured_results_path.strip()
                else Path(__file__).resolve().parents[1] / "data" / "autonomous_test_results.sqlite3"
            )
            repository_factory = autonomous_testing_repository_factory or SQLiteTestResultRepository
            app.state.autonomous_test_result_repository = repository_factory(results_path)
            if app.state.autonomous_test_result_repository is not None:
                owned_resources.append(app.state.autonomous_test_result_repository)
            manager_factory = autonomous_testing_manager_factory or AutonomousTestBatchManager
            app.state.autonomous_test_batch_manager = manager_factory(
                repository=app.state.autonomous_test_result_repository
            )
            app.state.autonomous_test_active_batches = set()
            app.state.autonomous_test_active_lock = Lock()

            yield
        finally:
            # Stop accepting new background work first, then ask the batch
            # manager to stop between cases. A running case is never killed.
            task_registry = getattr(app.state, "autonomous_test_background_tasks", None)
            if task_registry is not None:
                task_registry.stop_accepting()
            batch_manager = getattr(app.state, "autonomous_test_batch_manager", None)
            request_shutdown = getattr(batch_manager, "request_shutdown", None)
            if callable(request_shutdown):
                try:
                    request_shutdown()
                except Exception:
                    logger.exception("Failed to request autonomous batch shutdown.")
            if task_registry is not None:
                try:
                    shutdown_cancelled = await _await_background_drain(task_registry)
                except Exception:
                    logger.exception("Failed while draining autonomous background tasks.")

            lifecycle_manager = getattr(app.state, "capability_lifecycle_manager", None)
            adapter = getattr(lifecycle_manager, "adapter", None)
            detach = getattr(adapter, "detach", None)
            if callable(detach):
                try:
                    detach()
                except Exception:
                    logger.exception("Failed to detach capability lifecycle adapter.")

            # Reverse construction order, close each transferred resource once,
            # and continue even if an individual cleanup operation fails.
            try:
                cleanup_cancelled = await close_owned_resources(owned_resources, logger=logger)
                shutdown_cancelled = shutdown_cancelled or cleanup_cancelled
            finally:
                app.state.authentication_provider = None
                app.state.authorization_policy = None
                app.state.cognitive_capability_runtime = None
                app.state.environment_capability_discovery = None
                app.state.environment_lifecycle_manager = None
                app.state.cognitive_runtime = None
                app.state.execution_runtime = None
                app.state.capability_proposal_store = None
                app.state.capability_approval_manager = None
                app.state.capability_lifecycle_manager = None
                app.state.capability_lifecycle_authorizer = None
                app.state.capability_execution_gateway = None
                app.state.trusted_capability_invocation_port = None
                app.state.capability_execution_registry = None
                app.state.execution_trace_store = None
                app.state.table_report_service = None
                registry = getattr(app.state, "database_runtime_registry", None)
                if registry is not None:
                    registry.shutdown()
                app.state.database_runtime_registry = None
                app.state.autonomous_test_batch_manager = None
                app.state.autonomous_test_result_repository = None
                app.state.autonomous_test_active_batches = set()
                app.state.autonomous_test_active_lock = None
                app.state.autonomous_test_background_tasks = None
            if shutdown_cancelled:
                raise asyncio.CancelledError

    application = FastAPI(
        title="hAI Agent API",
        version="0.1.0",
        description=(
            "HTTP API for the hAI Agent cognitive pipeline, a separate allowlisted "
            "RoomEnvironment execution simulation, and composed trusted capability "
            "registry/lifecycle infrastructure. Capability execution remains gated."
        ),
        lifespan=lifespan,
    )
    application.state.database_onboarding_allowlists = {}
    application.state.database_onboarding_lock = Lock()

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:5174,http://localhost:5174",
        ).split(",")
        if origin.strip()
    ]

    # Central authentication/authorization boundary for the versioned API.
    # Capability handlers retain their operation-specific PBAC checks.
    # Register this first so CORSMiddleware is the outermost user middleware.
    from api.security.authentication_middleware import APIAuthenticationAuthorizationMiddleware
    application.add_middleware(APIAuthenticationAuthorizationMiddleware)

    # CORS must wrap the authentication boundary so early 401/403 responses
    # emitted by that middleware still receive the appropriate CORS headers.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Idempotency-Key"],
    )

    from api.routes.cognitive import router as cognitive_router
    from api.routes.execution import router as execution_router
    from api.routes.capabilities import router as capabilities_router
    from api.routes.capability_lifecycle import router as capability_lifecycle_router
    from api.routes.capability_execution import router as capability_execution_router
    from api.routes.environment_workspace import router as environment_workspace_router
    from api.routes.strategies import router as strategies_router
    from api.routes.telemetry import router as telemetry_router
    from api.routes.autonomous_testing import router as autonomous_testing_router
    from api.routes.trusted_capability_invocation import router as trusted_capability_invocation_router
    from api.routes.database_onboarding import router as database_onboarding_router
    from api.routes.table_reports import router as table_reports_router
    from api.routes.database_connections import router as database_connections_router
    from api.routes.data_changes import router as data_changes_router
    from api.routes.database_creation import router as database_creation_router
    from api.routes.database_engine_requests import router as database_engine_requests_router

    application.include_router(cognitive_router, prefix="/api/v1")
    application.include_router(execution_router, prefix="/api/v1")
    application.include_router(capabilities_router, prefix="/api/v1")
    application.include_router(capability_lifecycle_router, prefix="/api/v1")
    application.include_router(capability_execution_router, prefix="/api/v1")
    application.include_router(environment_workspace_router, prefix="/api/v1")
    application.include_router(telemetry_router, prefix="/api/v1")
    application.include_router(strategies_router, prefix="/api/v1")
    application.include_router(autonomous_testing_router, prefix="/api/v1")
    application.include_router(trusted_capability_invocation_router, prefix="/api/v1")
    application.include_router(database_onboarding_router, prefix="/api/v1")
    application.include_router(table_reports_router, prefix="/api/v1")
    application.include_router(database_connections_router, prefix="/api/v1")
    application.include_router(data_changes_router, prefix="/api/v1")
    application.include_router(database_creation_router, prefix="/api/v1")
    application.include_router(database_engine_requests_router, prefix="/api/v1")

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "hai-agent-api"}

    @application.exception_handler(LLMOrchestrationStageError)
    async def cognitive_stage_error_handler(
        request: Request,
        exc: LLMOrchestrationStageError,
    ) -> JSONResponse:
        logger.exception(
            "Cognitive stage failed: %s",
            exc.stage,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return JSONResponse(
            status_code=502,
            content={
                "error": "cognitive_stage_failed",
                "stage": exc.stage,
                "message": (
                    "The cognitive pipeline failed while processing "
                    "the request. Check the API server logs for details."
                ),
            },
        )

    return application



def create_app_from_environment() -> FastAPI:
    """Default app; table reporting turns on only if HAI_TABLE_REPORT_CONFIG is set."""
    config_path = os.getenv("HAI_TABLE_REPORT_CONFIG", "").strip()
    if not config_path:
        return create_app()
    if os.getenv("HAI_AUTH_ENABLED", "false").strip().lower() != "true":
        raise RuntimeError(
            "HAI_TABLE_REPORT_CONFIG requires HAI_AUTH_ENABLED=true: reports read "
            "business data and are only served to authenticated principals."
        )
    from api.dependencies import create_llm_provider
    from api.table_report_assembly import build_create_app_kwargs, load_config

    return create_app(**build_create_app_kwargs(
        load_config(config_path), llm_provider_factory=create_llm_provider))


app = create_app_from_environment()
