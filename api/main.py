
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.llm.orchestrator import LLMOrchestrationStageError
from api.dependencies import create_runtime
from api.execution_runtime import create_execution_runtime

logger = logging.getLogger(__name__)


def create_app(
    runtime_factory: Callable = create_runtime,
    execution_runtime_factory: Callable = create_execution_runtime,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cognitive_runtime = runtime_factory()
        app.state.execution_runtime = execution_runtime_factory()

        try:
            yield
        finally:
            app.state.cognitive_runtime = None
            app.state.execution_runtime = None

    application = FastAPI(
        title="hAI Agent API",
        version="0.1.0",
        description=(
            "HTTP API for the hAI Agent cognitive pipeline "
            "and a separate, allowlisted RoomEnvironment "
            "execution simulation."
        ),
        lifespan=lifespan,
    )

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:5174,http://localhost:5174",
        ).split(",")
        if origin.strip()
    ]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "Idempotency-Key",
        ],
    )

    from api.routes.cognitive import router as cognitive_router
    from api.routes.execution import router as execution_router
    from api.routes.strategies import router as strategies_router
    from api.routes.telemetry import router as telemetry_router

    application.include_router(cognitive_router, prefix="/api/v1")
    application.include_router(execution_router, prefix="/api/v1")
    application.include_router(telemetry_router, prefix="/api/v1")
    application.include_router(strategies_router, prefix="/api/v1")

    @application.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "hai-agent-api",
        }

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


app = create_app()
