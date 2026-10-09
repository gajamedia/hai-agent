from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

# from dotenv import load_dotenv

# # Resolve the project root independently of the current working directory.
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# # Load environment variables before importing runtime dependencies.
# # Existing OS environment variables take precedence over .env values.
# load_dotenv(
#     dotenv_path=PROJECT_ROOT / ".env",
#     override=False,
# )

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.llm.orchestrator import LLMOrchestrationStageError
from api.dependencies import create_runtime

logger = logging.getLogger(__name__)


def create_app(
    runtime_factory: Callable = create_runtime,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cognitive_runtime = runtime_factory()
        try:
            yield
        finally:
            app.state.cognitive_runtime = None

    application = FastAPI(
        title="hAI Agent API",
        version="0.1.0",
        description=(
            "HTTP API for the existing hAI Agent cognitive pipeline. "
            "Cognitive requests do not execute Agent actions."
        ),
        lifespan=lifespan,
    )

    from api.routes.cognitive import router as cognitive_router

    application.include_router(cognitive_router, prefix="/api/v1")

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


app = create_app()
