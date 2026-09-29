"""
AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System
backend/app/main.py

Minimal FastAPI application — Phase 1 skeleton.

Contains:
  - Application factory (create_app)
  - Lifespan context (startup / shutdown hooks)
  - A single /health endpoint to verify the server is running

Does NOT contain:
  - Business logic
  - Database connections (Phase 2)
  - ML or prediction logic (Phase 5+)
  - Data provider calls (Phase 3+)
  - Route registration for domain endpoints (Phase 13)
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Startup and shutdown logic.

    Startup:
      - Configure logging
      - (Phase 2) Database connection pool
      - (Phase 3) Data provider initialisation
      - (Phase 11) Event bus connection

    Shutdown:
      - (Phase 2) Close database pool
      - (Phase 11) Flush event bus
    """
    # --- Startup ---
    configure_logging(
        level=settings.LOG_LEVEL,
        json_mode=settings.LOG_JSON,
    )
    logger.info(
        "Starting %s v%s [%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("Shutting down %s", settings.APP_NAME)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Later phases will register routers here, e.g.:
        app.include_router(market_router, prefix=settings.API_V1_PREFIX)
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI Multi-Factor Stock Intelligence & Portfolio Decision-Support System. "
            "Phase 1 — skeleton only."
        ),
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Routes registered in this phase
    # -----------------------------------------------------------------------

    @app.get(
        "/health",
        tags=["System"],
        summary="Health check",
        response_description="Service status and version",
    )
    async def health_check() -> JSONResponse:
        """
        Returns 200 if the service is running.
        Kubernetes liveness/readiness probes point here.
        """
        return JSONResponse(
            content={
                "status": "ok",
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
            }
        )

    # -----------------------------------------------------------------------
    # Future: domain routers added here in Phase 13
    # Example:
    #   from app.api.routes import market, stocks, predictions
    #   app.include_router(market.router, prefix=settings.API_V1_PREFIX)
    # -----------------------------------------------------------------------

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (used by Uvicorn)
# ---------------------------------------------------------------------------

app = create_app()
