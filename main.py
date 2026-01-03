"""
V2 Main Entry Point - FastAPI Application.

Phase 19: Slim entry point that imports from modular v2 components.

This file contains:
- FastAPI app initialization
- Route mounting from v2/api/routes/
- Middleware setup from v2/api/middleware.py
- Health check endpoints

NO BUSINESS LOGIC - just wiring and setup.

Usage:
    uvicorn v2.main:app --reload
    python v2/main.py
"""

import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Load environment variables first
load_dotenv()

# ============================================================================
# NON-BLOCKING LOGGING SETUP (MUST BE BEFORE OTHER IMPORTS)
# ============================================================================
from utils.logger_config import configure_non_blocking_logging

_log_listener = configure_non_blocking_logging()

# Import routes
from api.routes import (
    calls_router,
    batch_router,
    agents_router,
    knowledge_base_router,
    oauth_router,
    oauth_microsoft_router,
    recordings_router,
)

# Import middleware setup
from api.middleware import setup_middlewares, setup_request_logging

logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN EVENTS
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("V2 Voice Agent API starting up...")
    yield
    logger.info("V2 Voice Agent API shutting down...")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Vonage Voice Agent API (V2)",
    description="Multi-tenant voice agent platform with LiveKit integration",
    version="2.0.0",
    lifespan=lifespan,
)

# Setup middlewares (CORS, security, rate limiting)
setup_middlewares(app)
setup_request_logging(app)


# =============================================================================
# ROUTES
# =============================================================================

# Call management (router has /calls prefix)
app.include_router(calls_router, tags=["Calls"])

# Batch operations (router has /batch prefix)
app.include_router(batch_router, tags=["Batch"])

# Agent CRUD (router has /agents prefix)
app.include_router(agents_router, tags=["Agents"])

# Knowledge base (router has /knowledge-base prefix)
app.include_router(knowledge_base_router, tags=["Knowledge Base"])

# OAuth - Google (router has /auth prefix)
app.include_router(oauth_router, tags=["OAuth - Google"])

# OAuth - Microsoft (router has /auth/microsoft prefix)
app.include_router(oauth_microsoft_router, tags=["OAuth - Microsoft"])

# Recordings (router has /recordings prefix)
app.include_router(recordings_router, tags=["Recordings"])



# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - health check."""
    return {"status": "ok", "version": "2.0.0", "service": "voice-agent-api"}


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Kubernetes liveness probe."""
    return {"status": "healthy"}


@app.get("/readyz", include_in_schema=False)
async def readyz():
    """Kubernetes readiness probe."""
    return {"status": "ready"}


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
