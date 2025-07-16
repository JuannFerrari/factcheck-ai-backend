import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.rate_limiter import limiter
from app.core.database import init_database, close_database
from app.api.routes import fact_check
from app.api.routes.health import root as health_root
from app.api.routes import vector_db

# Setup logging
setup_logging()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting application")

    # Initialize database
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue without database if initialization fails

    yield

    # Close database connections
    try:
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    logger.info("Shutting down application")


# Create FastAPI app with metadata, disable docs and openapi
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered fact-checking API using RAG with web search and "
        "Hugging Face Inference Providers"
    ),
    docs_url=None,  # Disable Swagger UI
    redoc_url=None,  # Disable ReDoc
    openapi_url=None,  # Disable OpenAPI schema
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware with configurable origins
logger.info(f"Configuring CORS with origins: {settings.cors_origins}")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_rate_limit_headers(request: Request, call_next):
    """Add rate limiting headers to all responses"""
    response = await call_next(request)

    # Add rate limit headers if not already present
    if "X-RateLimit-Limit" not in response.headers:
        response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)

    # Add remaining and reset headers (simplified for testing)
    if "X-RateLimit-Remaining" not in response.headers:
        response.headers["X-RateLimit-Remaining"] = str(
            settings.rate_limit_per_minute - 1
        )

    if "X-RateLimit-Reset" not in response.headers:
        # Set reset to 60 seconds from now
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)

    return response


app.include_router(fact_check.router, prefix="/api/v1", tags=["fact-checking"])
app.include_router(vector_db.router, prefix="/api/v1/vector", tags=["vector-database"])
app.add_api_route("/", health_root, methods=["GET"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": (
                str(exc)
                if settings.environment == "development"
                else "An unexpected error occurred"
            ),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request, exc):
    return JSONResponse(
        status_code=429,
        content={
            "error": (
                "Too many requests. Please wait a few seconds and try again. "
                "If you need higher limits, contact the site owner."
            )
        },
    )
