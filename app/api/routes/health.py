from datetime import datetime, UTC
from typing import Dict, Any
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()
router = APIRouter()


async def check_hf_health() -> Dict[str, Any]:
    """Check Hugging Face API health"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://router.huggingface.co/v1/models",
                headers={
                    "Authorization": f"Bearer {settings.huggingface_api_key}"},
                timeout=5.0
            )
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code
            }
    except Exception as e:
        logger.error("Hugging Face health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}


async def check_serper_health() -> Dict[str, Any]:
    """Check Serper.dev API health"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": settings.serper_api_key,
                    "Content-Type": "application/json"
                },
                json={"q": "test", "num": 1},
                timeout=5.0
            )
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code
            }
    except Exception as e:
        logger.error("Serper health check failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}


@router.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "message": f"{settings.app_name} is running ✅",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(UTC).isoformat()
    }


@router.get("/health")
async def health_check():
    """Detailed health check endpoint with service status"""
    try:
        # Check external services
        hf_status = await check_hf_health()
        serper_status = await check_serper_health()

        # Determine overall health
        overall_status = "healthy"
        if hf_status["status"] == "unhealthy" or serper_status["status"] == "unhealthy":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "timestamp": datetime.now(UTC).isoformat(),
            "services": {
                "huggingface": hf_status,
                "serper": serper_status
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check failed",
                "timestamp": datetime.now(UTC).isoformat()
            }
        )


@router.options("/health")
async def health_options(request: Request):
    """CORS preflight handler for /health endpoint"""
    response = JSONResponse(content=None, status_code=200)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


@router.get("/metrics")
async def metrics():
    """Basic operational metrics endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(UTC).isoformat(),
        "metrics": {
            "uptime": "TODO: Implement uptime tracking",
            "requests_total": "TODO: Implement request counting",
            "active_connections": "TODO: Implement connection tracking"
        }
    }
