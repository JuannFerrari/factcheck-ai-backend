from datetime import datetime, UTC
from app.core.config import settings
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def root():
    """Basic health check endpoint"""
    return {
        "message": f"{settings.app_name} is running ✅",
        "version": settings.app_version,
        "environment": settings.environment,
        "timestamp": datetime.now(UTC).isoformat()
    }
