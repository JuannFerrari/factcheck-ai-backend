from fastapi import APIRouter, Request, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from langserve import add_routes

from app.domain.models import FactCheckRequest
from app.services.fact_checking import fact_check_chain, store_fact_check_background
from app.core.logging import get_logger
from app.core.rate_limiter import limiter  # Import the limiter instance
from app.core.config import settings

logger = get_logger()
router = APIRouter()


def verify_api_key(request: Request):
    api_key = request.headers.get(settings.api_key_header)
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


@router.post("/factcheck")
@limiter.limit("10/minute;2/second")  # Burst/steady: 10/minute, 2/second per IP
async def factcheck_endpoint(
    request: Request,
    fact_check_request: FactCheckRequest,
    background_tasks: BackgroundTasks,
    _=Depends(verify_api_key),
):
    """Direct fact-checking endpoint"""
    try:
        logger.info(
            "Processing fact-check request", claim=fact_check_request.claim[:100]
        )

        # Use the LangServe chain
        result = await fact_check_chain.ainvoke({"claim": fact_check_request.claim})

        # Add background task to store result if vector storage is enabled
        if settings.enable_vector_storage and result.get("verdict") not in [
            "Rejected",
            "Unclear",
        ]:
            background_tasks.add_task(
                store_fact_check_background,
                claim=fact_check_request.claim,
                verdict=result.get("verdict"),
                confidence=result.get("confidence", 0),
                reasoning=result.get("reasoning", ""),
                tldr=result.get("tldr", ""),
                sources=result.get("sources", []),
            )
            logger.info("Added background task to store fact-check result")

        logger.info(
            "Fact-check completed",
            claim=fact_check_request.claim[:100],
            verdict=result.get("verdict"),
        )

        return result

    except Exception as e:
        logger.error(
            "Fact-check endpoint error",
            error=str(e),
            claim=fact_check_request.claim[:100],
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error during fact-checking",
                "detail": str(e),
            },
        )


# Add LangServe routes for the chain
def add_fact_check_routes(app):
    """Add LangServe routes to the FastAPI app"""
    add_routes(app, fact_check_chain, path="/verify")
