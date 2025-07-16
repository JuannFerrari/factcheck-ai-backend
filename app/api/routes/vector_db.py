from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.database import AsyncSessionLocal
from app.services.vector_database import vector_database_service
from app.services.hybrid_search import hybrid_search_service
from app.core.logging import get_logger
from app.core.rate_limiter import limiter
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


@router.get("/statistics")
@limiter.limit("30/minute;5/second")
async def get_statistics(request: Request, _=Depends(verify_api_key)):
    """Get vector database and search statistics"""
    if not settings.enable_vector_search:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Vector search is disabled",
                "detail": "Vector database features are not available in this deployment.",
            },
        )

    try:
        async with AsyncSessionLocal() as session:
            # Get vector database statistics
            vector_stats = await vector_database_service.get_statistics(session)

            # Get search statistics
            search_stats = await hybrid_search_service.get_search_statistics(session)

            return {
                "vector_database": vector_stats,
                "search": search_stats,
                "configuration": {
                    "vector_similarity_threshold": settings.vector_similarity_threshold,
                    "max_vector_results": settings.max_vector_results,
                    "enable_vector_search": settings.enable_vector_search,
                    "enable_vector_storage": settings.enable_vector_storage,
                },
            }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve statistics", "detail": str(e)},
        )


@router.get("/recent")
@limiter.limit("30/minute;5/second")
async def get_recent_fact_checks(
    request: Request, limit: int = 10, _=Depends(verify_api_key)
):
    """Get recent fact-check records"""
    if not settings.enable_vector_search:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Vector search is disabled",
                "detail": "Vector database features are not available in this deployment.",
            },
        )

    try:
        if limit > 50:
            limit = 50  # Cap at 50 records

        async with AsyncSessionLocal() as session:
            records = await vector_database_service.get_recent_fact_checks(
                session, limit
            )

            return {
                "records": [record.model_dump() for record in records],
                "count": len(records),
            }

    except Exception as e:
        logger.error(f"Error getting recent fact checks: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to retrieve recent fact checks",
                "detail": str(e),
            },
        )


@router.get("/search")
@limiter.limit("30/minute;5/second")
async def search_similar_claims(
    request: Request,
    claim: str,
    max_results: int = 5,
    similarity_threshold: float = 0.85,
    _=Depends(verify_api_key),
):
    """Search for similar claims in the vector database"""
    if not settings.enable_vector_search:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Vector search is disabled",
                "detail": "Vector database features are not available in this deployment.",
            },
        )

    try:
        if not claim.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Claim parameter is required",
            )

        if max_results > 20:
            max_results = 20  # Cap at 20 results

        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            similarity_threshold = 0.85

        async with AsyncSessionLocal() as session:
            results = await vector_database_service.find_similar_claims(
                session, claim, max_results, similarity_threshold
            )

            return {
                "query": claim,
                "results": [result.model_dump() for result in results],
                "count": len(results),
                "similarity_threshold": similarity_threshold,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching similar claims: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to search similar claims", "detail": str(e)},
        )


@router.get("/record/{record_id}")
@limiter.limit("30/minute;5/second")
async def get_fact_check_record(
    request: Request, record_id: int, _=Depends(verify_api_key)
):
    """Get a specific fact-check record by ID"""
    if not settings.enable_vector_search:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Vector search is disabled",
                "detail": "Vector database features are not available in this deployment.",
            },
        )

    try:
        async with AsyncSessionLocal() as session:
            record = await vector_database_service.get_fact_check_by_id(
                session, record_id
            )

            if not record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Fact-check record not found",
                )

            return record.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting fact check record: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve fact check record", "detail": str(e)},
        )
