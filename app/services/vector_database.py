import json
import asyncio
from typing import List, Optional, Union, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from app.core.database import FactCheckRecordDB
from app.domain.models import FactCheckRecord, VectorSearchResult, Source
from app.services.embedding_service import embedding_service
from app.core.config import settings
from app.core.logging import get_logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = get_logger()

DEFAULT_LIMIT = 10
MAX_CLAIM_LENGTH = 2000
EMBEDDING_DIMENSIONS = 384


class VectorDatabaseService:
    """Service for vector database operations with Neon PostgreSQL"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((SQLAlchemyError,)),
    )
    async def store_fact_check(
        self,
        session: AsyncSession,
        claim: str,
        verdict: str,
        confidence: int,
        reasoning: str,
        tldr: Optional[str],
        sources: List[Union[Source, Dict[str, Any]]],
    ) -> Optional[FactCheckRecord]:
        """Store a fact-check result with its embedding"""
        if not claim or not claim.strip():
            logger.warning("Attempted to store empty claim")
            return None

        try:
            # Validate input
            if len(claim) > MAX_CLAIM_LENGTH:
                claim = claim[:MAX_CLAIM_LENGTH]
                logger.warning(f"Claim truncated to {MAX_CLAIM_LENGTH} characters")

            normalized_claim = await embedding_service.normalize_text_for_embedding(
                claim
            )

            try:
                async with asyncio.timeout(30.0):
                    embedding = await embedding_service.get_embedding(normalized_claim)
            except asyncio.TimeoutError:
                logger.error("Embedding generation timed out")
                return None

            # Convert to proper format for pgvector
            import numpy as np

            embedding_array = np.array(embedding, dtype=np.float32)

            # Validate embedding dimensions
            if len(embedding) != EMBEDDING_DIMENSIONS:
                logger.error(
                    f"Invalid embedding dimensions: {len(embedding)}, expected {EMBEDDING_DIMENSIONS}"
                )
                return None

            # Convert sources to JSON - handle both Source objects and dictionaries
            sources_data = self._normalize_sources(sources)
            sources_json = json.dumps(sources_data)

            # Create database record
            db_record = FactCheckRecordDB(
                claim=claim,
                claim_embedding=embedding_array,
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                tldr=tldr,
                sources_json=sources_json,
            )

            session.add(db_record)
            await session.commit()
            await session.refresh(db_record)

            logger.info(f"Stored fact-check record with ID: {db_record.id}")

            return await self._db_to_domain_model(db_record)

        except IntegrityError as e:
            await session.rollback()
            logger.error(f"Integrity error storing fact-check record: {e}")
            return None
        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error storing fact-check record: {e}")
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Unexpected error storing fact-check record: {e}")
            return None

    def _normalize_sources(
        self, sources: List[Union[Source, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Normalize sources to consistent dictionary format"""
        sources_data = []
        for source in sources:
            try:
                if hasattr(source, "model_dump") and callable(
                    getattr(source, "model_dump")
                ):
                    # Source object
                    sources_data.append(source.model_dump())
                elif isinstance(source, dict):
                    # Dictionary (from API response)
                    sources_data.append(source)
                else:
                    # Fallback - try to convert to dict
                    sources_data.append(dict(source))
            except Exception as e:
                logger.warning(f"Failed to normalize source: {e}")
                continue
        return sources_data

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=6),
        retry=retry_if_exception_type((SQLAlchemyError,)),
    )
    async def find_similar_claims(
        self,
        session: AsyncSession,
        claim: str,
        max_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Find similar claims using vector similarity search"""
        if not claim or not claim.strip():
            logger.warning("Attempted to search with empty claim")
            return []

        if max_results is None:
            max_results = settings.max_vector_results

        if similarity_threshold is None:
            similarity_threshold = settings.vector_similarity_threshold

        try:
            normalized_claim = await embedding_service.normalize_text_for_embedding(
                claim
            )

            try:
                async with asyncio.timeout(30.0):
                    query_embedding = await embedding_service.get_embedding(
                        normalized_claim
                    )
            except asyncio.TimeoutError:
                logger.error("Embedding generation timed out for similarity search")
                return []

            # Convert embedding to PostgreSQL vector string
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

            # Perform vector similarity search using pgvector with optimized query
            stmt = (
                select(
                    FactCheckRecordDB.id,
                    FactCheckRecordDB.claim,
                    FactCheckRecordDB.claim_embedding,
                    FactCheckRecordDB.verdict,
                    FactCheckRecordDB.confidence,
                    FactCheckRecordDB.reasoning,
                    FactCheckRecordDB.tldr,
                    FactCheckRecordDB.sources_json,
                    FactCheckRecordDB.created_at,
                    FactCheckRecordDB.updated_at,
                    text("claim_embedding <-> :embedding as distance"),
                )
                .where(FactCheckRecordDB.claim_embedding.is_not(None))
                .order_by(text("claim_embedding <-> :embedding"))
                .limit(max_results * 2)
            )  # Get more results to filter by threshold

            result = await session.execute(stmt, {"embedding": embedding_str})
            rows = result.fetchall()

            # Convert to domain models and calculate similarities
            vector_results = []
            for row in rows:
                if row.claim_embedding is not None:
                    # Convert vector back to list for similarity calculation
                    stored_embedding = list(row.claim_embedding)
                    similarity = embedding_service.cosine_similarity(
                        query_embedding, stored_embedding
                    )

                    # Check if similarity meets threshold
                    if similarity >= similarity_threshold:
                        # Create domain record
                        domain_record = FactCheckRecord(
                            id=row.id,
                            claim=row.claim,
                            claim_embedding=row.claim_embedding,
                            verdict=row.verdict,
                            confidence=row.confidence,
                            reasoning=row.reasoning,
                            tldr=row.tldr,
                            sources_json=row.sources_json,
                            created_at=row.created_at,
                            updated_at=row.updated_at,
                        )

                        is_exact_match = (
                            claim.lower().strip() == row.claim.lower().strip()
                        )

                        vector_results.append(
                            VectorSearchResult(
                                record=domain_record,
                                similarity_score=similarity,
                                is_exact_match=is_exact_match,
                            )
                        )

            # Sort by similarity score (highest first) and limit results
            vector_results.sort(key=lambda x: x.similarity_score, reverse=True)
            vector_results = vector_results[:max_results]

            logger.info(
                f"Found {len(vector_results)} similar claims for: {claim[:50]}..."
            )
            return vector_results

        except SQLAlchemyError as e:
            logger.error(f"Database error finding similar claims: {e}")
            raise  # Retry on database errors
        except Exception as e:
            logger.error(f"Unexpected error finding similar claims: {e}")
            return []

    async def get_exact_match(
        self, session: AsyncSession, claim: str
    ) -> Optional[FactCheckRecord]:
        """Find exact match for a claim"""
        if not claim or not claim.strip():
            return None

        try:
            # Use case-insensitive exact match
            normalized_claim = claim.strip().lower()
            stmt = select(FactCheckRecordDB).where(
                func.lower(FactCheckRecordDB.claim) == normalized_claim
            )

            result = await session.execute(stmt)
            db_record = result.scalar_one_or_none()

            if db_record:
                logger.info(f"Found exact match for claim: {claim[:50]}...")
                return await self._db_to_domain_model(db_record)

            return None

        except SQLAlchemyError as e:
            logger.error(f"Database error finding exact match: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error finding exact match: {e}")
            return None

    async def get_recent_fact_checks(
        self, session: AsyncSession, limit: int = DEFAULT_LIMIT
    ) -> List[FactCheckRecord]:
        """Get recent fact-check records"""
        try:
            stmt = (
                select(FactCheckRecordDB)
                .order_by(FactCheckRecordDB.created_at.desc())
                .limit(limit)
            )

            result = await session.execute(stmt)
            db_records = result.scalars().all()

            domain_records = []
            for db_record in db_records:
                try:
                    domain_record = await self._db_to_domain_model(db_record)
                    domain_records.append(domain_record)
                except Exception as e:
                    logger.warning(f"Failed to convert record {db_record.id}: {e}")
                    continue

            return domain_records

        except SQLAlchemyError as e:
            logger.error(f"Database error getting recent fact checks: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting recent fact checks: {e}")
            return []

    async def get_fact_check_by_id(
        self, session: AsyncSession, record_id: int
    ) -> Optional[FactCheckRecord]:
        """Get fact-check record by ID"""
        try:
            stmt = select(FactCheckRecordDB).where(FactCheckRecordDB.id == record_id)
            result = await session.execute(stmt)
            db_record = result.scalar_one_or_none()

            if db_record:
                return await self._db_to_domain_model(db_record)

            return None

        except SQLAlchemyError as e:
            logger.error(f"Database error getting fact check by ID: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting fact check by ID: {e}")
            return None

    async def _db_to_domain_model(
        self, db_record: FactCheckRecordDB
    ) -> FactCheckRecord:
        """Convert database model to domain model"""
        try:
            # Parse sources from JSON with error handling
            if db_record.sources_json:
                try:
                    json.loads(db_record.sources_json)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse sources JSON for record {db_record.id}: {e}"
                    )

            return FactCheckRecord(
                id=db_record.id,
                claim=db_record.claim,
                claim_embedding=db_record.claim_embedding,
                verdict=db_record.verdict,
                confidence=db_record.confidence,
                reasoning=db_record.reasoning,
                tldr=db_record.tldr,
                sources_json=db_record.sources_json,
                created_at=db_record.created_at,
                updated_at=db_record.updated_at,
            )

        except Exception as e:
            logger.error(f"Error converting DB model to domain model: {e}")
            raise

    async def get_statistics(self, session: AsyncSession) -> dict:
        """Get database statistics with error handling"""
        try:
            # Total records
            total_stmt = select(func.count(FactCheckRecordDB.id))
            total_result = await session.execute(total_stmt)
            total_count = total_result.scalar() or 0

            # Records by verdict
            verdict_stmt = select(
                FactCheckRecordDB.verdict,
                func.count(FactCheckRecordDB.id).label("count"),
            ).group_by(FactCheckRecordDB.verdict)

            verdict_result = await session.execute(verdict_stmt)
            verdict_counts = {row.verdict: row.count for row in verdict_result}

            # Average confidence
            avg_confidence_stmt = select(func.avg(FactCheckRecordDB.confidence))
            avg_result = await session.execute(avg_confidence_stmt)
            avg_confidence = avg_result.scalar() or 0

            # Recent activity (last 24 hours)
            recent_stmt = select(func.count(FactCheckRecordDB.id)).where(
                FactCheckRecordDB.created_at >= func.now() - text("INTERVAL '24 hours'")
            )
            recent_result = await session.execute(recent_stmt)
            recent_count = recent_result.scalar() or 0

            return {
                "total_records": total_count,
                "verdict_distribution": verdict_counts,
                "average_confidence": round(float(avg_confidence), 2),
                "recent_activity_24h": recent_count,
            }

        except SQLAlchemyError as e:
            logger.error(f"Database error getting statistics: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting statistics: {e}")
            return {}

    async def cleanup_old_records(self, session: AsyncSession, days: int = 90) -> int:
        """Clean up old fact-check records (optional maintenance)"""
        try:
            stmt = select(FactCheckRecordDB.id).where(
                FactCheckRecordDB.created_at
                < func.now() - text(f"INTERVAL '{days} days'")
            )
            result = await session.execute(stmt)
            old_record_ids = result.scalars().all()

            if old_record_ids:
                delete_stmt = FactCheckRecordDB.__table__.delete().where(
                    FactCheckRecordDB.id.in_(old_record_ids)
                )
                await session.execute(delete_stmt)
                await session.commit()

                logger.info(f"Cleaned up {len(old_record_ids)} old records")
                return len(old_record_ids)

            return 0

        except SQLAlchemyError as e:
            await session.rollback()
            logger.error(f"Database error during cleanup: {e}")
            return 0
        except Exception as e:
            await session.rollback()
            logger.error(f"Unexpected error during cleanup: {e}")
            return 0


# Global instance
vector_database_service = VectorDatabaseService()
