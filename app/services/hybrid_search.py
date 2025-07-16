import asyncio
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.models import Source, HybridSearchResult, VectorSearchResult, FactCheckRecord
from app.services.web_search import web_search_service
from app.services.vector_database import vector_database_service
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class HybridSearchService:
    """Service that combines vector database search with web search for enhanced RAG"""
    
    async def search_claim(
        self, 
        session: AsyncSession, 
        claim: str, 
        num_web_results: int = 5,
        use_vector_cache: bool = True
    ) -> HybridSearchResult:
        """
        Perform hybrid search combining vector database and web search
        """
        vector_results = []
        web_results = []
        used_vector_cache = False
        
        try:
            # Step 1: Check for exact match in vector database
            exact_match = None
            if use_vector_cache:
                exact_match = await vector_database_service.get_exact_match(session, claim)
                if exact_match:
                    logger.info(f"Found exact match in vector database for: {claim[:50]}...")
                    used_vector_cache = True
            
            # Step 2: Search for similar claims in vector database
            if use_vector_cache and settings.enable_vector_search:
                vector_results = await vector_database_service.find_similar_claims(
                    session, 
                    claim,
                    max_results=settings.max_vector_results,
                    similarity_threshold=settings.vector_similarity_threshold
                )
                
                # Add exact match to the beginning if it exists
                if exact_match:
                    exact_vector_result = VectorSearchResult(
                        record=exact_match,
                        similarity_score=1.0,
                        is_exact_match=True
                    )
                    vector_results.insert(0, exact_vector_result)
                
                if vector_results:
                    logger.info(f"Found {len(vector_results)} similar claims in vector database")
                    used_vector_cache = True
            
            # Step 3: Perform web search
            web_results = await web_search_service.search_claim(claim, num_web_results)
            logger.info(f"Found {len(web_results)} web search results")
            
            # Step 4: Combine and deduplicate sources
            combined_sources = await self._combine_and_deduplicate_sources(
                vector_results, web_results
            )
            
            return HybridSearchResult(
                vector_results=vector_results,
                web_results=web_results,
                combined_sources=combined_sources,
                used_vector_cache=used_vector_cache
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to web search only
            try:
                web_results = await web_search_service.search_claim(claim, num_web_results)
                return HybridSearchResult(
                    vector_results=[],
                    web_results=web_results,
                    combined_sources=web_results,
                    used_vector_cache=False
                )
            except Exception as fallback_error:
                logger.error(f"Fallback web search also failed: {fallback_error}")
                return HybridSearchResult(
                    vector_results=[],
                    web_results=[],
                    combined_sources=[],
                    used_vector_cache=False
                )
    
    async def _combine_and_deduplicate_sources(
        self, 
        vector_results: List[VectorSearchResult], 
        web_results: List[Source]
    ) -> List[Source]:
        """Combine sources from vector and web search, removing duplicates"""
        combined_sources = []
        seen_urls = set()
        
        # Add sources from vector results (high similarity ones first)
        for vector_result in vector_results:
            if vector_result.similarity_score >= 0.9:  # High confidence matches
                import json
                sources_data = json.loads(vector_result.record.sources_json) if vector_result.record.sources_json else []
                for source_data in sources_data:
                    source = Source(**source_data)
                    if source.url not in seen_urls:
                        combined_sources.append(source)
                        seen_urls.add(source.url)
        
        # Add web search results
        for web_source in web_results:
            if web_source.url not in seen_urls:
                combined_sources.append(web_source)
                seen_urls.add(web_source.url)
        
        # Add remaining vector results (lower similarity)
        for vector_result in vector_results:
            if vector_result.similarity_score < 0.9:
                import json
                sources_data = json.loads(vector_result.record.sources_json) if vector_result.record.sources_json else []
                for source_data in sources_data:
                    source = Source(**source_data)
                    if source.url not in seen_urls:
                        combined_sources.append(source)
                        seen_urls.add(source.url)
        
        # Limit total sources to prevent overwhelming the AI
        max_total_sources = 8
        if len(combined_sources) > max_total_sources:
            combined_sources = combined_sources[:max_total_sources]
        
        logger.info(f"Combined {len(combined_sources)} unique sources from vector and web search")
        return combined_sources
    
    async def get_search_statistics(self, session: AsyncSession) -> dict:
        """Get statistics about search performance"""
        try:
            # Get vector database statistics
            vector_stats = await vector_database_service.get_statistics(session)
            
            return {
                "vector_database": vector_stats,
                "search_config": {
                    "vector_similarity_threshold": settings.vector_similarity_threshold,
                    "max_vector_results": settings.max_vector_results,
                    "enable_vector_search": settings.enable_vector_search,
                    "enable_vector_storage": settings.enable_vector_storage
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {}


# Global instance
hybrid_search_service = HybridSearchService() 