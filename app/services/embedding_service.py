import asyncio
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger()


class EmbeddingService:
    """Service for generating and comparing text embeddings"""

    def __init__(self):
        # Use a lightweight but effective model for embeddings
        self.model_name = "all-MiniLM-L6-v2"  # 384 dimensions, fast and accurate
        self.model: Optional[SentenceTransformer] = None
        self._model_loaded = False

    async def _load_model(self):
        """Load the embedding model asynchronously"""
        if not self._model_loaded:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = await asyncio.to_thread(
                    SentenceTransformer, self.model_name
                )
                self._model_loaded = True
                logger.info("Embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a text string"""
        await self._load_model()

        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        try:
            # Generate embedding
            embedding = await asyncio.to_thread(
                self.model.encode, text, convert_to_tensor=False
            )

            # Convert to list of floats
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        await self._load_model()

        if not self.model:
            raise RuntimeError("Embedding model not loaded")

        try:
            # Generate embeddings in batch
            embeddings = await asyncio.to_thread(
                self.model.encode, texts, convert_to_tensor=False
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def is_similar(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if two embeddings are similar based on threshold"""
        if threshold is None:
            threshold = settings.vector_similarity_threshold

        similarity = self.cosine_similarity(embedding1, embedding2)
        return similarity >= threshold

    async def normalize_text_for_embedding(self, text: str) -> str:
        """Normalize text for better embedding generation"""
        # Basic text normalization
        normalized = text.strip().lower()

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        # Truncate if too long (embeddings work best with reasonable length)
        if len(normalized) > 1000:
            normalized = normalized[:1000]

        return normalized


# Global instance
embedding_service = EmbeddingService()
