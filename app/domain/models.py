from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum
from datetime import datetime


class Verdict(str, Enum):
    TRUE = "True"
    FALSE = "False"
    UNCLEAR = "Unclear"
    DISPUTED = "Disputed"
    REJECTED = "Rejected"


class Source(BaseModel):
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: Optional[str] = Field(
        None, description="Relevant text snippet from the source"
    )


class FactCheckRequest(BaseModel):
    claim: str = Field(
        ..., min_length=1, max_length=1000, description="The claim to fact-check"
    )

    @field_validator("claim")
    @classmethod
    def validate_claim(cls, v):
        if not v.strip():
            raise ValueError("Claim cannot be empty")
        return v.strip()


class FactCheckResponse(BaseModel):
    verdict: Verdict = Field(..., description="Fact-check verdict")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score (0-100)")
    reasoning: str = Field(..., description="Detailed explanation of the verdict")
    tldr: Optional[str] = Field(None, description="TL;DR summary")
    sources: List[Source] = Field(
        default_factory=list, description="Sources used for verification"
    )
    claim: str = Field(..., description="The original claim that was fact-checked")


# Vector Database Models
class FactCheckRecord(BaseModel):
    id: Optional[int] = None
    claim: str = Field(..., description="The original claim")
    claim_embedding: Optional[List[float]] = Field(
        None, description="Claim embedding vector"
    )
    verdict: Verdict = Field(..., description="Fact-check verdict")
    confidence: int = Field(..., ge=0, le=100, description="Confidence score")
    reasoning: str = Field(..., description="Detailed explanation")
    tldr: Optional[str] = Field(None, description="TL;DR summary")
    sources_json: str = Field(..., description="JSON string of sources")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class VectorSearchResult(BaseModel):
    record: FactCheckRecord
    similarity_score: float = Field(..., description="Cosine similarity score")
    is_exact_match: bool = Field(
        False, description="Whether this is an exact claim match"
    )


class HybridSearchResult(BaseModel):
    vector_results: List[VectorSearchResult] = Field(default_factory=list)
    web_results: List[Source] = Field(default_factory=list)
    combined_sources: List[Source] = Field(default_factory=list)
    used_vector_cache: bool = Field(False, description="Whether vector cache was used")
