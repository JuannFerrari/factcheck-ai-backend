from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
from enum import Enum


class Verdict(str, Enum):
    TRUE = "True"
    FALSE = "False"
    UNCLEAR = "Unclear"


class Source(BaseModel):
    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: Optional[str] = Field(
        None, description="Relevant text snippet from the source")


class FactCheckRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=1000,
                       description="The claim to fact-check")

    @field_validator('claim')
    @classmethod
    def validate_claim(cls, v):
        if not v.strip():
            raise ValueError('Claim cannot be empty')
        return v.strip()


class FactCheckResponse(BaseModel):
    verdict: Verdict = Field(..., description="Fact-check verdict")
    confidence: int = Field(..., ge=0, le=100,
                            description="Confidence score (0-100)")
    reasoning: str = Field(...,
                           description="Detailed explanation of the verdict")
    sources: List[Source] = Field(
        default_factory=list, description="Sources used for verification")
    claim: str = Field(...,
                       description="The original claim that was fact-checked")
