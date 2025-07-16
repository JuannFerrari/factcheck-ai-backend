import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import settings
import structlog

logger = structlog.get_logger()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.huggingface_api_key,
)


class ModerationDecision(Enum):
    ALLOW = "allow"
    REJECT = "reject"


@dataclass
class ModerationResult:
    decision: ModerationDecision
    confidence: float  # 0.0–1.0
    reason: Optional[str] = None
    flagged_categories: Optional[list[str]] = None


class ContentModerationService:
    """Evaluate claims for appropriateness before fact-checking."""

    def __init__(self):
        self.enable_moderation = settings.enable_content_moderation
        self.strictness = settings.content_moderation_strictness

        self.prompt = ChatPromptTemplate.from_template(
            """
<s>[INST] You are a content moderation expert. Decide if the claim is
appropriate for fact-checking.

CLAIM: {claim}

INSTRUCTIONS:
- REJECT claims about: illegal activities, violence, explicit content, self-harm,
  harmful instructions, or personal attacks.
- ALLOW claims about: conspiracy theories, misinformation, sports, history, science,
  health, politics, geography, entertainment, business, technology, etc.
- Fact-checking is meant to debunk false claims, so allow controversial or false
  claims to be fact-checked.

RESPONSE FORMAT:
- Only respond with "true" (safe) or "false" (unsafe). No extra text.

EXAMPLES:
- "How to hack a computer" → false
- "How to make a bomb" → false
- "Is Argentina the biggest country in South America?" → true
- "What are the health benefits of running?" → true
- "Messi is the best player" → true
- "Which team won the World Cup?" → true
- "What is the capital of France?" → true
- "Vaccines cause autism" → true (allow for fact-checking)
- "The Earth is flat" → true (allow for fact-checking)
- "5G causes COVID" → true (allow for fact-checking)

[/INST]
"""
        )

    # --- Main entry ---
    async def evaluate_claim(self, claim: str) -> ModerationResult:
        """Evaluate whether a claim is appropriate for fact-checking."""
        if not self.enable_moderation:
            return self._allow("Moderation disabled")

        if not claim or not claim.strip():
            return self._reject("Empty or whitespace-only claim")

        try:
            ai_result = await self._query_ai(claim)
            return self._parse_ai_response(claim, ai_result)

        except Exception as e:
            logger.error("Content moderation failed", error=str(e), claim=claim[:100])
            return self._allow(
                f"Moderation failed ({e}), defaulting to allow", confidence=0.3
            )

    # --- Internal AI call ---
    async def _query_ai(self, claim: str) -> str:
        """Send moderation prompt and return raw AI response."""
        prompt_text = self.prompt.format(claim=claim)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.huggingface_model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.1,
            max_tokens=20,
        )
        content = (
            response.choices[0].message.content if response and response.choices else ""
        )
        return content.strip().lower() if content else ""

    # --- Response parsing ---
    def _parse_ai_response(self, claim: str, result: str) -> ModerationResult:
        """Interpret AI output into a ModerationResult."""
        if not result:
            logger.warning("No AI response for content moderation", claim=claim[:100])
            return self._allow("No AI response, defaulting to allow", confidence=0.5)

        if result.startswith("true"):
            return self._allow("Claim approved by AI moderation")
        elif result.startswith("false"):
            return self._reject("Claim rejected by AI moderation")
        else:
            logger.warning(
                "Unexpected AI moderation response",
                claim=claim[:100],
                response=result,
            )
            return self._allow(
                f"Unexpected AI response '{result}', defaulting to allow",
                confidence=0.5,
            )

    # --- Helper decisions ---
    def _allow(self, reason: str, confidence: float = 0.95) -> ModerationResult:
        return ModerationResult(ModerationDecision.ALLOW, confidence, reason)

    def _reject(self, reason: str, confidence: float = 0.95) -> ModerationResult:
        return ModerationResult(ModerationDecision.REJECT, confidence, reason)

    # --- Public helpers ---
    def is_appropriate(self, result: ModerationResult) -> bool:
        return result.decision == ModerationDecision.ALLOW

    def is_enabled(self) -> bool:
        return self.enable_moderation


# Global instance
content_moderation_service = ContentModerationService()
