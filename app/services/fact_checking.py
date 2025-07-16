import asyncio
import re
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.hybrid_search import hybrid_search_service
from app.services.content_moderation import content_moderation_service
from app.services.vector_database import vector_database_service
from app.core.config import settings
from app.core.database import AsyncSessionLocal
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import APITimeoutError, APIConnectionError, APIStatusError

logger = structlog.get_logger()

# Initialize OpenAI client for Hugging Face Inference Providers with retry configuration
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.huggingface_api_key,
    timeout=30.0,  # 30 second timeout
    max_retries=2,  # Built-in retry logic
)

# Constants for better maintainability
SIMILARITY_THRESHOLD = 0.55
MAX_SOURCES = 5
CACHE_TIMEOUT = 25.0  # seconds
DEFAULT_CONFIDENCE = 50
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1

FACT_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """<s>[INST] You are an expert fact-checker. Verify the claim below using the provided sources.

CLAIM: {claim}

SOURCES:
{sources}

Your task:
1. Compare the claim with the sources.
2. Decide the verdict:
   - TRUE → Most credible sources strongly support the claim.
   - FALSE → Most credible sources strongly reject the claim.
   - DISPUTED → Credible sources directly contradict each other.
   - UNCLEAR → There isn't enough evidence or the claim is ambiguous.
3. Always prioritize credible sources (e.g. CDC, WHO, government, universities).
4. Ignore conspiracy theories or unreliable sources unless no other information exists.
5. If ALL sources are from social media (Twitter, Facebook, Instagram, TikTok, etc.), mark as UNCLEAR due to lack of reliable verification.
6. Provide detailed analysis of the specific sources - explain which sources support/reject the claim and why they are credible or not. READ SOURCES CAREFULLY.

OUTPUT FORMAT (must follow this exact structure):
Verdict: [TRUE/FALSE/DISPUTED/UNCLEAR]
Confidence: [0-100]
Reasoning: [One short sentence]

TL;DR_START
[2-3 simple sentences for a quick summary]
TL;DR_END

[Detailed analysis: Explain which specific sources support or reject the claim, their credibility, and why you reached this verdict. Be specific about source numbers and content. Quote or paraphrase exactly what each source says - do not misrepresent their content.]

Sources:
- [Source title] ([URL])
- [Source title] ([URL])

EXAMPLE:
Verdict: FALSE
Confidence: 95
Reasoning: CDC and WHO confirm vaccines don't cause autism.

TL;DR_START
Vaccines do NOT cause autism. Major studies show no link.
TL;DR_END

Source 1 (CDC) provides comprehensive data showing no correlation between vaccines and autism in large-scale studies. Source 2 (WHO) confirms this with international research findings and explicitly states that the original study linking vaccines to autism was fraudulent and retracted. Both sources are highly credible government health organizations with extensive research backing their conclusions.

Sources:
- CDC - Vaccine Safety (https://www.cdc.gov/vaccinesafety/concerns/autism.html)
- WHO - Vaccine Myths (https://www.who.int/news-room/questions-and-answers/item/vaccines-and-autism)
[/INST]"""
)


def normalize_paragraphs(text: str) -> str:
    """Normalize paragraph spacing in text."""
    return re.sub(r"\n{2,}", "\n\n", text).strip()


def parse_tldr(analysis: str) -> tuple[str, str]:
    """Extract TL;DR section and return (tldr, remaining_analysis)."""
    tldr_match = re.search(r"TL;DR_START\s*(.+?)\s*TL;DR_END", analysis, re.DOTALL)
    if not tldr_match:
        return "", analysis
    tldr = tldr_match.group(1).strip()
    cleaned_analysis = re.sub(
        r"TL;DR_START\s*.+?\s*TL;DR_END", "", analysis, flags=re.DOTALL
    ).strip()
    return tldr, cleaned_analysis


def extract_verdict(analysis: str) -> str:
    """Extract verdict from model response."""
    match = re.search(
        r"Verdict:\s*(True|False|Unclear|Disputed|Rejected)", analysis, re.IGNORECASE
    )
    return match.group(1).capitalize() if match else "Unclear"


def extract_confidence(analysis: str) -> int:
    """Extract numeric confidence score from analysis."""
    match = re.search(r"Confidence:\s*(\d{1,3})", analysis, re.IGNORECASE)
    return int(match.group(1)) if match else DEFAULT_CONFIDENCE


def extract_reasoning(analysis: str) -> str:
    """Extract reasoning after 'Reasoning:' and clean up duplicate sections."""
    lower = analysis.lower()
    if "reasoning:" not in lower:
        return analysis.strip()

    start = lower.index("reasoning:") + len("reasoning:")

    # Find the end of reasoning section (before TL;DR or other sections)
    end = len(analysis)
    for marker in ["\ntl;dr_start", "\ndetailed explanation:", "\nsources:", "\n-"]:
        pos = lower.find(marker, start)
        if pos != -1 and pos < end:
            end = pos

    reasoning = analysis[start:end].strip()

    # Clean up any remaining "Detailed explanation:" or "Sources:" sections
    reasoning = re.sub(
        r"\n\s*Detailed explanation:\s*", "\n", reasoning, flags=re.IGNORECASE
    )
    reasoning = re.sub(r"\n\s*Sources:\s*", "", reasoning, flags=re.IGNORECASE)

    return reasoning


def format_sources(sources: List[Any]) -> str:
    """Format search results into prompt-friendly text."""
    return "\n\n".join(
        f"Source {i+1}: {s.title}\nURL: {s.url}\nContent: {s.snippet}"
        for i, s in enumerate(sources)
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (APITimeoutError, APIConnectionError, APIStatusError)
    ),
)
async def call_model(
    claim: str, sources_text: str, similar_claims_context: str = ""
) -> str:
    """Call Hugging Face model with retry logic and return raw response text."""
    # Add similar claims context to sources if available
    full_sources_text = sources_text
    if similar_claims_context:
        full_sources_text = sources_text + similar_claims_context

    prompt = FACT_CHECK_PROMPT.format(claim=claim, sources=full_sources_text)

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=settings.huggingface_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return (
            response.choices[0].message.content or ""
            if response and response.choices
            else ""
        )
    except (APITimeoutError, APIConnectionError, APIStatusError) as e:
        logger.warning(f"Model call failed, retrying: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in model call: {e}")
        raise


async def store_fact_check_background(
    claim: str,
    verdict: str,
    confidence: int,
    reasoning: str,
    tldr: str,
    sources: List[Any],
) -> None:
    """Background task to store fact-check result in vector database with timeout protection."""
    if not settings.enable_vector_storage:
        return

    # Skip storage for certain verdicts to save resources
    if verdict in ["Rejected", "Unclear"] and confidence < 50:
        logger.info(
            f"Background: Skipping storage for low-confidence {verdict} verdict"
        )
        return

    # Add timeout for Render compatibility
    try:
        async with asyncio.timeout(CACHE_TIMEOUT):
            async with AsyncSessionLocal() as session:
                stored_record = await vector_database_service.store_fact_check(
                    session, claim, verdict, confidence, reasoning, tldr, sources
                )
                if stored_record:
                    logger.info(
                        f"Background: Stored fact-check in vector database with ID: {stored_record.id}"
                    )
                else:
                    logger.warning(
                        "Background: Failed to store fact-check in vector database"
                    )
    except asyncio.TimeoutError:
        logger.warning("Background: Storage task timed out (Render timeout limit)")
    except Exception as e:
        logger.error(f"Background: Failed to store fact-check in vector database: {e}")
        # Don't raise the exception - background tasks should not affect main flow


def build_similar_claims_context(hybrid_result) -> tuple[str, bool]:
    """Build context for similar claims and detect contradictions."""
    similar_claims = []
    for vector_result in hybrid_result.vector_results:
        if vector_result.similarity_score >= SIMILARITY_THRESHOLD:
            similar_claims.append(
                {
                    "claim": vector_result.record.claim,
                    "verdict": vector_result.record.verdict,
                    "confidence": vector_result.record.confidence,
                    "similarity": vector_result.similarity_score,
                    "is_exact": vector_result.is_exact_match,
                }
            )

    has_contradictions = (
        len(set(s["verdict"] for s in similar_claims)) > 1
        if len(similar_claims) > 1
        else False
    )

    similar_claims_context = ""
    if similar_claims:
        similar_claims_context = "\n\nSIMILAR PREVIOUS CLAIMS:\n"
        for i, similar in enumerate(similar_claims, 1):
            exact_marker = " (EXACT MATCH)" if similar.get("is_exact", False) else ""
            similar_claims_context += (
                f"{i}. Claim: '{similar['claim']}' - Verdict: {similar['verdict']} "
                f"(Confidence: {similar['confidence']}%, Similarity: {similar['similarity']:.2f})"
                f"{exact_marker}\n"
            )
        if has_contradictions:
            similar_claims_context += (
                "\n⚠️ CONTRADICTION DETECTED: Multiple similar claims have different "
                "verdicts. This suggests the topic may be disputed or requires careful "
                "investigation of which claim is actually correct.\n"
            )
        similar_claims_context += (
            "\nThis context may be relevant to your analysis. Pay special attention "
            "to contradictions and investigate which claim is actually correct."
        )

    return similar_claims_context, has_contradictions


def should_return_cached_result(hybrid_result, has_contradictions: bool) -> bool:
    """Determine if we should return cached result based on exact match and contradictions."""
    return (
        hybrid_result.vector_results
        and hybrid_result.vector_results[0].is_exact_match
        and not has_contradictions
    )


# --- Main Logic ---
async def fact_check_chain_logic(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main fact-checking logic with comprehensive error handling and optimization."""
    claim = str(input_data.get("claim", "") or "").strip()
    if not claim:
        return {
            "verdict": "Unclear",
            "confidence": 0,
            "reasoning": "No claim provided",
            "sources": [],
            "claim": "",
        }

    logger.info("Processing fact-check request", claim=claim[:100])

    # --- 1. Content moderation ---
    try:
        moderation = await content_moderation_service.evaluate_claim(claim)
        if not content_moderation_service.is_appropriate(moderation):
            return {
                "verdict": "Rejected",
                "confidence": 100,
                "reasoning": f"Inappropriate/unsafe content: {moderation.reason}",
                "sources": [],
                "claim": claim,
            }
    except Exception as e:
        logger.error(f"Content moderation failed: {e}")
        # Continue with fact-checking if moderation fails

    # --- 2. Hybrid search (vector + web) ---
    try:
        async with AsyncSessionLocal() as session:
            # Perform hybrid search (vector + web)
            hybrid_result = await hybrid_search_service.search_claim(
                session, claim, num_web_results=5, use_vector_cache=True
            )

            sources = hybrid_result.combined_sources

            # Log search results
            if hybrid_result.used_vector_cache:
                logger.info(f"Used vector cache for claim: {claim[:50]}...")
                if hybrid_result.vector_results:
                    logger.info(
                        f"Found {len(hybrid_result.vector_results)} similar claims"
                    )

            if not sources:
                return {
                    "verdict": "Unclear",
                    "confidence": 0,
                    "reasoning": "No relevant sources found",
                    "sources": [],
                    "claim": claim,
                }

            # Build similar claims context and check for contradictions
            similar_claims_context, has_contradictions = build_similar_claims_context(
                hybrid_result
            )

            # If exact match and no contradictions, return cache
            if should_return_cached_result(hybrid_result, has_contradictions):
                cached_record = hybrid_result.vector_results[0].record
                logger.info(f"Returning cached result for exact match: {claim[:50]}...")
                return {
                    "verdict": cached_record.verdict,
                    "confidence": cached_record.confidence,
                    "reasoning": cached_record.reasoning,
                    "tldr": cached_record.tldr,
                    "sources": sources,
                    "claim": claim,
                }

    except Exception as e:
        logger.error("Hybrid search failed", error=str(e))
        return {
            "verdict": "Unclear",
            "confidence": 0,
            "reasoning": f"Search error: {e}",
            "sources": [],
            "claim": claim,
        }

    # --- 3. AI analysis ---
    try:
        analysis = await call_model(
            claim, format_sources(sources), similar_claims_context
        )
    except Exception as e:
        logger.error("Model call failed", error=str(e))
        return {
            "verdict": "Unclear",
            "confidence": 0,
            "reasoning": f"AI model error: {e}",
            "sources": [s.model_dump() for s in sources],
            "claim": claim,
        }

    if not analysis:
        return {
            "verdict": "Unclear",
            "confidence": 0,
            "reasoning": "No response from AI model",
            "sources": [s.model_dump() for s in sources],
            "claim": claim,
        }

    # --- 4. Parse response ---
    tldr, cleaned = parse_tldr(analysis)
    verdict = extract_verdict(cleaned)
    confidence = extract_confidence(cleaned)
    reasoning = normalize_paragraphs(extract_reasoning(cleaned))

    logger.info(
        "Fact-check complete",
        claim=claim[:100],
        verdict=verdict,
        confidence=confidence,
    )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "tldr": tldr,
        "sources": [s.model_dump() for s in sources],
        "claim": claim,
    }


# --- LangServe chain with retry logic ---
fact_check_chain = RunnableLambda(fact_check_chain_logic).with_retry(
    retry_if_exception_type=(Exception,),
    stop_after_attempt=2,
    wait_exponential_jitter=True,
)
