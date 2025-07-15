import asyncio
import re
from typing import Dict, Any, List
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.web_search import web_search_service
from app.services.content_moderation import content_moderation_service
from app.core.config import settings
import structlog

logger = structlog.get_logger()

# Initialize OpenAI client for Hugging Face Inference Providers
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.huggingface_api_key,
)

FACT_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """
<s>[INST] You are an expert fact-checker. Your task is to verify the following claim using the provided sources.

CLAIM TO VERIFY: {claim}

SOURCES:
{sources}

INSTRUCTIONS:

1. Analyze the claim against the provided sources.
2. Prioritize information from reputable, authoritative sources (e.g., academic publications, government websites, major news organizations).
3. Disregard or downweight information from less reliable sources such as Reddit, forums, social media, or user-generated content, unless no other sources are available.
4. Determine the verdict for the claim as one of the following:
   - TRUE → The claim is factually correct with strong supporting evidence.
   - FALSE → The claim is factually incorrect with strong contradictory evidence from authoritative sources.
   - UNCLEAR → Insufficient evidence or contradictory information prevents a clear conclusion.
   - DISPUTED → The claim relates to a politically sensitive, controversial, or context-dependent topic (e.g., territorial disputes, historical conflicts, ideological debates) where multiple perspectives exist.

5. If the claim is DISPUTED or involves controversy/ambiguity:
   - Present **both sides of the argument**.
   - Clearly explain the different viewpoints and who holds them.
   - Note the current legal/de facto status vs. claimed status, if relevant.
   - Cite sources supporting each perspective.

6. If the answer depends on definitions or context, clearly state this and explain the different interpretations.

7. Provide a confidence score (0-100) based on these guidelines:
   - **95-100**: Overwhelming evidence from multiple authoritative sources with no significant contradictions
   - **85-94**: Strong evidence from reputable sources with minor discrepancies in details
   - **75-84**: Good evidence but some uncertainty or limited sources
   - **60-74**: Moderate evidence with some contradictions or unreliable sources
   - **40-59**: Weak evidence, contradictory sources, or unclear information
   - **20-39**: Very limited evidence or highly unreliable sources
   - **0-19**: Insufficient evidence to make any determination

   For DISPUTED claims, the confidence reflects certainty about the existence of the dispute, not which side is "correct."

8. Write a clear, detailed explanation of your reasoning, citing the most credible sources. Include direct quotes or evidence where possible.

9. If only less reliable sources are available, mention this in your explanation.

**IMPORTANT VERDICT GUIDELINES:**
- Use FALSE when authoritative sources (CDC, WHO, major universities, government agencies) clearly debunk the claim
- Use FALSE for conspiracy theories that have been thoroughly debunked by credible sources
- Use FALSE for claims that contradict established scientific consensus
- Use UNCLEAR only when there is genuine lack of evidence or credible disagreement between authoritative sources
- Use DISPUTED for topics with legitimate competing viewpoints from credible sources

**EXAMPLES:**
- "Vaccines cause autism" → FALSE (debunked by CDC, WHO, major studies)
- "Vaccines contain microchips" → FALSE (conspiracy theory, no credible evidence)
- "The Earth is flat" → FALSE (contradicts established science)
- "Coffee is good for health" → UNCLEAR (mixed evidence, depends on amount/context)

OUTPUT FORMAT:
Your response MUST start with these three lines, each on their own line:
Verdict: [True/False/Unclear/Disputed/Rejected]
Confidence: [0-100]
Reasoning: [One-sentence summary of your reasoning]

Then provide a TL;DR section in this exact format:
TL;DR_START
[2-3 sentence summary of the key finding, written in simple language for immediate understanding]
TL;DR_END

After the TL;DR section, provide your detailed explanation and cite sources as needed.

**CONFIDENCE GUIDANCE:**
- Be confident (85-100) when multiple authoritative sources agree on the core fact, even if there are minor discrepancies in details
- Don't penalize confidence for minor ranking variations or small measurement differences
- Focus on whether the core claim is supported or contradicted, not on peripheral details
- If the evidence clearly shows a claim is false, use high confidence (90-100) even with minor data inconsistencies

If sources are insufficient or contradictory, mark as UNCLEAR with low confidence.
If the claim is political, territorial, or ideological in nature and has competing viewpoints, mark it as DISPUTED and explain all perspectives neutrally. [/INST]
"""  # noqa: E501
)


def normalize_paragraphs(text: str) -> str:
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
    return int(match.group(1)) if match else 50


def extract_reasoning(analysis: str) -> str:
    """Extract reasoning after 'Reasoning:'."""
    lower = analysis.lower()
    if "reasoning:" not in lower:
        return analysis.strip()
    start = lower.index("reasoning:") + len("reasoning:")
    # end at next section marker
    end = lower.find("\n-", start)
    return analysis[start:end].strip() if end != -1 else analysis[start:].strip()


def format_sources(sources: List[Any]) -> str:
    """Format search results into prompt-friendly text."""
    return "\n\n".join(
        f"Source {i+1}: {s.title}\nURL: {s.url}\nContent: {s.snippet}"
        for i, s in enumerate(sources)
    )


async def call_model(claim: str, sources_text: str) -> str:
    """Call Hugging Face model and return raw response text."""
    prompt = FACT_CHECK_PROMPT.format(claim=claim, sources=sources_text)
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=settings.huggingface_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=512,
    )
    return (
        response.choices[0].message.content or ""
        if response and response.choices
        else ""
    )


# --- Main Logic ---
async def fact_check_chain_logic(input_data: Dict[str, Any]) -> Dict[str, Any]:
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
    moderation = await content_moderation_service.evaluate_claim(claim)
    if not content_moderation_service.is_appropriate(moderation):
        return {
            "verdict": "Rejected",
            "confidence": 100,
            "reasoning": f"Inappropriate/unsafe content: {moderation.reason}",
            "sources": [],
            "claim": claim,
        }

    # --- 2. Web search ---
    try:
        sources = await web_search_service.search_claim(claim, num_results=5)
        if not sources:
            return {
                "verdict": "Unclear",
                "confidence": 0,
                "reasoning": "No relevant sources found",
                "sources": [],
                "claim": claim,
            }
    except Exception as e:
        logger.error("Web search failed", error=str(e))
        return {
            "verdict": "Unclear",
            "confidence": 0,
            "reasoning": f"Web search error: {e}",
            "sources": [],
            "claim": claim,
        }

    # --- 3. AI analysis ---
    try:
        analysis = await call_model(claim, format_sources(sources))
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
        "Fact-check complete", claim=claim[:100], verdict=verdict, confidence=confidence
    )

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": reasoning,
        "tldr": tldr,
        "sources": [s.model_dump() for s in sources],
        "claim": claim,
    }


# --- LangServe chain ---
fact_check_chain = RunnableLambda(fact_check_chain_logic)
