import asyncio
import hashlib
import json
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
import re

from app.domain.models import FactCheckRequest, FactCheckResponse, Verdict, Source
from app.services.web_search import web_search_service
from app.core.config import settings
import structlog

logger = structlog.get_logger()

# Initialize OpenAI client for Hugging Face Inference Providers
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=settings.huggingface_api_key,
)

# RAG Prompt Template for Mistral
FACT_CHECK_PROMPT = ChatPromptTemplate.from_template("""
<s>[INST] You are an expert fact-checker. Your task is to verify the following claim using the provided sources.

CLAIM TO VERIFY: {claim}

SOURCES:
{sources}

INSTRUCTIONS:
1. Analyze the claim against the provided sources.
2. Prioritize information from reputable, authoritative sources (e.g., academic publications, government websites, major news organizations).
3. Disregard or downweight information from less reliable sources such as Reddit, forums, social media, or user-generated content, unless no other sources are available.
4. Determine if the claim is TRUE, FALSE, or UNCLEAR.
5. If the sources disagree, or if there is controversy or ambiguity, explain both sides and cite sources for each.
6. If the answer depends on definitions or context, clearly state this and explain the different interpretations.
7. Provide a confidence score (0-100).
8. Write a clear, detailed explanation of your reasoning, citing the most credible sources. Include direct quotes or evidence where possible.
9. If only less reliable sources are available, mention this in your explanation.

OUTPUT FORMAT:
- Verdict: [True/False/Unclear]
- Confidence: [0-100]
- Reasoning: [Detailed explanation, including both sides if relevant]

If sources are insufficient or contradictory, mark as UNCLEAR with low confidence. [/INST]
""")


async def fact_check_chain_logic(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main fact-checking logic using RAG approach with Hugging Face
    """
    try:
        # Extract claim from input
        claim = input_data.get("claim", "")
        # Ensure claim is a string
        if not isinstance(claim, str):
            claim = str(claim) if claim is not None else ""

        if not claim or not claim.strip():
            return {
                "verdict": "Unclear",
                "confidence": 0,
                "reasoning": "No claim provided",
                "sources": [],
                "claim": claim
            }

        logger.info("Processing fact-check request",
                    claim=claim[:100])

        # Step 1: Web search for relevant information
        print(f"Searching for information about: {claim}")
        try:
            sources = await web_search_service.search_claim(claim, num_results=5)
            print(f"Found {len(sources)} sources")

            if not sources:
                return {
                    "verdict": "Unclear",
                    "confidence": 0,
                    "reasoning": "Unable to find relevant sources to verify this claim. Please check your Serper.dev API key.",
                    "sources": [],
                    "claim": claim
                }
        except Exception as e:
            logger.error("Web search failed", error=str(e), claim=claim[:100])
            return {
                "verdict": "Unclear",
                "confidence": 0,
                "reasoning": f"Error during web search: {str(e)}. Please check your Serper.dev API key.",
                "sources": [],
                "claim": claim
            }

        # Step 2: Format sources for the prompt
        sources_text = "\n\n".join([
            f"Source {i+1}: {source.title}\nURL: {source.url}\nContent: {source.snippet}"
            for i, source in enumerate(sources)
        ])

        # Step 3: Generate fact-check using Hugging Face
        print("Generating fact-check analysis with Mistral-7B...")
        try:
            prompt_text = FACT_CHECK_PROMPT.format(
                claim=claim,
                sources=sources_text
            )
            print(f"Prompt being sent: {prompt_text[:200]}...")

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=settings.huggingface_model,
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.1,
                max_tokens=512
            )

            print(f"Response type: {type(response)}")
            print(f"Response: {response}")

            # Step 4: Parse the response
            if response and response.choices:
                analysis = response.choices[0].message.content
            else:
                analysis = "Error: No response received from Hugging Face API"

            print(f"Final analysis: {analysis}")

        except Exception as e:
            logger.error("Hugging Face API call failed",
                         error=str(e), claim=claim[:100])
            print(f"Error calling Hugging Face API: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            analysis = f"Error calling AI model: {str(e)}"

        # Simple parsing (in production, you'd want more robust parsing)
        verdict = "Unclear"
        confidence = 50
        reasoning = analysis

        if analysis:
            # Try to extract the reasoning part only
            lower = analysis.lower()
            if "reasoning:" in lower:
                try:
                    # Find the start of the reasoning
                    start = lower.index("reasoning:") + len("reasoning:")
                    # Find the next section or end of string
                    next_section = lower.find("\n-", start)
                    if next_section == -1:
                        reasoning = analysis[start:].strip()
                    else:
                        reasoning = analysis[start:next_section].strip()
                except Exception:
                    reasoning = analysis
            else:
                reasoning = analysis

        # --- FIXED VERDICT EXTRACTION ---
        def extract_verdict(analysis: str) -> str:
            match = re.search(r'Verdict:\s*(True|False|Unclear)', analysis, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
            return "Unclear"

        if analysis:
            verdict = extract_verdict(analysis)

        # Extract confidence if mentioned
        if analysis and "confidence:" in analysis.lower():
            try:
                conf_part = analysis.lower().split(
                    "confidence:")[1].split("\n")[0]
                conf_num = int(''.join(filter(str.isdigit, conf_part)))
                if 0 <= conf_num <= 100:
                    confidence = conf_num
            except:
                pass

        result = {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "sources": [source.model_dump() for source in sources],
            "claim": claim
        }

        logger.info("Fact-check completed",
                    claim=claim[:100],
                    verdict=verdict,
                    confidence=confidence)

        return result

    except Exception as e:
        logger.error("Fact-checking failed", error=str(e),
                     claim=str(input_data.get("claim", ""))[:100] if input_data.get("claim") else "")
        return {
            "verdict": "Unclear",
            "confidence": 0,
            "reasoning": f"Error occurred during fact-checking: {str(e)}",
            "sources": [],
            "claim": str(input_data.get("claim", "")) if input_data.get("claim") else ""
        }

# Create the LangServe chain
fact_check_chain = RunnableLambda(fact_check_chain_logic)
