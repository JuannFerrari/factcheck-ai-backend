"""
Pytest configuration and shared fixtures for FactCheck AI Backend tests.
"""

import pytest
import asyncio
import httpx
from typing import Dict, Any
from unittest.mock import MagicMock
from app.domain.models import Source
from app.core.config import settings

# --- Test constants ---
TEST_BASE_URL = "http://localhost:8000"

TEST_CLAIMS = [
    "The Earth is flat.",
    "The Eiffel Tower is in Berlin.",
    "Water boils at 100°C at sea level.",
    "The Great Wall of China is visible from space.",
    "Humans use only 10% of their brain.",
]

EXPECTED_VERDICTS = ["True", "False", "Unclear", "Disputed", "Rejected"]
CONFIDENCE_RANGE = (0, 100)


# -------------------------
# Test configuration
# -------------------------
@pytest.fixture(scope="session", autouse=True)
def disable_vector_storage():
    """Disable vector storage during tests to prevent test data persistence"""
    original_setting = settings.enable_vector_storage
    settings.enable_vector_storage = False
    yield
    settings.enable_vector_storage = original_setting


# -------------------------
# Event loop for async tests
# -------------------------
@pytest.fixture(scope="session")
def event_loop():
    """Provide a session-wide event loop for asyncio tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# -------------------------
# Quick reusable mocks
# -------------------------
@pytest.fixture
def mock_huggingface_response():
    """Mock a typical Hugging Face LLM response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[
        0
    ].message.content = """
        Verdict: FALSE
        Confidence: 95
        Reasoning: Based on scientific evidence, the Earth is not flat.
    """
    return resp


@pytest.fixture
def mock_serper_response():
    """Mock Serper.dev search API response."""
    return {
        "organic": [
            {
                "title": "Earth - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Earth",
                "snippet": (
                    "Earth is the third planet from the Sun and the only "
                    "astronomical object known to harbor life."
                ),
            },
            {
                "title": "Flat Earth - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Flat_Earth",
                "snippet": (
                    "The flat Earth model is an archaic conception of Earth's "
                    "shape as a plane or disk."
                ),
            },
        ]
    }


# -------------------------
# Mock sources utilities
# -------------------------
def make_mock_source(
    title="Test Source", url="https://example.com", snippet="Test snippet"
) -> Source:
    return Source(title=title, url=url, snippet=snippet)


def make_mock_sources(n=1) -> list[Source]:
    return [
        make_mock_source(f"Test Source {i}", f"https://example{i}.com", f"Snippet {i}")
        for i in range(1, n + 1)
    ]


# -------------------------
# Test data helper class
# -------------------------
class TestData:
    VALID_CLAIMS = TEST_CLAIMS
    INVALID_CLAIMS = ["", "   ", "a" * 1001]  # empty, whitespace, too long

    @staticmethod
    def get_claim(idx: int = 0) -> str:
        """Return a valid claim by index."""
        return TEST_CLAIMS[idx % len(TEST_CLAIMS)]

    @staticmethod
    def factcheck_payload(claim: str) -> Dict[str, Any]:
        """Build a fact-check request payload."""
        return {"input": {"claim": claim}}

    @staticmethod
    def create_mock_sources(count=1) -> list[Source]:
        return make_mock_sources(count)


# -------------------------
# Assertion helpers
# -------------------------
class AssertionHelpers:
    @staticmethod
    def assert_valid_fact_check_response(resp: Dict[str, Any], expect_metadata=True):
        """Check that a fact-check response has valid structure."""
        output = resp

        for key in ("verdict", "confidence", "reasoning", "sources", "claim"):
            assert key in output, f"Missing output key: {key}"

        # verdict must be one of expected
        assert (
            output["verdict"] in EXPECTED_VERDICTS
        ), f"Invalid verdict: {output['verdict']}"
        # confidence must be int in valid range
        assert isinstance(output["confidence"], int), "Confidence must be int"
        assert (
            CONFIDENCE_RANGE[0] <= output["confidence"] <= CONFIDENCE_RANGE[1]
        ), f"Confidence out of range: {output['confidence']}"
        # sources must be list with required fields
        assert isinstance(output["sources"], list)
        for src in output["sources"]:
            assert "title" in src and "url" in src

        if expect_metadata:
            assert "metadata" in resp

    @staticmethod
    def assert_rate_limit_headers(response: httpx.Response):
        """Check that rate limit headers are present."""
        for key in (
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ):
            assert key in response.headers, f"Missing header {key}"

    @staticmethod
    def assert_health_check_response(data: Dict[str, Any]):
        """Validate a health check endpoint response."""
        for k in ("status", "service", "version", "environment", "timestamp"):
            assert k in data, f"Health check missing '{k}'"
        assert data["status"] in ("healthy", "degraded", "unhealthy")


def mock_fact_check(
    mock_llm,
    mock_search,
    sources=1,
    verdict="TRUE",
    confidence=85,
    reasoning="This claim is true.",
):
    # Create a HybridSearchResult object for the mock
    from app.domain.models import HybridSearchResult
    mock_sources = TestData.create_mock_sources(sources)
    mock_search.return_value = HybridSearchResult(
        vector_results=[],
        web_results=mock_sources,
        combined_sources=mock_sources,
        used_vector_cache=False
    )
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = f"""
        Verdict: {verdict}
        Confidence: {confidence}
        Reasoning: {reasoning}
    """
    mock_llm.return_value = mock_response


def post_factcheck(client, claim, status=200):
    headers = {settings.api_key_header: settings.api_key}
    payload = {"claim": claim}
    response = client.post("/api/v1/factcheck", json=payload, headers=headers)
    assert response.status_code == status
    return response
