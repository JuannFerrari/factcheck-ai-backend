"""
Pytest configuration and shared fixtures for FactCheck AI Backend tests.
"""
import pytest
import pytest_asyncio
import asyncio
import httpx
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock
from app.domain.models import Source

# Test configuration
TEST_BASE_URL = "http://localhost:8000"
TEST_CLAIMS = [
    "The Earth is flat.",
    "The Eiffel Tower is in Berlin.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Great Wall of China is visible from space.",
    "Humans use only 10% of their brain."
]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_fact_check_request() -> Dict[str, Any]:
    """Provide a sample fact-check request payload."""
    return {
        "input": {
            "claim": "The Earth is flat."
        }
    }


@pytest.fixture
def sample_fact_check_response() -> Dict[str, Any]:
    """Provide a sample fact-check response structure."""
    return {
        "output": {
            "verdict": "False",
            "confidence": 95,
            "reasoning": "Scientific evidence overwhelmingly supports that Earth is spherical.",
            "sources": [
                {
                    "title": "Earth - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Earth",
                    "snippet": "Earth is the third planet from the Sun..."
                }
            ],
            "claim": "The Earth is flat."
        },
        "metadata": {
            "run_id": "test-run-id",
            "feedback_tokens": []
        }
    }


@pytest.fixture
def mock_huggingface_response():
    """Mock Hugging Face API response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """
    Verdict: FALSE
    Confidence: 95
    Reasoning: Based on scientific evidence, the Earth is not flat.
    """
    return mock_response


@pytest.fixture
def mock_serper_response():
    """Mock Serper.dev API response."""
    return {
        "organic": [
            {
                "title": "Earth - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Earth",
                "snippet": "Earth is the third planet from the Sun and the only astronomical object known to harbor life."
            },
            {
                "title": "Flat Earth - Wikipedia",
                "link": "https://en.wikipedia.org/wiki/Flat_Earth",
                "snippet": "The flat Earth model is an archaic conception of Earth's shape as a plane or disk."
            }
        ]
    }


@pytest.fixture
def mock_source_dict():
    """Create a mock source as a dictionary that can be serialized."""
    return {
        "title": "Test Source",
        "url": "https://example.com",
        "snippet": "Test snippet"
    }


@pytest.fixture
def mock_sources_list():
    """Create a list of mock sources as dictionaries."""
    return [
        {
            "title": "Test Source 1",
            "url": "https://example1.com",
            "snippet": "Test snippet 1"
        },
        {
            "title": "Test Source 2",
            "url": "https://example2.com",
            "snippet": "Test snippet 2"
        }
    ]


class TestData:
    """Test data constants and utilities."""

    VALID_CLAIMS = TEST_CLAIMS
    INVALID_CLAIMS = ["", "   ", "a" * 1001]  # Empty, whitespace, too long

    EXPECTED_VERDICTS = ["True", "False", "Unclear"]
    CONFIDENCE_RANGE = (0, 100)

    @staticmethod
    def get_test_claim(index: int = 0) -> str:
        """Get a test claim by index."""
        return TEST_CLAIMS[index % len(TEST_CLAIMS)]

    @staticmethod
    def create_fact_check_payload(claim: str) -> Dict[str, Any]:
        """Create a fact-check request payload."""
        return {"input": {"claim": claim}}

    @staticmethod
    def create_mock_source(title: str = "Test Source", url: str = "https://example.com", snippet: str = "Test snippet") -> Source:
        """Create a mock Source object."""
        return Source(
            title=title,
            url=url,
            snippet=snippet
        )

    @staticmethod
    def create_mock_sources(count: int = 1) -> list[Source]:
        """Create a list of mock Source objects."""
        return [
            TestData.create_mock_source(
                f"Test Source {i}", f"https://example{i}.com", f"Test snippet {i}")
            for i in range(1, count + 1)
        ]


class AssertionHelpers:
    """Helper methods for common test assertions."""

    @staticmethod
    def assert_valid_fact_check_response(response_data: Dict[str, Any]) -> None:
        """Assert that a fact-check response has valid structure and data."""
        assert "output" in response_data, "Response missing 'output' key"

        output = response_data["output"]
        assert "verdict" in output, "Output missing 'verdict' key"
        assert "confidence" in output, "Output missing 'confidence' key"
        assert "reasoning" in output, "Output missing 'reasoning' key"
        assert "sources" in output, "Output missing 'sources' key"
        assert "claim" in output, "Output missing 'claim' key"

        # Validate verdict
        assert output[
            "verdict"] in TestData.EXPECTED_VERDICTS, f"Invalid verdict: {output['verdict']}"

        # Validate confidence
        assert isinstance(output["confidence"],
                          int), "Confidence must be an integer"
        assert TestData.CONFIDENCE_RANGE[0] <= output["confidence"] <= TestData.CONFIDENCE_RANGE[1], \
            f"Confidence out of range: {output['confidence']}"

        # Validate sources
        assert isinstance(output["sources"], list), "Sources must be a list"
        for source in output["sources"]:
            assert "title" in source, "Source missing 'title'"
            assert "url" in source, "Source missing 'url'"

    @staticmethod
    def assert_rate_limit_headers(response: httpx.Response) -> None:
        """Assert that rate limit headers are present."""
        assert "X-RateLimit-Limit" in response.headers, "Missing X-RateLimit-Limit header"
        assert "X-RateLimit-Remaining" in response.headers, "Missing X-RateLimit-Remaining header"
        assert "X-RateLimit-Reset" in response.headers, "Missing X-RateLimit-Reset header"

    @staticmethod
    def assert_health_check_response(response_data: Dict[str, Any]) -> None:
        """Assert that a health check response has valid structure."""
        assert "status" in response_data, "Health check missing 'status' key"
        assert "service" in response_data, "Health check missing 'service' key"
        assert "version" in response_data, "Health check missing 'version' key"
        assert "environment" in response_data, "Health check missing 'environment' key"
        assert "timestamp" in response_data, "Health check missing 'timestamp' key"

        # Validate status
        valid_statuses = ["healthy", "degraded", "unhealthy"]
        assert response_data[
            "status"] in valid_statuses, f"Invalid status: {response_data['status']}"
