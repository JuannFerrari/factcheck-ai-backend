"""
End-to-end system integration tests using FastAPI TestClient with mocks.
"""
import pytest
import pytest_asyncio
import asyncio
import time
import psutil
import os
import threading
import queue
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from tests.conftest import TestData, AssertionHelpers


@pytest.fixture
def client():
    """Create a test client with lifespan events enabled."""
    from app.core.rate_limiter import limiter
    # Reset the rate limiter before each test to avoid interference
    limiter.reset()
    with TestClient(app) as client:
        yield client


class TestSystemIntegration:
    """End-to-end system integration tests."""

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_full_fact_check_workflow(self, mock_llm, mock_search, client: TestClient):
        """Test the complete fact-check workflow from request to response."""
        # Mock web search with multiple Source objects
        mock_search.return_value = TestData.create_mock_sources(3)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: FALSE
        Confidence: 95
        Reasoning: This claim is false based on multiple sources.
        """
        mock_llm.return_value = mock_response

        # Test multiple claims
        test_claims = [
            "The Earth is flat.",
            "Water boils at 100 degrees Celsius.",
            "The Great Wall of China is visible from space."
        ]

        for claim in test_claims:
            # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
            payload = {"claim": claim}
            headers = {settings.api_key_header: settings.api_key}
            response = client.post("/api/v1/factcheck", json=payload, headers=headers)

            # Verify response
            assert response.status_code == 200
            data = response.json()
            # The response structure is different for the direct endpoint vs LangServe
            assert "verdict" in data, "Response missing 'verdict' key"
            assert "confidence" in data, "Response missing 'confidence' key"
            assert "reasoning" in data, "Response missing 'reasoning' key"
            assert "sources" in data, "Response missing 'sources' key"
            assert "claim" in data, "Response missing 'claim' key"

            # Verify specific values
            assert data["claim"] == claim
            assert data["verdict"] == "False"
            assert data["confidence"] == 95
            assert len(data["sources"]) == 3
            time.sleep(0.6)

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_rate_limiting_behavior(self, mock_llm, mock_search, client: TestClient):
        """Test rate limiting behavior under load."""
        # Mock web search with proper Source objects
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # Make multiple requests quickly
        responses = []
        for i in range(10):
            response = client.post("/api/v1/factcheck", json=payload, headers=headers)
            responses.append(response)
            time.sleep(0.6)

        # Check that we have some successful responses
        successful_responses = [r for r in responses if r.status_code == 200]
        assert len(
            successful_responses) > 0, "Should have some successful responses"

        # Verify rate limit headers are present
        for response in successful_responses:
            AssertionHelpers.assert_rate_limit_headers(response)

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_concurrent_requests(self, mock_llm, mock_search, client: TestClient):
        """Test handling of concurrent requests."""
        # Mock web search with proper Source objects
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # Make concurrent requests
        results = queue.Queue()
        errors = queue.Queue()

        def make_request():
            try:
                response = client.post("/api/v1/factcheck", json=payload, headers=headers)
                results.put(response)
            except Exception as e:
                errors.put(e)

        # Start multiple threads with delays to respect rate limiting
        threads = []
        for i in range(3):  # Reduced from 5 to 3 to work better with rate limiting
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
            time.sleep(0.3)  # Small delay between thread starts

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert errors.empty(
        ), f"Errors occurred: {[errors.get() for _ in range(errors.qsize())]}"

        # Verify responses - some may be rate limited (429), others successful (200)
        successful_responses = 0
        rate_limited_responses = 0
        
        while not results.empty():
            response = results.get()
            if response.status_code == 200:
                successful_responses += 1
                data = response.json()
                # The response structure is different for the direct endpoint vs LangServe
                assert "verdict" in data, "Response missing 'verdict' key"
                assert "confidence" in data, "Response missing 'confidence' key"
                assert "reasoning" in data, "Response missing 'reasoning' key"
                assert "sources" in data, "Response missing 'sources' key"
                assert "claim" in data, "Response missing 'claim' key"
            elif response.status_code == 429:
                rate_limited_responses += 1
            else:
                assert False, f"Unexpected status code: {response.status_code}"
        
        # Verify we got at least some successful responses and some rate limited
        assert successful_responses > 0, "No successful responses received"
        assert rate_limited_responses > 0, "No rate limiting occurred - test may not be working correctly"
        assert successful_responses + rate_limited_responses == 3, f"Expected 3 total responses, got {successful_responses + rate_limited_responses}"

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_real_world_scenarios(self, mock_llm, mock_search, client: TestClient):
        """Test with realistic real-world scenarios."""
        # Mock web search with different sources for different claims
        def mock_search_side_effect(claim):
            if "Earth" in claim:
                return TestData.create_mock_sources(2)
            elif "water" in claim.lower():
                return TestData.create_mock_sources(1)
            else:
                return TestData.create_mock_sources(3)

        mock_search.side_effect = mock_search_side_effect

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response

        # Test various real-world claims
        real_world_claims = [
            "The Earth is the third planet from the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
            "The Great Wall of China is the longest wall in the world.",
            "Humans have 206 bones in their adult body.",
            "The speed of light is approximately 299,792 kilometers per second."
        ]

        for claim in real_world_claims:
            # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
            payload = {"claim": claim}
            headers = {settings.api_key_header: settings.api_key}
            response = client.post("/api/v1/factcheck", json=payload, headers=headers)

            assert response.status_code == 200
            data = response.json()
            # The response structure is different for the direct endpoint vs LangServe
            assert "verdict" in data, "Response missing 'verdict' key"
            assert "confidence" in data, "Response missing 'confidence' key"
            assert "reasoning" in data, "Response missing 'reasoning' key"
            assert "sources" in data, "Response missing 'sources' key"
            assert "claim" in data, "Response missing 'claim' key"
            assert data["claim"] == claim
            assert data["verdict"] in ["True", "False", "Unclear"]
            assert 0 <= data["confidence"] <= 100
            time.sleep(0.6)


class TestPerformanceCharacteristics:
    """Test performance characteristics."""

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_response_time_consistency(self, mock_llm, mock_search, client: TestClient):
        """Test that response times are consistent."""
        # Mock web search with proper Source objects
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # Make multiple requests and measure response times
        response_times = []
        for i in range(5):
            start_time = time.time()
            response = client.post("/api/v1/factcheck", json=payload, headers=headers)
            end_time = time.time()

            assert response.status_code == 200
            response_times.append(end_time - start_time)
            time.sleep(0.6)

        # Check that response times are reasonable (less than 5 seconds each)
        for response_time in response_times:
            assert response_time < 5.0, f"Response time {response_time} is too slow"

        # Check that response times are consistent (within 2 seconds of each other)
        max_time = max(response_times)
        min_time = min(response_times)
        assert max_time - min_time < 2.0, "Response times are too inconsistent"

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_memory_usage_stability(self, mock_llm, mock_search, client: TestClient):
        """Test that memory usage remains stable under load."""
        # Mock web search with proper Source objects
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make multiple requests
        for i in range(10):
            response = client.post("/api/v1/factcheck", json=payload, headers=headers)
            assert response.status_code == 200
            time.sleep(0.6)

        # Get final memory usage
        final_memory = process.memory_info().rss

        # Check that memory usage hasn't increased dramatically (less than 50MB increase)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 50 * 1024 * \
            1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.2f}MB"




class TestErrorRecovery:
    """Test error recovery scenarios."""

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_recovery_after_web_search_failure(self, mock_llm, mock_search, client: TestClient):
        """Test recovery after web search failure."""
        # Mock web search to fail first, then succeed
        mock_search.side_effect = [
            Exception("Web search failed"),  # First call fails
            TestData.create_mock_sources(1)  # Second call succeeds
        ]

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # First request should fail gracefully
        response1 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["verdict"] == "Unclear"
        assert data1["confidence"] == 0

        # Second request should succeed
        response2 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["verdict"] == "True"
        assert data2["confidence"] == 85

    @pytest.mark.e2e
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_recovery_after_llm_failure(self, mock_llm, mock_search, client: TestClient):
        """Test recovery after LLM failure."""
        # Mock web search
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM to fail first, then succeed
        mock_llm.side_effect = [
            Exception("LLM service unavailable"),  # First call fails
            MagicMock(choices=[MagicMock(message=MagicMock(content="""
                Verdict: TRUE
                Confidence: 85
                Reasoning: This claim is true.
            """))])  # Second call succeeds
        ]

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # First request should fail gracefully
        response1 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["verdict"] == "Unclear"
        assert "Error calling AI model" in data1["reasoning"]

        # Second request should succeed
        response2 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["verdict"] == "True"
        assert data2["confidence"] == 85
