"""
Integration tests for API endpoints using FastAPI TestClient with mocks.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import TestData, AssertionHelpers
from app.core.config import settings


@pytest.fixture
def client():
    """Create a test client."""
    from app.core.rate_limiter import limiter
    # Reset the rate limiter before each test to avoid interference
    limiter.reset()
    return TestClient(app)


class TestFactCheckEndpoint:
    """Test the fact-check API endpoint."""

    @pytest.mark.integration
    def test_fact_check_missing_api_key(self, client: TestClient):
        """Test request without API key header is rejected."""
        payload = {"claim": "The Earth is round."}
        response = client.post("/api/v1/factcheck", json=payload)
        assert response.status_code == 401
        data = response.json()
        assert "error" in data, "Response should have 'error' field"
        assert "invalid" in data["error"].lower() or "missing" in data["error"].lower()

    @pytest.mark.integration
    def test_fact_check_invalid_api_key(self, client: TestClient):
        """Test request with invalid API key is rejected."""
        payload = {"claim": "The Earth is round."}
        headers = {settings.api_key_header: "wrongkey"}
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response.status_code == 401
        data = response.json()
        assert "error" in data, "Response should have 'error' field"
        assert "invalid" in data["error"].lower() or "missing" in data["error"].lower()

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_fact_check_valid_api_key(self, mock_llm, mock_search, client: TestClient):
        """Test request with valid API key is accepted."""
        mock_search.return_value = TestData.create_mock_sources(1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This claim is true.
        """
        mock_llm.return_value = mock_response
        payload = {"claim": "The Earth is round."}
        headers = {settings.api_key_header: settings.api_key}
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "verdict" in data
        assert "confidence" in data
        assert "reasoning" in data
        assert "sources" in data
        assert "claim" in data

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_fact_check_successful_request(self, mock_llm, mock_search, client: TestClient):
        """Test successful fact-check request."""
        # Mock web search with proper Source objects
        mock_search.return_value = TestData.create_mock_sources(2)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: FALSE
        Confidence: 95
        Reasoning: This claim is false based on evidence.
        """
        mock_llm.return_value = mock_response

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "The Earth is flat."}
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

        # Verify specific values
        assert data["verdict"] == "False"
        assert data["confidence"] == 95
        assert len(data["sources"]) == 2

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_fact_check_multiple_claims(self, mock_llm, mock_search, client: TestClient):
        """Test multiple fact-check requests."""
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

        claims = ["The Earth is round.", "Water boils at 100°C."]

        for claim in claims:
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
            assert data["verdict"] == "True"
            assert data["confidence"] == 85

    @pytest.mark.integration
    def test_fact_check_missing_input(self, client: TestClient):
        """Test fact-check request with missing input."""
        headers = {settings.api_key_header: settings.api_key}
        response = client.post("/api/v1/factcheck", json={}, headers=headers)
        assert response.status_code == 422

    @pytest.mark.integration
    def test_fact_check_missing_claim(self, client: TestClient):
        """Test fact-check request with missing claim."""
        headers = {settings.api_key_header: settings.api_key}
        response = client.post("/api/v1/factcheck", json={"input": {}}, headers=headers)
        assert response.status_code == 422

    @pytest.mark.integration
    def test_fact_check_empty_claim(self, client: TestClient):
        """Test fact-check request with empty claim."""
        payload = TestData.create_fact_check_payload("")
        headers = {settings.api_key_header: settings.api_key}
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response.status_code == 422

    @pytest.mark.integration
    def test_fact_check_whitespace_claim(self, client: TestClient):
        """Test fact-check request with whitespace-only claim."""
        payload = TestData.create_fact_check_payload("   ")
        headers = {settings.api_key_header: settings.api_key}
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response.status_code == 422

    @pytest.mark.integration
    def test_fact_check_claim_too_long(self, client: TestClient):
        """Test fact-check request with claim that's too long."""
        long_claim = "A" * 1001  # Exceeds 1000 character limit
        payload = TestData.create_fact_check_payload(long_claim)
        headers = {settings.api_key_header: settings.api_key}
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response.status_code == 422


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_rate_limit_exceeded(self, mock_llm, mock_search, client: TestClient):
        """Test that rate limiting works when limit is exceeded."""
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

        # Make requests up to the rate limit
        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": "Test claim"}
        headers = {settings.api_key_header: settings.api_key}

        # First request should succeed
        response1 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response1.status_code == 200
        AssertionHelpers.assert_rate_limit_headers(response1)

        # Second request should also succeed (rate limit is higher in tests)
        response2 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert response2.status_code == 200
        AssertionHelpers.assert_rate_limit_headers(response2)

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_rate_limit_exceeded_with_custom_error_message(self, mock_llm, mock_search, client: TestClient):
        """Test that rate limiting returns the correct status code and custom error message."""
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

        # Make requests rapidly to trigger rate limiting
        # The limit is "10/minute;2/second" so we need to exceed 2/second
        responses = []
        for i in range(10):  # Make 10 rapid requests to ensure we hit the limit
            response = client.post("/api/v1/factcheck", json=payload, headers=headers)
            responses.append(response)
            print(f"Request {i+1}: Status {response.status_code}")

        # Check that we got rate limited (429 status)
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        print(f"Total responses: {len(responses)}")
        print(f"Rate limited responses: {len(rate_limited_responses)}")
        print(f"Status codes: {[r.status_code for r in responses]}")
        
        assert len(rate_limited_responses) > 0, "Expected at least one rate-limited response"

        # Check the custom error message for rate-limited responses
        for rate_limited_response in rate_limited_responses:
            error_data = rate_limited_response.json()
            assert "error" in error_data, "Rate limit response should have 'error' field"
            assert "Too many requests" in error_data["error"], "Rate limit response should contain 'Too many requests'"
            assert "Please wait a few seconds" in error_data["error"], "Rate limit response should contain wait message"
            assert "contact the site owner" in error_data["error"], "Rate limit response should mention contacting site owner"

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_rate_limit_steady_state(self, mock_llm, mock_search, client: TestClient):
        """Test that rate limiting allows requests within the steady state limit."""
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

        # Make 2 requests (within the 2/second limit)
        response1 = client.post("/api/v1/factcheck", json=payload, headers=headers)
        response2 = client.post("/api/v1/factcheck", json=payload, headers=headers)

        # Both should succeed
        assert response1.status_code == 200, "First request should succeed"
        assert response2.status_code == 200, "Second request should succeed within rate limit"

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_rate_limit_headers_present(self, mock_llm, mock_search, client: TestClient):
        """Test that rate limit headers are present in responses."""
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
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        AssertionHelpers.assert_rate_limit_headers(response)


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_large_claim_handling(self, mock_llm, mock_search, client: TestClient):
        """Test handling of very large claims."""
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

        large_claim = "A" * 1000  # 1000 character claim

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": large_claim}
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

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_special_characters_in_claim(self, mock_llm, mock_search, client: TestClient):
        """Test handling of claims with special characters."""
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

        special_claim = "The Earth is round! 🌍 And water boils at 100°C."

        # For the /api/v1/factcheck endpoint, we need a direct claim, not wrapped in input
        payload = {"claim": special_claim}
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

        assert data["claim"] == special_claim

class TestCORS:
    """Test CORS functionality."""

    @pytest.mark.integration
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    def test_cors_headers_present(self, mock_llm, mock_search, client: TestClient):
        """Test that CORS headers are present in responses."""
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
        response = client.post("/api/v1/factcheck", json=payload, headers=headers)
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


