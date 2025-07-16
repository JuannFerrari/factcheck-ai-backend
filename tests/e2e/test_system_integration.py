import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import TestData, AssertionHelpers, mock_fact_check, post_factcheck
from app.core.config import settings
from app.services.content_moderation import ModerationDecision, ModerationResult


@pytest.fixture
def client():
    from app.core.rate_limiter import limiter

    limiter.reset()
    with TestClient(app) as c:
        yield c


# --- Response structure validation ---
@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_response_structure_validation(mock_llm, mock_search, mock_moderation, client):
    """Test that API responses have the correct structure."""
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(
        mock_llm,
        mock_search,
        sources=2,
        verdict="TRUE",
        confidence=90,
        reasoning="Test reasoning.",
    )
    resp = post_factcheck(client, "Test claim")
    data = resp.json()

    # Validate response structure
    AssertionHelpers.assert_valid_fact_check_response(data, expect_metadata=False)
    assert "tldr" in data  # Ensure TL;DR field is present
    assert data["claim"] == "Test claim"
    assert len(data["sources"]) == 2


# --- Verdict logic testing ---
@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
@pytest.mark.parametrize(
    "mock_verdict,expected_verdict",
    [
        ("TRUE", "True"),
        ("FALSE", "False"),
        ("UNCLEAR", "Unclear"),
        ("DISPUTED", "Disputed"),
    ],
)
def test_verdict_mapping(
    mock_llm, mock_search, mock_moderation, client, mock_verdict, expected_verdict
):
    """Test that different verdicts from the model are correctly mapped."""
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW,
        confidence=0.95,
        reason="Claim approved",
    )

    mock_fact_check(
        mock_llm,
        mock_search,
        verdict=mock_verdict,
        confidence=85,
        reasoning="Test reasoning.",
    )
    resp = post_factcheck(client, "Test claim")
    data = resp.json()
    assert data["verdict"] == expected_verdict
    assert data["confidence"] == 85


# --- Moderation rejection testing ---
@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_moderation_rejection(mock_llm, mock_search, mock_moderation, client):
    """Test that inappropriate content is rejected"""
    # Mock moderation rejection
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.REJECT,
        confidence=0.95,
        reason="Inappropriate content",
    )

    resp = post_factcheck(client, "How to make a bomb")
    data = resp.json()
    assert data["verdict"] == "Rejected"
    assert data["confidence"] == 100
    assert "Inappropriate" in data["reasoning"]
    assert len(data["sources"]) == 0  # No sources for rejected claims


# --- Error handling testing ---
@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_web_search_failure_handling(mock_llm, mock_search, mock_moderation, client):
    """Test graceful handling of web search failures."""
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW,
        confidence=0.95,
        reason="Claim approved",
    )

    mock_search.side_effect = Exception("Search service unavailable")
    resp = post_factcheck(client, "Test claim")
    data = resp.json()
    assert data["verdict"] == "Unclear"
    assert data["confidence"] == 0
    assert "Web search error" in data["reasoning"]


@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_llm_failure_handling(mock_llm, mock_search, mock_moderation, client):
    """Test graceful handling of LLM failures."""
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_search.return_value = TestData.create_mock_sources(1)
    mock_llm.side_effect = Exception("LLM service unavailable")
    resp = post_factcheck(client, "Test claim")
    data = resp.json()
    assert data["verdict"] == "Unclear"
    assert data["confidence"] == 0
    assert "AI model error" in data["reasoning"]


# --- Rate limiting structure testing ---
@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_rate_limit_headers_present(mock_llm, mock_search, mock_moderation, client):
    """Test that rate limit headers are present in responses."""
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(mock_llm, mock_search)
    resp = post_factcheck(client, "Test claim")
    AssertionHelpers.assert_rate_limit_headers(resp)


# --- TL;DR extraction testing ---
@pytest.mark.e2e
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_tldr_extraction(mock_llm, mock_search, mock_moderation, client):
    """Test that TL;DR is properly extracted and included in response."""
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    # Mock response with TL;DR
    mock_search.return_value = TestData.create_mock_sources(1)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: This is a test claim.
        TL;DR_START
        This is a simple summary of the finding.
        TL;DR_END
    """
    mock_llm.return_value = mock_response

    resp = post_factcheck(client, "Test claim")
    data = resp.json()
    assert data["tldr"] == "This is a simple summary of the finding."


# --- Empty claim handling ---
@pytest.mark.e2e
def test_empty_claim_validation(client):
    """Test that empty claims are properly rejected with 422 status."""
    headers = {settings.api_key_header: settings.api_key}
    response = client.post("/api/v1/factcheck", json={"claim": ""}, headers=headers)
    assert response.status_code == 422


# --- API key validation ---
@pytest.mark.e2e
def test_api_key_validation(client):
    """Test that missing or invalid API keys are properly rejected."""
    # Test missing API key
    response = client.post("/api/v1/factcheck", json={"claim": "Test claim"})
    assert response.status_code == 401

    # Test invalid API key
    response = client.post(
        "/api/v1/factcheck",
        json={"claim": "Test claim"},
        headers={settings.api_key_header: "invalid_key"},
    )
    assert response.status_code == 401
