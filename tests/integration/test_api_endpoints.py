import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app
from tests.conftest import AssertionHelpers, mock_fact_check, post_factcheck
from app.core.config import settings
from app.services.content_moderation import ModerationDecision, ModerationResult


@pytest.fixture
def client():
    from app.core.rate_limiter import limiter

    limiter.reset()
    return TestClient(app)


# --- Root health check ---
def test_root_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    for key in ("message", "version", "environment", "timestamp"):
        assert key in data


# --- API key handling ---
@pytest.mark.parametrize(
    "headers,expected_error",
    [
        ({}, "missing"),
        ({settings.api_key_header: "wrong"}, "invalid"),
    ],
)
def test_fact_check_api_key_errors(client, headers, expected_error):
    response = client.post("/api/v1/factcheck", json={"claim": "test"}, headers=headers)
    assert response.status_code == 401
    assert expected_error in response.json()["error"].lower()


# --- Valid claim flow ---
@pytest.mark.integration
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_fact_check_successful(mock_llm, mock_search, mock_moderation, client):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(
        mock_llm,
        mock_search,
        sources=2,
        verdict="FALSE",
        confidence=95,
        reasoning="This claim is false.",
    )
    response = post_factcheck(client, "The Earth is flat.")
    data = response.json()

    # Validate response structure
    AssertionHelpers.assert_valid_fact_check_response(data, expect_metadata=False)
    assert data["verdict"] == "False"
    assert data["confidence"] == 95
    assert len(data["sources"]) == 2
    assert "tldr" in data  # Ensure TL;DR field is present


# --- Multiple claims with different verdicts ---
@pytest.mark.integration
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
@pytest.mark.parametrize(
    "claim,expected_verdict",
    [
        ("The Earth is round.", "True"),
        ("Water boils at 100°C.", "True"),
    ],
)
def test_fact_check_multiple_verdicts(
    mock_llm, mock_search, mock_moderation, client, claim, expected_verdict
):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(
        mock_llm,
        mock_search,
        sources=1,
        verdict="TRUE",
        confidence=85,
        reasoning="This claim is true.",
    )
    response = post_factcheck(client, claim)
    data = response.json()
    assert data["claim"] == claim
    assert data["verdict"] == expected_verdict
    assert data["confidence"] == 85


# --- Invalid claims (empty, whitespace, too long) ---
@pytest.mark.parametrize("claim", ["", "   ", "A" * 1001])
def test_invalid_claims_return_422(client, claim):
    headers = {settings.api_key_header: settings.api_key}
    response = client.post("/api/v1/factcheck", json={"claim": claim}, headers=headers)
    assert response.status_code == 422


# --- Rate limiting headers ---
@pytest.mark.integration
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_rate_limit_headers_present(mock_llm, mock_search, mock_moderation, client):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(mock_llm, mock_search)
    response = post_factcheck(client, "Test claim")
    AssertionHelpers.assert_rate_limit_headers(response)


# --- Special characters and long claims ---
@pytest.mark.integration
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_special_characters_and_long_claims(
    mock_llm, mock_search, mock_moderation, client
):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(mock_llm, mock_search)
    special_claims = [
        "A" * 1000,  # max allowed length
        "The Earth is round! 🌍 And water boils at 100°C.",
    ]
    for claim in special_claims:
        response = post_factcheck(client, claim)
        data = response.json()
        AssertionHelpers.assert_valid_fact_check_response(data, expect_metadata=False)
        assert data["claim"] == claim


# --- CORS headers ---
@pytest.mark.integration
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_cors_headers(mock_llm, mock_search, mock_moderation, client):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(mock_llm, mock_search)
    origin = "http://localhost:3000"

    options_headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    options_response = client.options("/api/v1/factcheck", headers=options_headers)
    assert options_response.status_code == 200
    assert "access-control-allow-origin" in options_response.headers
    assert "access-control-allow-methods" in options_response.headers
    assert "access-control-allow-headers" in options_response.headers

    post_headers = {
        settings.api_key_header: settings.api_key,
        "Origin": origin,
    }
    post_response = client.post(
        "/api/v1/factcheck", json={"claim": "Test claim"}, headers=post_headers
    )
    assert post_response.status_code == 200
    assert "access-control-allow-origin" in post_response.headers


# --- TL;DR field validation ---
@pytest.mark.integration
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
def test_tldr_field_present(mock_llm, mock_search, mock_moderation, client):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_fact_check(
        mock_llm,
        mock_search,
        sources=1,
        verdict="TRUE",
        confidence=90,
        reasoning="Test reasoning.",
    )
    response = post_factcheck(client, "Test claim")
    data = response.json()
    assert "tldr" in data
    assert isinstance(data["tldr"], str)
