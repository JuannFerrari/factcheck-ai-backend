import pytest
from unittest.mock import MagicMock, patch
from app.services.content_moderation import (
    ContentModerationService,
    ModerationDecision,
    ModerationResult,
    content_moderation_service,
)


# --- Helpers ---
def mock_ai_response(mock_llm, response_text: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock_llm.return_value = mock_response


def assert_moderation(
    result, expected_decision, expected_confidence=None, contains=None
):
    assert result.decision == expected_decision
    if expected_confidence is not None:
        assert result.confidence == expected_confidence
    if contains:
        assert contains.lower() in result.reason.lower()


# --- Tests ---
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ai_text, expected_decision, confidence, reason_part",
    [
        ("true", ModerationDecision.ALLOW, 0.95, "approved"),
        ("false", ModerationDecision.REJECT, 0.95, "rejected"),
        ("maybe", ModerationDecision.ALLOW, 0.5, "unexpected"),
        ("", ModerationDecision.ALLOW, 0.5, "no ai response"),
    ],
)
@patch("app.services.content_moderation.client.chat.completions.create")
async def test_ai_responses(
    mock_llm, ai_text, expected_decision, confidence, reason_part
):
    service = ContentModerationService()
    mock_ai_response(mock_llm, ai_text)
    result = await service.evaluate_claim("Some test claim")
    assert_moderation(result, expected_decision, confidence, reason_part)


@pytest.mark.asyncio
@patch(
    "app.services.content_moderation.client.chat.completions.create",
    side_effect=Exception("AI fail"),
)
async def test_ai_exception_defaults_to_allow(mock_llm):
    service = ContentModerationService()
    result = await service.evaluate_claim("Some test claim")
    assert_moderation(result, ModerationDecision.ALLOW, 0.3, "failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("claim", ["", "   "])
async def test_empty_or_whitespace_claims_are_rejected(claim):
    service = ContentModerationService()
    result = await service.evaluate_claim(claim)
    assert_moderation(result, ModerationDecision.REJECT, 0.95, "empty")


@pytest.mark.parametrize(
    "decision,expected",
    [
        (ModerationDecision.ALLOW, True),
        (ModerationDecision.REJECT, False),
    ],
)
def test_is_appropriate_logic(decision, expected):
    service = ContentModerationService()
    result = ModerationResult(decision=decision, confidence=0.95, reason="Test")
    assert service.is_appropriate(result) is expected


@pytest.mark.asyncio
@patch("app.services.content_moderation.client.chat.completions.create")
async def test_global_instance_allows_safe_claim(mock_llm):
    mock_ai_response(mock_llm, "true")
    result = await content_moderation_service.evaluate_claim("Safe claim")
    assert_moderation(result, ModerationDecision.ALLOW, 0.95)
    assert content_moderation_service.is_appropriate(result)


@pytest.mark.asyncio
@patch("app.services.content_moderation.client.chat.completions.create")
async def test_multiple_safe_claims_in_loop(mock_llm):
    """Safe claims should always allow."""
    mock_ai_response(mock_llm, "true")
    service = ContentModerationService()
    safe_claims = [
        "Vaccines cause autism",
        "The Earth is flat",
        "Climate change is a hoax",
        "5G causes COVID-19",
    ]
    for claim in safe_claims:
        result = await service.evaluate_claim(claim)
        assert result.decision == ModerationDecision.ALLOW


@pytest.mark.asyncio
@patch("app.services.content_moderation.client.chat.completions.create")
async def test_multiple_inappropriate_claims_in_loop(mock_llm):
    """Dangerous claims should reject."""
    mock_ai_response(mock_llm, "false")
    service = ContentModerationService()
    bad_claims = [
        "How to hack into someone's computer",
        "How to make a bomb",
        "How to commit tax fraud",
    ]
    for claim in bad_claims:
        result = await service.evaluate_claim(claim)
        assert result.decision == ModerationDecision.REJECT
