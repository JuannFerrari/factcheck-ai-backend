import pytest
from unittest.mock import MagicMock, patch
from app.services.fact_checking import fact_check_chain_logic
from tests.conftest import TestData, AssertionHelpers
from app.services.content_moderation import ModerationDecision, ModerationResult


def make_llm_response(
    verdict="True", confidence=90, reasoning="Reasoning text", tldr=None
):
    tldr_text = f"\n\nTL;DR_START\n{tldr}\nTL;DR_END" if tldr else ""
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = (
        f"Verdict: {verdict}\nConfidence: {confidence}\n"
        f"Reasoning: {reasoning}.{tldr_text}"
    )
    return mock


def assert_fact_check_result(result, verdict=None, confidence=None, sources=None):
    # For unit tests, result is already flat (no output wrapper)
    AssertionHelpers.assert_valid_fact_check_response(result, expect_metadata=False)
    if verdict:
        assert result["verdict"] == verdict
    if confidence is not None:
        assert result["confidence"] == confidence
    if sources is not None:
        assert len(result["sources"]) == sources


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict,confidence",
    [("True", 90), ("False", 95), ("Unclear", 45), ("Disputed", 85)],
)
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
async def test_various_verdicts(
    mock_llm, mock_search, mock_moderation, verdict, confidence
):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_search.return_value = TestData.create_mock_sources(2)
    mock_llm.return_value = make_llm_response(
        verdict, confidence, f"This is {verdict.lower()}."
    )

    result = await fact_check_chain_logic({"claim": "Some claim"})
    assert_fact_check_result(result, verdict=verdict, confidence=confidence, sources=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tldr_input, expected_tldr",
    [
        ("TL;DR_START\nThis is short.\nTL;DR_END", "This is short."),
        ("TL;DR_START\nlowercase works too.\nTL;DR_END", "lowercase works too."),
        (
            "TL;DR_START\nMulti-line\nstill works.\nTL;DR_END",
            "Multi-line\nstill works.",
        ),
        (None, ""),
    ],
)
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
async def test_tldr_variations(
    mock_llm, mock_search, mock_moderation, tldr_input, expected_tldr
):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_search.return_value = TestData.create_mock_sources(1)
    reasoning = "Reasoning: Example reasoning."
    content = f"Verdict: True\nConfidence: 80\n{reasoning}\n\n{tldr_input or ''}"
    mock = MagicMock()
    mock.choices = [MagicMock()]
    mock.choices[0].message.content = content
    mock_llm.return_value = mock

    result = await fact_check_chain_logic({"claim": "Test claim"})
    assert_fact_check_result(result, verdict="True", confidence=80)
    assert result["tldr"] == expected_tldr


@pytest.mark.asyncio
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
async def test_no_sources(mock_llm, mock_search, mock_moderation):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    mock_search.return_value = []  # no sources
    result = await fact_check_chain_logic({"claim": "Test claim"})
    assert_fact_check_result(result, verdict="Unclear", confidence=0, sources=0)
    assert "No relevant sources found" in result["reasoning"]


@pytest.mark.asyncio
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch(
    "app.services.fact_checking.hybrid_search_service.search_claim",
    side_effect=Exception("Search fail"),
)
async def test_web_search_error(mock_search, mock_moderation):
    # Mock moderation to allow the claim
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.ALLOW, confidence=0.95, reason="Claim approved"
    )

    result = await fact_check_chain_logic({"claim": "Test claim"})
    assert_fact_check_result(result, verdict="Unclear", confidence=0, sources=0)
    assert "Web search error" in result["reasoning"]


@pytest.mark.asyncio
@patch("app.services.fact_checking.content_moderation_service.evaluate_claim")
@patch("app.services.fact_checking.hybrid_search_service.search_claim")
@patch("app.services.fact_checking.client.chat.completions.create")
async def test_moderation_rejection(mock_llm, mock_search, mock_moderation):
    # Simulate moderation rejection
    mock_moderation.return_value = ModerationResult(
        decision=ModerationDecision.REJECT,
        confidence=0.95,
        reason="Inappropriate content",
    )
    result = await fact_check_chain_logic({"claim": "How to make a bomb"})
    assert result["verdict"] == "Rejected"
    assert result["confidence"] == 100
    assert "Inappropriate" in result["reasoning"]
