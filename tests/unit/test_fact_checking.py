"""
Unit tests for the fact-checking service.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.services.fact_checking import fact_check_chain_logic
from tests.conftest import TestData, AssertionHelpers


class TestFactChecking:
    """Test the fact-checking service logic."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_successful_response(self, mock_llm, mock_search):
        """Test successful fact-checking with valid response."""
        # Mock web search with proper dictionary structure
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: FALSE
        Confidence: 90
        Reasoning: This claim is false based on evidence.
        """
        mock_llm.return_value = mock_response

        result = await fact_check_chain_logic({"claim": "Test claim"})

        # Verify response structure
        AssertionHelpers.assert_valid_fact_check_response({"output": result})

        # Verify specific values
        assert result["verdict"] == "False"
        assert result["confidence"] == 90
        assert "reasoning" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["title"] == "Test Source 1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_multiple_sources(self, mock_llm, mock_search):
        """Test fact-checking with multiple sources."""
        # Mock web search with multiple sources
        mock_search.return_value = TestData.create_mock_sources(3)

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: TRUE
        Confidence: 85
        Reasoning: Multiple sources confirm this claim.
        """
        mock_llm.return_value = mock_response

        result = await fact_check_chain_logic({"claim": "Test claim"})

        # Verify response structure
        AssertionHelpers.assert_valid_fact_check_response({"output": result})

        # Verify multiple sources
        assert len(result["sources"]) == 3
        assert result["sources"][0]["title"] == "Test Source 1"
        assert result["sources"][1]["title"] == "Test Source 2"
        assert result["sources"][2]["title"] == "Test Source 3"

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_unclear_verdict(self, mock_llm, mock_search):
        """Test fact-checking with unclear verdict."""
        # Mock web search
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response with unclear verdict
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: UNCLEAR
        Confidence: 45
        Reasoning: Insufficient evidence to determine truth.
        """
        mock_llm.return_value = mock_response

        result = await fact_check_chain_logic({"claim": "Test claim"})

        # Verify response structure
        AssertionHelpers.assert_valid_fact_check_response({"output": result})

        # Verify unclear verdict
        assert result["verdict"] == "Unclear"
        assert result["confidence"] == 45

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_no_sources_found(self, mock_llm, mock_search):
        """Test fact-checking when no sources are found."""
        # Mock web search with no results
        mock_search.return_value = []

        # Mock LLM response (should not be called)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: UNCLEAR
        Confidence: 30
        Reasoning: No sources found to verify this claim.
        """
        mock_llm.return_value = mock_response

        result = await fact_check_chain_logic({"claim": "Test claim"})

        # Verify response structure
        AssertionHelpers.assert_valid_fact_check_response({"output": result})

        # Verify no sources and default values
        assert len(result["sources"]) == 0
        assert result["verdict"] == "Unclear"
        assert result["confidence"] == 0  # Default for no sources
        assert "Unable to find relevant sources" in result["reasoning"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_web_search_error(self, mock_llm, mock_search):
        """Test fact-checking when web search fails."""
        # Mock web search to raise an exception
        mock_search.side_effect = Exception("Web search service unavailable")

        # Mock LLM response (should not be called)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: UNCLEAR
        Confidence: 0
        Reasoning: Web search failed.
        """
        mock_llm.return_value = mock_response

        result = await fact_check_chain_logic({"claim": "Test claim"})

        # Verify error handling
        AssertionHelpers.assert_valid_fact_check_response({"output": result})
        assert result["verdict"] == "Unclear"
        assert result["confidence"] == 0
        assert "Error during web search" in result["reasoning"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_llm_error(self, mock_llm, mock_search):
        """Test fact-checking when LLM fails."""
        # Mock web search
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM to raise an exception
        mock_llm.side_effect = Exception("LLM service unavailable")

        # Test that the function handles LLM errors gracefully
        result = await fact_check_chain_logic({"claim": "Test claim"})

        # Verify error handling
        AssertionHelpers.assert_valid_fact_check_response({"output": result})
        assert result["verdict"] == "Unclear"
        assert result["confidence"] == 50  # Default confidence for LLM errors
        assert "Error calling AI model" in result["reasoning"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_empty_claim(self, mock_llm, mock_search):
        """Test fact-checking with empty claim."""
        # Mock web search (should not be called)
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response (should not be called)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: UNCLEAR
        Confidence: 0
        Reasoning: No claim provided to verify.
        """
        mock_llm.return_value = mock_response

        # Test with empty claim
        result = await fact_check_chain_logic({"claim": ""})

        # Verify response structure
        AssertionHelpers.assert_valid_fact_check_response({"output": result})

        # Verify empty claim handling
        assert result["verdict"] == "Unclear"
        assert result["confidence"] == 0
        assert "No claim provided" in result["reasoning"]
        assert len(result["sources"]) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    @patch('app.services.fact_checking.web_search_service.search_claim')
    @patch('app.services.fact_checking.client.chat.completions.create')
    async def test_whitespace_claim(self, mock_llm, mock_search):
        """Test fact-checking with whitespace-only claim."""
        # Mock web search (should not be called)
        mock_search.return_value = TestData.create_mock_sources(1)

        # Mock LLM response (should not be called)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        Verdict: UNCLEAR
        Confidence: 0
        Reasoning: No claim provided to verify.
        """
        mock_llm.return_value = mock_response

        # Test with whitespace-only claim
        result = await fact_check_chain_logic({"claim": "   "})

        # Verify response structure
        AssertionHelpers.assert_valid_fact_check_response({"output": result})

        # Verify whitespace claim handling
        assert result["verdict"] == "Unclear"
        assert result["confidence"] == 0
        assert "No claim provided" in result["reasoning"]
        assert len(result["sources"]) == 0
