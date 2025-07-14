"""
Unit tests for the web search service.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from app.services.web_search import WebSearchService
from app.domain.models import Source


class TestWebSearch:
    """Test the web search service."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_successful_search(self):
        """Test successful web search with results."""
        service = WebSearchService()

        mock_response_data = {
            "organic": [
                {
                    "title": "Test Source 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a test snippet for source 1"
                },
                {
                    "title": "Test Source 2",
                    "link": "https://example.com/2",
                    "snippet": "This is a test snippet for source 2"
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            result = await service.search_claim("test claim", num_results=2)

            assert len(result) == 2
            assert isinstance(result[0], Source)
            assert result[0].title == "Test Source 1"
            assert result[0].url == "https://example.com/1"
            assert result[0].snippet == "This is a test snippet for source 1"
            assert result[1].title == "Test Source 2"
            assert result[1].url == "https://example.com/2"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_results(self):
        """Test web search with no results."""
        service = WebSearchService()

        mock_response_data = {"organic": []}

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            result = await service.search_claim("test claim")

            assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_http_error(self):
        """Test web search with HTTP error."""
        service = WebSearchService()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "404 Not Found",
                request=AsyncMock(),
                response=AsyncMock()
            )

            result = await service.search_claim("test claim")

            assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_request_error(self):
        """Test web search with request error."""
        service = WebSearchService()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.RequestError(
                "Connection failed")

            result = await service.search_claim("test claim")

            assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        """Test web search with unexpected error."""
        service = WebSearchService()

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = Exception("Unexpected error")

            result = await service.search_claim("test claim")

            assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_missing_organic_key(self):
        """Test web search response without organic key."""
        service = WebSearchService()

        mock_response_data = {"other_key": []}

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            result = await service.search_claim("test claim")

            assert len(result) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_partial_source_data(self):
        """Test web search with partial source data."""
        service = WebSearchService()

        mock_response_data = {
            "organic": [
                {
                    "title": "Test Source",
                    "link": "https://example.com",
                    # Missing snippet
                }
            ]
        }

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = MagicMock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response

            result = await service.search_claim("test claim")

            assert len(result) == 1
            assert result[0].title == "Test Source"
            assert result[0].url == "https://example.com"
            assert result[0].snippet == ""  # Default value for missing snippet
