import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.web_search import WebSearchService
from app.domain.models import Source


def mock_http_response(mock_client_class, response_data=None, side_effect=None):
    """Patch httpx.AsyncClient to return a fake JSON or raise an error."""
    mock_client = AsyncMock()
    mock_client_class.return_value.__aenter__.return_value = mock_client

    if side_effect:
        mock_client.post.side_effect = side_effect
    else:
        mock_response = MagicMock()
        mock_response.json.return_value = response_data or {"organic": []}
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_successful_search(mock_client_class):
    """Web search returns multiple sources."""
    service = WebSearchService()
    mock_response_data = {
        "organic": [
            {
                "title": "Test Source 1",
                "link": "https://example.com/1",
                "snippet": "Snippet 1",
            },
            {
                "title": "Test Source 2",
                "link": "https://example.com/2",
                "snippet": "Snippet 2",
            },
        ]
    }
    mock_http_response(mock_client_class, response_data=mock_response_data)

    result = await service.search_claim("test claim", num_results=2)
    assert len(result) == 2
    assert isinstance(result[0], Source)
    assert result[0].title == "Test Source 1"
    assert result[0].url == "https://example.com/1"
    assert result[0].snippet == "Snippet 1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_data",
    [
        {"organic": []},  # no results
        {"other_key": []},  # missing organic key
    ],
)
@patch("httpx.AsyncClient")
async def test_empty_results(mock_client_class, response_data):
    """No results or missing organic key returns empty list."""
    service = WebSearchService()
    mock_http_response(mock_client_class, response_data=response_data)

    result = await service.search_claim("test claim")
    assert result == []  # no results


@pytest.mark.asyncio
@patch("httpx.AsyncClient")
async def test_partial_source_data(mock_client_class):
    """Missing snippet defaults to empty string."""
    service = WebSearchService()
    mock_response_data = {
        "organic": [{"title": "Test Source", "link": "https://example.com"}]
    }
    mock_http_response(mock_client_class, response_data=mock_response_data)

    result = await service.search_claim("test claim")
    assert len(result) == 1
    assert result[0].title == "Test Source"
    assert result[0].url == "https://example.com"
    assert result[0].snippet == ""  # fallback


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side_effect",
    [
        httpx.HTTPStatusError(
            "404 Not Found", request=AsyncMock(), response=AsyncMock()
        ),
        httpx.RequestError("Connection failed"),
        Exception("Unexpected error"),
    ],
)
@patch("httpx.AsyncClient")
async def test_error_conditions_return_empty(mock_client_class, side_effect):
    """HTTP errors, request errors, or unexpected exceptions → return empty list."""
    service = WebSearchService()
    mock_http_response(mock_client_class, side_effect=side_effect)

    result = await service.search_claim("test claim")
    assert result == []  # graceful fallback
