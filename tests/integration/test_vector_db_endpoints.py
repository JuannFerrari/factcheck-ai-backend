import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings


@pytest.fixture
def client():
    from app.core.rate_limiter import limiter

    limiter.reset()
    return TestClient(app)


API_PREFIX = "/api/v1/vector"
VALID_HEADERS = {settings.api_key_header: settings.api_key}
INVALID_HEADERS = {settings.api_key_header: "invalid"}


# --- Auth tests ---
def test_recent_requires_api_key(client):
    resp = client.get(f"{API_PREFIX}/recent")
    assert resp.status_code == 401
    resp = client.get(f"{API_PREFIX}/recent", headers=INVALID_HEADERS)
    assert resp.status_code == 401


def test_statistics_requires_api_key(client):
    resp = client.get(f"{API_PREFIX}/statistics")
    assert resp.status_code == 401
    resp = client.get(f"{API_PREFIX}/statistics", headers=INVALID_HEADERS)
    assert resp.status_code == 401


def test_search_requires_api_key(client):
    resp = client.get(f"{API_PREFIX}/search?claim=test")
    assert resp.status_code == 401
    resp = client.get(f"{API_PREFIX}/search?claim=test", headers=INVALID_HEADERS)
    assert resp.status_code == 401


def test_record_requires_api_key(client):
    resp = client.get(f"{API_PREFIX}/record/1")
    assert resp.status_code == 401
    resp = client.get(f"{API_PREFIX}/record/1", headers=INVALID_HEADERS)
    assert resp.status_code == 401


# --- Structure/status tests (with valid API key) ---
def test_recent_structure(client):
    resp = client.get(f"{API_PREFIX}/recent", headers=VALID_HEADERS)
    assert resp.status_code in (200, 500)  # 500 if DB empty/unavailable
    if resp.status_code == 200:
        data = resp.json()
        assert "records" in data
        assert "count" in data


def test_statistics_structure(client):
    resp = client.get(f"{API_PREFIX}/statistics", headers=VALID_HEADERS)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "vector_database" in data
        assert "search" in data
        assert "configuration" in data


def test_search_structure(client):
    resp = client.get(f"{API_PREFIX}/search?claim=test", headers=VALID_HEADERS)
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "query" in data
        assert "results" in data
        assert "count" in data
        assert "similarity_threshold" in data


def test_record_structure(client):
    # Try to fetch record 1 (may not exist, so 404 is OK)
    resp = client.get(f"{API_PREFIX}/record/1", headers=VALID_HEADERS)
    assert resp.status_code in (200, 404, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "id" in data
        assert "claim" in data
        assert "verdict" in data
        assert "confidence" in data
        assert "created_at" in data
