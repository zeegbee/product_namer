from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.errors import BlockedURLError, ConfigError, FetchError, FetchTimeoutError, LLMResponseError
from app.main import create_app
from app.schemas import GenerateResponse, NameSuggestion

NAMES = [NameSuggestion(name=f"Name{i}", reason=f"Причина {i}") for i in range(1, 6)]


@pytest.fixture
def agent() -> MagicMock:
    agent = MagicMock()
    agent.run.side_effect = lambda url: GenerateResponse(source_url=url, names=NAMES)
    return agent


@pytest.fixture
def client(settings, agent):
    app = create_app(settings=settings, agent=agent)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "1.0.0", "model": "gpt-test"}


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Product Namer" in response.text
    assert "Подобрать названия" in response.text
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_generate_success(client, agent):
    response = client.post("/api/generate", json={"url": "https://example.com/product"})
    assert response.status_code == 200
    body = response.json()
    assert body["source_url"] == "https://example.com/product"
    assert len(body["names"]) == 5
    assert body["names"][0] == {"name": "Name1", "reason": "Причина 1"}
    agent.run.assert_called_once_with("https://example.com/product")


@pytest.mark.parametrize(
    "payload",
    [{}, {"url": ""}, {"url": "not-a-url"}, {"url": "ftp://example.com"}, {"url": 123},
     {"url": "https://example.com", "extra": 1}, {"url": "https://example.com/" + "a" * 3000}],
)  # fmt: skip
def test_generate_rejects_invalid_input(client, agent, payload):
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 422
    assert response.json()["error_type"] == "validation_error"
    agent.run.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status", "error_type"),
    [
        (BlockedURLError(), 422, "blocked_url"),
        (FetchError(), 502, "fetch_failed"),
        (FetchTimeoutError(), 504, "fetch_timeout"),
        (LLMResponseError(), 502, "llm_invalid_response"),
        (RuntimeError("boom"), 500, "internal_error"),
    ],
)
def test_generate_maps_errors_to_status_codes(client, agent, error, status, error_type):
    agent.run.side_effect = error
    response = client.post("/api/generate", json={"url": "https://example.com/"})
    assert response.status_code == status
    body = response.json()
    assert body["error_type"] == error_type
    assert body["detail"]
    assert "boom" not in body["detail"]  # внутренние подробности не уходят клиенту


def test_generate_blocks_localhost_end_to_end(settings):
    """Реальный агент (без моков загрузчика) не обращается к localhost."""
    app = create_app(settings=settings, agent=None)
    with TestClient(app) as test_client:
        response = test_client.post("/api/generate", json={"url": "http://127.0.0.1:8000/health"})
    assert response.status_code == 422
    assert response.json()["error_type"] == "blocked_url"


def test_load_settings_requires_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_settings(env_file=None)


def test_load_settings_validates_numbers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("MAX_RETRIES", "abc")
    with pytest.raises(ConfigError, match="MAX_RETRIES"):
        load_settings(env_file=None)


def test_load_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-custom")
    monkeypatch.setenv("MAX_RETRIES", "2")
    settings = load_settings(env_file=None)
    assert settings.openai_model == "gpt-custom"
    assert settings.max_retries == 2
    assert "sk-real" not in repr(settings)


def test_app_fails_to_start_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir("tests")  # чтобы не подхватить локальный .env
    with pytest.raises(ConfigError), TestClient(create_app()):
        pass
