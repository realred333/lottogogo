"""Tests for MVP FastAPI layer."""

from __future__ import annotations

from fastapi.testclient import TestClient

from lottogogo.mvp import api


class StubService:
    """Simple fake service for API contract tests."""

    def recommend(self, preset: str, games: int, seed: int | None = None) -> dict[str, object]:
        _ = (games, seed)
        return {
            "meta": {"preset": preset, "percentile": 17},
            "recommendations": [
                {
                    "numbers": [1, 9, 17, 23, 34, 41],
                    "score": 12.3456,
                    "tags": ["hmm_hot"],
                    "reasons": ["AC>=7", "sum 100-175"],
                }
            ],
        }


def test_home_page_includes_donate_and_backend_url(monkeypatch) -> None:
    monkeypatch.setenv("DONATE_URL", "https://example.com/donate")
    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:9000")
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "https://example.com/donate" in response.text
    assert "http://127.0.0.1:9000" in response.text


def test_recommend_endpoint_returns_expected_shape(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_service", lambda: StubService())
    client = TestClient(api.app)

    response = client.post("/api/recommend", json={"preset": "A", "games": 5, "seed": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["preset"] == "A"
    assert payload["meta"]["percentile"] == 17
    assert payload["recommendations"][0]["numbers"] == [1, 9, 17, 23, 34, 41]


def test_recommend_endpoint_rejects_invalid_games() -> None:
    client = TestClient(api.app)

    response = client.post("/api/recommend", json={"preset": "A", "games": 7})

    assert response.status_code == 422


def test_cors_preflight_allows_local_dev_origin() -> None:
    client = TestClient(api.app)

    response = client.options(
        "/api/recommend",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
