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

    def warmup(self, blocking: bool = False) -> dict[str, int]:
        _ = blocking
        return {"A_5": 2, "A_10": 2, "B_5": 2, "B_10": 2}

    def pool_status(self) -> dict[str, int]:
        return {"A_5": 2, "A_10": 2, "B_5": 2, "B_10": 2}


class StrictStubService:
    """Service with no seed argument to ensure API ignores payload seed."""

    def recommend(self, preset: str, games: int) -> dict[str, object]:
        return {
            "meta": {"preset": preset, "percentile": 1},
            "recommendations": [
                {
                    "numbers": [1, 2, 3, 4, 5, 6],
                    "score": 1.0,
                    "tags": [],
                    "reasons": [],
                }
            ],
        }


def test_home_page_includes_donate_and_model_url(monkeypatch) -> None:
    monkeypatch.setenv("DONATE_URL", "https://example.com/donate")
    monkeypatch.setenv("MODEL_URL", "/custom/model.json")
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "https://example.com/donate" in response.text
    assert "/custom/model.json" in response.text


def test_home_page_includes_seo_meta(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://lottogogo.example")
    monkeypatch.setenv("GOOGLE_SITE_VERIFICATION", "google-token")
    monkeypatch.setenv("NAVER_SITE_VERIFICATION", "naver-token")
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert '<link rel="canonical" href="https://lottogogo.example/" />' in response.text
    assert '<meta property="og:image" content="https://lottogogo.example/assets/og-image.png" />' in response.text
    assert '<meta name="twitter:image" content="https://lottogogo.example/assets/og-image.png" />' in response.text
    assert '<meta name="google-site-verification" content="google-token" />' in response.text
    assert '<meta name="naver-site-verification" content="naver-token" />' in response.text
    assert "__SEO_" not in response.text
    assert "application/ld+json" in response.text


def test_recommend_endpoint_returns_expected_shape(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_service", lambda: StubService())
    client = TestClient(api.app)

    response = client.post("/api/recommend", json={"preset": "A", "games": 5, "seed": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["preset"] == "A"
    assert payload["meta"]["percentile"] == 17
    assert payload["recommendations"][0]["numbers"] == [1, 9, 17, 23, 34, 41]


def test_recommend_endpoint_ignores_seed_payload(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_service", lambda: StrictStubService())
    client = TestClient(api.app)

    response = client.post("/api/recommend", json={"preset": "A", "games": 5, "seed": 42})

    assert response.status_code == 200
    payload = response.json()
    assert payload["recommendations"][0]["numbers"] == [1, 2, 3, 4, 5, 6]


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


def test_robots_txt_exposes_sitemap_url() -> None:
    client = TestClient(api.app)

    response = client.get("/robots.txt")

    assert response.status_code == 200
    assert "User-agent: *" in response.text
    assert "Allow: /" in response.text
    assert "Sitemap: http://testserver/sitemap.xml" in response.text


def test_sitemap_xml_lists_home_url() -> None:
    client = TestClient(api.app)

    response = client.get("/sitemap.xml")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert "<loc>http://testserver/</loc>" in response.text


def test_warmup_endpoint_returns_pool_status(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_service", lambda: StubService())
    client = TestClient(api.app)

    response = client.get("/api/warmup")

    assert response.status_code == 200
    payload = response.json()
    assert payload["detail"] == "warmup triggered"
    assert payload["pool"]["A_5"] == 2


def test_warmup_endpoint_requires_token_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("WARMUP_TOKEN", "secret-token")
    monkeypatch.setattr(api, "get_service", lambda: StubService())
    client = TestClient(api.app)

    unauthorized = client.get("/api/warmup")
    assert unauthorized.status_code == 401

    authorized = client.get("/api/warmup", headers={"x-warmup-token": "secret-token"})
    assert authorized.status_code == 200


def test_pool_status_endpoint_returns_status(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_service", lambda: StubService())
    client = TestClient(api.app)

    response = client.get("/api/pool-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["pool"]["B_10"] == 2
