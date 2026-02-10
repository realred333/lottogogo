"""FastAPI app for LottoGoGo MVP."""

from __future__ import annotations

import json
import html
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator

from .service import RecommendationService

APP_ROOT = Path(__file__).resolve().parent
INDEX_HTML_PATH = APP_ROOT / "static" / "index.html"
DEFAULT_SEO_TITLE = "LottoGoGo | 로또 확률 실험 기반 선택 보조 도구"
DEFAULT_SEO_DESCRIPTION = (
    "로또 번호 선택을 위한 확률 실험 기반 보조 도구입니다. "
    "당첨을 보장하지 않으며 통계 필터로 무근거 선택을 줄입니다."
)


class RecommendRequest(BaseModel):
    """Request payload for recommendation endpoint."""

    preset: Literal["A", "B"] = "A"
    games: Literal[5, 10] = 5
    seed: int | None = Field(default=None, ge=0, le=2_147_483_647)


class RecommendationItem(BaseModel):
    """Single recommendation entry."""

    numbers: list[int]
    score: float
    tags: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @field_validator("numbers")
    @classmethod
    def _validate_numbers(cls, values: list[int]) -> list[int]:
        if len(values) != 6:
            raise ValueError("numbers must include exactly 6 values.")
        if sorted(values) != values:
            raise ValueError("numbers must be sorted in ascending order.")
        return values


class RecommendMeta(BaseModel):
    """Metadata for recommendation response."""

    preset: Literal["A", "B"]
    percentile: int | None = Field(default=None, ge=1, le=100)


class RecommendResponse(BaseModel):
    """Response payload for recommendation endpoint."""

    meta: RecommendMeta
    recommendations: list[RecommendationItem]


app = FastAPI(title="LottoGoGo MVP", version="0.1.0")


def get_cors_origins() -> list[str]:
    """Resolve allowed CORS origins from env."""

    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [item.strip() for item in raw.split(",") if item.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_service() -> RecommendationService:
    """Return singleton recommendation service."""

    history_csv = os.getenv("LOTTO_HISTORY_CSV", "history.csv")
    service = RecommendationService(history_csv=history_csv)
    # Trigger async warmup once at process bootstrap.
    service.warmup(blocking=False)
    return service


def require_warmup_token(request: Request) -> None:
    """Optionally protect warmup endpoint with token."""

    expected = os.getenv("WARMUP_TOKEN", "").strip()
    if not expected:
        return

    supplied = request.headers.get("x-warmup-token", "").strip() or request.query_params.get("token", "").strip()
    if supplied != expected:
        raise HTTPException(status_code=401, detail="warmup token mismatch")


def get_donate_url() -> str:
    """Resolve donate URL for CTA button."""

    return os.getenv("DONATE_URL", "https://buymeacoffee.com/lottogogo")


def get_backend_url() -> str:
    """Resolve optional backend base URL for frontend API calls."""

    return os.getenv("BACKEND_URL", "")


def get_public_base_url(request: Request) -> str:
    """Resolve canonical base URL from env or incoming request."""

    configured = os.getenv("PUBLIC_BASE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    return f"{request.url.scheme}://{request.url.netloc}"


def get_seo_title() -> str:
    """Resolve SEO page title."""

    return os.getenv("SEO_TITLE", DEFAULT_SEO_TITLE).strip() or DEFAULT_SEO_TITLE


def get_seo_description() -> str:
    """Resolve SEO page description."""

    return os.getenv("SEO_DESCRIPTION", DEFAULT_SEO_DESCRIPTION).strip() or DEFAULT_SEO_DESCRIPTION


def get_optional_env(name: str) -> str:
    """Read optional env var as stripped string."""

    return os.getenv(name, "").strip()


def build_optional_meta_tag(attr: str, key: str, value: str) -> str:
    """Build optional single-line meta tag."""

    if not value:
        return ""
    escaped = html.escape(value, quote=True)
    return f'<meta {attr}="{key}" content="{escaped}" />'


def build_structured_data(base_url: str) -> str:
    """Build JSON-LD schema for Google/Naver crawlers."""

    description = get_seo_description()
    data = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "WebSite",
                "name": "LottoGoGo",
                "url": f"{base_url}/",
                "inLanguage": "ko-KR",
                "description": description,
            },
            {
                "@type": "SoftwareApplication",
                "name": "LottoGoGo",
                "applicationCategory": "UtilitiesApplication",
                "operatingSystem": "Web",
                "url": f"{base_url}/",
                "description": description,
                "inLanguage": "ko-KR",
                "offers": {
                    "@type": "Offer",
                    "price": "0",
                    "priceCurrency": "KRW",
                },
            },
        ],
    }
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    """Serve single-page MVP UI."""

    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="index.html 파일을 찾을 수 없습니다.")

    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")
    base_url = get_public_base_url(request)
    seo_title = get_seo_title()
    seo_description = get_seo_description()
    canonical_url = f"{base_url}/"
    sitemap_url = f"{base_url}/sitemap.xml"

    og_image = get_optional_env("OG_IMAGE_URL")
    twitter_image = get_optional_env("TWITTER_IMAGE_URL") or og_image
    google_verification = get_optional_env("GOOGLE_SITE_VERIFICATION")
    naver_verification = get_optional_env("NAVER_SITE_VERIFICATION")

    replacements = {
        "__DONATE_URL__": html.escape(get_donate_url(), quote=True),
        "__BACKEND_URL__": html.escape(get_backend_url(), quote=True),
        "__SEO_TITLE__": html.escape(seo_title, quote=True),
        "__SEO_DESCRIPTION__": html.escape(seo_description, quote=True),
        "__SEO_CANONICAL_URL__": html.escape(canonical_url, quote=True),
        "__SEO_SITEMAP_URL__": html.escape(sitemap_url, quote=True),
        "__SEO_OG_IMAGE_META__": build_optional_meta_tag("property", "og:image", og_image),
        "__SEO_TWITTER_IMAGE_META__": build_optional_meta_tag("name", "twitter:image", twitter_image),
        "__SEO_GOOGLE_SITE_VERIFICATION_META__": build_optional_meta_tag(
            "name", "google-site-verification", google_verification
        ),
        "__SEO_NAVER_SITE_VERIFICATION_META__": build_optional_meta_tag(
            "name", "naver-site-verification", naver_verification
        ),
        "__SEO_STRUCTURED_DATA_JSON__": build_structured_data(base_url),
    }
    for token, value in replacements.items():
        html_text = html_text.replace(token, value)

    return HTMLResponse(content=html_text)


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots_txt(request: Request) -> PlainTextResponse:
    """Expose robots.txt for Googlebot/Naver Yeti crawlers."""

    base_url = get_public_base_url(request)
    sitemap_url = f"{base_url}/sitemap.xml"
    content = "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            f"Sitemap: {sitemap_url}",
        ]
    )
    return PlainTextResponse(f"{content}\n")


@app.get("/sitemap.xml", response_class=Response, include_in_schema=False)
def sitemap_xml(request: Request) -> Response:
    """Expose XML sitemap for crawler discovery."""

    base_url = get_public_base_url(request)
    page_url = html.escape(f"{base_url}/", quote=True)
    today = datetime.now(timezone.utc).date().isoformat()
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        "  <url>\n"
        f"    <loc>{page_url}</loc>\n"
        f"    <lastmod>{today}</lastmod>\n"
        "    <changefreq>weekly</changefreq>\n"
        "    <priority>1.0</priority>\n"
        "  </url>\n"
        "</urlset>\n"
    )
    return Response(content=xml, media_type="application/xml")


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    """Generate recommendations via engine adapter."""

    try:
        result = get_service().recommend(
            preset=payload.preset,
            games=payload.games,
            seed=payload.seed,
        )
        return RecommendResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"데이터 파일을 찾을 수 없습니다: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(status_code=500, detail=f"추천 생성 실패: {exc}") from exc


@app.get("/api/warmup")
def warmup(request: Request) -> dict[str, object]:
    """Trigger async pre-generation for recommendation pools."""

    require_warmup_token(request)
    status = get_service().warmup(blocking=False)
    return {"detail": "warmup triggered", "pool": status}


@app.get("/api/pool-status")
def pool_status(request: Request) -> dict[str, object]:
    """Inspect current recommendation pool status."""

    require_warmup_token(request)
    return {"pool": get_service().pool_status()}
