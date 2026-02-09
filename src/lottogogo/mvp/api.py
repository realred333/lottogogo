"""FastAPI app for LottoGoGo MVP."""

from __future__ import annotations

import html
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from .service import RecommendationService

APP_ROOT = Path(__file__).resolve().parent
INDEX_HTML_PATH = APP_ROOT / "static" / "index.html"


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
    return RecommendationService(history_csv=history_csv)


def get_donate_url() -> str:
    """Resolve donate URL for CTA button."""

    return os.getenv("DONATE_URL", "https://buymeacoffee.com/lottogogo")


def get_backend_url() -> str:
    """Resolve optional backend base URL for frontend API calls."""

    return os.getenv("BACKEND_URL", "")


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    """Serve single-page MVP UI."""

    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="index.html 파일을 찾을 수 없습니다.")

    html_text = INDEX_HTML_PATH.read_text(encoding="utf-8")
    html_text = html_text.replace("__DONATE_URL__", html.escape(get_donate_url(), quote=True))
    html_text = html_text.replace("__BACKEND_URL__", html.escape(get_backend_url(), quote=True))
    return HTMLResponse(content=html_text)


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
