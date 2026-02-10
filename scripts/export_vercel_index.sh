#!/usr/bin/env sh
set -eu

TEMPLATE="src/lottogogo/mvp/static/index.html"
OUTPUT="index.html"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

"$PYTHON_BIN" - "$TEMPLATE" "$OUTPUT" <<'PY'
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys


def optional_meta(attr: str, key: str, value: str) -> str:
    if not value:
        return ""
    return f'<meta {attr}="{key}" content="{html.escape(value, quote=True)}" />'


template_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
text = template_path.read_text(encoding="utf-8")

base_url = os.getenv("PUBLIC_BASE_URL", "https://lottogogo.vercel.app").strip().rstrip("/")
donate_url = os.getenv("DONATE_URL", "https://buymeacoffee.com/lottogogo").strip()
model_url = os.getenv("MODEL_URL", "/data/model.json").strip() or "/data/model.json"
seo_title = os.getenv("SEO_TITLE", "LottoGoGo | 로또 확률 실험 기반 선택 보조 도구").strip()
seo_description = os.getenv(
    "SEO_DESCRIPTION",
    "로또 번호 선택을 위한 확률 실험 기반 보조 도구입니다. 당첨을 보장하지 않으며 통계 필터로 무근거 선택을 줄입니다.",
).strip()

canonical_url = f"{base_url}/"
sitemap_url = f"{base_url}/sitemap.xml"

og_image = os.getenv("OG_IMAGE_URL", "").strip()
twitter_image = os.getenv("TWITTER_IMAGE_URL", "").strip() or og_image

google_verification = os.getenv("GOOGLE_SITE_VERIFICATION", "PbH0mni_SrDNFhpznnlQNcklvSnXtGVP9GhOhedijYQ").strip()
naver_verification = os.getenv("NAVER_SITE_VERIFICATION", "bfd8bd236fd5f7ac42c026fbb28dfd437612f0fd").strip()

structured = json.dumps(
    {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "LottoGoGo",
        "url": canonical_url,
        "inLanguage": "ko-KR",
        "description": seo_description,
    },
    ensure_ascii=False,
    separators=(",", ":"),
)

replacements = {
    "__DONATE_URL__": html.escape(donate_url, quote=True),
    "__BACKEND_URL__": "",
    "__MODEL_URL__": html.escape(model_url, quote=True),
    "__SEO_TITLE__": html.escape(seo_title, quote=True),
    "__SEO_DESCRIPTION__": html.escape(seo_description, quote=True),
    "__SEO_CANONICAL_URL__": html.escape(canonical_url, quote=True),
    "__SEO_SITEMAP_URL__": html.escape(sitemap_url, quote=True),
    "__SEO_OG_IMAGE_META__": optional_meta("property", "og:image", og_image),
    "__SEO_TWITTER_IMAGE_META__": optional_meta("name", "twitter:image", twitter_image),
    "__SEO_GOOGLE_SITE_VERIFICATION_META__": optional_meta(
        "name", "google-site-verification", google_verification
    ),
    "__SEO_NAVER_SITE_VERIFICATION_META__": optional_meta(
        "name", "naver-site-verification", naver_verification
    ),
    "__SEO_STRUCTURED_DATA_JSON__": structured,
}

for token, value in replacements.items():
    text = text.replace(token, value)

output_path.write_text(text, encoding="utf-8")
print(f"Exported {output_path} from {template_path}")
PY
