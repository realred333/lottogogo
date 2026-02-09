#!/usr/bin/env sh
set -eu

TEMPLATE="src/lottogogo/mvp/static/index.html"
OUTPUT="index.html"

cp "$TEMPLATE" "$OUTPUT"

perl -0pi -e '
  my $structured = q({"@context":"https://schema.org","@type":"WebSite","name":"LottoGoGo","url":"/"});
  s{__DONATE_URL__}{https://buymeacoffee.com/lottogogo}g;
  s{__BACKEND_URL__}{}g;
  s{__SEO_TITLE__}{LottoGoGo | 로또 확률 실험 기반 선택 보조 도구}g;
  s{__SEO_DESCRIPTION__}{로또 번호 선택을 위한 확률 실험 기반 보조 도구입니다. 당첨을 보장하지 않으며 통계 필터로 무근거 선택을 줄입니다.}g;
  s{__SEO_CANONICAL_URL__}{/}g;
  s{__SEO_SITEMAP_URL__}{/sitemap.xml}g;
  s{__SEO_OG_IMAGE_META__}{}g;
  s{__SEO_TWITTER_IMAGE_META__}{}g;
  s{__SEO_GOOGLE_SITE_VERIFICATION_META__}{}g;
  s{__SEO_NAVER_SITE_VERIFICATION_META__}{}g;
  s{__SEO_STRUCTURED_DATA_JSON__}{$structured}g;
' "$OUTPUT"

echo "Exported $OUTPUT from $TEMPLATE"
