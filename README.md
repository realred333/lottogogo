# LottoGoGo v2

확률 실험 기반 선택 보조 도구입니다.

> A small probability experiment.
> No prediction.
> No guarantees.

이 프로젝트는 과거 회차 데이터를 바탕으로 번호 점수를 계산하고,
샘플링/필터링/랭킹을 통해 조합을 제안합니다.

## 원칙

- 당첨을 보장하지 않습니다.
- 결과는 통계 실험용 참고 정보입니다.
- 구매 판단과 책임은 사용자에게 있습니다.

## 주요 기능

- `POST /api/recommend` 단일 API
- 단일 웹 페이지(`/`)에서 Preset A/B, 5/10게임 선택
- 추천 근거(tags/reasons) 표시
- 후원 버튼(`DONATE_URL`) 지원

## 프로젝트 구조

```text
history.csv
recommend.py
backtest.py
src/lottogogo/
  data/
  engine/
  mvp/
    api.py
    service.py
    static/index.html
tests/unit/
```

## 빠른 시작

### 1) 의존성 설치

```bash
uv sync
```

### 2) 테스트

```bash
uv run pytest -q
```

### 3) 서버 실행

```bash
DONATE_URL='https://buymeacoffee.com/lottogogo' \
uv run uvicorn lottogogo.mvp.api:app --app-dir src --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 `http://127.0.0.1:8000` 접속.

## 환경변수

`.env.example`를 참고해 `.env`를 만들고, 실제 배포 환경(Render)에서는 환경변수로 주입하세요.

- `DONATE_URL` : 후원 버튼 링크
- `BACKEND_URL` : 프론트에서 호출할 API 베이스 URL (비우면 same-origin)
- `PUBLIC_BASE_URL` : canonical/robots/sitemap 생성에 사용할 공개 URL
- `SEO_TITLE` : 페이지 `<title>` 커스텀
- `SEO_DESCRIPTION` : 페이지 meta description 커스텀
- `OG_IMAGE_URL` : Open Graph 이미지 URL
- `TWITTER_IMAGE_URL` : Twitter 이미지 URL (비우면 `OG_IMAGE_URL` 사용)
- `GOOGLE_SITE_VERIFICATION` : Google Search Console 인증값
- `NAVER_SITE_VERIFICATION` : 네이버 서치어드바이저 인증값
- `LOTTO_HISTORY_CSV` : 기본 `history.csv` 대신 사용할 CSV 경로
- `CORS_ALLOW_ORIGINS` : CORS 허용 origin 목록(쉼표 구분, 기본 `*`)

## API 예시

### Request

```bash
curl -X POST 'http://127.0.0.1:8000/api/recommend' \
  -H 'content-type: application/json' \
  -d '{"preset":"A","games":5,"seed":42}'
```

### Response (shape)

```json
{
  "meta": {
    "preset": "A",
    "percentile": 12
  },
  "recommendations": [
    {
      "numbers": [1, 2, 3, 4, 5, 6],
      "score": 0.123,
      "tags": ["hmm_hot"],
      "reasons": ["AC>=7", "sum 100-175", "zone<=3"]
    }
  ]
}
```

## 운영 체크리스트 (주간)

1. `history.csv` 최신 회차 반영
2. 커밋/푸시
3. Render 재배포(또는 자동 배포 확인)
4. 배포 URL에서 추천 버튼/후원 버튼 동작 확인

## Render 배포

- Build Command:

```bash
pip install .
```

- Start Command:

```bash
uvicorn lottogogo.mvp.api:app --app-dir src --host 0.0.0.0 --port $PORT
```

- 필수 환경변수:
  - `DONATE_URL`

## Vercel + Render 분리 배포 (권장)

- 목적:
  - 프론트는 Vercel 정적 배포로 즉시 로드
  - 백엔드는 Render 유지

- 이 저장소에서 사용하는 파일:
  - `index.html` : Vercel이 바로 서빙할 정적 프론트
  - `api/recommend.js` : Vercel 서버리스 프록시 (`/api/recommend` -> Render)
  - `api/robots.js`, `api/sitemap.js` : 도메인 기준 robots/sitemap 동적 생성
  - `vercel.json` : `/robots.txt`, `/sitemap.xml` 라우팅
  - `scripts/export_vercel_index.sh` : 템플릿(`src/lottogogo/mvp/static/index.html`)을 Vercel용 `index.html`로 변환

- 프론트 템플릿 변경 후 동기화:

```bash
./scripts/export_vercel_index.sh
```

- Vercel 프로젝트 환경변수:
  - `RENDER_BACKEND_URL=https://<your-render-service>.onrender.com`
  - (선택) `RENDER_WARMUP_TOKEN=<임의의긴토큰>`

- Render 프로젝트 환경변수:
  - (선택) `WARMUP_TOKEN=<Vercel의 RENDER_WARMUP_TOKEN 과 동일한 값>`
  - (선택) `RECOMMEND_POOL_TARGET=4` (미리 준비할 추천 결과 개수)
  - (선택) `RECOMMEND_POOL_MAX=8` (메모리 내 최대 저장 개수)

- 동작 방식:
  - 브라우저는 Vercel 도메인의 `/api/recommend`를 호출
  - Vercel 함수가 Render API로 프록시 호출
  - Render가 슬립 상태면 첫 요청이 느릴 수 있고, UI에 안내 패널이 자동 표시됨
  - Render `/api/warmup`는 비동기로 추천 풀(pre-generated sets)을 채워서 첫 사용자 지연을 줄임

## SEO 점검 포인트

- `GET /robots.txt` 노출
- `GET /sitemap.xml` 노출
- 메타 태그/OG/JSON-LD(구조화 데이터) 기본 포함
- Search Console 등록 시:
  - Google: `GOOGLE_SITE_VERIFICATION` 설정
  - Naver: `NAVER_SITE_VERIFICATION` 설정

## 공개 운영 가이드

- 공개해도 되는 것:
  - 엔진 구조, 필터 아이디어, 실험 방식
- 공개하면 안 되는 것:
  - 토큰/키/비밀번호/내부 운영 비밀값
- 권장:
  - `.env`는 커밋 금지, `.env.example`만 커밋

## 라이선스

MIT License
