# LottoGoGo v2

로또 번호 **선택 보조용 확률 실험 프로젝트**입니다.
당첨을 보장하지 않으며, 통계 실험 결과를 참고 정보로 제공합니다.

## 핵심 변경사항 (현재 운영 구조)

- `Render` 의존 제거
- `Vercel + GitHub Actions` 중심 운영
- 매 요청 서버 계산 대신:
  - 주간 배치에서 `history.csv` + `model.json` 갱신
  - 프론트에서 Web Worker로 즉시 조합 생성/필터링

## 아키텍처

1. GitHub Actions (토요일)
- 신규 회차만 `history.csv` 업데이트 (증분)
- Python으로 preset별 100k Monte Carlo 실행
- 결과를 `data/model.json`으로 생성
- 변경이 있을 때만 커밋/푸시

2. Frontend (Vercel 정적)
- 브라우저가 `data/model.json` 로드
- `assets/recommend-worker.js`에서 비동기 생성
- AC/총합/홀짝/고저/구간/히스토리 필터 적용
- 다양성(max overlap) 적용 후 5/10게임 반환
- seed 입력 없이도 즉시 결과 생성

## 프로젝트 구조

```text
.github/workflows/
  lotto-history-update.yml

assets/
  recommend-worker.js

scripts/
  update_history_csv.py
  build_frontend_model.py
  export_vercel_index.sh

src/lottogogo/
  data/
  engine/
  mvp/
    api.py
    service.py
    static/index.html

data/
  model.json

history.csv
index.html
vercel.json
```

## 빠른 시작 (로컬)

### 1) 의존성 설치

```bash
uv sync
```

### 2) 테스트

```bash
uv run pytest -q
```

### 3) 데이터 업데이트 (증분)

```bash
uv run python scripts/update_history_csv.py --csv history.csv --workers 8
```

- 기본 동작: 마지막 회차 다음부터만 조회
- `history.csv`가 없거나 비정상일 때 전체 부트스트랩을 자동으로 하지 않음

### 4) 프론트 모델 생성

```bash
uv run python scripts/build_frontend_model.py --history-csv history.csv --output data/model.json
```

### 5) Vercel용 정적 index 내보내기

```bash
./scripts/export_vercel_index.sh
```

## GitHub Actions (주간 자동 갱신)

워크플로: `.github/workflows/lotto-history-update.yml`

스케줄:
- 매주 토요일(KST 저녁) 3회 재시도 창

동작:
1. `history.csv` 증분 업데이트
2. `data/model.json` 재생성 (preset별 100k)
3. 두 파일 중 변경이 있을 때만 커밋

커밋 없음(no-op) 조건:
- 신규 회차 없음
- 모델 변화 없음

## 모델(`data/model.json`)에 들어가는 내용

- 최신 회차 메타
- 번호별 raw score
- preset A/B별:
  - 필터 파라미터
  - 생성 확률 가중치
  - diversity/ranking 파라미터
  - Monte Carlo 요약 통계
- 히스토리 중복 방지 인덱스
  - exact 조합
  - 5개 부분집합 키

## 프론트 동작 상세

- 버튼 클릭 시 메인 스레드는 즉시 반환, 계산은 Worker에서 수행
- Worker가 확률 샘플링 → 필터 → 점수화 → 다양성 선택
- 최근 추천 재노출 완화:
  - `localStorage`의 최근 키를 우선 회피
- 결과 없을 때 fallback 경로로 무한 대기 방지

## SEO

- 정적 `index.html`에 다음 메타 포함:
  - Google verification
  - Naver verification
  - canonical / OG / Twitter / JSON-LD
- 기본 OG/Twitter 이미지는 `assets/og-image.png`를 사용
  - 필요하면 `OG_IMAGE_URL`, `TWITTER_IMAGE_URL`로 교체 가능
- `robots.txt`, `sitemap.xml`은 `vercel.json` 리라이트로 `api/robots.js`, `api/sitemap.js` 제공

## 환경변수

`.env.example` 참고

주요 항목:
- `DONATE_URL`
- `PUBLIC_BASE_URL`
- `MODEL_URL`
- `GOOGLE_SITE_VERIFICATION`
- `NAVER_SITE_VERIFICATION`
- `LOTTO_HISTORY_CSV`
- `FRONTEND_MODEL_PATH`

## 배포

### Vercel

- 이 저장소 루트를 배포 대상으로 설정
- 정적 `index.html` + `data/model.json` + `assets/recommend-worker.js` 배포
- GitHub push 시 자동 배포

## 주의사항

- 이 프로젝트는 예측 서비스가 아니라 **실험/참고 도구**입니다.
- 실제 구매 판단과 결과 책임은 사용자에게 있습니다.

## 라이선스

MIT
