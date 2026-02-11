# LottoGoGo v2

로또 번호 **선택 보조용 확률 실험 프로젝트**입니다.
당첨을 보장하지 않으며, 통계 실험 결과를 참고 정보로 제공합니다.

## 핵심 변경사항 (현재 운영 구조)

- `Render` 의존 제거
- `Vercel + GitHub Actions` 중심 운영
- 매 요청 서버 계산 대신:
  - 주간 배치에서 `history.csv` + `model.json` 갱신
  - 프론트에서 Web Worker로 즉시 조합 생성/필터링

---

## 확률 엔진 아키텍처

### 전체 파이프라인

```
[history.csv]
    ↓
[1. Base Score]   Beta-Bernoulli 베이지안 출현 확률
    ↓
[2. Boost]        Hot/Cold/Carryover/Neighbor/Reverse 휴리스틱
    ↓
[3. HMM]          은닉 마르코프 모델 상태 추론 (Hot/Neutral/Cold)
    ↓
[4. Penalty]      Poisson/Markov 페널티 (현재 비활성화)
    ↓
[5. Ensemble]     최종 점수 = Base + Boost + HMM - Penalty
    ↓
[6. Softmax]      Temperature 적용 확률 변환
    ↓
[7. Sampling]     Monte Carlo 10만개 가중 비복원 추출 (Gumbel Top-K)
    ↓
[8. Filter]       기본 필터 + 도박사의 오류 필터
    ↓
[9. Rank]         조합 점수 기반 랭킹
    ↓
[10. Diversity]   번호 겹침 최소화 다양성 선택 → 최종 5게임
```

### 1. Base Score (`calculator.py`)

**Beta-Bernoulli 모델**로 각 번호의 출현 확률 계산.

```
posterior_alpha = prior_alpha(1) + 나온 횟수
posterior_beta  = prior_beta(1)  + 안 나온 횟수
Base Score = alpha / (alpha + beta)
```

- `prior_alpha=1, prior_beta=1`: 라플라스 스무딩
- `recent_n=50`: 최근 50회차 데이터 사용
- 범위: 약 0.08 ~ 0.21

### 2. Boost (`booster.py`)

휴리스틱 기반 가중치 부여:

| Boost 유형 | 조건 | 가중치 |
|-----------|------|-------|
| **Hot** | 최근 5회 중 2회 이상 출현 | `+0.40` |
| **Cold** | 최근 10회 미출현 | `+0.15` |
| **Neighbor** | Hot 번호의 ±1 이웃 | `+0.30` |
| **Carryover** | 직전 회차 당첨번호 | `+0.40` |
| **Carryover2** | 2회차 전 당첨번호 | `+0.40` |
| **Reverse** | 46 - Hot 번호 | `+0.10` |

### 3. HMM Scorer (`hmm_scorer.py`)

**Hidden Markov Model**로 각 번호의 상태 추론:

- 각 번호(1-45)마다 출현/미출현 이진 시퀀스 생성
- 3-state HMM 학습 (최근 100회차):
  - **Hot state** → `+0.30` boost
  - **Neutral state** → `0`
  - **Cold state** → `+0.15` boost (기회로 간주)
- 상태 분류: emission probability 기반 (방출 확률이 높으면 Hot)

### 4. Penalty (`penalizer.py`)

Poisson/Markov 기반 페널티 (현재 비활성화):

| Penalty 유형 | 설명 | 현재 λ |
|-------------|------|--------|
| **Poisson** | 빈도 초과분 페널티 | `0.0` (OFF) |
| **Markov** | 전이 확률 기반 페널티 | `0.0` (OFF) |

### 5. Score Ensemble (`calculator.py`)

```
최종 점수 = max(0, Base + Boost + HMM - Penalty)
```

### 6. Probability Normalizer (`normalizer.py`)

```
확률 = Softmax(최종 점수 / temperature)
```

- `temperature=0.5`: 점수 차이 증폭 (낮을수록 확률 편차 커짐)
- `min_prob_floor=0.005`: 모든 번호 최소 0.5% 확률 보장

### 7. Monte Carlo Sampler (`monte_carlo.py`)

- **Gumbel Top-K**: 가중 비복원 추출 벡터화 구현
- 10만 개 조합을 청크(2만)로 나눠 생성
- 확률 높은 번호가 더 자주 조합에 포함됨

### 8. Filters

#### 기본 필터 (FilterPipeline)

| 필터 | 조건 |
|------|------|
| **SumFilter** | 합계 100~175 |
| **ACFilter** | AC값 7 이상 |
| **ZoneFilter** | 10단위 구간 분포 |
| **TailFilter** | 끝자리 분포 |
| **OddEvenFilter** | 홀짝 비율 |
| **HighLowFilter** | 고저 비율 |
| **HistoryFilter** | 과거 당첨 조합 제외 |

#### 도박사의 오류 필터 (Custom)

| 필터 | 설명 |
|------|------|
| **희귀 쌍** | 22개 희귀 쌍 조합 제외 |
| **제외 번호** | 특정 번호(`{8}`) 포함 조합 제외 |
| **이월수 제한** | 직전+2주전 이월수 최대 2개까지 허용 |

### 9-10. Ranking & Diversity

- **CombinationRanker**: 조합 내 번호 확률 합산으로 점수 산정
- **DiversitySelector**: 번호 겹침 최소화 (max overlap 제한)

---

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
    loader.py               # CSV 데이터 로더
    fetcher.py               # 동행복권 API 데이터 수집
  engine/
    score/
      calculator.py          # Base Score (Beta-Bernoulli)
      booster.py             # Hot/Cold/Carryover 휴리스틱 부스트
      hmm_scorer.py          # HMM 상태 추론 스코어러
      penalizer.py           # Poisson/Markov 페널티
      normalizer.py          # Softmax + Floor 확률 정규화
    sampler/
      monte_carlo.py         # Gumbel Top-K 몬테카를로 샘플링
    filters/
      sum_filter.py          # 합계 범위 필터
      ac_filter.py           # AC값 필터
      zone_filter.py         # 10단위 구간 필터
      tail_filter.py         # 끝자리 필터
      odd_even_filter.py     # 홀짝 비율 필터
      high_low_filter.py     # 고저 비율 필터
      history_filter.py      # 과거 당첨 필터
      pipeline.py            # 필터 파이프라인
    ranker/
      scorer.py              # 조합 랭킹
      diversity.py           # 다양성 선택
  mvp/
    api.py
    service.py
    static/index.html

data/
  model.json

recommend.py                 # CLI 추천기
backtest.py                  # CLI 백테스트
history.csv
index.html
vercel.json
```

---

## 빠른 시작 (로컬)

### 1) 의존성 설치

```bash
uv sync
```

### 2) 테스트

```bash
uv run pytest -q
```

### 3) 추천 번호 생성

```bash
uv run recommend.py
```

출력 예시:
```
🔥 Hot: [1, 17, 27, 38, 42]
❄️  Cold: [11, 13, 14, 15, 19, 25, 34, 43]
🔄 Carryover (직전): [2, 17, 20, 35, 37, 39]
🔄 Carryover2 (2주전): [6, 27, 30, 36, 38, 42]
🧠 HMM Hot: [2, 3, 4, 5, 6, 9, 10, 13, 14, 15]...
🧠 HMM Cold: [8, 11, 12, 16, 21, 22, 23, 41]

🎯 1210회 추천 번호 (5게임)
  1게임: [ 1, 13, 18, 34, 36, 38]
  ...
```

### 4) 백테스트

```bash
uv run backtest.py --round 1100
```

- 1~N 회차 데이터로 학습 후 N+1 회차 예측
- 미래 데이터 유출 없음 (시점 분리)
- Seed 기본 랜덤 (고정: `--seed 42`)

### 5) 데이터 업데이트 (증분)

```bash
uv run python scripts/update_history_csv.py --csv history.csv --workers 8
```

### 6) 프론트 모델 생성

```bash
uv run python scripts/build_frontend_model.py --history-csv history.csv --output data/model.json
```

---

## GitHub Actions (주간 자동 갱신)

워크플로: `.github/workflows/lotto-history-update.yml`

스케줄:
- 매주 토요일(KST 저녁) 3회 재시도 창

동작:
1. `history.csv` 증분 업데이트
2. `data/model.json` 재생성 (preset별 100k)
3. 두 파일 중 변경이 있을 때만 커밋

---

## 프론트 동작 상세

- 버튼 클릭 시 메인 스레드는 즉시 반환, 계산은 Worker에서 수행
- Worker가 확률 샘플링 → 필터 → 점수화 → 다양성 선택
- 최근 추천 재노출 완화: `localStorage` 활용
- 결과 없을 때 fallback 경로로 무한 대기 방지

---

## 현재 파라미터 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Base prior | `alpha=1, beta=1` | 균등 사전 분포 |
| Base recent_n | `50` | 최근 50회차 |
| Hot threshold | `2회/5회` | 5회 중 2회 이상 출현 |
| Cold window | `10` | 10회 미출현 |
| Carryover weight | `0.40` | 이월수 가중치 |
| HMM hot_boost | `0.30` | HMM Hot 부스트 |
| HMM cold_boost | `0.15` | HMM Cold 부스트 |
| HMM window | `100` | 최근 100회차 학습 |
| Temperature | `0.5` | Softmax 온도 |
| Min prob floor | `0.005` | 최소 확률 바닥 |
| Sample size | `100,000` | 조합 생성 수 |
| Max carryover | `2` | 이월수 최대 허용 수 |

---

## 환경변수

`.env.example` 참고

주요 항목:
- `DONATE_URL`, `PUBLIC_BASE_URL`, `MODEL_URL`
- `GOOGLE_SITE_VERIFICATION`, `NAVER_SITE_VERIFICATION`
- `LOTTO_HISTORY_CSV`, `FRONTEND_MODEL_PATH`

---

## 배포

### Vercel

- 이 저장소 루트를 배포 대상으로 설정
- 정적 `index.html` + `data/model.json` + `assets/recommend-worker.js` 배포
- GitHub push 시 자동 배포

---

## 주의사항

- 이 프로젝트는 예측 서비스가 아니라 **실험/참고 도구**입니다.
- 실제 구매 판단과 결과 책임은 사용자에게 있습니다.
- 로또는 **독립시행**입니다. 과거 데이터 분석은 참고 용도로만 활용하세요.

## 라이선스

MIT
