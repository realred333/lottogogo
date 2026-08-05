# LottoGoGo v2

로또 번호 **선택 보조용 확률 실험 프로젝트**입니다.
당첨을 보장하지 않으며, 통계 실험 결과를 참고 정보로 제공합니다.
로또는 독립시행이고, 이 저장소의 어떤 코드도 그 사실을 바꾸지 않습니다. 여기 있는 것은 "무근거 선택을 줄이는 필터"와 "그 필터가 실제로 도움이 되는지 측정하는 도구"입니다.

---

## 목차

1. [운영 구조 한눈에 보기](#1-운영-구조-한눈에-보기)
2. [저장소 지도](#2-저장소-지도)
3. [확률 엔진 상세](#3-확률-엔진-상세)
4. [필터 상세](#4-필터-상세)
5. [랭킹과 다양성 선택](#5-랭킹과-다양성-선택)
6. [실행 경로 3종 비교 (중요)](#6-실행-경로-3종-비교-중요)
7. [프론트엔드 동작 상세](#7-프론트엔드-동작-상세)
8. [튜닝: GA / XGBoost / 페널티 그리드](#8-튜닝-ga--xgboost--페널티-그리드)
9. [백테스트 도구](#9-백테스트-도구)
10. [풀 제외(pool exclusion) 실험 트랙](#10-풀-제외pool-exclusion-실험-트랙)
11. [데이터 파이프라인과 GitHub Actions](#11-데이터-파이프라인과-github-actions)
12. [로컬 개발](#12-로컬-개발)
13. [배포](#13-배포)
14. [현재 상태와 알려진 이슈](#14-현재-상태와-알려진-이슈)

---

## 1. 운영 구조 한눈에 보기

`Render` 같은 상시 백엔드에 의존하지 않습니다. **주간 배치로 모델 JSON을 굽고, 브라우저가 그 JSON으로 직접 계산**하는 구조입니다.

```
┌─ GitHub Actions (주 1회, 토요일) ────────────────────────┐
│  scripts/update_history_csv.py    → history.csv 증분 갱신  │
│  scripts/build_frontend_model.py  → data/model_ga.json    │
│                                    data/model_xgb.json    │
│                                    data/model.json (레거시)│
│  변경이 있을 때만 커밋 → Vercel 자동 배포                    │
└───────────────────────────────────────────────────────────┘
                            ↓
┌─ GitHub Actions (주 1회, 월요일) ────────────────────────┐
│  lottogogo.tuning.ga_optimizer → data/optimized_weights.json│
│  (다음 주 모델 빌드 때 이 가중치가 반영됨)                   │
└───────────────────────────────────────────────────────────┘
                            ↓
┌─ 브라우저 (Vercel 정적 호스팅) ─────────────────────────┐
│  index.html → model_ga.json 또는 model_xgb.json fetch     │
│             → assets/recommend-worker.js (Web Worker)     │
│             → 가중 샘플링 → 필터 → 점수 → 다양성 → 5/10게임 │
│  서버 계산 없음. 요청당 지연 없음.                          │
└───────────────────────────────────────────────────────────┘
```

로컬에는 이와 **별개로** CLI 추천기(`recommend.py`, `recommend_ml.py`, `recommend_pool.py`)가 있습니다. 프론트와 CLI는 같은 엔진 모듈을 쓰지만 **파이프라인 세부가 다릅니다.** → [6장](#6-실행-경로-3종-비교-중요) 필독.

---

## 2. 저장소 지도

```text
├─ src/lottogogo/                엔진 라이브러리 (설치 가능한 패키지)
│  ├─ config/
│  │  ├─ schema.py               EngineConfig (pydantic, 기본값 + 범위 검증)
│  │  └─ loader.py               YAML/JSON 설정 로더
│  ├─ data/
│  │  ├─ loader.py               history.csv 로드 + 스키마/범위/중복 검증
│  │  └─ fetcher.py              동행복권 API 수집, 증분 갱신, 최신 회차 탐색
│  ├─ engine/
│  │  ├─ score/
│  │  │  ├─ calculator.py        BaseScoreCalculator(베타-베르누이) + ScoreEnsembler
│  │  │  ├─ booster.py           Hot/Cold/Neighbor/Carryover/Reverse 휴리스틱
│  │  │  ├─ hmm_scorer.py        번호별 3-state HMM 상태 추론
│  │  │  ├─ penalizer.py         Poisson/Markov 페널티
│  │  │  └─ normalizer.py        Softmax + 확률 하한(floor)
│  │  ├─ sampler/monte_carlo.py  Gumbel Top-K 벡터화 가중 비복원 추출
│  │  ├─ filters/                조합 필터 8종 + 파이프라인
│  │  ├─ ranker/                 조합 점수 랭킹 + 다양성 선택
│  │  └─ backtester/             워크포워드 백테스터 + 지표 + 리포트 + 랜덤 베이스라인
│  ├─ tuning/
│  │  ├─ fitness.py              GA 적합도(hit@K, mean_rank) + 가중치 탐색 범위
│  │  ├─ ga_optimizer.py         DEAP 기반 GA + 체크포인트 + 수렴 그래프
│  │  ├─ feature_builder.py      XGBoost용 28개 특징 추출
│  │  ├─ xgb_ranker.py           XGBoost 학습/평가 + Optuna 튜닝
│  │  └─ penalty_search.py       Poisson/Markov λ 그리드 서치(멀티프로세스)
│  └─ mvp/                       FastAPI 앱 (현재 배포에는 미사용, 로컬 실행용)
│     ├─ api.py                  /, /api/recommend, /api/warmup, /api/pool-status
│     ├─ service.py              프리셋 A/B 정의 + 결과 풀 캐싱
│     └─ static/index.html       프론트 원본 템플릿 (플레이스홀더 포함)
│
├─ recommend.py                  CLI 추천기 — GA/엔진 기반
├─ recommend_ml.py               CLI 추천기 — XGBoost 기반
├─ recommend_pool.py             CLI 추천기 — 풀 제외 실험 (별도 트랙)
├─ backtest.py                   단일 회차 백테스트 (엔진 기반)
├─ backtest_compare.py           단일 회차 GA vs XGBoost 비교
├─ backtest_multi.py             다회차 GA vs XGBoost 평균 비교
├─ backtest_pool.py              풀 제외 실험 백테스트
│
├─ scripts/
│  ├─ update_history_csv.py      history.csv 증분 갱신 CLI
│  ├─ build_frontend_model.py    model_ga.json / model_xgb.json 빌드
│  └─ export_vercel_index.sh     mvp/static/index.html → 루트 index.html 렌더
│
├─ assets/recommend-worker.js    브라우저 Web Worker 추천 엔진 (JS 재구현)
├─ index.html                    배포되는 정적 페이지 (플레이스홀더 치환 완료본)
├─ api/robots.js, api/sitemap.js Vercel 서버리스 함수 (robots.txt / sitemap.xml)
├─ vercel.json                   위 두 함수로의 rewrite 규칙
│
├─ history.csv                   1회~최신 회차 당첨번호 (round,n1..n6,bonus)
├─ data/
│  ├─ model_ga.json              프론트용 GA/엔진 모델 (기본 선택)
│  ├─ model_xgb.json             프론트용 XGBoost 모델
│  ├─ model.json                 레거시 호환 (GA 모델 사본)
│  ├─ optimized_weights.json     GA 최적화 결과 (Actions가 매주 갱신)
│  ├─ optimized_weights_no_hmm.json  로컬 run_ga.bat 결과물
│  ├─ ga_checkpoint.json         GA 25세대마다 저장되는 체크포인트
│  ├─ fitness_history.png        GA 수렴 그래프
│  ├─ xgb_model.pkl              XGBoost 학습 모델 캐시
│  ├─ xgb_model_metadata.json    캐시 유효성 판정용 (학습 회차/시각/가중치 경로)
│  ├─ xgb_best_params.json       Optuna 튜닝 결과
│  └─ xgb_feature_importance.json 특징 중요도
│
├─ docs/pool_exclusion_experiment_2026-08-01.md  풀 제외 실험 노트
├─ tests/unit/                   pytest 유닛 테스트 19개 파일
├─ about.html / methodology.html / privacy.html / terms.html  정적 문서 페이지
└─ run_ga.bat                    Windows용 GA 실행 배치 (로그: ga_log.txt)
```

---

## 3. 확률 엔진 상세

전체 흐름 (CLI `recommend.py` 기준):

```
[history.csv]
  ↓
[1] Base Score      베타-베르누이 사후 평균 (최근 50회차)
  ↓
[2] Boost           Hot / Cold / Neighbor / Carryover / Carryover2 / Reverse
  ↓
[3] HMM             번호별 3-state HMM → Hot/Cold 상태에 가산
  ↓
[4] Penalty         Poisson / Markov (기본 λ=0 이지만 가중치 파일에서 켜질 수 있음)
  ↓
[5] Ensemble        raw = max(0, Base + Boost + HMM − Penalty)
  ↓
[6] Softmax         prob = softmax(raw / temperature), 이후 최소확률 floor 적용
  ↓
[6b] 고구간 하향     40~45번 확률 ×0.5 후 재정규화   ← CLI/프론트에만 있음
  ↓
[7] Sampling        Gumbel Top-K로 10만 조합 가중 비복원 추출
  ↓
[8] Filter          기본 필터 8종 + 도박사의 오류 필터 3종
  ↓
[9] Rank            조합 점수 = 포함된 번호들의 raw score 합
  ↓
[10] Diversity      번호 겹침 최소화 → 최종 N게임
```

### 3.1 Base Score — `engine/score/calculator.py`

각 번호가 얼마나 자주 나오는지를 베타-베르누이 모델의 사후 평균으로 추정합니다.

```
posterior_alpha = prior_alpha + (해당 번호가 나온 회차 수)
posterior_beta  = prior_beta  + (총 회차 수 − 나온 회차 수)
base_score      = alpha / (alpha + beta)
```

- `prior_alpha = prior_beta = 1.0` → 라플라스 스무딩(균등 사전분포)
- `recent_n = 50` → 최근 50회차만 사용
- 따라서 실제 값은 `(1 + count) / 52`. 45개 번호가 50회차에서 총 300번 등장하므로 count 기댓값은 약 6.7 → 대부분 **0.10 ~ 0.17** 범위에 몰립니다.
- 이 값 자체의 편차는 작습니다. 실질적인 순위 차이는 대부분 [2] Boost 단계에서 생깁니다.

`ScoreEnsembler.combine()`은 `max(minimum_score, base + boost − penalty)`를 계산합니다 (`minimum_score = 0.0`). 즉 **최종 점수는 음수가 되지 않습니다.**

### 3.2 Boost — `engine/score/booster.py`

| Boost | 발동 조건 | 기본 가중치 |
|---|---|---|
| **hot** | 최근 `hot_window=5`회 중 `hot_threshold=2`회 이상 출현 | `+0.40` |
| **cold** | 최근 `cold_window=10`회 동안 **한 번도** 출현하지 않음 | `+0.15` |
| **carryover** | 직전 회차(N−1) 당첨번호 | `+0.40` |
| **carryover2** | 2회차 전(N−2) 당첨번호. **단 직전 회차와 겹치면 미적용** | `+0.40` |
| **neighbor** | 직전 회차 번호의 ±1. **단 직전 회차 번호 자신은 제외** | `+0.30` |
| **reverse** | `46 − (직전 회차 번호)` | `+0.10` |

주의할 점:

- 이 부스트들은 **배타적이지 않습니다.** 한 번호가 hot + carryover + neighbor를 동시에 받으면 `0.40 + 0.40 + 0.30 = 1.10`이 base score(≈0.13)에 더해집니다. 즉 **부스트가 base score를 압도**하는 것이 정상 동작입니다.
- `carryover_weight`의 기본값이 코드 위치마다 다릅니다. `BoostCalculator` 자체 기본값과 `recommend.py`는 `0.40`, `config/schema.py`의 `EngineConfig`와 `penalty_search.py`는 `0.20`입니다.
- 각 번호에 어떤 부스트가 걸렸는지는 `tags` 딕셔너리로 함께 반환되며, 프론트의 배지 라벨("최근 빈도 상승", "직전 회차 연결" 등)이 여기서 나옵니다.

### 3.3 HMM Scorer — `engine/score/hmm_scorer.py`

`hmmlearn.CategoricalHMM`으로 **번호마다 독립적인 3-state HMM을 학습**합니다.

1. 번호 n에 대해 최근 `window=100`회차의 출현 여부를 0/1 시퀀스로 만듭니다.
2. 3-state HMM을 `n_iter=100` EM으로 적합합니다.
3. `emissionprob_[state, 1]`(= 그 상태에서 "출현"을 방출할 확률)로 상태를 정렬합니다.
   - 가장 높은 상태 → **Hot**
   - 가장 낮은 상태 → **Cold**
   - 나머지 → Neutral
4. 시퀀스의 **마지막 시점 상태**를 현재 상태로 보고 부스트를 줍니다.

| 상태 | 부스트 | 해석 |
|---|---|---|
| Hot | `+0.30` | 출현 흐름이 살아있는 구간 |
| Neutral | `0` | — |
| Cold | `+0.15` | "잠복 후 반등" 가정 — 페널티가 아니라 **기회로 취급** |

- 45개 번호 × EM 100회이므로 **느립니다.** 이 때문에 아래처럼 곳곳에서 꺼져 있습니다:
  - `tuning/fitness.py`의 `CachedScoreComputer`는 HMM을 **완전히 건너뜁니다** (주석: `Skip HMM boosts`). 즉 **GA 최적화는 HMM 없는 엔진을 대상으로 진행됩니다.**
  - `scripts/build_frontend_model.py`는 `--use-hmm` 플래그를 줄 때만 켭니다. 현재 GitHub Actions 워크플로는 이 플래그를 **주지 않습니다** → 배포되는 `model_ga.json`에는 HMM 신호가 반영되어 있지 않습니다 (`hmm_hot_numbers`/`hmm_cold_numbers`가 빈 배열).
  - 반대로 `recommend.py`, `backtest.py`, `mvp/service.py`는 HMM을 **켠 채** 동작합니다.
- 학습 실패 시 조용히 Neutral로 폴백합니다(`except Exception: return None, STATE_NEUTRAL`).

### 3.4 Penalty — `engine/score/penalizer.py`

| 페널티 | 계산 | 비고 |
|---|---|---|
| **Poisson** | 최근 `poisson_window=20`회 출현 수가 기대값(`20×6/45 = 2.67`)을 초과한 만큼 `λ_poisson × 초과분` | 과열 번호 억제 |
| **Markov** | 46×46 전이행렬을 만들고, 직전 회차 번호들에서 해당 번호로 가는 전이확률 평균 × `λ_markov` | 직전 회차 연동 억제 |

- 두 λ 모두 **0.0 ~ 0.5로 제한**되며, 범위를 벗어나면 생성자가 `ValueError`를 던집니다.
- `recommend.py` / `build_frontend_model.py`의 기본값은 **0.0(비활성)** 입니다. 그러나 `--weights`로 GA 결과를 주입하면 **가중치 파일의 값이 그대로 들어갑니다.** 현재 `data/optimized_weights.json`은 `poisson_lambda = 0.343`, `markov_lambda = 0.124`이므로 **가중치를 쓰는 순간 페널티는 켜집니다.**
- 반면 `feature_builder.py`는 항상 `0.5 / 0.3` 하드코딩 값으로 페널티 특징을 만듭니다.
- Markov 전이행렬은 회차 목록 해시 기반으로 캐시됩니다.

### 3.5 확률 변환 — `engine/score/normalizer.py`

```
prob = softmax(raw_score / temperature)
prob = min_prob_floor + (1 − min_prob_floor × 45) × prob   # floor 적용 후 재정규화
```

- `temperature = 0.5` 고정 (`normalizer.DEFAULT_TEMPERATURE`) → 점수 차이를 2배로 증폭. **낮을수록 편차가 커집니다.**
  프리셋 B만 0.65를 씁니다.

> **temperature는 튜닝 대상이 아닙니다.** 예전에는 GA 탐색 공간에 들어 있었는데, GA 적합도는 raw score의 *순위*만 보고 softmax를 아예 거치지 않기 때문에 이 유전자에는 기울기가 없었습니다. 최적화되지 못한 채 표류하다 탐색 하한 `0.1`에 멈췄고, 그 값이 `recommend.py`와 프론트 모델 빌드로 흘러들어가 **한 번호에 샘플링 확률의 31%가 쏠렸습니다**(프리셋 A 기준, 32번). 지금은 탐색 공간에서 제거하고 상수로 고정했으며, 다시 유전자로 추가되면 `test_temperature_is_not_a_tunable_weight`가 실패합니다. 다시 튜닝하려면 적합도 함수가 실제로 샘플링까지 수행하도록 먼저 바꿔야 합니다.
- `min_prob_floor = 0.005` → 모든 번호에 최소 0.5% 확률 보장. floor는 softmax **이후** 선형 혼합 방식으로 적용되므로 합은 항상 1입니다.
- `min_prob_floor × 45 >= 1.0`이면 `ValueError`.

### 3.6 40~45번 확률 하향 (문서화되지 않았던 동작)

`recommend.py`, `recommend_ml.py`, `assets/recommend-worker.js` 세 곳 모두, **정규화가 끝난 뒤** 40~45번의 확률에 `×0.5`를 곱하고 전체를 다시 정규화합니다.

```python
HIGH_ZONE_PENALTY = 0.5
for num in range(40, 46):
    probs[num] *= HIGH_ZONE_PENALTY
probs = {n: p / sum(probs.values()) for n, p in probs.items()}
```

- 엔진 모듈이 아니라 **호출부에 하드코딩**되어 있습니다. 설정으로 뺄 수 없습니다.
- `backtest.py`에는 **없습니다.** 따라서 `backtest.py`의 결과는 `recommend.py`의 성능을 그대로 대변하지 않습니다.

### 3.7 몬테카를로 샘플러 — `engine/sampler/monte_carlo.py`

가중 비복원 추출을 **Gumbel Top-K 트릭**으로 벡터화했습니다.

```
score_i = log(p_i) + Gumbel(0, 1)
→ 상위 6개 index 선택 == 확률 p로 비복원 추출한 것과 동일 분포
```

- `sample_size = 100,000`, `chunk_size = 20,000` → 5개 청크로 나눠 생성(메모리 관리).
- `np.argpartition`으로 상위 6개만 뽑으므로 전체 정렬 없이 O(45) 수준.
- 결과 조합은 **중복될 수 있습니다.** 이후 `dict.fromkeys()`로 중복 제거합니다.

---

## 4. 필터 상세

모든 필터는 `BaseFilter`를 상속하고 `FilterDecision(passed, reason)`을 반환합니다.
`FilterPipeline`은 **순서대로 평가하고 첫 실패에서 즉시 탈락(early exit)** 시키며, 필터별 탈락 횟수를 `rejection_counts`에 누적합니다.

### 4.1 기본 필터 8종 — `engine/filters/`

| 클래스 | `name` | 기본 조건 | 설명 |
|---|---|---|---|
| `SumFilter` | `sum` | 합계 100~175 | 6개 합계의 현실적 구간 |
| `ACFilter` | `ac` | AC ≥ 7 | AC = (서로 다른 차이값 개수) − 5. 번호가 고르게 퍼졌는지의 지표 (최대 10) |
| `ZoneFilter` | `zone` | 구간당 ≤ 3개 | 구간은 **1–11 / 12–22 / 23–33 / 34–45** (10단위가 아니라 4등분) |
| `TailFilter` | `ending` | 같은 끝자리 ≤ 2개 | `number % 10` 기준 |
| `OddEvenFilter` | `odd_even` | 홀수 2~4개 | 6:0, 5:1, 1:5, 0:6 배제 |
| `HighLowFilter` | `high_low` | 고구간(23 이상) 2~4개 | `high_start = 23` |
| `HistoryFilter` | `history` | 과거 당첨과 5개 이상 겹치면 탈락 | 아래 참고 |
| `ArithmeticProgressionFilter` | `ap` | 5개 이상이 등차수열이면 탈락 | 예: `[1,3,5,7,9,43]` 탈락 |

`HistoryFilter`의 `match_threshold=5` 동작을 정확히 쓰면:

- 초기화 시 모든 과거 당첨조합의 **5개 부분집합**(1회차당 6개)을 전부 set에 넣습니다.
- 후보 조합의 5개 부분집합 중 하나라도 그 set에 있으면 탈락합니다.
- `match_threshold=6`이면 완전 일치만 탈락, 그 외 값이면 비트마스크 `popcount`로 겹침 수를 셉니다.
- 1234회차 기준 5-부분집합은 약 7,400개이며, 이를 프론트 모델 JSON에도 통째로 실어 보냅니다(`history.five_subset_keys`).

### 4.2 "도박사의 오류" 필터 3종 (호출부 하드코딩)

엔진 모듈이 아니라 `recommend.py` / `recommend_ml.py` / `mvp/service.py` / 워커에 각각 정의되어 있습니다.

| 필터 | 내용 |
|---|---|
| **희귀 쌍 제외** | 역대 동반 출현이 7~10회에 그친 22개 쌍 (`(8,12)` 7회가 최저, `(8,26)`, `(24,43)`, `(26,32)` 8회 …) 중 하나라도 포함되면 탈락 |
| **제외 번호** | `{8}` 포함 조합 탈락 (8이 희귀 쌍에 가장 많이 등장) |
| **이월수 제한** | `carryover ∪ carryover2`(직전 + 2주 전, 최대 12개) 중 조합에 포함된 개수가 **2개 초과면 탈락** |

> 이름 그대로 이 세 가지는 **도박사의 오류(gambler's fallacy)를 의도적으로 코드화한 것**입니다. 독립시행에서 "최근 안 나온 쌍"이 앞으로도 안 나올 이유는 없습니다. 통계적 근거가 아니라 "체감상 납득 가능한 조합"을 만들기 위한 장치로 이해하세요.

---

## 5. 랭킹과 다양성 선택

### `CombinationRanker` — `engine/ranker/scorer.py`

```
combo_score = Σ raw_scores[number]   (조합에 포함된 6개 번호)
```
내림차순 정렬 후 `top_k`로 자릅니다. 호출부마다 `top_k`가 다릅니다:

| 호출부 | `top_k` |
|---|---|
| `recommend.py`, `recommend_ml.py`, `backtest.py` | 없음 (전체) |
| `mvp/service.py` 프리셋 A / B | 180 / 220 |
| `penalty_search.py` | 300 |

조합 점수가 **번호 점수의 단순 합**이라는 점이 중요합니다. 제약이 없으면 최고점 번호를 포함한 조합이 랭킹 상위를 통째로 차지하고, 그 번호가 모든 게임에 들어갑니다.

### `DiversitySelector` — `engine/ranker/diversity.py`

랭킹 상위부터 훑으면서 두 가지 제약을 겁니다.

| 제약 | 내용 |
|---|---|
| `max_overlap` | 이미 선택된 조합과 이 값을 초과해 겹치면 탈락 (3이면 4개 이상 겹칠 때) |
| `max_number_frequency` | **한 번호가 등장할 수 있는 최대 게임 수** |

`max_overlap`만으로는 번호 단위 쏠림을 막지 못합니다. 32번 하나만 공유하는 조합들은 겹침이 1이라 전부 통과하기 때문입니다. 그래서 번호별 등장 횟수 상한이 따로 필요합니다.

기본 상한은 `default_number_frequency(games, ratio)` = `max(2, ceil(games × ratio))`:

| 게임 수 | 프리셋 A (ratio 0.4) | 프리셋 B (ratio 0.6) |
|---|---|---|
| 5게임 | 2 | 3 |
| 10게임 | 4 | 6 |

### `select_with_relaxation()` — 조건 완화 사다리

제약이 빡빡해 게임 수를 못 채우면 순서대로 완화합니다.

```
빈도 상한 +1씩 증가 → 빈도 상한 해제 → max_overlap 4 → 5 → 6
```

끝까지 못 채우면 **가장 많이 채운 결과**를 반환하므로 빈 리스트가 나오지 않습니다.
`recommend.py` / `recommend_ml.py` / `backtest.py` / `mvp/service.py`가 모두 이 함수를 씁니다. (`penalty_search.py`만 랭킹 → 샘플 → 랜덤 베이스라인 순의 자체 폴백을 유지합니다.)

---

## 6. 실행 경로 3종 비교 (중요)

같은 엔진 모듈을 쓰지만 **조합 방식이 서로 다릅니다.** 이 표를 보지 않으면 "왜 CLI 결과와 웹 결과가 다르지?"에 답할 수 없습니다.

| 항목 | `recommend.py` | `recommend_ml.py` | 프론트 워커 |
|---|---|---|---|
| 점수 원천 | 엔진 (Base+Boost+HMM−Penalty) | XGBoost `predict_proba` | 사전 계산된 모델 JSON의 weights |
| HMM | **켬** | 특징으로만 사용 | 배포 모델은 **꺼짐** (`--use-hmm` 미지정) |
| GA 가중치 주입 | `--weights` 지원 | `--weights` 지원 (특징 추출에 적용) | 빌드 시 반영 |
| Softmax temperature | 0.5 고정 | 0.5 고정 | 프리셋 A 0.5 / B 0.65 |
| 40–45번 ×0.5 | 적용 | 적용 | 적용 |
| 샘플링 | Gumbel Top-K 10만 개 일괄 | 동일 | **JS 루프 기각 샘플링** (max_attempts까지) |
| AP 필터 | 있음 | 있음 | 있음 |
| 도박사 필터 | 희귀쌍 + `{8}` + 이월수≤2 | 동일 | 프리셋 A만 (B는 전부 해제) |
| 랭킹 `top_k` | 전체 | 전체 | 프리셋별 180/220 |
| 번호 빈도 상한 | `ceil(games×0.4)` | 동일 | 프리셋 A 0.4 / B 0.6 |
| 게임 수 | 임의 | 임의 | 5 또는 10 |

### `backtest.py`는 별도 경로가 아닙니다

`backtest.py`는 파이프라인을 자체 구현하지 않고 **`recommend.generate_recommendations()`를 그대로 호출합니다.** 따라서 백테스트는 정의상 `recommend.py`와 동일한 조건으로 돌아갑니다.

> 과거에는 `backtest.py`가 파이프라인을 복사해 갖고 있었고, AP 필터 · 40–45번 하향 · 가중치 주입 · 중복 제거 · 다양성 완화가 빠진 채 조용히 갈라져 있었습니다. 백테스트 점수가 좋아져도 실제 추천이 좋아진 것인지 알 수 없는 상태였습니다.

**파이프라인을 바꿀 때는 반드시 `recommend.generate_recommendations()` 안에서 바꾸세요.** 그래야 두 경로가 다시 갈라지지 않습니다.

---

## 7. 프론트엔드 동작 상세

### 7.1 모델 JSON 스키마 — `scripts/build_frontend_model.py`가 생성

```jsonc
{
  "schema_version": 1,
  "generated_at_utc": "2026-...Z",
  "source": { "history_csv": "history.csv", "rows": 1234, "latest_round": 1234 },
  "numbers": [1, ..., 45],
  "raw_scores": [45개 float],                  // 조합 점수 계산용
  "boost_tags_by_number": [45개 string[]],     // ["hot","carryover"] 등
  "tag_labels": { "hot": "최근 빈도 상승", ... },
  "signals": { "hot_numbers": [], "cold_numbers": [], "hmm_hot_numbers": [], "hmm_cold_numbers": [] },
  "carryover_numbers": [직전+2주전 번호],
  "rare_pairs": [[8,12], ...],
  "history": {
    "draws": [[1,15,19,31,35,43], ...],
    "exact_keys": ["1-15-19-31-35-43", ...],
    "five_subset_keys": ["1-15-19-31-35", ...],   // 약 7천 개
    "match_threshold": 5
  },
  "presets": {
    "A": {
      "filters":  { "min_sum":100, "max_sum":175, "min_ac":7, "max_per_zone":3, ... },
      "special":  { "rare_pair_filter": true, "excluded_numbers": [8], "max_carryover_in_combo": 2 },
      "ranking":  { "top_k":180, "max_overlap":3, "percentile_bias":-5, "number_frequency_ratio":0.4 },
      "sampling": {
        "sample_size": 100000,
        "max_attempts": 추정치,                 // 수용률로부터 역산
        "blend_base_weight": 0.65,
        "weights": [45개],                      // 브라우저가 실제로 쓰는 확률
        "base_probabilities": [45개]
      },
      "reasons": ["AC>=7", "sum 100-175", ...],
      "monte_carlo": { "seed":..., "acceptance_rate":..., "score_quantiles":{...}, "top_numbers":[...] }
    },
    "B": { ... }
  }
}
```

**`sampling.weights`는 두 분포의 혼합**입니다 (`--base-weight`, 기본 0.65):

```
weights = 0.65 × (softmax 기반 확률) + 0.35 × (실제로 필터를 통과한 조합에서의 번호 등장 빈도)
```

즉 빌드 스크립트가 서버에서 10만 개를 미리 돌려보고, "필터를 통과하기 쉬운 번호"에 가중치를 얹어서 브라우저에 넘깁니다. 브라우저의 기각률을 낮추기 위한 장치입니다.

`model_xgb.json`은 GA 모델을 복사한 뒤 `sampling.weights` / `base_probabilities`만 XGBoost 예측 확률로 교체합니다. **나머지 필드(raw_scores, 필터, 태그)는 GA 모델과 동일합니다.** 따라서 조합 랭킹 점수는 두 모델이 같고, 차이는 샘플링 확률에서만 발생합니다.

### 7.2 프리셋 A / B

| 항목 | A (기본, 보수적) | B (완화) |
|---|---|---|
| 합계 | 100–175 | 90–185 |
| AC | ≥ 7 | ≥ 5 |
| 구간당 최대 | 3 | 4 |
| 같은 끝자리 | ≤ 2 | ≤ 3 |
| 홀수 | 2–4 | 1–5 |
| 고구간(23+) | 2–4 | 1–5 |
| temperature | 0.5 | 0.65 |
| min_prob_floor | 0.005 | 0.003 |
| 희귀 쌍 필터 | 켬 | **끔** |
| 제외 번호 | `{8}` | **없음** |
| 이월수 최대 | 2 | 3 |
| `top_k` / `max_overlap` | 180 / 3 | 220 / 4 |
| 번호 빈도 비율 | 0.4 (5게임 중 2) | 0.6 (5게임 중 3) |
| `percentile_bias` | −5 | +8 |

`percentile_bias`는 UI에 표시되는 "상위 N%" 배지를 프리셋 성격에 맞게 보정하는 값입니다(A는 더 좋아 보이게, B는 더 느슨하게).

### 7.3 Web Worker 알고리즘 — `assets/recommend-worker.js`

Python 쪽과 **알고리즘이 다릅니다.** Gumbel Top-K가 아니라 단순 기각 샘플링입니다.

```
1. weights 로드 → 40~45번 ×0.5 → 재정규화
2. targetCandidates = max(350, games × 70)
3. while (attempts < max_attempts && 후보 < targetCandidates):
     - 가중 비복원 추출로 6개 뽑기 (룰렛 휠 방식, 매 픽마다 배열에서 제거)
     - 이미 본 조합 / localStorage 최근 조합이면 skip
     - 필터 전부 통과해야 후보에 추가
4. 후보가 games보다 적으면: 필터를 "과거 당첨 중복"만 남기고 전부 풀어 재시도
5. 점수 내림차순 정렬 → DiversitySelector와 동일한 겹침 규칙 → 상위 N게임
```

- **시드가 없습니다.** `Math.random()`을 쓰므로 브라우저에서는 매번 결과가 달라집니다.
- 최근 노출된 조합은 `localStorage`(`lottogogo_recent_combos_v1`, 최대 120개)에 저장되어 다음 생성에서 제외됩니다.
- 후보가 부족할 때의 폴백(4단계)이 있어 무한 대기에 빠지지 않습니다.
- 모델 전환(GA ↔ XGBoost) 시 `modelPromise`를 비우고 워커를 재초기화합니다.

---

## 8. 튜닝: GA / XGBoost / 페널티 그리드

### 8.1 GA 가중치 최적화 — `lottogogo.tuning.ga_optimizer`

**탐색 공간은 기본 7차원입니다** (`--use-hmm`을 주면 HMM 2개를 추가해 9차원).

| 유전자 | 범위 | 7D 기본 |
|---|---|---|
| `hot_weight` | 0.0 – 1.0 | ✅ |
| `cold_weight` | 0.0 – 0.5 | ✅ |
| `neighbor_weight` | 0.0 – 1.0 | ✅ |
| `carryover_weight` | 0.0 – 1.0 | ✅ |
| `reverse_weight` | 0.0 – 0.5 | ✅ |
| `poisson_lambda` | 0.0 – 0.5 | ✅ |
| `markov_lambda` | 0.0 – 0.5 | ✅ |
| `hmm_hot_boost` | 0.0 – 1.0 | `--use-hmm` 시에만 |
| `hmm_cold_boost` | 0.0 – 0.5 | `--use-hmm` 시에만 |

`temperature`는 의도적으로 제외되어 있습니다. 이유는 [3.5절](#35-확률-변환--enginescorenormalizerpy) 참조.

GA 연산자 (`GAConfig`):

- 선택: 토너먼트 (`tournament_size=3`)
- 교차: `cxBlend(alpha=0.5)`, `crossover_rate=0.8`
- 변이: 가우시안, `mutation_rate=0.15`, `indpb=0.2`,
  **적응형 σ**: 세대 진행에 따라 `0.1 → 0.02`로 선형 감소
- 엘리트 보존: 상위 5개
- 매 세대 후 모든 유전자를 범위로 클램프
- 25세대마다 `data/ga_checkpoint.json`에 저장, 재시작 시 자동 이어받기
- 병렬화는 `ThreadPoolExecutor`(`--jobs`). GIL이 있지만 numpy/pandas 구간에서 일부 해제됩니다.

**적합도 함수** (`tuning/fitness.py`):

```
train_fitness = mean(hit@15)  over 학습 구간 마지막 20회차
val_fitness   = mean(hit@15)  over 검증 구간 전체
rank_bonus    = (45 − mean_rank) / 45
combined      = 0.4 × train_fitness + 0.5 × val_fitness + 0.1 × rank_bonus
```

- `hit@K` = 점수 상위 K개 번호 중 실제 당첨번호가 몇 개 들어있는지 (높을수록 좋음)
- `mean_rank` = 실제 당첨번호 6개의 평균 순위 (낮을수록 좋음, 랜덤 기대값 23)
- 랜덤 기대값: `hit@15 = 15×6/45 = 2.0`, `hit@20 = 2.667`
- 각 검증 회차마다 **그 회차 이전 데이터만** 사용합니다(시점 분리, 미래 정보 유출 없음).
- 평가 시 `CachedScoreComputer`가 학습 데이터 해시로 계산 결과를 캐시하며, **HMM은 건너뜁니다.**

실행:

```bash
uv run python -m lottogogo.tuning.ga_optimizer \
  --csv history.csv \
  --train-end 900 --val-end 1100 \
  --population 100 --generations 200 \
  --jobs 4 \
  --output data/optimized_weights.json \
  --plot data/fitness_history.png
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--csv` | `history.csv` | 과거 추첨 데이터 |
| `--train-end` | `900` | 학습 마지막 회차 |
| `--val-end` | `1100` | 검증 마지막 회차 |
| `--population` | `100` | GA 개체 수 (CLI 기본값. `GAConfig` 자체 기본은 50) |
| `--generations` | `200` | 세대 수 |
| `--jobs` | `1` | 병렬 평가 스레드 수 |
| `--seed` | `42` | 랜덤 시드 |
| `--checkpoint` | `data/ga_checkpoint.json` | 25세대마다 저장 |
| `--output` | `data/optimized_weights.json` | 결과 JSON |
| `--plot` | `data/fitness_history.png` | 수렴 그래프 |
| `--cycle-label` | `ga-weight-optimization-20260215` | 결과에 기록될 실행 라벨 |
| `--use-hmm` | off | HMM 가중치를 탐색 공간에 포함 (10D) |
| `--quiet` | off | 진행 출력 최소화 |

Windows에서는 `run_ga.bat`으로 백그라운드 실행 + `ga_log.txt` 기록이 가능합니다. 실시간 확인:

```powershell
Get-Content .\ga_log.txt -Wait -Tail 50
```

**현재 결과 해석 (`data/optimized_weights.json`, 2026-07-27 갱신):**

```
hot 0.730 / cold 0.367 / neighbor 0.850 / carryover 0.223 / reverse 0.436
poisson_λ 0.343 / markov_λ 0.124 / temperature 0.100

hit@15 = 2.085   (랜덤 2.000)
hit@20 = 2.775   (랜덤 2.667)
mean_rank = 22.61 (랜덤 22.5)
```

솔직하게 읽으면: **랜덤 대비 개선폭이 거의 없습니다.** `hit@15` +0.085, `mean_rank`는 오히려 랜덤보다 나쁩니다. `temperature`가 탐색 범위 하한(0.1)에 붙어 있는 것도 GA가 유의미한 신호를 못 찾고 분포를 극단적으로 뾰족하게 만드는 쪽으로 수렴했다는 신호로 볼 수 있습니다. 이 프로젝트를 예측기가 아니라 **실험 도구**로 부르는 이유입니다.

### 8.2 XGBoost 랭커 — `lottogogo.tuning.xgb_ranker`

각 (회차 × 번호) 쌍을 하나의 샘플로 보고, "이 번호가 이번 회차에 나왔는가(0/1)"를 **이진 분류**합니다.

**28개 특징** (`tuning/feature_builder.py`의 `FEATURE_NAMES`):

| 그룹 | 특징 |
|---|---|
| 엔진 출력 (10) | `base_score`, `hot_boost`, `cold_boost`, `neighbor_boost`, `carryover_boost`, `reverse_boost`, `hmm_hot_prob`, `hmm_cold_prob`, `poisson_penalty`, `markov_penalty` |
| 빈도/간격 (5) | `frequency_recent_10`, `frequency_recent_20`, `frequency_all`, `gap_since_last`, `number_value` |
| 추가 (13) | `streak_length`, `pair_freq_recent`, `zone_low`, `zone_mid`, `zone_high`, `lag_1_appeared`, `lag_2_appeared`, `odd_even`, `frequency_variance`, `rank_percentile`, `recency_score`, `gap_variance`, `cycle_phase` |

- `zone_low/mid/high`는 여기서만 **1–15 / 16–30 / 31–45**로 3등분합니다 (ZoneFilter의 4등분과 다릅니다).
- `cycle_phase = (n % 7) / 7` — "요일 주기 가설"에 기반한 특징. 근거는 없고 실험용입니다.
- 라벨 불균형(6/45)은 `scale_pos_weight = n_neg/n_pos ≈ 6.5`로 보정합니다.
- 특징 추출은 **매 회차마다 그 이전 데이터로만** 수행되므로 유출이 없습니다. 대신 느립니다.

기본 하이퍼파라미터: `max_depth=6`, `learning_rate=0.1`, `n_estimators=200`, `objective=binary:logistic`.
`--tune`을 주면 Optuna + `TimeSeriesSplit` CV(logloss 최소화)로 탐색하고 `data/xgb_best_params.json`에 저장합니다. 현재 저장된 튜닝 결과는 `max_depth=10`, `lr=0.114`, `n_estimators=391`입니다.

```bash
# GA와 성능 비교
uv run python -m lottogogo.tuning.xgb_ranker \
  --csv history.csv --train-end 900 --val-end 1100 \
  --ga-weights data/optimized_weights.json

# Optuna 튜닝 후 비교
uv run python -m lottogogo.tuning.xgb_ranker --tune --n-trials 50 --cv-folds 3
```

실행하면 `data/xgb_feature_importance.json`이 항상 갱신됩니다.

### 8.3 페널티 λ 그리드 서치 — `lottogogo.tuning.penalty_search`

`WalkForwardBacktester`로 `poisson_lambda × markov_lambda` 격자를 전부 돌려 `P(match≥3)` 기준으로 정렬합니다. `ProcessPoolExecutor` 멀티프로세스.

```bash
uv run python -m lottogogo.tuning.penalty_search \
  --history history.csv --start-round 1000 --end-round 1200 \
  --poisson-step 0.1 --markov-step 0.1 --workers 8 \
  --save-json data/penalty_grid.json
```

---

## 9. 백테스트 도구

| 스크립트 | 하는 일 | 지표 |
|---|---|---|
| `backtest.py` | `--round N`까지 학습 → `N+1`회 예측. `--last N`으로 연속 실행 | 일치 개수 + 당첨 등수 + **랜덤 기준선 대비** |
| `backtest_compare.py` | `--round N`을 CSV에서 **제거**하고 GA/XGBoost 양쪽 실행 | 모델별 최고 일치 |
| `backtest_multi.py` | 여러 회차 반복 (`--last 10` 또는 `--rounds ...`) | 모델별 최고/평균 일치, 보너스 포함 게임 수 |
| `backtest_pool.py` | 풀 제외 실험 전용 → [10장](#10-풀-제외pool-exclusion-실험-트랙) | 게임별 일치 개수 |
| `engine/backtester/` | 라이브러리 수준 워크포워드 (튜닝 스크립트가 사용) | `P(≥3)`, `P(≥4)`, 평균/표준편차, 분포 |

```bash
# 단일 회차 (1100회까지 학습 → 1101회 예측)
uv run python backtest.py --round 1100 --games 5 --seed 42

# 최근 10회차 연속 + 랜덤 기준선 비교
uv run python backtest.py --last 10 --games 5 --weights data/optimized_weights.json

uv run python backtest_compare.py --round 1200 --weights data/optimized_weights.json
uv run python backtest_multi.py --last 10 --games 10
```

`backtest.py`는 `--weights`를 지원하므로 **실제 배포 설정 그대로** 백테스트할 수 있습니다.

### 랜덤 기준선 읽는 법

6개를 무작위로 고르면 당첨번호 6개 중 평균 **0.8개**가 맞습니다 (`6 × 6 / 45`). `backtest.py`는 이 값을 항상 같이 출력합니다.

주의할 점은 **표본이 작으면 차이가 전부 노이즈**라는 것입니다. 게임당 일치 수의 표준편차는 약 0.78이므로:

| 게임 수 | 평균의 오차범위(1σ) |
|---|---|
| 5게임 (1회차) | ±0.35 |
| 50게임 (10회차) | ±0.11 |
| 250게임 (50회차) | ±0.05 |

한 회차만 돌려서 나온 ±0.4 차이는 아무 의미가 없습니다. 같은 회차 내 게임들은 서로 상관돼 있으므로 실제 오차범위는 위 값보다 조금 더 큽니다.

### 그 밖의 주의

- `backtest.py`는 `--round N`까지 학습하고 `N+1`을 예측합니다. `backtest_compare.py`/`backtest_multi.py`는 `--round N` **자신을** 가리고 나머지 전체(미래 회차 포함!)로 학습합니다. **의미가 다릅니다.** 후자는 미래 데이터가 학습에 섞이므로 낙관 편향이 있습니다.
- `backtest_compare.py`와 `backtest_multi.py`는 임시 CSV를 `/tmp/`에 씁니다. **Windows에서는 실패하거나 `C:\tmp`에 만들어집니다.**
- 시드 기본값: `backtest.py`는 시간 기반 랜덤, 나머지는 42.
- HMM 때문에 회차당 3~4분 걸립니다. `--last 50`은 몇 시간 단위입니다.

---

## 10. 풀 제외(pool exclusion) 실험 트랙

`recommend.py` / `backtest.py`를 건드리지 않고 별도 파일로 진행 중인 실험입니다. 상세 기록은 `docs/pool_exclusion_experiment_2026-08-01.md`.

**발상 전환**: "최종 5게임이 맞았는가"가 아니라 **"제외하고 남은 풀에 당첨번호 6개 중 몇 개가 살아남았는가(survival)"** 를 1차 지표로 삼습니다. 풀이 당첨번호를 죽이고 있으면 그 뒤 파이프라인은 볼 필요가 없기 때문입니다.

### `recommend_pool.py`

```
1. 1~45에서 hard_exclude 제거          → {2, 5, 8, 9, 22, 32, 39} (수동 고정)
2. 남은 38개에 risk_score 계산
3. 위험도 상위 N개를 자동 제거          → pool_size=30이면 8개
4. 남은 풀에서 균등에 가깝게 광범위 샘플링
5. 기존 필터 파이프라인 + 희귀 쌍 제외 → 랭킹 → 다양성
```

risk_score 구성 (모두 후보 내에서 min-max 정규화 후 가중합):

```
risk_score = 0.40·long_gap + 0.25·silence_low + 0.25·rare_pair + 0.10·overheat

long_gap    = 0.35·avg_gap + 0.25·gap_ge30 + 0.20·max_gap + 0.20·p90_gap
silence_low = 0.55·current_gap + 0.45·(recent50 빈도 낮음)
rare_pair   = 0.50·(pair≤10 개수) + 0.25·(pair≤12 개수) + 0.25·(평균 pair 낮음)
overheat    = 정규화(recent20×0.65 + recent5×1.4)
```

이 실험은 carryover / neighbor / reverse 신호를 **의도적으로 쓰지 않습니다.** 랭킹 점수도 `1 − risk_score`로 약하게만 씁니다.

### 현재까지의 결과 (최근 100회차, 1135–1234, `pool_size=30`)

| 생존 수 | 회차 수 |
|---|---|
| 6개 | 9 |
| 5개 | 28 |
| 4개 | 30 |
| 3개 | 20 |
| 2개 | 10 |
| 1개 | 3 |
| 0개 | 0 |

평균 생존 **3.970 / 6**. 4개 이상 생존 67%, 2개 이하 13%.

당첨번호를 죽인 원인 분석(5·4개 생존 58회차, 총 88개 제거):

- `risk_exclude` 46개 / `hard_exclude` 42개
- risk 제거 46개 중 주요 원인: `long_gap` 42, `silence_low` 3, `rare_pair` 1
- 세부 원인 1위는 `avg_gap` 27건

`avg_gap` 가중치를 0으로 껐을 때 평균 생존은 3.960으로 **오히려 소폭 하락**했습니다. `pool_size=30`이 8개 강제 제거를 요구하는 한, 어떤 항목을 끄든 다른 항목이 그 자리를 채우기 때문입니다.

### 사용법

```bash
# 다음 회차 풀 + 추천
uv run python recommend_pool.py --pool-size 30 --games 5 --show-risk 12

# 특정 회차 백테스트 (해당 회차 이전 데이터만 학습)
uv run python backtest_pool.py --target-round 1234
uv run python backtest_pool.py --last 20 --pool-size 30
```

**알려진 간극**: 실험 문서는 "생존 분포로만 판단하라"고 명시하지만, `backtest_pool.py`는 아직 **최종 게임의 적중 개수만 출력**합니다. 문서에 실린 생존 분포 표를 재현하는 코드 경로가 저장소에 없습니다. 문서의 다음 단계(pool_size 35/32/30/28/25 비교)를 진행하려면 생존 카운트 모드를 먼저 추가해야 합니다.

---

## 11. 데이터 파이프라인과 GitHub Actions

### `history.csv`

```csv
round,n1,n2,n3,n4,n5,n6,bonus
1,10,23,29,33,37,40,16
...
1234,1,15,19,31,35,43,27
```

`LottoHistoryLoader`가 로드 시 검증합니다: 필수 컬럼 존재, 전부 숫자형, `n1~n6`가 1–45 범위, 한 회차 내 번호 중복 없음. `round_id` 컬럼명은 `round`로 자동 정규화됩니다.

### 수집 — `data/fetcher.py`

- 엔드포인트: `https://dhlottery.co.kr/lt645/selectPstLt645Info.do?srchLtEpsd={round}`
- 최신 회차 탐색: 지수 증가 → 이진 탐색
- 신규 회차만 `ThreadPoolExecutor`로 병렬 수집 후 append (`--workers 8`)
- 재시도 2회, 지수 백오프
- **기본적으로 TLS 검증이 꺼져 있습니다** (`verify_ssl=False`). `--verify-ssl`로 켤 수 있습니다.
- CSV가 없거나 비어 있으면 전량 백필을 **거부**합니다. 의도적 백필은 `--allow-bootstrap` 필요.

```bash
uv run python scripts/update_history_csv.py --csv history.csv --workers 8
```

### 모델 빌드

```bash
uv run python scripts/build_frontend_model.py \
  --history-csv history.csv \
  --weights data/optimized_weights.json \
  --output-ga data/model_ga.json \
  --output-xgb data/model_xgb.json \
  --output data/model.json
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--weights` | `data/optimized_weights.json` | 없으면 기본 가중치로 진행 |
| `--base-weight` | `0.65` | 기본확률 vs 필터통과빈도 혼합 비율 |
| `--chunk-size` | `20000` | 샘플러 청크 |
| `--seed` | 최신 회차 | 미지정 시 회차에서 유도 → **같은 데이터면 같은 모델** |
| `--use-hmm` | off | HMM 활성화 (느림) |
| `--sample-size-override` | — | 디버그용 |

XGBoost 모델 생성이 실패해도 GA 모델만 저장하고 계속 진행합니다.

### 워크플로

**`lotto-history-update.yml`** — 매주 토요일 KST 21:20 / 21:50 / 22:20 (UTC 12:20/12:50/13:20) 3회 재시도 창

1. `pip install .`
2. `history.csv` 증분 갱신
3. `model_ga.json` + `model_xgb.json` + `model.json` 재빌드
4. `git diff`로 변경 여부 확인 → **변경 시에만** 커밋/푸시
5. Discord 웹훅 알림 (성공 / 새 데이터 없음 각각)

**`ga-optimize.yml`** — 매주 월요일 UTC 00:00

1. `uv sync --frozen`
2. GA 실행 (기본 50세대 × 100개체, `--jobs 4`, `cycle_label`에 커밋 SHA 기록)
3. `optimized_weights.json` + `fitness_history.png` 커밋 (`[skip ci]`)
4. Discord 알림

필요한 시크릿: `DISCORD_WEBHOOK_URL`

---

## 12. 로컬 개발

### 설치

```bash
uv sync
```

Python ≥ 3.11. 주요 의존성: pandas, numpy, scipy, pydantic, PyYAML, hmmlearn, xgboost, scikit-learn, optuna, deap, matplotlib, tqdm, fastapi, uvicorn.
개발 추가: pytest, mypy, httpx (`uv sync --extra dev`).

### 테스트

```bash
uv run pytest -q
```

`pyproject.toml`에서 `pythonpath = ["src"]`, `testpaths = ["tests"]`로 설정되어 있습니다.
GA/fitness 테스트는 실제로 GA를 돌리므로 **전체 스위트는 10분 이상 걸립니다.** 빠른 확인:

```bash
uv run pytest -q --ignore=tests/unit/test_ga_optimizer.py --ignore=tests/unit/test_fitness.py
```

**현재 결과: 112 passed, 8 failed (약 11분).** 실패 8건은 전부 코드 변경에 테스트가 따라오지 못한 것이며, 원인은 [14장](#14-현재-상태와-알려진-이슈)에 정리했습니다.

### CLI 추천

```bash
# GA/엔진 기반 (해석 가능, 빠름)
uv run python recommend.py --games 5 --weights data/optimized_weights.json

# XGBoost 기반 (첫 실행 학습 ~3분, 이후 캐시 로드 ~8초)
uv run python recommend_ml.py --games 5
uv run python recommend_ml.py --games 5 --weights data/optimized_weights.json

# 풀 제외 실험
uv run python recommend_pool.py --pool-size 30 --games 5
```

공통 옵션: `--csv`(기본 `history.csv`), `--games`(기본 5), `--seed`(미지정 시 시간 기반), `--weights`.

`recommend_ml.py`의 **모델 캐싱 규칙**: `data/xgb_model_metadata.json`의 `trained_until_round`가 현재 최신 회차 이상이고 `weights_path`가 동일하면 `xgb_model.pkl`을 재사용합니다. 둘 중 하나라도 다르면 재학습합니다. 즉 **가중치를 바꾸면 자동 재학습**됩니다.

### FastAPI 앱 (선택)

현재 배포 경로에는 쓰이지 않지만 로컬에서 API 형태로 돌릴 수 있습니다.

```bash
uv run uvicorn lottogogo.mvp.api:app --reload
```

| 엔드포인트 | 설명 |
|---|---|
| `GET /` | 플레이스홀더가 치환된 index.html |
| `POST /api/recommend` | `{"preset": "A" 또는 "B", "games": 5 또는 10}` → 추천 결과 (`seed`는 무시됨) |
| `GET /api/warmup` | 결과 풀 비동기 예열 (`WARMUP_TOKEN` 설정 시 보호) |
| `GET /api/pool-status` | 프리셋×게임수별 풀 잔량 |
| `GET /robots.txt`, `/sitemap.xml` | 크롤러용 |

`RecommendationService`는 **요청마다 계산하지 않습니다.** 프리셋×게임수 조합별로 최대 8개(`RECOMMEND_POOL_MAX`)의 결과를 미리 만들어 데크에 넣고 순환 반환하며, 백그라운드 스레드가 목표치(`RECOMMEND_POOL_TARGET=4`)까지 다시 채웁니다. 그래서 `seed`가 무시됩니다.

### 프론트 정적 파일 재생성

```bash
sh scripts/export_vercel_index.sh
```

원래 설계: `src/lottogogo/mvp/static/index.html`(템플릿)의 `__DONATE_URL__`, `__MODEL_URL__`, `__SEO_*__` 플레이스홀더를 환경변수 값으로 치환해 루트 `index.html`을 생성.

> ⚠️ **현재는 동작하지 않습니다.** 템플릿과 배포본이 동일 파일이 되면서 플레이스홀더가 전부 사라졌습니다. 지금 이 스크립트를 돌리면 템플릿을 그대로 복사할 뿐입니다. 프론트를 수정하려면 당분간 **두 파일을 함께** 고쳐야 합니다. → [14장 3번](#14-현재-상태와-알려진-이슈)

---

## 13. 배포

### Vercel

- 저장소 루트를 정적 배포 대상으로 설정
- 배포되는 것: `index.html`, `about.html` 등 정적 페이지, `data/model_*.json`, `assets/recommend-worker.js`, `api/*.js`
- `vercel.json`이 `/robots.txt` → `/api/robots`, `/sitemap.xml` → `/api/sitemap`으로 rewrite
- GitHub push 시 자동 배포 (Actions가 데이터를 커밋하면 그대로 반영)

### 환경변수

`.env.example` 참고. 프론트 빌드/FastAPI에서 사용합니다.

| 변수 | 용도 |
|---|---|
| `PUBLIC_BASE_URL` | canonical / sitemap / OG 이미지 기준 URL |
| `DONATE_URL` | 후원 CTA 링크 |
| `MODEL_URL` | 워커가 fetch할 모델 경로 (레거시 폴백) |
| `SEO_TITLE`, `SEO_DESCRIPTION` | 메타 태그 |
| `OG_IMAGE_URL`, `TWITTER_IMAGE_URL` | 소셜 카드 이미지 (미지정 시 `{BASE}/assets/og-image.png`) |
| `GOOGLE_SITE_VERIFICATION`, `NAVER_SITE_VERIFICATION` | 검색엔진 소유 확인 |
| `LOTTO_HISTORY_CSV` | FastAPI가 읽을 CSV 경로 |
| `CORS_ALLOW_ORIGINS` | FastAPI CORS (기본 `*`) |
| `WARMUP_TOKEN` | `/api/warmup`, `/api/pool-status` 보호 (미설정 시 무인증) |
| `RECOMMEND_POOL_TARGET` / `RECOMMEND_POOL_MAX` / `RECOMMEND_BOOTSTRAP_SAMPLE_SIZE` | 결과 풀 튜닝 |

---

## 14. 현재 상태와 알려진 이슈

### 데이터

- `history.csv` 최신 회차: **1234회**
- `data/xgb_model_metadata.json`의 학습 회차: **1211회** → 다음 `recommend_ml.py` 실행 시 자동 재학습됩니다.
- `data/optimized_weights.json`: 2026-07-27 자동 갱신. **`temperature: 0.1` 키가 아직 남아 있지만 이제 어디서도 읽지 않습니다.** 다음 GA 실행 때 사라집니다.

### 최근에 고친 것 (2026-08-05)

| 문제 | 조치 |
|---|---|
| GA가 최적화할 수 없는 `temperature`가 표류해 프리셋 A 샘플링 확률의 **31%가 32번 한 개에 쏠림** | 탐색 공간에서 제거하고 0.5로 고정 → **6.4%**. 회귀 방지 테스트 추가 |
| 얕은 복사 때문에 `data/model.json`(GA 사본이어야 함)에 **XGBoost 가중치가 들어감** | `deepcopy`로 수정 |
| `max_overlap`만으로는 번호 단위 쏠림을 못 막아 최고점 번호가 **모든 게임에 등장** | `DiversitySelector`에 번호별 등장 횟수 상한 추가 (엔진·CLI·프론트 워커 전부) |
| `backtest.py`가 파이프라인을 복사해 갖고 있어 실제 추천과 **조건이 갈라짐** | `recommend.generate_recommendations()`를 직접 호출하도록 통합 |
| 결과가 랜덤보다 나은지 판단할 기준이 없음 | `backtest.py`가 랜덤 기준선(0.8개)을 항상 함께 출력 |

최근 10회차(1225~1234, 50게임) 백테스트 결과는 평균 **0.960개** vs 랜덤 0.800개(+0.16)입니다. 다만 50게임의 오차범위가 ±0.11이라 **통계적으로 유의하지 않습니다.** 판단하려면 50회차 이상이 필요합니다.

### 알려진 이슈

1. **깨진 서브모듈 2개** — `datadata/lotto_data`와 `pipline-kit`이 gitlink(mode 160000)로 커밋되어 있는데 `.gitmodules`가 없습니다. 클론하면 빈 디렉터리만 생깁니다. 실수로 커밋된 것으로 보이며, `git rm --cached datadata/lotto_data pipline-kit`로 정리 가능합니다. (`pipline-kit`은 `pipeline` 오타이기도 합니다.)

2. **테스트 8건 실패 (112 passed / 8 failed).** 전부 코드가 바뀌고 테스트가 안 따라온 경우입니다.

   *HMM을 탐색 공간에서 뺀 리팩터링의 여파 (6건)*
   - `test_ga_optimizer.py::test_vec_to_weights_round_trip`, `test_clamp_individual_clips_out_of_bounds`
     — `_vec_to_weights` / `_weights_to_vec` / `_clamp_individual`이 이제 `weight_keys`·`weight_bounds` 인자를 요구하는데 테스트는 옛 단일 인자로 호출 → `TypeError`
   - `test_ga_optimizer.py::test_ga_config_defaults` — `population_size == 100`을 기대하지만 `GAConfig` 기본값은 `50`
   - `test_ga_optimizer.py::test_ga_optimizer_runs_small` — 가중치 10개를 기대하지만 기본(`use_hmm=False`)은 이제 7개 반환
   - `test_fitness.py::test_evaluator_returns_fitness_result` — `combined = 0.6·train + 0.4·val`을 기대하지만 현재 공식은 `0.4·train + 0.5·val + 0.1·rank_bonus`
   - `test_fitness.py::test_evaluator_rejects_missing_weight` — `_validate_weights`가 이제 **존재하는 키만 검증**하므로(7D/9D 양쪽 지원 목적) 키 누락 시 더 이상 예외를 던지지 않음

   *프론트 템플릿이 플레이스홀더를 잃은 여파 (2건)* — 아래 3번 참조
   - `test_mvp_api.py::test_home_page_includes_donate_and_model_url`, `test_home_page_includes_seo_meta`

3. **프론트 템플릿과 배포본이 동일 파일이 되어, 템플릿 렌더링 경로가 죽어 있습니다.**
   `src/lottogogo/mvp/static/index.html`(템플릿)과 루트 `index.html`(배포본)이 **바이트 단위로 완전히 동일**합니다. 어느 시점에 렌더링 결과가 템플릿 위에 덮어써진 것으로 보입니다. 결과적으로:
   - 템플릿에 `__DONATE_URL__` / `__MODEL_URL__` / `__SEO_*__` 플레이스홀더가 **하나도 남아 있지 않습니다** (`grep -c` = 0).
   - `scripts/export_vercel_index.sh`는 치환할 대상이 없어 **사실상 no-op**입니다.
   - `mvp/api.py`의 SEO/도네이트/모델 URL 치환 로직도 **동작하지 않습니다.** 환경변수를 바꿔도 페이지가 변하지 않고, canonical URL은 `https://lottogogo.vercel.app/`로 하드코딩되어 있습니다.
   - 위 `test_mvp_api` 2건은 정확히 이것을 잡아낸 것입니다 — 테스트가 옳고 코드가 깨진 쪽입니다.

   복구하려면 템플릿 파일의 해당 값들을 플레이스홀더로 되돌려야 합니다.

4. **`backtest_compare.py` / `backtest_multi.py`는 POSIX 전용입니다.** 임시 CSV를 `/tmp/`에 씁니다. 또한 대상 회차만 제거하고 **미래 회차를 학습에 포함**하므로 낙관 편향이 있습니다. (`backtest.py`는 해당 없음 — 시점 분리가 지켜집니다.)

5. **배포 모델에 HMM이 빠져 있습니다.** `lotto-history-update.yml`이 `build_frontend_model.py`에 `--use-hmm`을 주지 않으므로, 프론트가 쓰는 `model_ga.json`의 `hmm_hot_numbers` / `hmm_cold_numbers`는 비어 있습니다. UI의 "AI 흐름 강세" 배지가 표시되지 않는 원인입니다.

6. **GA 결과가 랜덤 대비 유의미하지 않습니다.** `hit@15` 2.085 vs 랜덤 2.0, `mean_rank` 22.61 vs 랜덤 22.5. 이 수치를 근거로 예측 성능을 주장할 수 없습니다.
   덧붙여 적합도의 노이즈가 신호보다 큽니다. 학습 표본이 **마지막 20회차뿐**이라 `combined_fitness`의 표준오차가 약 0.105인데, 얻은 개선폭은 0.085입니다. `fitness.py`의 `train_rounds[-20:]`를 늘리는 것이 가장 싼 개선입니다.

7. **조합 랭킹이 점수를 이중 반영합니다.** 샘플링에서 이미 점수 비례로 뽑아 놓고, 랭킹에서 같은 점수의 합으로 다시 정렬합니다. 그 결과 시드를 바꿔도 상위 게임이 거의 동일하게 나옵니다. 번호 단위 쏠림은 빈도 상한으로 해결됐지만 **조합 단위 쏠림은 남아 있습니다.**

8. **풀 실험의 1차 지표(생존 분포)를 계산하는 코드가 없습니다.** → [10장](#10-풀-제외pool-exclusion-실험-트랙)
   참고로 생존 지표에는 닫힌 형태의 랜덤 기준선이 있습니다: `6 × pool_size / 45`. `pool_size=30`이면 **4.000**이고, 현재 실험 결과는 3.970으로 **정확히 랜덤 수준**입니다.

9. **TLS 검증 기본 비활성** — `LottoHistoryFetcher(verify_ssl=False)`가 기본값입니다. 동행복권 인증서 체인 문제를 우회하기 위한 것이나, 보안상 `--verify-ssl` 사용이 권장됩니다.

---

## 주의사항

- 이 프로젝트는 예측 서비스가 아니라 **실험/참고 도구**입니다.
- 실제 구매 판단과 결과 책임은 사용자에게 있습니다.
- 로또는 **독립시행**입니다. 과거 데이터 분석은 참고 용도로만 활용하세요.
- 저장소의 "도박사의 오류 필터"는 이름 그대로 통계적 근거가 없는 휴리스틱입니다.

## 라이선스

MIT
