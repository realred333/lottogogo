# LottoGoGo v2

CSV 기반 로또 당첨 이력을 바탕으로 번호별 확률 점수를 만들고, 몬테카를로 샘플링 + 필터링 + 랭킹/다양성 제약으로 최종 추천 조합을 생성하는 Python 엔진입니다.  
이 저장소는 엔진 코드(`src/lottogogo`), 실행 스크립트(`recommend.py`, `backtest.py`), 튜닝 유틸(`penalty_search.py`), 프로젝트 산출 문서(`docs/`), 프롬프트 자산(`pipeline_prompts/`), 외부 데이터 서브폴더(`datadata/lotto_data`)를 함께 포함합니다.

## Table of Contents

1. [프로젝트 목적](#프로젝트-목적)
2. [현재 저장소 스냅샷](#현재-저장소-스냅샷)
3. [기술 스택](#기술-스택)
4. [빠른 시작](#빠른-시작)
5. [실행 방법](#실행-방법)
6. [아키텍처와 데이터 흐름](#아키텍처와-데이터-흐름)
7. [모듈별 상세](#모듈별-상세)
8. [데이터 자산](#데이터-자산)
9. [테스트](#테스트)
10. [문서/프롬프트 자산](#문서프롬프트-자산)
11. [전체 파일 인벤토리](#전체-파일-인벤토리)
12. [트러블슈팅](#트러블슈팅)
13. [운영/배포 메모](#운영배포-메모)
14. [면책](#면책)

## 프로젝트 목적

- 과거 회차 데이터를 바탕으로 번호별 점수(`Base + Boost - Penalty`)를 계산
- 점수를 확률로 정규화하고, 대량 조합을 생성
- 필터(합계/AC/구간/끝수/홀짝/고저/과거중복)를 적용
- 점수 기반 랭킹 + 조합 간 중복 억제(다양성)로 최종 추천
- 워크포워드 백테스트로 성능 지표(`P(match>=3)` 등)를 비교

## 현재 저장소 스냅샷

- 언어/런타임: Python 3.12 계열에서 검증
- 핵심 엔진 패키지 위치: `src/lottogogo`
- 루트 실행 스크립트:
  - `recommend.py`: 추천 조합 생성
  - `backtest.py`: 특정 기준 회차 백테스트
- 하이퍼파라미터 탐색:
  - `src/lottogogo/tuning/penalty_search.py`
- 데이터:
  - `history.csv`: 1회~1209회 (`round,n1..n6,bonus`)
  - `datadata/lotto_data/lotto_data.db`: SQLite (1회~1204회)
- 테스트:
  - `tests/unit` 내 14개 테스트 파일
  - 로컬 실행 결과: 총 60개 중 59개 통과, 1개 실패(의존성 미설치 환경에서 `PyYAML` 누락)

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| 패키징 | `pyproject.toml` + `hatchling` |
| 데이터 처리 | `pandas` |
| 확률/통계 | `scipy`, `numpy`, `hmmlearn` |
| 설정 검증 | `pydantic` |
| 설정 파일 파싱 | `PyYAML`, `json` |
| 테스트 | `pytest` |
| 타입 점검(옵션) | `mypy` |
| 패키지 관리 | `uv` 권장 |

## 빠른 시작

### 1) 클론 후 진입

```bash
git clone <repo-url>
cd lottogogo_v2
```

### 2) 의존성 설치 (권장: uv)

```bash
uv sync
```

`uv`를 쓰지 않는 경우:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3) 테스트

```bash
pytest -q
```

### 4) 추천 실행

```bash
.venv/bin/python recommend.py --csv history.csv --games 5 --seed 42
```

## 실행 방법

### 추천 번호 생성

```bash
.venv/bin/python recommend.py --csv history.csv --games 5 --seed 42
```

인자:

- `--csv`: 이력 CSV 경로 (기본 `history.csv`)
- `--games`: 출력 게임 수 (기본 5)
- `--seed`: 랜덤 시드 (미지정 시 현재 시간 기반)

### 특정 회차 백테스트

```bash
.venv/bin/python backtest.py --csv history.csv --round 1180 --games 5 --seed 42
```

인자:

- `--round`: 학습 마지막 회차 (`round+1`을 예측 대상으로 평가)
- `--games`: 추천 게임 수
- `--seed`: 랜덤 시드

### Penalty 람다 그리드 서치

```bash
PYTHONPATH=src .venv/bin/python -m lottogogo.tuning.penalty_search \
  --history history.csv \
  --poisson-start 0.0 --poisson-end 0.5 --poisson-step 0.1 \
  --markov-start 0.0 --markov-end 0.5 --markov-step 0.1 \
  --top-n 10 \
  --save-json docs/final/penalty_tuning_results_manual.json
```

### 코드에서 직접 사용

```python
from lottogogo.data.loader import LottoHistoryLoader
from lottogogo.engine.score import (
    BaseScoreCalculator, BoostCalculator, PenaltyCalculator,
    ScoreEnsembler, ProbabilityNormalizer
)
from lottogogo.engine.sampler import MonteCarloSampler
from lottogogo.engine.filters import (
    FilterPipeline, SumFilter, ACFilter, ZoneFilter,
    TailFilter, OddEvenFilter, HighLowFilter, HistoryFilter
)
from lottogogo.engine.ranker import CombinationRanker, DiversitySelector

history = LottoHistoryLoader().load_and_validate("history.csv").reset_index(drop=True)

base = BaseScoreCalculator().calculate_scores(history, recent_n=50)
boost, tags = BoostCalculator().calculate_boosts(history)
penalty = PenaltyCalculator(poisson_lambda=0.0, markov_lambda=0.0).calculate_penalties(history)
raw = ScoreEnsembler(minimum_score=0.0).combine(base, boost, penalty)
probs = ProbabilityNormalizer.to_sampling_probabilities(raw, temperature=0.5, min_prob_floor=0.005)

sampled = MonteCarloSampler(sample_size=100000, chunk_size=20000).sample(probs, seed=42)
historical_draws = [tuple(row[["n1","n2","n3","n4","n5","n6"]].astype(int).tolist()) for _, row in history.iterrows()]
pipeline = FilterPipeline([
    SumFilter(100, 175),
    ACFilter(7),
    ZoneFilter(max_per_zone=3),
    TailFilter(max_same_tail=2),
    OddEvenFilter(min_odd=2, max_odd=4),
    HighLowFilter(min_high=2, max_high=4),
    HistoryFilter(historical_draws, match_threshold=5),
])
filtered = pipeline.filter_combinations(sampled)
ranked = CombinationRanker().rank(filtered, raw, top_k=100)
final = DiversitySelector(max_overlap=3).select([r.numbers for r in ranked], output_count=5)
print(final)
```

## 아키텍처와 데이터 흐름

```text
history.csv
  -> LottoHistoryLoader (스키마/범위/중복 검증)
  -> BaseScoreCalculator (베이지안 기본 점수)
  -> BoostCalculator + HMMScorer (가중치/상태 보정)
  -> PenaltyCalculator (Poisson/Markov 페널티)
  -> ScoreEnsembler (base + boost - penalty)
  -> ProbabilityNormalizer (softmax + floor)
  -> MonteCarloSampler (대량 조합 샘플링)
  -> FilterPipeline (7개 필터)
  -> CombinationRanker (조합 점수화)
  -> DiversitySelector (중복/과도한 교집합 억제)
  -> 추천 조합 출력
```

## 모듈별 상세

### `lottogogo.config`

- `schema.py`
  - `EngineConfig`: 엔진 파라미터 스키마와 범위 검증
- `loader.py`
  - `load_config(path)`: YAML/JSON 로드 + `EngineConfig` 검증
  - 예외: `ConfigLoadError`, `ValidationError`

### `lottogogo.data`

- `loader.py`
  - `LottoHistoryLoader`: CSV 로드/컬럼 정규화/범위 검증/회차 인덱싱
  - 예외: `DataValidationError`
- `fetcher.py`
  - 동행복권 API(`https://dhlottery.co.kr/lt645/selectPstLt645Info.do`)에서 회차 데이터 수집
  - 최신 회차 탐색(지수 + 이진 탐색), 누락 회차 병합 저장
  - 예외: `LottoFetchError`, `LottoRoundNotFoundError`

### `lottogogo.engine.score`

- `calculator.py`
  - `BaseScoreCalculator`: Beta posterior mean 기반 번호 점수
  - `ScoreEnsembler`: `base + boost - penalty`, 음수 방지/정규화
- `booster.py`
  - Hot/Cold, Neighbor, Carryover(직전+2주전), Reverse boost
- `hmm_scorer.py`
  - 번호별 3-state HMM으로 `hmm_hot`, `hmm_cold` 태그 부여
- `penalizer.py`
  - Poisson 과출현/Markov 전이 기반 penalty
- `normalizer.py`
  - temperature softmax + min floor + 재정규화

### `lottogogo.engine.sampler`

- `monte_carlo.py`
  - Gumbel top-k 기반 벡터화 무복원 샘플링
  - 대량 생성 시 `chunk_size`로 메모리 제어

### `lottogogo.engine.filters`

- `base.py`: `BaseFilter`, `FilterDecision`, 조합 유효성 체크
- `sum_filter.py`: 합계 범위
- `ac_filter.py`: AC값 하한
- `zone_filter.py`: 구간 쏠림 방지
- `tail_filter.py`: 동일 끝수 제한
- `odd_even_filter.py`: 홀짝 비율 제한
- `high_low_filter.py`: 고저 비율 제한
- `history_filter.py`: 과거 당첨과 과도한 중복 제거
- `pipeline.py`: 순차 실행 + 탈락 통계(`rejection_counts`)

### `lottogogo.engine.ranker`

- `scorer.py`
  - `CombinationRanker`: 조합별 점수 합산, `top_k` 정렬
- `diversity.py`
  - `DiversitySelector`: 중복 제거 + 최대 교집합 제한

### `lottogogo.engine.backtester`

- `walk_forward.py`: 회차 순차 학습/평가 루프
- `baseline.py`: 랜덤 기준선 생성
- `metrics.py`: `P(match>=3)`, `P(match>=4)`, 평균/표준편차/분포
- `report.py`: JSON + Markdown 리포트 출력

### `lottogogo.tuning`

- `penalty_search.py`
  - `PenaltyTuneConfig`, `PenaltyTuneResult`
  - Poisson/Markov lambda 조합 병렬 탐색
  - 결과를 JSON으로 저장 가능

## 데이터 자산

### `history.csv`

- 컬럼: `round,n1,n2,n3,n4,n5,n6,bonus`
- 현재 행 수: 1209
- 회차 범위: 1~1209
- 주요 사용 컬럼: 엔진 핵심은 `round,n1..n6`, `bonus`는 보조 정보

### `datadata/lotto_data/lotto_data.db`

- SQLite DB 파일
- 테이블: `tb_lotto_list`
- 레코드 수: 1204
- 회차 범위: 1~1204
- 서브 프로젝트 `datadata/lotto_data/`는 별도 README/스크립트/requirements 포함

### `docs/final/penalty_tuning_results*.json`

- 람다 탐색 결과 스냅샷
- 파일별 최고 결과(첫 row 기준):
  - `penalty_tuning_results_0_0.5.json`: `(poisson=0.2, markov=0.05, p>=3=0.0204)`
  - `penalty_tuning_results_coarse.json`: `(1.0, 0.7, 0.0245)`
  - `penalty_tuning_results_refined.json`: `(1.6, 0.9, 0.0286, p>=4=0.0041)`
  - `penalty_tuning_results_refined2.json`: `(2.3, 1.2, 0.0367, p>=4=0.0041)`

## 테스트

### 실행

```bash
pytest -q
```

### 테스트 범위

- 데이터 로더/Fetcher
- Config 로더
- Base/Boost/Penalty/Normalizer/Ensembler
- 샘플러 성능/중복 검증
- 필터 개별 동작 + 파이프라인 조기탈락/거절 통계
- 랭커/다양성
- 워크포워드 백테스터/리포트
- penalty search 보조 로직

### 로컬 검증 메모

- 현재 환경에서 `pytest -q` 실행 시:
  - `59 passed`
  - `1 failed`: YAML 로딩 테스트 (`PyYAML` 미설치 환경)
- 해결:
  - `uv sync` 또는 `pip install -e ".[dev]"` 후 재실행

## 문서/프롬프트 자산

### `docs/`

- `docs/1`: TRD/아키텍처/모듈 트리 초안
- `docs/2`: API/데이터모델/에러 스펙 초안
- `docs/3`: 계획/테스트/릴리즈 체크리스트 초안
- `docs/final`: 통합 최종 문서, 스프린트 결과, 튜닝 결과 JSON

주의:

- `docs/*`는 설계/기록 산출물이며, 실제 코드 구조와 1:1 일치하지 않는 항목이 일부 있습니다.

### `pipeline_prompts/`

- 문서 생성/스프린트 구현/QA 릴리즈에 쓰인 프롬프트 집합
- `pipeline_prompts/docs/*`에는 각 문서 에이전트용 세부 프롬프트 포함

## 전체 파일 인벤토리

아래는 `.git` object 파일, `.venv/site-packages`, `__pycache__` 캐시류를 제외한 실사용 파일 기준 인벤토리입니다.

### 루트

```text
.gitignore
PRD.md
backtest.py
history.csv
pyproject.toml
recommend.py
uv.lock
```

### `src/lottogogo`

```text
src/lottogogo/__init__.py
src/lottogogo/config/__init__.py
src/lottogogo/config/loader.py
src/lottogogo/config/schema.py
src/lottogogo/data/__init__.py
src/lottogogo/data/fetcher.py
src/lottogogo/data/loader.py
src/lottogogo/engine/__init__.py
src/lottogogo/engine/backtester/__init__.py
src/lottogogo/engine/backtester/baseline.py
src/lottogogo/engine/backtester/metrics.py
src/lottogogo/engine/backtester/report.py
src/lottogogo/engine/backtester/walk_forward.py
src/lottogogo/engine/filters/__init__.py
src/lottogogo/engine/filters/ac_filter.py
src/lottogogo/engine/filters/base.py
src/lottogogo/engine/filters/high_low_filter.py
src/lottogogo/engine/filters/history_filter.py
src/lottogogo/engine/filters/odd_even_filter.py
src/lottogogo/engine/filters/pipeline.py
src/lottogogo/engine/filters/sum_filter.py
src/lottogogo/engine/filters/tail_filter.py
src/lottogogo/engine/filters/zone_filter.py
src/lottogogo/engine/ranker/__init__.py
src/lottogogo/engine/ranker/diversity.py
src/lottogogo/engine/ranker/scorer.py
src/lottogogo/engine/sampler/__init__.py
src/lottogogo/engine/sampler/monte_carlo.py
src/lottogogo/engine/score/__init__.py
src/lottogogo/engine/score/booster.py
src/lottogogo/engine/score/calculator.py
src/lottogogo/engine/score/hmm_scorer.py
src/lottogogo/engine/score/normalizer.py
src/lottogogo/engine/score/penalizer.py
src/lottogogo/tuning/__init__.py
src/lottogogo/tuning/penalty_search.py
```

### `tests/unit`

```text
tests/unit/test_backtester.py
tests/unit/test_base_score_calculator.py
tests/unit/test_booster.py
tests/unit/test_config_loader.py
tests/unit/test_data_fetcher.py
tests/unit/test_data_loader.py
tests/unit/test_filter_pipeline.py
tests/unit/test_filters.py
tests/unit/test_normalizer.py
tests/unit/test_penalizer.py
tests/unit/test_penalty_search.py
tests/unit/test_ranker.py
tests/unit/test_sampler.py
tests/unit/test_score_ensembler.py
```

### `docs/1`, `docs/2`, `docs/3`

```text
docs/1/ARCHITECTURE.md
docs/1/MODULE_TREE.txt
docs/1/TRD.md
docs/2/API_SPEC.md
docs/2/DATA_MODEL.md
docs/2/ERRORS.md
docs/3/PLAN.md
docs/3/RELEASE_CHECKLIST.md
docs/3/TEST_PLAN.md
```

### `docs/final`

```text
docs/final/API_SPEC.md
docs/final/BACKLOG.md
docs/final/OPEN_QUESTIONS.md
docs/final/PLAN.md
docs/final/RESULT.md
docs/final/SPRINT1_RESULT.md
docs/final/SPRINT2_RESULT.md
docs/final/SPRINT3_RESULT.md
docs/final/SPRINT4_RESULT.md
docs/final/TRD.md
docs/final/penalty_tuning_results_0_0.5.json
docs/final/penalty_tuning_results_coarse.json
docs/final/penalty_tuning_results_refined.json
docs/final/penalty_tuning_results_refined2.json
```

### `pipeline_prompts`

```text
pipeline_prompts/FINAL_QA_RELEASE.prompt
pipeline_prompts/IMPLEMENT_SPRINT1.prompt
pipeline_prompts/IMPLEMENT_SPRINT{N}.prompt
pipeline_prompts/SPRINT_QA.prompt
pipeline_prompts/docs/0_PROCESS_CONCERNS.md
pipeline_prompts/docs/1_AGENT_TRD.prompt
pipeline_prompts/docs/2_AGENT_API.prompt
pipeline_prompts/docs/3_AGENT_PLAN.prompt
pipeline_prompts/docs/FINAL_BACKLOG.prompt
pipeline_prompts/docs/FINAL_MERGE.prompt
```

### `datadata/lotto_data` (서브 프로젝트)

```text
datadata/lotto_data/README.md
datadata/lotto_data/lotto_data.py
datadata/lotto_data/lotto_data.db
datadata/lotto_data/requirements.txt
datadata/lotto_data/.github/workflows/run_lotto_data_collection.yml
datadata/lotto_data/.github/workflows/run_release_tag.yml
datadata/lotto_data/.github/workflows/test_lotto_data_collection.yml
```

참고:

- `datadata/lotto_data/` 내부에는 독립 `.git/` 디렉터리가 존재합니다(서브 저장소 형태).

### 로컬/캐시 디렉터리 (참고)

```text
.venv/
.pytest_cache/
src/**/__pycache__/
tests/**/__pycache__/
pipeline_prompts/.DS_Store
```

## 트러블슈팅

### `python: command not found`

- 환경에 `python` 심볼릭 링크가 없을 수 있습니다.
- `python3` 또는 `.venv/bin/python`을 사용하세요.

### `ConfigLoadError: PyYAML is required`

- 원인: `PyYAML` 미설치
- 해결:

```bash
uv sync
# 또는
pip install PyYAML
```

### `ModuleNotFoundError: No module named 'lottogogo'`

- 해결 1: editable 설치

```bash
pip install -e .
```

- 해결 2: 임시 실행 시 `PYTHONPATH=src` 사용

```bash
PYTHONPATH=src .venv/bin/python -m lottogogo.tuning.penalty_search --help
```

### 필터 후 후보가 너무 적음

- `sample_size`를 증가
- 필터 임계값 완화(`sum/ac/zone/...`)
- `Penalty` 람다를 낮추거나 `temperature`를 올려 분산 확대

## 운영/배포 메모

- 이 저장소는 현재 엔진 중심의 로컬 실행/실험 구조입니다.
- 프로덕션 웹 서비스 배포 구성(`Dockerfile`, `k8s`, `render.yaml`, `vercel.json`)은 포함되어 있지 않습니다.
- 배포형 서비스로 확장 시 권장 순서:
  1. FastAPI 래퍼 API 추가
  2. 배치/스케줄러로 데이터 업데이트 자동화
  3. 결과/로그 저장소 분리(PostgreSQL 등)
  4. CI에서 테스트 + 린트 + 타입체크 고정

## 면책

- 이 프로젝트는 통계/실험 목적의 추천 엔진입니다.
- 당첨을 보장하지 않습니다.
- 실제 구매/투자 판단은 사용자 책임입니다.
