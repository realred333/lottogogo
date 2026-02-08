# LottoGoGo Probability Engine - 통합 기술 설계 문서 (TRD)

> **Version:** 1.0.0 Final  
> **최종 병합일:** 2026-02-08

---

## 1. 목표 / 범위

### 1.1 목표
- **Primary:** 3개 이상 적중 확률 P(match ≥ 3) 최대화
- **Secondary:** 평균 적중 개수 유지, 결과 분산 최소화, 재현성 보장 (동일 seed → 동일 결과)

### 1.2 범위

#### In Scope (포함)
- CSV 기반 과거 로또 데이터 로딩 및 검증
- 번호별 통계/확률 기반 점수 계산 (베이지안 + 휴리스틱)
- 몬테카를로 샘플링 기반 조합 생성
- 필터 파이프라인 (합계, AC값, 구간분산, 끝수, 홀짝, 고저, 과거당첨)
- 조합 랭킹 및 다양성 제약 적용
- 백테스터를 통한 성능 평가
- Config 기반 파라미터 관리

#### Out of Scope (제외)
- Web UI / API 서버 (Phase 1 범위 외)
- 사용자 인증/인가
- 데이터베이스 영구 저장 (엔진은 파일 기반으로만 동작)
- 외부 API 연동

---

## 2. 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LottoGoGo Probability Engine                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Input Layer  │    │ Config Layer │    │    Output Layer          │  │
│  │  (CSV Data)   │    │ (YAML/JSON)  │    │    (Recommendations)     │  │
│  └──────┬───────┘    └──────┬───────┘    └────────────▲─────────────┘  │
│         │                   │                         │                  │
│         ▼                   ▼                         │                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                        Core Engine                                │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │                                                                    │  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐  │  │
│  │  │ Data Loader │──▶│  Feature    │──▶│    Score Calculator     │  │  │
│  │  │  & Validator│   │  Extractor  │   │ (Bayes + Boost - Penalty)│  │  │
│  │  └─────────────┘   └─────────────┘   └───────────┬─────────────┘  │  │
│  │                                                   │                │  │
│  │                                                   ▼                │  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐  │  │
│  │  │  Backtester │◀──│   Ranker    │◀──│  Monte Carlo Sampler    │  │  │
│  │  │             │   │ + Diversity │   │  + Filter Pipeline      │  │  │
│  │  └─────────────┘   └─────────────┘   └─────────────────────────┘  │  │
│  │                                                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 주요 모듈 책임

| 모듈 | 책임 | 주요 기능 |
|------|------|-----------|
| `data_loader` | 데이터 입출력 | CSV 파싱, 검증, Config 로딩 |
| `feature_extractor` | 통계 분석 | 빈도 계산, Hot/Cold 분석, Neighbor/Carryover 분석 |
| `score_calculator` | 점수 계산 | 베이지안 점수, Boost/Penalty 계산, Softmax 정규화 |
| `sampler` | 조합 생성 | 몬테카를로 샘플링, 필터 파이프라인 적용 |
| `filters` | 조합 필터링 | 합계, AC값, 구간분산, 끝수, 홀짝, 고저, 과거당첨 필터 |
| `ranker` | 랭킹/다양성 | 조합 점수 계산, 다양성 제약 적용 |
| `backtester` | 성능 평가 | Walk-forward 테스트, 지표 계산 |
| `config` | 설정 관리 | 파라미터 로딩, 기본값 적용, 검증 |

---

## 4. 데이터 흐름

```
CSV 파일 ─────────────────────────────────────────────────────────────────▶

[1. 로딩/검증]
    ├── history.csv 로딩
    ├── 컬럼 검증 (round, n1~n6)
    └── 범위 검증 (1~45)
         │
         ▼
[2. Feature 계산]
    ├── 최근 N회 빈도 집계
    ├── Hot/Cold 플래그 계산
    ├── Neighbor/Carryover 계산
    └── Markov 전이 행렬 계산
         │
         ▼
[3. 점수 계산]
    ├── BaseScore: Beta Posterior Mean
    ├── Boost: Hot/Cold/Neighbor/Carryover/Reverse
    ├── Penalty: Poisson + Markov
    └── Softmax → 샘플링 확률 p(i)
         │
         ▼
[4. 조합 생성]
    ├── Monte Carlo: M개 후보 생성 (50,000~100,000)
    └── 중복 없이 6개씩 샘플링
         │
         ▼
[5. 필터링]
    ├── 합계 필터 (100~175)
    ├── AC값 필터 (≥7)
    ├── 구간분산 필터 (구간당 ≤3)
    ├── 끝수 필터 (동일 끝수 ≤2)
    ├── 홀짝 필터 (2:4~4:2)
    ├── 고저 필터 (균형)
    └── 과거당첨 필터 (5개 이상 일치 시 폐기)
         │
         ▼
[6. 랭킹 & 다양성]
    ├── 조합 점수 = Σ raw_score(i)
    ├── 상위 K개 선택
    └── 교집합 ≥4 조합 제거
         │
         ▼
─────────────────────────────────────────────────────────▶ 추천 조합 출력
```

---

## 5. 기술 스택

| 영역 | 기술 | 비고 |
|------|------|------|
| **언어** | Python 3.11+ | 타입 힌트 적극 활용 |
| **패키지/실행** | uv | 빠른 의존성 관리 |
| **수치 연산** | NumPy, SciPy | 베이지안/확률 계산 |
| **데이터 처리** | Pandas | CSV 로딩, 집계 |
| **Config** | YAML (PyYAML) | 파라미터 관리 |
| **스키마 검증** | Pydantic | Config/Output 스키마 |
| **테스트** | pytest + pytest-asyncio | 단위/통합 테스트 |
| **타입 체크** | mypy | 정적 타입 검사 |
| **로깅** | Python logging | 표준 라이브러리 |

---

## 6. 외부 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `numpy` | ^2.0 | 수치 연산, 확률 계산 |
| `scipy` | ^1.14 | Beta 분포, 통계 함수 |
| `pandas` | ^2.2 | CSV 파싱, 데이터 처리 |
| `pyyaml` | ^6.0 | Config 파일 파싱 |
| `pydantic` | ^2.0 | Config/Output 스키마 검증 |

### Dev Dependencies
| 패키지 | 용도 |
|--------|------|
| `pytest` | 테스트 프레임워크 |
| `pytest-cov` | 커버리지 측정 |
| `mypy` | 타입 체크 |
| `ruff` | 린팅/포매팅 |

---

## 7. 핵심 데이터 모델

### 7.1 Round (회차)
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| round_id | integer | Y | 회차 번호 (PK) |
| n1~n6 | integer | Y | 당첨 번호 (1~45) |
| bonus | integer | N | 보너스 번호 |
| draw_date | date | N | 추첨일 |

### 7.2 NumberScore (번호 점수)
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| number | integer | Y | 번호 (1~45) |
| raw_score | float | Y | 최종 점수 |
| probability | float | Y | 샘플링 확률 |
| active_boosts | array | Y | 적용된 Boost 목록 |
| status | string | Y | hot/warm/neutral/cold |

### 7.3 Combination (추천 조합)
| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| rank | integer | Y | 순위 |
| numbers | array[int] | Y | 6개 번호 (정렬됨) |
| combo_score | float | Y | 조합 총점 |
| filters_passed | array | Y | 통과한 필터 목록 |

---

## 8. 비기능 요구사항

| 항목 | 요구사항 | 측정 방법 |
|------|----------|-----------|
| **재현성** | 동일 seed → 동일 결과 100% | 테스트 케이스 검증 |
| **성능** | 100,000 샘플 생성 < 5초 | 벤치마크 테스트 |
| **안정성** | 모든 입력에 대해 최소 K개 결과 반환 | 엣지 케이스 테스트 |
| **확장성** | 새 필터/Boost 추가 시 기존 코드 수정 최소화 | 플러그인 패턴 적용 |
| **가독성** | 모든 public 함수에 docstring | CI 검사 |
| **테스트** | 커버리지 85% 이상 | pytest-cov |

---

## 9. 모듈 트리

```
lottogogo/
├── pyproject.toml
├── README.md
├── config/
│   └── default.yaml
├── src/lottogogo/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── defaults.py
│   ├── data/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── feature_extractor.py
│   ├── engine/
│   │   ├── processor.py
│   │   ├── score/
│   │   │   ├── calculator.py
│   │   │   ├── booster.py
│   │   │   ├── penalizer.py
│   │   │   └── normalizer.py
│   │   ├── sampler/
│   │   │   ├── monte_carlo.py
│   │   │   └── probability.py
│   │   ├── filters/
│   │   │   ├── base.py
│   │   │   ├── sum_filter.py
│   │   │   ├── ac_filter.py
│   │   │   ├── zone_filter.py
│   │   │   ├── tail_filter.py
│   │   │   ├── odd_even_filter.py
│   │   │   ├── high_low_filter.py
│   │   │   ├── history_filter.py
│   │   │   └── pipeline.py
│   │   ├── ranker/
│   │   │   ├── scorer.py
│   │   │   └── diversity.py
│   │   └── backtester/
│   │       ├── walk_forward.py
│   │       ├── metrics.py
│   │       └── baseline.py
│   └── types/
│       ├── models.py
│       ├── enums.py
│       └── constants.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

---

## 10. Config 인터페이스

```python
@dataclass
class EngineConfig:
    # 베이지안
    recent_n: int = 50
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    
    # Boost
    hot_threshold: int = 2
    hot_window: int = 5
    hot_weight: float = 0.4
    cold_window: int = 10
    cold_weight: float = 0.15
    neighbor_weight: float = 0.3
    carryover_weight: float = 0.2
    reverse_weight: float = 0.1
    
    # Penalty
    poisson_window: int = 20
    poisson_lambda: float = 0.5
    markov_lambda: float = 0.3
    
    # Sampling
    sample_size: int = 50000
    min_prob_floor: float = 0.001
    output_count: int = 5
    
    # Diversity
    max_overlap: int = 3
    
    # Reproducibility
    seed: int = 42
```

---

## 11. 확장 계획

```
Phase 1 (현재)        Phase 2 (향후)         Phase 3 (향후)
┌─────────────┐      ┌─────────────────┐    ┌───────────────────┐
│ CLI Engine  │  →   │ + REST API      │ →  │ + Web Dashboard   │
│ (Python)    │      │   (FastAPI)     │    │   (React/Next.js) │
└─────────────┘      └─────────────────┘    └───────────────────┘
```
