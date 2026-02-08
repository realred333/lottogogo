# LottoGoGo Probability Engine - 아키텍처 문서

## 1. 컴포넌트 관계

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Application Layer                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                              main.py                                  │   │
│  │                    (CLI Entry Point / Orchestrator)                   │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Engine Layer                                   │
│                                                                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────────┐   │
│  │  Processor    │◀──▶│   Ranker      │◀──▶│    Backtester            │   │
│  │  (Pipeline    │    │   (Ranking    │    │    (Evaluation)          │   │
│  │   Orchestrator)│    │   + Diversity)│    │                          │   │
│  └───────┬───────┘    └───────────────┘    └───────────────────────────┘   │
│          │                                                                   │
│          ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Score Module                                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │ ScoreCalcula │  │   Booster    │  │  Penalizer   │                │  │
│  │  │ tor (Bayes)  │  │ (Heuristics) │  │ (Reduction)  │                │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Sampling Module                                 │  │
│  │  ┌──────────────┐  ┌──────────────────────────────────────────────┐  │  │
│  │  │ MonteCarlo   │  │           Filter Pipeline                      │  │  │
│  │  │ Sampler      │──▶  Sum │ AC │ Zone │ Tail │ OddEven │ HighLow  │  │  │
│  │  └──────────────┘  └──────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Data Layer                                     │
│                                                                              │
│  ┌───────────────┐    ┌───────────────┐    ┌────────────────────────────┐  │
│  │  DataLoader   │    │ FeatureExtrac │    │    Config                  │  │
│  │  (CSV I/O)    │    │ tor (Stats)   │    │    (YAML Params)           │  │
│  └───────────────┘    └───────────────┘    └────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 레이어 구조

### 2.1 계층 구조

```
┌─────────────────────────────────────────────┐
│           Presentation Layer                │  ← (향후 확장: CLI/API)
│           (main.py - CLI)                   │
├─────────────────────────────────────────────┤
│           Application Layer                 │
│           (Engine Orchestration)            │
│  • Processor: 파이프라인 전체 조율          │
│  • Ranker: 결과 정렬 및 다양성 적용         │
│  • Backtester: 성능 평가 실행               │
├─────────────────────────────────────────────┤
│           Domain Layer                      │
│           (Core Business Logic)             │
│  • ScoreCalculator: 베이지안 점수 계산      │
│  • Booster: 휴리스틱 가중치 적용            │
│  • Penalizer: 제외수 패널티 적용            │
│  • Sampler: 몬테카를로 샘플링               │
│  • Filters: 조합 필터링 규칙                │
├─────────────────────────────────────────────┤
│           Data Access Layer                 │
│           (I/O & Configuration)             │
│  • DataLoader: CSV 파일 읽기/검증           │
│  • FeatureExtractor: 통계 지표 계산         │
│  • Config: 파라미터 로딩/관리               │
├─────────────────────────────────────────────┤
│           Infrastructure Layer              │
│           (Cross-cutting Concerns)          │
│  • Logging: 실행 로그 기록                  │
│  • Types: 공통 타입/상수 정의               │
└─────────────────────────────────────────────┘
```

### 2.2 의존성 방향

```
Presentation → Application → Domain → Data Access
                   ↓            ↓          ↓
              Infrastructure (Logging, Types)
```

- **상위 레이어**는 하위 레이어에만 의존
- **Infrastructure**는 모든 레이어에서 사용 가능
- **순환 의존성 금지**

---

## 3. 모듈 상세 관계

### 3.1 컴포넌트 의존성 다이어그램

```
                        ┌─────────────┐
                        │    main     │
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         ┌───────────┐  ┌───────────┐  ┌───────────┐
         │ processor │  │  ranker   │  │backtester │
         └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
               │              │              │
    ┌──────────┴──────────┐   │              │
    ▼                     ▼   ▼              ▼
┌────────┐           ┌─────────────┐    ┌─────────────┐
│ score  │           │   sampler   │    │data_loader  │
│ module │           │   module    │    │             │
├────────┤           ├─────────────┤    └──────┬──────┘
│ - calc │           │ - monte     │           │
│ - boost│           │   carlo     │           ▼
│ - penal│           │ - filters   │    ┌─────────────┐
└────┬───┘           └──────┬──────┘    │  feature    │
     │                      │           │  extractor  │
     └──────────┬───────────┘           └──────┬──────┘
                ▼                              │
         ┌─────────────┐                       │
         │   config    │◀──────────────────────┘
         └─────────────┘
```

### 3.2 데이터 타입 흐름

```
CSV File
    │
    ▼
LottoRound[]           ← DataLoader 출력
    │
    ▼
FeatureSet             ← FeatureExtractor 출력
    │   (frequencies, hot_flags, cold_flags, etc.)
    ▼
NumberScore[]          ← ScoreCalculator 출력
    │   (number, raw_score, probability)
    ▼
Combination[]          ← Sampler 출력
    │   (numbers[], combo_score, filter_results)
    ▼
Recommendation[]       ← Ranker 출력
    │   (combination, rank, meta_info)
    ▼
JSON/Console Output
```

---

## 4. 배포 형태

### 4.1 패키지 구조

```
lottogogo/
├── pyproject.toml        # uv 패키지 정의
├── README.md
├── config/
│   └── default.yaml      # 기본 설정
├── src/
│   └── lottogogo/
│       ├── __init__.py
│       ├── main.py       # CLI 엔트리포인트
│       ├── config/       # 설정 관리
│       ├── data/         # 데이터 로딩/추출
│       ├── engine/       # 핵심 엔진 로직
│       └── types/        # 공통 타입 정의
├── tests/
│   ├── unit/
│   └── integration/
└── data/
    └── history.csv       # 샘플 데이터
```

### 4.2 실행 방식

```bash
# 1. CLI 직접 실행
uv run python -m lottogogo --config config/default.yaml

# 2. 패키지 설치 후 실행
uv pip install .
lottogogo --config config/default.yaml

# 3. 모듈로 임포트
from lottogogo.engine import generate_recommendations
results = generate_recommendations(csv_path, config)
```

### 4.3 배포 형태

| 형태 | 설명 | 사용 시나리오 |
|------|------|---------------|
| **CLI Tool** | 커맨드라인에서 직접 실행 | 개발/테스트/수동 분석 |
| **Python Package** | pip/uv로 설치 가능한 라이브러리 | 다른 프로젝트에서 임포트 |
| **Docker Image** | 컨테이너화된 실행 환경 | CI/CD, 재현 가능한 실행 |

### 4.4 확장 계획

```
Phase 1 (현재)        Phase 2 (향후)         Phase 3 (향후)
┌─────────────┐      ┌─────────────────┐    ┌───────────────────┐
│ CLI Engine  │  →   │ + REST API      │ →  │ + Web Dashboard   │
│ (Python)    │      │   (FastAPI)     │    │   (React/Next.js) │
└─────────────┘      └─────────────────┘    └───────────────────┘
```

---

## 5. 주요 인터페이스

### 5.1 Config 인터페이스

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

### 5.2 출력 인터페이스

```python
@dataclass
class Recommendation:
    rank: int
    numbers: list[int]  # 정렬된 6개 번호
    combo_score: float
    number_scores: dict[int, NumberScoreDetail]
    filter_results: dict[str, bool]
```
