# LottoGoGo Data Model

> **Version:** 1.0.0

---

## 1. 개요

LottoGoGo Probability Engine의 핵심 데이터 모델을 정의합니다.

---

## 2. 핵심 엔티티

### 2.1 Round (회차)

과거 로또 당첨 회차 데이터를 저장합니다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| round_id | integer | Y | 회차 번호 (PK) |
| n1 | integer | Y | 첫 번째 당첨 번호 (1~45) |
| n2 | integer | Y | 두 번째 당첨 번호 (1~45) |
| n3 | integer | Y | 세 번째 당첨 번호 (1~45) |
| n4 | integer | Y | 네 번째 당첨 번호 (1~45) |
| n5 | integer | Y | 다섯 번째 당첨 번호 (1~45) |
| n6 | integer | Y | 여섯 번째 당첨 번호 (1~45) |
| bonus | integer | N | 보너스 번호 (1~45) |
| draw_date | date | N | 추첨일 |
| created_at | timestamp | Y | 생성 시각 |

#### 제약 조건

- `1 ≤ n1 < n2 < n3 < n4 < n5 < n6 ≤ 45`
- 모든 번호는 정렬된 상태로 저장

```json
{
  "round_id": 1150,
  "n1": 5,
  "n2": 12,
  "n3": 18,
  "n4": 29,
  "n5": 35,
  "n6": 42,
  "bonus": 7,
  "draw_date": "2026-02-01",
  "created_at": "2026-02-01T21:00:00+09:00"
}
```

---

### 2.2 NumberScore (번호 점수)

각 번호(1~45)의 계산된 점수 정보입니다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| number | integer | Y | 번호 (1~45, PK) |
| raw_score | float | Y | 최종 원시 점수 |
| base_score | float | Y | 베이지안 기본 점수 |
| boost_score | float | Y | 휴리스틱 가중치 합계 |
| penalty_score | float | Y | 패널티 합계 |
| probability | float | Y | 정규화된 샘플링 확률 |
| active_boosts | array[string] | Y | 적용된 Boost 목록 |
| status | string | Y | 번호 상태 (hot/warm/neutral/cold) |
| calculated_at | timestamp | Y | 계산 시각 |
| context_rounds | integer | Y | 분석 대상 회차 수 |

#### Boost 타입

| 값 | 조건 |
|----|------|
| hot | 최근 5회 중 ≥2회 등장 |
| cold | 최근 10회 중 0회 |
| neighbor | 직전 회차 번호 ±1 |
| carryover | 직전 회차 번호 |
| reverse | 역수 관계 |

```json
{
  "number": 17,
  "raw_score": 0.85,
  "base_score": 0.52,
  "boost_score": 0.38,
  "penalty_score": 0.05,
  "probability": 0.0278,
  "active_boosts": ["hot", "neighbor"],
  "status": "hot",
  "calculated_at": "2026-02-08T22:38:00+09:00",
  "context_rounds": 50
}
```

---

### 2.3 Combination (추천 조합)

생성된 추천 조합 정보입니다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| combo_id | string | Y | 조합 식별자 (UUID) |
| rank | integer | Y | 순위 |
| numbers | array[integer] | Y | 6개 번호 (정렬됨) |
| combo_score | float | Y | 조합 총점 |
| breakdown | object | Y | 번호별 점수 상세 |
| filters_passed | array[string] | Y | 통과한 필터 목록 |
| generated_at | timestamp | Y | 생성 시각 |
| seed | integer | N | 사용된 랜덤 시드 |

```json
{
  "combo_id": "c_a1b2c3d4",
  "rank": 1,
  "numbers": [3, 15, 22, 31, 38, 44],
  "combo_score": 4.82,
  "breakdown": {
    "3": {"raw_score": 0.85, "boosts": ["hot"], "penalty": 0.02},
    "15": {"raw_score": 0.78, "boosts": ["neighbor"], "penalty": 0.0}
  },
  "filters_passed": ["sum", "ac", "zone", "ending", "odd_even", "high_low", "history"],
  "generated_at": "2026-02-08T22:38:00+09:00",
  "seed": 42
}
```

---

### 2.4 Config (설정)

엔진 동작 설정입니다.

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| recent_rounds | integer | 50 | 분석 대상 최근 회차 수 |
| sample_size | integer | 100000 | 몬테카를로 샘플 수 |
| seed | integer | null | 랜덤 시드 |
| min_prob_floor | float | 0.001 | 최소 확률 바닥값 |
| weights | WeightConfig | - | Boost 가중치 |
| penalty_lambda | PenaltyConfig | - | Penalty 람다 |
| filters | FilterConfig | - | 필터 설정 |
| diversity | DiversityConfig | - | 다양성 제약 |

#### WeightConfig

| 필드 | 타입 | 기본값 |
|------|------|--------|
| hot | float | 0.4 |
| cold | float | 0.15 |
| neighbor | float | 0.3 |
| carryover | float | 0.2 |
| reverse | float | 0.1 |

#### PenaltyConfig

| 필드 | 타입 | 기본값 |
|------|------|--------|
| poisson | float | 0.5 |
| markov | float | 0.5 |

#### FilterConfig

| 필드 | 타입 | 기본값 |
|------|------|--------|
| sum_range | [int, int] | [100, 175] |
| ac_min | integer | 7 |
| max_per_zone | integer | 3 |
| max_same_ending | integer | 2 |
| odd_even_range | array[string] | ["2:4", "3:3", "4:2"] |
| max_history_overlap | integer | 4 |

#### DiversityConfig

| 필드 | 타입 | 기본값 |
|------|------|--------|
| max_intersection | integer | 3 |

---

### 2.5 BacktestResult (백테스트 결과)

백테스트 실행 결과를 저장합니다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| test_id | string | Y | 테스트 식별자 |
| start_round | integer | Y | 시작 회차 |
| end_round | integer | Y | 종료 회차 |
| total_rounds_tested | integer | Y | 테스트 회차 수 |
| match_distribution | object | Y | 적중 분포 (0~6개) |
| p_match_gte_3 | float | Y | 3개 이상 적중 확률 |
| avg_matches | float | Y | 평균 적중 개수 |
| p_match_gte_4 | float | Y | 4개 이상 적중 확률 |
| std_dev | float | Y | 표준편차 |
| baseline_p_match_gte_3 | float | Y | 기준선 3개 이상 적중 확률 |
| improvement | string | Y | 개선율 |
| config_snapshot | Config | Y | 사용된 설정 스냅샷 |
| started_at | timestamp | Y | 시작 시각 |
| completed_at | timestamp | Y | 완료 시각 |
| duration_seconds | integer | Y | 소요 시간(초) |

---

### 2.6 FilterResult (필터 결과)

조합 필터 검증 결과입니다.

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| filter_name | string | Y | 필터 이름 |
| passed | boolean | Y | 통과 여부 |
| value | any | Y | 측정된 값 |
| threshold | any | Y | 기준값 |
| reason | string | N | 탈락 사유 (실패 시) |

#### 필터 종류

| 필터명 | 검증 내용 |
|--------|----------|
| sum | 번호 합계: 100~175 |
| ac | AC값: ≥7 |
| zone | 구간당 최대 3개 |
| ending | 동일 끝수 ≤2 |
| odd_even | 홀짝 비율: 2:4 ~ 4:2 |
| high_low | 고저 균형 |
| history | 과거 당첨과 5개 이상 일치 시 폐기 |

---

## 3. 관계도

```
┌──────────────┐     ┌───────────────┐
│    Round     │────▶│  NumberScore  │
│ (회차 데이터) │     │  (번호 점수)   │
└──────────────┘     └───────────────┘
                            │
                            ▼
┌──────────────┐     ┌───────────────┐
│   Config     │────▶│  Combination  │
│  (설정)       │     │  (추천 조합)   │
└──────────────┘     └───────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌───────────────┐
│BacktestResult│◀────│ FilterResult  │
│(백테스트 결과)│     │  (필터 결과)   │
└──────────────┘     └───────────────┘
```

- **Round → NumberScore**: 회차 데이터로부터 번호별 점수 계산
- **Config → Combination**: 설정 기반으로 조합 생성
- **Combination → FilterResult**: 조합에 필터 적용
- **Config → BacktestResult**: 설정 스냅샷 포함

---

## 4. 저장소 가정

### 4.1 Primary Storage

- **파일 기반 (MVP)**
  - CSV: 회차 데이터
  - YAML/JSON: 설정 파일
  - JSON: 결과 캐시

### 4.2 확장 가능 옵션

| 저장소 | 용도 | 비고 |
|--------|------|------|
| SQLite | 로컬 개발/단일 사용자 | MVP용 권장 |
| PostgreSQL | 멀티 사용자/프로덕션 | 확장 시 |
| Redis | 점수 캐시/세션 | 성능 최적화 시 |

### 4.3 데이터 흐름

```
CSV Input → Round Table → Feature Calculation → NumberScore Cache
                                      ↓
                              Monte Carlo Sampling
                                      ↓
                              Filter Pipeline
                                      ↓
                              Combination Output
```

---

## 5. 인덱스 권장사항

| 엔티티 | 인덱스 | 목적 |
|--------|--------|------|
| Round | round_id (PK) | 회차 조회 |
| Round | draw_date | 날짜 범위 조회 |
| NumberScore | number, calculated_at | 최신 점수 조회 |
| BacktestResult | test_id, completed_at | 결과 조회 |
