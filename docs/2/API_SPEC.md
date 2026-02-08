# LottoGoGo API Specification

> **Version:** 1.0.0  
> **Base URL:** `/api/v1`  
> **인증 방식:** API Key (Header: `X-API-Key`)

---

## 1. 개요

LottoGoGo Probability Engine의 API 인터페이스입니다.  
모든 응답은 JSON 형식이며, UTF-8 인코딩을 사용합니다.

---

## 2. 인증

| 방식 | 설명 |
|------|------|
| API Key | `X-API-Key` 헤더에 발급받은 키 포함 |
| Rate Limit | 분당 60회 요청 제한 |

```http
X-API-Key: your_api_key_here
```

---

## 3. 상태 코드 규칙

| 코드 | 의미 | 설명 |
|------|------|------|
| 200 | OK | 요청 성공 |
| 201 | Created | 리소스 생성 성공 |
| 400 | Bad Request | 잘못된 요청 파라미터 |
| 401 | Unauthorized | 인증 실패 |
| 404 | Not Found | 리소스 없음 |
| 422 | Unprocessable Entity | 유효성 검사 실패 |
| 429 | Too Many Requests | Rate Limit 초과 |
| 500 | Internal Server Error | 서버 내부 오류 |

---

## 4. 엔드포인트 목록

### 4.1 추천 조합 생성

**POST** `/recommendations`

번호 확률 기반 추천 조합을 생성합니다.

#### Request

```json
{
  "count": 5,
  "config": {
    "recent_rounds": 50,
    "sample_size": 100000,
    "seed": 42,
    "weights": {
      "hot": 0.4,
      "cold": 0.15,
      "neighbor": 0.3,
      "carryover": 0.2,
      "reverse": 0.1
    },
    "penalty_lambda": {
      "poisson": 0.5,
      "markov": 0.5
    }
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| count | integer | N | 생성할 조합 수 (기본: 5, 최대: 20) |
| config | object | N | 커스텀 설정 |
| config.recent_rounds | integer | N | 분석 대상 최근 회차 수 (기본: 50) |
| config.sample_size | integer | N | 몬테카를로 샘플 수 (기본: 100000) |
| config.seed | integer | N | 랜덤 시드 (재현성용) |
| config.weights | object | N | Boost 가중치 |
| config.penalty_lambda | object | N | Penalty 람다 값 |

#### Response

```json
{
  "success": true,
  "data": {
    "recommendations": [
      {
        "rank": 1,
        "numbers": [3, 15, 22, 31, 38, 44],
        "combo_score": 4.82,
        "breakdown": {
          "3": {"raw_score": 0.85, "boosts": ["hot"], "penalty": 0.02},
          "15": {"raw_score": 0.78, "boosts": ["neighbor"], "penalty": 0.0},
          "22": {"raw_score": 0.72, "boosts": [], "penalty": 0.05},
          "31": {"raw_score": 0.81, "boosts": ["carryover"], "penalty": 0.0},
          "38": {"raw_score": 0.88, "boosts": ["hot", "neighbor"], "penalty": 0.01},
          "44": {"raw_score": 0.78, "boosts": ["cold"], "penalty": 0.0}
        },
        "filters_passed": ["sum", "ac", "zone", "ending", "odd_even", "high_low", "history"]
      }
    ],
    "meta": {
      "generated_at": "2026-02-08T22:38:00+09:00",
      "total_samples": 100000,
      "passed_filters": 8543,
      "seed_used": 42
    }
  }
}
```

---

### 4.2 데이터 업로드

**POST** `/data/upload`

CSV 형식의 로또 회차 데이터를 업로드합니다.

#### Request

```http
Content-Type: multipart/form-data

file: lotto_data.csv
```

CSV 형식:
```csv
round,n1,n2,n3,n4,n5,n6
1,10,23,29,33,37,40
2,9,13,21,25,32,42
```

#### Response

```json
{
  "success": true,
  "data": {
    "upload_id": "data_20260208_001",
    "rounds_count": 1150,
    "date_range": {
      "first": 1,
      "last": 1150
    },
    "validation": {
      "valid": true,
      "warnings": []
    }
  }
}
```

---

### 4.3 데이터 조회

**GET** `/data`

업로드된 데이터의 요약 정보를 조회합니다.

#### Query Parameters

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| limit | integer | N | 조회할 회차 수 (기본: 10) |
| offset | integer | N | 시작 오프셋 (기본: 0) |

#### Response

```json
{
  "success": true,
  "data": {
    "total_rounds": 1150,
    "recent_rounds": [
      {"round": 1150, "numbers": [5, 12, 18, 29, 35, 42], "date": "2026-02-01"},
      {"round": 1149, "numbers": [3, 17, 22, 31, 38, 44], "date": "2026-01-25"}
    ],
    "statistics": {
      "most_frequent": [34, 17, 1, 40, 12],
      "least_frequent": [9, 22, 41, 28, 36]
    }
  }
}
```

---

### 4.4 번호별 점수 조회

**GET** `/scores`

각 번호(1~45)의 현재 점수와 확률을 조회합니다.

#### Query Parameters

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| recent_rounds | integer | N | 분석 대상 최근 회차 수 (기본: 50) |

#### Response

```json
{
  "success": true,
  "data": {
    "scores": [
      {
        "number": 1,
        "raw_score": 0.72,
        "probability": 0.0234,
        "base_score": 0.52,
        "boost": 0.25,
        "penalty": 0.05,
        "active_boosts": ["hot"],
        "status": "hot"
      }
    ],
    "meta": {
      "analyzed_rounds": 50,
      "calculated_at": "2026-02-08T22:38:00+09:00"
    }
  }
}
```

---

### 4.5 백테스트 실행

**POST** `/backtest`

과거 데이터를 기반으로 엔진 성능을 평가합니다.

#### Request

```json
{
  "start_round": 100,
  "end_round": 1140,
  "config": {
    "recent_rounds": 50,
    "sample_size": 50000
  }
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "test_id": "bt_20260208_001",
    "results": {
      "total_rounds_tested": 1040,
      "match_distribution": {
        "0": 0.412,
        "1": 0.328,
        "2": 0.168,
        "3": 0.072,
        "4": 0.018,
        "5": 0.002,
        "6": 0.0
      },
      "primary_metric": {
        "p_match_gte_3": 0.092
      },
      "secondary_metrics": {
        "avg_matches": 1.08,
        "p_match_gte_4": 0.020,
        "std_dev": 0.94
      },
      "baseline_comparison": {
        "random_p_match_gte_3": 0.062,
        "improvement": "+48.4%"
      }
    },
    "meta": {
      "started_at": "2026-02-08T22:30:00+09:00",
      "completed_at": "2026-02-08T22:38:00+09:00",
      "duration_seconds": 480
    }
  }
}
```

---

### 4.6 필터 검증

**POST** `/validate`

특정 조합이 필터 조건을 통과하는지 검증합니다.

#### Request

```json
{
  "numbers": [5, 12, 22, 31, 38, 44]
}
```

#### Response

```json
{
  "success": true,
  "data": {
    "valid": true,
    "checks": {
      "sum": {"passed": true, "value": 152, "range": [100, 175]},
      "ac": {"passed": true, "value": 9, "min": 7},
      "zone_variance": {"passed": true, "distribution": [1, 2, 1, 2]},
      "ending": {"passed": true, "max_same_ending": 2},
      "odd_even": {"passed": true, "ratio": "3:3"},
      "high_low": {"passed": true, "ratio": "3:3"},
      "history": {"passed": true, "max_overlap": 3}
    }
  }
}
```

---

### 4.7 설정 조회/갱신

**GET** `/config`

현재 엔진 설정을 조회합니다.

**PUT** `/config`

엔진 설정을 갱신합니다.

#### Response (GET)

```json
{
  "success": true,
  "data": {
    "recent_rounds": 50,
    "sample_size": 100000,
    "min_prob_floor": 0.001,
    "weights": {
      "hot": 0.4,
      "cold": 0.15,
      "neighbor": 0.3,
      "carryover": 0.2,
      "reverse": 0.1
    },
    "penalty_lambda": {
      "poisson": 0.5,
      "markov": 0.5
    },
    "filters": {
      "sum_range": [100, 175],
      "ac_min": 7,
      "max_per_zone": 3,
      "max_same_ending": 2,
      "odd_even_range": ["2:4", "3:3", "4:2"],
      "max_history_overlap": 4
    },
    "diversity": {
      "max_intersection": 3
    }
  }
}
```

---

## 5. 공통 응답 형식

### 성공 응답

```json
{
  "success": true,
  "data": { ... }
}
```

### 에러 응답

```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAM",
    "message": "파라미터 형식이 올바르지 않습니다.",
    "details": {
      "field": "count",
      "reason": "1~20 범위의 정수여야 합니다."
    }
  }
}
```
