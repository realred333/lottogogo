# LottoGoGo API Specification (통합본)

> **Version:** 1.0.0 Final  
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
          "3": {"raw_score": 0.85, "boosts": ["hot"], "penalty": 0.02}
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

---

### 4.3 데이터 조회

**GET** `/data`

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| limit | integer | N | 조회할 회차 수 (기본: 10) |
| offset | integer | N | 시작 오프셋 (기본: 0) |

---

### 4.4 번호별 점수 조회

**GET** `/scores`

| 파라미터 | 타입 | 필수 | 설명 |
|----------|------|------|------|
| recent_rounds | integer | N | 분석 대상 최근 회차 수 (기본: 50) |

---

### 4.5 백테스트 실행

**POST** `/backtest`

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

---

### 4.6 필터 검증

**POST** `/validate`

```json
{
  "numbers": [5, 12, 22, 31, 38, 44]
}
```

---

### 4.7 설정 조회/갱신

**GET** `/config` - 현재 엔진 설정 조회  
**PUT** `/config` - 엔진 설정 갱신

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
    "code": "ERROR_CODE",
    "message": "사용자 친화적 메시지",
    "details": {
      "developer_message": "개발자를 위한 상세 정보"
    }
  }
}
```

---

## 6. 에러 코드 체계

### 6.1 인증/권한 에러 (AUTH_)

| 코드 | HTTP | 사용자 메시지 | 재시도 |
|------|------|--------------|--------|
| AUTH_MISSING_KEY | 401 | API 키가 필요합니다. | ❌ |
| AUTH_INVALID_KEY | 401 | 유효하지 않은 API 키입니다. | ❌ |
| AUTH_RATE_LIMITED | 429 | 요청이 너무 많습니다. | ✅ |

### 6.2 입력 유효성 에러 (PARAM_)

| 코드 | HTTP | 사용자 메시지 | 재시도 |
|------|------|--------------|--------|
| PARAM_MISSING | 400 | 필수 항목이 누락되었습니다. | ❌ |
| PARAM_INVALID_TYPE | 400 | 입력 형식이 올바르지 않습니다. | ❌ |
| PARAM_OUT_OF_RANGE | 400 | 허용 범위를 벗어났습니다. | ❌ |

### 6.3 데이터 에러 (DATA_)

| 코드 | HTTP | 사용자 메시지 | 재시도 |
|------|------|--------------|--------|
| DATA_NOT_FOUND | 404 | 요청한 데이터를 찾을 수 없습니다. | ❌ |
| DATA_EMPTY | 400 | 데이터가 비어있습니다. | ❌ |
| DATA_INVALID_CSV | 422 | CSV 파일 형식이 올바르지 않습니다. | ❌ |
| DATA_INVALID_NUMBER | 422 | 번호가 유효하지 않습니다. | ❌ |

### 6.4 필터 에러 (FILTER_)

| 코드 | HTTP | 사용자 메시지 | 재시도 |
|------|------|--------------|--------|
| FILTER_SUM_FAIL | 422 | 번호 합계 조건을 만족하지 않습니다. | ❌ |
| FILTER_AC_FAIL | 422 | AC값 조건을 만족하지 않습니다. | ❌ |
| FILTER_NO_CANDIDATE | 500 | 조건에 맞는 조합을 생성하지 못했습니다. | ✅ |

### 6.5 엔진 에러 (ENGINE_)

| 코드 | HTTP | 사용자 메시지 | 재시도 |
|------|------|--------------|--------|
| ENGINE_CALCULATION_ERR | 500 | 계산 중 오류가 발생했습니다. | ✅ |
| ENGINE_TIMEOUT | 504 | 처리 시간이 초과되었습니다. | ✅ |

### 6.6 시스템 에러 (SYS_)

| 코드 | HTTP | 사용자 메시지 | 재시도 |
|------|------|--------------|--------|
| SYS_INTERNAL | 500 | 시스템 오류가 발생했습니다. | ✅ |
| SYS_MAINTENANCE | 503 | 서비스 점검 중입니다. | ✅ |

---

## 7. 재시도 가이드라인

| 에러 코드 | 권장 재시도 간격 | 최대 재시도 |
|-----------|-----------------|-------------|
| AUTH_RATE_LIMITED | 60초 | 3회 |
| ENGINE_* | 5초 | 3회 |
| SYS_* | 10초 exponential backoff | 5회 |

### Exponential Backoff 공식

```
wait_time = base_delay * (2 ^ attempt) + random_jitter
```

---

## 8. 데이터 모델 상세

### 8.1 핵심 엔티티 관계

```
Round (회차 데이터) → NumberScore (번호 점수)
                          ↓
Config (설정) → Combination (추천 조합)
    ↓                    ↓
BacktestResult ← FilterResult (필터 결과)
```

### 8.2 필터 종류

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

## 9. 저장소 가정

| 저장소 | 용도 | 비고 |
|--------|------|------|
| **파일 기반 (MVP)** | CSV/YAML/JSON | 현재 Phase |
| SQLite | 로컬 개발 | 확장 옵션 |
| PostgreSQL | 프로덕션 | 확장 시 |
