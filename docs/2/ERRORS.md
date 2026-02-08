# LottoGoGo Error Codes

> **Version:** 1.0.0

---

## 1. 개요

LottoGoGo API의 에러 코드 체계를 정의합니다.  
모든 에러는 일관된 형식으로 반환됩니다.

---

## 2. 에러 응답 형식

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "사용자 친화적 메시지",
    "details": {
      "developer_message": "개발자를 위한 상세 정보",
      "field": "문제가 발생한 필드 (해당 시)",
      "timestamp": "2026-02-08T22:38:00+09:00"
    }
  }
}
```

---

## 3. 에러 코드 테이블

### 3.1 인증/권한 에러 (AUTH_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| AUTH_MISSING_KEY | 401 | API 키가 필요합니다. | X-API-Key 헤더가 누락되었습니다. | ❌ |
| AUTH_INVALID_KEY | 401 | 유효하지 않은 API 키입니다. | 제공된 API 키가 데이터베이스에 존재하지 않습니다. | ❌ |
| AUTH_EXPIRED_KEY | 401 | API 키가 만료되었습니다. | API 키 만료일: {expiry_date} | ❌ |
| AUTH_RATE_LIMITED | 429 | 요청이 너무 많습니다. 잠시 후 다시 시도해주세요. | Rate limit 초과: {current}/{limit} per minute | ✅ |

---

### 3.2 입력 유효성 에러 (PARAM_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| PARAM_MISSING | 400 | 필수 항목이 누락되었습니다. | 필수 파라미터 '{field}'가 누락되었습니다. | ❌ |
| PARAM_INVALID_TYPE | 400 | 입력 형식이 올바르지 않습니다. | 필드 '{field}'는 {expected_type} 타입이어야 합니다. (받은 값: {actual_type}) | ❌ |
| PARAM_OUT_OF_RANGE | 400 | 허용 범위를 벗어났습니다. | 필드 '{field}'는 {min}~{max} 범위여야 합니다. (받은 값: {value}) | ❌ |
| PARAM_INVALID_FORMAT | 400 | 형식이 올바르지 않습니다. | 필드 '{field}'의 형식이 맞지 않습니다. 예상 형식: {expected_format} | ❌ |

---

### 3.3 데이터 에러 (DATA_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| DATA_NOT_FOUND | 404 | 요청한 데이터를 찾을 수 없습니다. | 리소스 '{resource_id}'가 존재하지 않습니다. | ❌ |
| DATA_EMPTY | 400 | 데이터가 비어있습니다. | 처리할 데이터가 없습니다. 먼저 데이터를 업로드해주세요. | ❌ |
| DATA_INVALID_CSV | 422 | CSV 파일 형식이 올바르지 않습니다. | CSV 파싱 오류 at line {line}: {reason} | ❌ |
| DATA_DUPLICATE_ROUND | 422 | 이미 존재하는 회차입니다. | 회차 {round_id}가 이미 존재합니다. | ❌ |
| DATA_INVALID_NUMBER | 422 | 번호가 유효하지 않습니다. | 번호는 1~45 범위의 정수여야 합니다. (받은 값: {value}) | ❌ |
| DATA_DUPLICATE_NUMBER | 422 | 중복된 번호가 있습니다. | 동일 회차 내 번호가 중복됩니다: {numbers} | ❌ |
| DATA_INSUFFICIENT | 400 | 데이터가 부족합니다. | 분석에 최소 {min_rounds}회차 데이터가 필요합니다. (현재: {current_rounds}) | ❌ |

---

### 3.4 필터 에러 (FILTER_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| FILTER_SUM_FAIL | 422 | 번호 합계 조건을 만족하지 않습니다. | 합계 {sum}이 허용 범위 [100, 175]를 벗어남 | ❌ |
| FILTER_AC_FAIL | 422 | AC값 조건을 만족하지 않습니다. | AC값 {ac}이 최소값 7 미만 | ❌ |
| FILTER_ZONE_FAIL | 422 | 구간 분포 조건을 만족하지 않습니다. | 구간 {zone}에 {count}개 번호 (최대 3개) | ❌ |
| FILTER_ENDING_FAIL | 422 | 끝수 조건을 만족하지 않습니다. | 동일 끝수 {ending}이 {count}개 (최대 2개) | ❌ |
| FILTER_ODD_EVEN_FAIL | 422 | 홀짝 비율 조건을 만족하지 않습니다. | 홀짝 비율 {ratio} (허용: 2:4 ~ 4:2) | ❌ |
| FILTER_HIGH_LOW_FAIL | 422 | 고저 균형 조건을 만족하지 않습니다. | 고저 비율 {ratio} | ❌ |
| FILTER_HISTORY_FAIL | 422 | 과거 당첨 번호와 너무 유사합니다. | 회차 {round}와 {overlap}개 일치 (최대 4개) | ❌ |
| FILTER_NO_CANDIDATE | 500 | 조건에 맞는 조합을 생성하지 못했습니다. | 필터 통과 조합 0개. 샘플 수 증가 또는 필터 완화 필요 | ✅ |

---

### 3.5 엔진 에러 (ENGINE_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| ENGINE_CALCULATION_ERR | 500 | 계산 중 오류가 발생했습니다. | 점수 계산 실패: {reason} | ✅ |
| ENGINE_SAMPLING_ERR | 500 | 조합 생성 중 오류가 발생했습니다. | Monte Carlo 샘플링 오류: {reason} | ✅ |
| ENGINE_TIMEOUT | 504 | 처리 시간이 초과되었습니다. | 작업 타임아웃: {elapsed}초 (제한: {limit}초) | ✅ |
| ENGINE_CONFIG_INVALID | 500 | 설정값이 유효하지 않습니다. | 설정 검증 실패: {field} = {value} | ❌ |

---

### 3.6 백테스트 에러 (BACKTEST_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| BACKTEST_RANGE_INVALID | 400 | 테스트 범위가 유효하지 않습니다. | start_round({start}) >= end_round({end}) | ❌ |
| BACKTEST_DATA_MISSING | 400 | 테스트 데이터가 부족합니다. | 회차 {round}의 데이터가 없습니다. | ❌ |
| BACKTEST_ALREADY_RUNNING | 409 | 이미 테스트가 진행 중입니다. | 진행 중인 테스트 ID: {test_id} | ✅ |
| BACKTEST_FAILED | 500 | 백테스트 실행에 실패했습니다. | 백테스트 오류: {reason} | ✅ |

---

### 3.7 시스템 에러 (SYS_)

| 코드 | HTTP | 사용자 메시지 | 개발자 메시지 | 재시도 |
|------|------|--------------|--------------|--------|
| SYS_INTERNAL | 500 | 시스템 오류가 발생했습니다. | 내부 서버 오류: {error_id} | ✅ |
| SYS_MAINTENANCE | 503 | 서비스 점검 중입니다. | 예정 종료 시각: {end_time} | ✅ |
| SYS_UNAVAILABLE | 503 | 서비스를 일시적으로 사용할 수 없습니다. | 서비스 상태: {status} | ✅ |

---

## 4. 재시도 가이드라인

### 4.1 재시도 가능 에러 (✅)

| 에러 코드 | 권장 재시도 간격 | 최대 재시도 |
|-----------|-----------------|-------------|
| AUTH_RATE_LIMITED | 60초 | 3회 |
| FILTER_NO_CANDIDATE | 즉시 (설정 변경 후) | 3회 |
| ENGINE_* | 5초 | 3회 |
| BACKTEST_ALREADY_RUNNING | 30초 | 10회 |
| SYS_* | 10초 exponential backoff | 5회 |

### 4.2 Exponential Backoff 공식

```
wait_time = base_delay * (2 ^ attempt) + random_jitter

예: base_delay=1초, attempt=3
    wait_time ≈ 1 * 8 + 0~1초 = 8~9초
```

---

## 5. 에러 처리 예시

### Python

```python
import requests
import time

def call_api_with_retry(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url)
        data = response.json()
        
        if data.get("success"):
            return data
        
        error = data.get("error", {})
        code = error.get("code", "")
        
        # 재시도 불가능한 에러
        if code.startswith(("PARAM_", "DATA_")):
            raise ValueError(error.get("message"))
        
        # 재시도 가능한 에러
        if code in ["AUTH_RATE_LIMITED", "SYS_INTERNAL"]:
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        
        raise Exception(error.get("message"))
    
    raise Exception("최대 재시도 횟수 초과")
```

### JavaScript

```javascript
async function callApiWithRetry(url, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const response = await fetch(url, { method: 'POST' });
    const data = await response.json();
    
    if (data.success) return data;
    
    const { code, message } = data.error || {};
    
    // 재시도 불가능한 에러
    if (code?.startsWith('PARAM_') || code?.startsWith('DATA_')) {
      throw new Error(message);
    }
    
    // 재시도 가능한 에러
    if (['AUTH_RATE_LIMITED', 'SYS_INTERNAL'].includes(code)) {
      await new Promise(r => setTimeout(r, 2 ** attempt * 1000));
      continue;
    }
    
    throw new Error(message);
  }
  
  throw new Error('최대 재시도 횟수 초과');
}
```

---

## 6. 로깅 권장사항

모든 에러는 다음 정보를 포함하여 로깅:

```json
{
  "timestamp": "2026-02-08T22:38:00+09:00",
  "error_id": "err_abc123",
  "code": "ENGINE_CALCULATION_ERR",
  "http_status": 500,
  "request_id": "req_xyz789",
  "user_id": "user_001",
  "endpoint": "/api/v1/recommendations",
  "method": "POST",
  "duration_ms": 1234,
  "stack_trace": "..."
}
```
