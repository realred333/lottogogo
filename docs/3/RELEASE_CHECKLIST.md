# LottoGoGo 릴리즈 체크리스트 (RELEASE_CHECKLIST.md)

## 1. 배포 전 체크리스트

### 1.1 코드 품질
- [ ] 모든 단위 테스트 통과 (pytest)
- [ ] 통합 테스트 통과
- [ ] E2E 테스트 통과
- [ ] 코드 커버리지 85% 이상
- [ ] 핵심 테스트 10개 전체 통과
- [ ] 린터 경고 0개 (flake8/pylint)
- [ ] 타입 힌트 검증 (mypy)

### 1.2 문서화
- [ ] README.md 업데이트
- [ ] API 문서 최신화
- [ ] CHANGELOG.md 작성
- [ ] 버전 번호 업데이트 (pyproject.toml)

### 1.3 보안
- [ ] 하드코딩된 민감정보 없음
- [ ] Config 파일 템플릿 제공
- [ ] .gitignore 유효성 확인

### 1.4 성능
- [ ] 벤치마크 테스트 통과
  - [ ] 점수 계산 < 100ms
  - [ ] 샘플링 100K < 5s
  - [ ] 전체 추천 < 10s
- [ ] 메모리 사용량 적정 (< 500MB)

### 1.5 재현성
- [ ] Seed 고정 테스트 통과
- [ ] 동일 입력 → 동일 출력 검증

### 1.6 백테스트 결과
- [ ] P(match ≥ 3) 기준선 대비 개선
- [ ] 결과 리포트 생성 및 저장

---

## 2. 배포 절차

### 2.1 릴리즈 브랜치 준비
```bash
# 1. 메인에서 릴리즈 브랜치 생성
git checkout main
git pull origin main
git checkout -b release/v1.0.0

# 2. 버전 업데이트
# pyproject.toml의 version 필드 수정
```

### 2.2 최종 검증
```bash
# 3. 전체 테스트 실행
pytest --cov=src --cov-report=html

# 4. 린터 확인
flake8 src/
mypy src/

# 5. 패키지 빌드 테스트
python -m build
```

### 2.3 태그 및 릴리즈
```bash
# 6. 태그 생성
git tag -a v1.0.0 -m "Release v1.0.0"

# 7. 푸시
git push origin release/v1.0.0
git push origin v1.0.0
```

### 2.4 배포 후 확인
- [ ] 패키지 설치 테스트 (새 환경)
- [ ] 기본 사용 시나리오 확인
- [ ] 문서 접근 확인

---

## 3. 롤백 절차

### 3.1 즉시 롤백 (Hotfix)
```bash
# 이전 안정 버전으로 복귀
git checkout v0.x.x  # 이전 버전 태그
git checkout -b hotfix/rollback
```

### 3.2 롤백 트리거 조건
| 조건 | 심각도 | 대응 |
|------|--------|------|
| 핵심 기능 실패 | Critical | 즉시 롤백 |
| 성능 50% 이상 저하 | High | 4시간 내 롤백 |
| 재현성 실패 | High | 조사 후 롤백 결정 |
| 비핵심 버그 | Medium | 패치 릴리즈 |

### 3.3 롤백 검증
- [ ] 이전 버전 테스트 통과
- [ ] 데이터 호환성 확인
- [ ] 사용자 영향 범위 파악

### 3.4 롤백 커뮤니케이션
```markdown
## 롤백 공지 템플릿
- 릴리즈 버전: v1.0.0
- 롤백 버전: v0.x.x
- 사유: [상세 설명]
- 영향 범위: [설명]
- 예상 복구: [일정]
```

---

## 4. 모니터링

### 4.1 핵심 메트릭
| 메트릭 | 설명 | 임계값 |
|--------|------|--------|
| 실행 성공률 | 추천 생성 성공 비율 | > 99% |
| 평균 실행 시간 | 전체 파이프라인 시간 | < 10s |
| 메모리 사용량 | 피크 메모리 | < 500MB |
| 필터 통과율 | 필터 후 남은 비율 | > 1% |

### 4.2 로깅 수준
```yaml
logging:
  level: INFO  # 운영
  debug_mode: false
  
  # 상세 로깅 대상
  modules:
    - score_engine: DEBUG  # 점수 분석 시
    - filter_pipeline: INFO
    - sampler: INFO
```

### 4.3 알림 설정
| 이벤트 | 알림 채널 | 우선순위 |
|--------|-----------|----------|
| 실행 실패 | Slack + Email | High |
| 성능 저하 | Slack | Medium |
| 필터 통과율 저하 | Dashboard | Low |

### 4.4 건강성 체크
```python
# 자가 진단 스크립트
def health_check():
    checks = {
        "csv_loadable": test_csv_load(),
        "config_valid": test_config(),
        "score_engine": test_score(),
        "sampler": test_sample(),
        "filters": test_filters(),
    }
    return all(checks.values()), checks
```

### 4.5 정기 점검 항목
- [ ] 일간: 실행 로그 확인
- [ ] 주간: 성능 메트릭 리뷰
- [ ] 월간: 백테스트 재실행 및 비교

---

## 5. 긴급 연락망

| 역할 | 담당 | 연락처 |
|------|------|--------|
| 1차 대응 | 개발자 | - |
| 에스컬레이션 | 팀 리드 | - |
| 최종 결정권자 | PM | - |

---

## 6. 릴리즈 승인

| 항목 | 확인자 | 서명 | 날짜 |
|------|--------|------|------|
| 코드 리뷰 완료 | | | |
| QA 테스트 통과 | | | |
| 성능 기준 충족 | | | |
| 문서화 완료 | | | |
| 최종 릴리즈 승인 | | | |
