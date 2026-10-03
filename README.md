# UZET — ALS 기반 개인화 위젯 추천 엔진

금융 앱의 위젯을 사용자 선호와 요청 시점의 맥락에 따라 정렬하는 Python/FastAPI 학습 프로젝트입니다.
2025년 팀 프로젝트에서 담당한 추천 엔진 영역을 개인적으로 재구현했습니다. 실서비스 로그 대신 합성 데이터를 사용합니다.

## 구현된 최소 흐름

```text
합성 로그 → ALS 학습 → 사용자별 후보·사용 이력 저장
                       ↓
FastAPI → 파일 저장소 조회 → Context/Recency/Fatigue 재랭킹 → 추천 응답
```

- Implicit ALS: 클릭 횟수를 confidence로 변환하고 user×item 희소 행렬 학습
- Context: 날짜·시간·요일·로밍 규칙을 YAML에서 읽어 부스팅 합산
- Recency: 마지막 클릭의 지수 감쇠 / Fatigue: 미클릭 노출의 포화 페널티
- 신규 사용자: 페르소나를 추정하지 않고 카탈로그의 기본 인기 가중치와 Context로 추천
- 후보 카테고리 상한 준수: 후보가 부족하면 요청한 수보다 적게 반환
- Recall@K, NDCG@K, Coverage@K와 시간 기반 사용자별 홀드아웃 평가

`parts`는 가중치가 반영된 각 항의 기여도이며 합계가 최종 점수입니다.
신규 사용자 응답은 ALS 대신 `prior` 항을 표시합니다.

## 빠른 실행

Python 3.12 기준입니다. Redis·Jupyter·데모 도구 없이 실행하려면:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
python -m src.datagen.log_generator --n-users 100 --days 14 --seed 42
python -m src.batch.precompute --backend file --factors 16
uvicorn src.api.main:app --reload
```

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommend/u000001?n=6&now=2025-10-15T10:30:00"
curl "http://localhost:8000/recommend/new-user?n=6&now=2025-10-25T10:00:00"
pytest -m "not redis"
python -m src.model.evaluator --factors 16 --k 10
```

시간대 없는 시각은 한국 시간으로 해석하고, 시간대가 있으면 Asia/Seoul로 변환합니다.
과거 데이터로 미래 시각을 재현하는 실험용 `now` 파라미터이며 운영 환경용 인증은 포함하지 않습니다.

## 검증 결과

합성 유저 100명·14일 로그로 학습→후보 저장을 실행했고, 파일 저장소 재로딩과 추천 API까지 통합 테스트했습니다.
ALS Recall@10=0.8200, 인기순=0.6500이며 **작은 합성 데이터 실험 결과**입니다.
[평가 조건과 상세 결과](docs/evaluation.md)를 함께 확인하세요. 실서비스 성능·CTR 개선은 주장하지 않습니다.

## 범위와 한계

- Redis 저장소와 Streamlit 결과 렌더링은 미구현이며 파일 저장소만 검증했습니다.
- 파일 저장소는 시작할 때 읽습니다. 배치 갱신 후 API를 다시 시작해야 합니다.
- pickle 산출물은 이 프로젝트에서 직접 생성한 신뢰할 수 있는 로컬 파일만 사용합니다.
- Fatigue는 전체 기간의 미클릭 노출 단순 집계입니다. 세션 단위 상쇄·시간 감쇠는 추후 개선 대상입니다.
- 급여일·Context는 고정 규칙이며 월 경계를 넘는 날짜 근접 판정은 지원하지 않습니다.
- 평가 분할은 사용자별 기준이며 전역 시간 분할·실사용 A/B 테스트는 아직 없습니다.
- 인증·운영 배포·부하 테스트는 범위 밖입니다.

## 코드 살펴보기

- [학습](src/model/als_trainer.py) / [평가](src/model/evaluator.py)
- [규칙](src/scoring/context_rules.py) / [점수화](src/scoring/hybrid_scorer.py)
- [배치](src/batch/precompute.py) / [API](src/api/main.py)
- [통합 테스트](tests/test_pipeline.py)
