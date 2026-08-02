# UZET — ALS 기반 개인화 위젯 추천 엔진

> 원본은 2025.10~12 2인 팀 프로젝트로 진행했으며, 본 레포는 당시 제가 담당한
> 추천 엔진 파트를 개인 학습 목적으로 **재구현**한 것입니다.
> 실서비스 로그는 사용할 수 없어 페르소나 기반 합성 데이터로 대체했습니다.

<!-- TODO: demo.gif 를 여기에. 시각 슬라이더를 움직이면 추천이 바뀌는 장면. -->

---

## 무엇을 푸는가

금융 앱의 메뉴는 깊습니다. 자주 쓰는 기능이 5-Depth 안쪽에 있으면 사용자는
같은 경로를 매번 헤맵니다. 그렇다고 모든 기능을 첫 화면에 늘어놓으면 아무것도
눈에 안 들어옵니다.

**질문: 이 사용자가 _지금_ 쓸 확률이 높은 기능 N개를 어떻게 고를까?**

신호는 두 종류입니다.

| 신호 | 성격 | 예시 | 담당 |
|---|---|---|---|
| 장기 선호 | 느리게 변함 | 이 사람은 평소 환전을 자주 쓴다 | Implicit ALS |
| 단기 맥락 | 빠르게 변함 | 지금은 급여일, 지금은 장중 | 규칙 기반 부스팅 |

둘을 하나의 스코어로 합칩니다.

```
final_score(u, w, ctx) = α·ALS_norm(u,w) + β·Context(w,ctx) + γ·Recency(u,w) − δ·Fatigue(u,w)
```

가중치 α~δ는 전부 [`config/scoring.yaml`](config/scoring.yaml)에 있습니다.
코드에 하드코딩된 튜닝 값은 없습니다.

---

## 왜 Implicit ALS인가

사용자는 위젯에 별점을 주지 않습니다. 클릭했거나 안 했거나뿐입니다.
암묵적 피드백(implicit feedback)에는 명시적 평점 알고리즘을 그대로 쓸 수 없습니다.

1. **부정 신호가 없다.** 클릭 안 한 게 "싫어서"인지 "몰라서"인지 구분 불가
2. **결측이 아니라 0이다.** 안 본 항목이 압도적 다수라 희소 행렬이 됨

Hu-Koren-Volinsky(2008)는 이걸 선호와 신뢰도로 분해합니다.

```
선호   p_ui = 1 if r_ui > 0 else 0
신뢰도 c_ui = 1 + α · r_ui
```

"클릭 0 → 선호 0이지만 확신은 약함", "클릭 50 → 선호 1이고 확신이 강함"으로
다룹니다. 손실 함수에 **관측되지 않은 0까지 전부 포함**되는 게 explicit MF와의
결정적 차이입니다. ALS를 택한 건 한쪽 행렬을 고정하면 다른 쪽이 닫힌 해를 갖는
최소제곱 문제가 되어 이 계산이 병렬화되기 때문입니다.

---

핵심은 **런타임에 모델을 돌리지 않는 것**입니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/architecture-dark.svg">
  <img alt="오프라인 배치로 ALS를 학습해 CandidateStore에 후보를 적재하고, 온라인에서는 조회와 재랭킹만 수행하는 구조" src="docs/images/architecture-light.svg" width="100%">
</picture>

"추천을 실시간으로 계산한다"가 아니라 "실시간 요소만 실시간으로 반영한다"입니다.

저장소는 인터페이스로 추상화해 파일 캐시와 Redis를 바꿔 끼울 수 있습니다
([`src/store/candidate_store.py`](src/store/candidate_store.py)).
Redis 없이도 개발과 테스트가 되고, 같은 계약 테스트를 두 구현에 돌립니다.

캐시에 후보가 없는 신규 유저는 페르소나 기본 위젯 세트로 폴백합니다.

---

## 실행

```bash
pip install -r requirements.txt

# 1. 합성 로그 생성
python -m src.datagen.log_generator --n-users 2000 --days 60

# 2. 학습 + 후보 사전계산
python -m src.batch.precompute --backend file

# 3. 서빙
uvicorn src.api.main:app --reload
```

```bash
# 평일 장중 → 투자 위젯이 위로
curl "http://localhost:8000/recommend/u000001?now=2025-10-15T10:30:00"

# 급여일 → 월급관리/적금이 위로
curl "http://localhost:8000/recommend/u000001?now=2025-10-25T10:00:00"

# 로밍 중 → 환전/해외송금이 위로
curl "http://localhost:8000/recommend/u000001?is_roaming=true"
```

응답의 `parts` 필드에 항별 기여도가 들어 있어 **왜 이 순서인지** 확인할 수 있습니다.

```bash
pytest                 # 전체
pytest -m redis        # Redis 구현 (로컬 Redis 필요)
```

---

## 평가

<!-- TODO: docs/evaluation.md 를 채운 뒤 핵심 표만 여기로 옮기세요 -->

leave-one-out(유저별 **시간상** 마지막 클릭 1건 홀드아웃)으로 측정합니다.
랜덤 분할은 미래 정보가 학습에 새어 들어가 지표를 부풀립니다.

| 모델 | Recall@10 | NDCG@10 | Coverage@10 |
|---|---|---|---|
| 인기순 기준선 | _TBD_ | _TBD_ | _TBD_ |
| ALS (factors=64) | _TBD_ | _TBD_ | _TBD_ |
| ALS + Context | _TBD_ | _TBD_ | _TBD_ |

Coverage를 같이 보는 이유: Recall만 보면 인기 위젯만 계속 미는 모델이
좋아 보입니다. 실제로는 롱테일 기능이 영원히 노출되지 않는 상태입니다.

응답 시간(p50/p95/p99): _TBD_ — `docs/evaluation.md`

---

## 알려진 한계

정직하게 적어둡니다. 면접에서 먼저 말하는 편이 낫습니다.

- **합성 데이터입니다.** 페르소나로 심어 넣은 구조를 모델이 되찾는 구성이라
  실제 사용자 행동의 복잡도(계절성, 이벤트, 유저 취향 변화)는 없습니다.
- **Context 규칙은 하드코딩된 도메인 지식입니다.** 데이터로 학습한 게 아닙니다.
  규칙 수가 늘면 상호작용을 사람이 관리하기 어려워집니다. 다음 단계는
  규칙을 피처로 넣은 랭킹 모델(LambdaMART 등)입니다.
- **오프라인 지표만 있습니다.** 실제 CTR 개선은 A/B 테스트 없이는 알 수 없습니다.
- **급여일을 25일 고정**으로 두었습니다. 실제로는 유저별로 다르고, 입금 내역에서
  추정해야 합니다.

---

## 구조

```
config/     personas.yaml · widgets.yaml · scoring.yaml   ← 모든 튜닝 값
src/
  datagen/  합성 로그 생성기
  model/    ALS 학습 · 평가 지표
  scoring/  Context 규칙 엔진 · 하이브리드 스코어러
  store/    후보 저장소 (파일 / Redis)
  batch/    오프라인 파이프라인
  api/      FastAPI 서빙
tests/      계약 테스트
docs/       구현순서.md · evaluation.md
```

## 참고

- Hu, Koren, Volinsky (2008), *Collaborative Filtering for Implicit Feedback Datasets*
- [`implicit`](https://github.com/benfred/implicit) 라이브러리
