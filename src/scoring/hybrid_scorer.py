"""Step 3 — 하이브리드 스코어러.

    final_score(u, w, ctx) = alpha * ALS_norm(u, w)
                           + beta  * Context_boost(w, ctx)
                           + gamma * Recency(u, w)
                           - delta * Fatigue(u, w)

이 프로젝트의 핵심 아이디어가 이 한 줄입니다. 느린 신호(ALS)와 빠른
신호(Context)를 **하나의 스코어로 합친다**는 것.

면접에서 나올 반문 두 개를 미리 준비하세요.
  Q. "그냥 규칙만 쓰면 안 되나요?"
     → 규칙은 '모두에게 같은' 추천만 합니다. 급여일에 전 유저가 같은 화면을
       봅니다. 개인화가 없습니다.
  Q. "그냥 ALS 만 쓰면 안 되나요?"
     → ALS 는 하루 한 번 학습되므로 '지금 장이 열렸다'를 모릅니다.
       그리고 콜드 스타트 유저에게 줄 게 없습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config import ScoringConfig, WidgetCatalog, load_catalog, load_scoring
from src.scoring.context_rules import ContextEngine, RequestContext


@dataclass
class UserState:
    """온라인 재랭킹에 필요한 유저별 최소 상태. 후보와 함께 캐시에 저장됩니다."""

    als_scores: dict[str, float] = field(default_factory=dict)
    last_used: dict[str, datetime] = field(default_factory=dict)
    unclicked_impressions: dict[str, int] = field(default_factory=dict)


@dataclass
class ScoredWidget:
    widget_id: str
    score: float
    parts: dict[str, float]  # 항별 기여도 — 디버깅과 데모용

    def as_dict(self) -> dict:
        return {"widget_id": self.widget_id, "score": round(self.score, 5),
                "parts": {k: round(v, 5) for k, v in self.parts.items()}}


class HybridScorer:
    def __init__(
        self,
        config: ScoringConfig | None = None,
        catalog: WidgetCatalog | None = None,
        engine: ContextEngine | None = None,
    ) -> None:
        self.config = config or load_scoring()
        self.catalog = catalog or load_catalog()
        self.engine = engine or ContextEngine(self.config)

    # ------------------------------------------------------------------
    # 여기서부터 직접 구현
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_als(scores: dict[str, float]) -> dict[str, float]:
        """ALS raw 점수를 0~1 로 min-max 정규화.

        왜 정규화가 필요한가: ALS 점수의 절대 스케일은 factors, regularization,
        학습 데이터 양에 따라 매번 달라집니다. 정규화 없이 beta 를 튜닝하면
        재학습할 때마다 가중치를 다시 잡아야 합니다.

        주의: 후보가 1개뿐이거나 전부 같은 값이면 max == min 이라 0으로 나눕니다.
        그 경우 전부 1.0 을 주든 0.5 를 주든 정하고, 이유를 주석에 남기세요.

        TODO: 구현.
        """
        raise NotImplementedError

    def recency(self, last_used: datetime | None, now: datetime) -> float:
        """마지막 사용 이후 경과시간의 지수 감쇠. 0~1.

            score = 0.5 ** (경과시간_시간 / half_life_hours)

        한 번도 안 쓴 위젯(last_used is None)은 0.0.
        미래 시각이 들어오면(시계 오차) 1.0 으로 클램프하세요.

        TODO: 구현. (self.config.half_life_hours)
        """
        raise NotImplementedError

    def fatigue(self, unclicked: int) -> float:
        """노출됐는데 클릭 안 한 횟수 → 0~1 페널티.

            score = min(1.0, unclicked / saturation)

        같은 위젯을 계속 위에 띄웠는데 아무도 안 누르면 자리를 내주게 만드는
        장치입니다. 추천 시스템의 '필터 버블/고착' 완화 파트로 설명하면 좋습니다.

        TODO: 구현. (self.config.fatigue_saturation)
        """
        raise NotImplementedError

    def score(
        self,
        state: UserState,
        ctx: RequestContext,
        candidates: list[str] | None = None,
    ) -> list[ScoredWidget]:
        """후보 위젯들의 최종 스코어를 계산해 **내림차순 정렬**해 반환합니다.

        순서
            1. candidates 가 None 이면 state.als_scores 의 키를 후보로 사용
            2. als_norm = self.normalize_als(후보에 한정한 als_scores)
            3. boosts = self.engine.evaluate(ctx)   ← 요청당 1회만 호출!
               (위젯마다 호출하면 규칙 평가가 N배로 늘어납니다)
            4. 각 후보 w 에 대해
                 s = alpha * als_norm.get(w, 0)
                   + beta  * boosts.get(w, 0)
                   + gamma * self.recency(state.last_used.get(w), ctx.now)
                   - delta * self.fatigue(state.unclicked_impressions.get(w, 0))
            5. ScoredWidget(w, s, parts={"als":..., "context":..., ...}) 로 담기
               parts 를 채워두면 데모에서 "왜 이게 1위인지" 를 보여줄 수 있습니다.
               이거 하나가 README 설득력의 절반입니다.

        TODO: 구현.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 아래는 완성된 배관
    # ------------------------------------------------------------------

    def diversify(self, scored: list[ScoredWidget], n: int) -> list[ScoredWidget]:
        """카테고리 쏠림 방지 — 점수 순으로 담되 같은 카테고리는 상한까지만.

        max_per_category = 0 이면 그냥 상위 n 개.
        """
        cap = self.config.max_per_category
        if cap <= 0:
            return scored[:n]

        picked: list[ScoredWidget] = []
        used: dict[str, int] = {}
        overflow: list[ScoredWidget] = []

        for sw in scored:
            cat = self.catalog.category_of(sw.widget_id)
            if used.get(cat, 0) < cap:
                picked.append(sw)
                used[cat] = used.get(cat, 0) + 1
            else:
                overflow.append(sw)
            if len(picked) == n:
                return picked

        # 상한 때문에 n 개를 못 채웠으면 밀려난 것들로 채웁니다.
        picked.extend(overflow[: n - len(picked)])
        return picked

    def recommend(
        self,
        state: UserState,
        ctx: RequestContext,
        n: int | None = None,
        candidates: list[str] | None = None,
    ) -> list[ScoredWidget]:
        n = n or self.config.default_n
        return self.diversify(self.score(state, ctx, candidates), n)


def fallback_widgets(persona_id: str, n: int) -> list[str]:
    """콜드 스타트 폴백 — 캐시에 후보가 없을 때 페르소나 기본 세트를 내려줍니다.

    설계서 5장. 신규 유저에게 빈 화면을 주는 것보다 훨씬 낫고, 면접에서
    "콜드 스타트는 어떻게 처리했나요?" 에 답할 재료가 됩니다.
    """
    from src.config import persona_by_id

    return persona_by_id(persona_id).fallback[:n]
