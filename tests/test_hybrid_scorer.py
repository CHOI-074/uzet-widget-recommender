"""하이브리드 스코어러 — 구현하면 통과합니다.

마지막 테스트(test_context_changes_ranking)가 이 프로젝트의 핵심 주장을
검증합니다. **같은 유저, 다른 시각 → 다른 추천.** 이게 안 되면 Context 결합이
동작하지 않는다는 뜻이고, 프로젝트의 존재 이유가 사라집니다.
"""

from datetime import datetime, timedelta

import pytest

from src.config import load_scoring
from src.scoring.context_rules import RequestContext
from src.scoring.hybrid_scorer import HybridScorer, UserState

NOW = datetime(2025, 10, 15, 10, 30)  # 평일 장중


@pytest.fixture
def scorer():
    return HybridScorer()


def test_normalize_als_maps_to_unit_range(scorer):
    out = scorer.normalize_als({"a": 2.0, "b": 0.0, "c": 1.0})
    assert out["a"] == pytest.approx(1.0)
    assert out["b"] == pytest.approx(0.0)
    assert out["c"] == pytest.approx(0.5)


def test_normalize_als_handles_flat_input(scorer):
    """전부 같은 값이면 0으로 나누게 됩니다. 죽지만 않으면 됩니다."""
    out = scorer.normalize_als({"a": 1.0, "b": 1.0})
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_recency_decays_by_half_life(scorer):
    half_life = load_scoring().half_life_hours
    assert scorer.recency(NOW, NOW) == pytest.approx(1.0)
    assert scorer.recency(NOW - timedelta(hours=half_life), NOW) == pytest.approx(0.5)
    assert scorer.recency(None, NOW) == 0.0
    assert scorer.recency(NOW - timedelta(hours=half_life * 4), NOW) < 0.1


def test_fatigue_saturates(scorer):
    sat = load_scoring().fatigue_saturation
    assert scorer.fatigue(0) == 0.0
    assert scorer.fatigue(sat) == pytest.approx(1.0)
    assert scorer.fatigue(sat * 10) == pytest.approx(1.0)  # 1.0 을 넘으면 안 됨


def test_score_is_sorted_descending(scorer):
    state = UserState(als_scores={"acct_balance": 1.0, "savings": 0.5, "fund": 0.1})
    scored = scorer.score(state, RequestContext(now=NOW))
    assert [s.score for s in scored] == sorted([s.score for s in scored], reverse=True)


def test_score_parts_are_reported(scorer):
    state = UserState(als_scores={"acct_balance": 1.0, "stock_status": 0.9})
    top = scorer.score(state, RequestContext(now=NOW))[0]
    assert set(top.parts) >= {"als", "context", "recency", "fatigue"}


def test_context_changes_ranking(scorer):
    """이 프로젝트의 핵심 주장을 검증하는 테스트.

    ALS 점수는 동일한데 시각만 바꿉니다.
    장중에는 stock_status 가, 급여일에는 salary_manage 가 위로 와야 합니다.
    """
    state = UserState(
        als_scores={"stock_status": 0.55, "salary_manage": 0.55, "acct_balance": 0.6}
    )
    market = datetime(2025, 10, 15, 10, 30)   # 평일 장중
    payday = datetime(2025, 10, 25, 10, 0)    # 급여일

    rank_market = [s.widget_id for s in scorer.score(state, RequestContext(now=market))]
    rank_payday = [s.widget_id for s in scorer.score(state, RequestContext(now=payday))]

    assert rank_market.index("stock_status") < rank_market.index("salary_manage")
    assert rank_payday.index("salary_manage") < rank_payday.index("stock_status")


def test_fatigue_pushes_widget_down(scorer):
    """노출만 되고 안 눌린 위젯은 옆자리에 밀려야 합니다.

    1·2위 ALS 점수를 아주 가깝게(정규화 후 차이 0.02) 두었습니다.
    delta 페널티가 그보다 크므로 순위가 뒤집혀야 정상입니다.
    """
    scores = {"acct_balance": 1.0, "acct_transfer": 0.98, "savings": 0.0}
    ctx = RequestContext(now=NOW)

    normal = UserState(als_scores=dict(scores))
    fatigued = UserState(
        als_scores=dict(scores),
        unclicked_impressions={"acct_balance": 99},
    )

    assert scorer.score(normal, ctx)[0].widget_id == "acct_balance"
    assert scorer.score(fatigued, ctx)[0].widget_id == "acct_transfer"


def test_diversify_respects_category_cap(scorer):
    cap = load_scoring().max_per_category
    if cap <= 0:
        pytest.skip("max_per_category 비활성")
    # 계좌 카테고리 4개 + 다른 카테고리
    state = UserState(als_scores={
        "acct_balance": 1.0, "acct_transfer": 0.99, "acct_history": 0.98,
        "acct_auto_transfer": 0.97, "savings": 0.5,
    })
    picked = scorer.recommend(state, RequestContext(now=NOW), n=4)
    accounts = [p for p in picked if p.widget_id.startswith("acct_")]
    assert len(accounts) <= cap
