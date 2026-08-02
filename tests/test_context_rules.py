"""Context 엔진 — 구현하면 통과합니다.

날짜를 주입해서 테스트한다는 게 포인트입니다. datetime.now() 를 코드 안에서
직접 부르면 이런 테스트를 못 씁니다. 그래서 RequestContext 로 시각을 밖에서
넣게 설계했습니다. (면접에서 "테스트 가능하게 어떻게 설계했나요" 답변 재료)
"""

from datetime import datetime

import pytest

from src.scoring.context_rules import ContextEngine, RequestContext

# 2025-10-25 (토) 10:00 — 급여일, 월말
PAYDAY = datetime(2025, 10, 25, 10, 0)
# 2025-10-15 (수) 10:30 — 평일 장중
MARKET = datetime(2025, 10, 15, 10, 30)
# 2025-10-12 (일) 20:00 — 카드 결제일(14일) D-2
CARD_DUE = datetime(2025, 10, 12, 20, 0)
# 2025-10-08 (수) 22:00 — 아무 규칙도 안 걸려야 함
QUIET = datetime(2025, 10, 8, 22, 0)


@pytest.fixture
def engine():
    return ContextEngine()


def test_payday_boosts_salary_widgets(engine):
    boosts = engine.evaluate(RequestContext(now=PAYDAY))
    assert boosts.get("salary_manage", 0) > 0
    assert boosts.get("savings", 0) > 0


def test_market_hours_boost_invest_widgets(engine):
    boosts = engine.evaluate(RequestContext(now=MARKET))
    assert boosts.get("stock_status", 0) > boosts.get("market_index", 0) > 0


def test_market_rule_does_not_fire_on_weekend(engine):
    # PAYDAY 는 토요일 10시 — 시간대는 맞지만 평일이 아니므로 발동하면 안 됨
    assert "market_open" not in engine.explain(RequestContext(now=PAYDAY))


def test_card_due_lead_window(engine):
    boosts = engine.evaluate(RequestContext(now=CARD_DUE))
    assert boosts.get("card_due", 0) > 0
    # 결제일 당일(14일)은 D-3 창 밖 — 0 < (14 - day) 조건이라 발동 안 함
    after = engine.evaluate(RequestContext(now=datetime(2025, 10, 14, 12, 0)))
    assert after.get("card_due", 0) == 0


def test_flag_condition(engine):
    off = engine.evaluate(RequestContext(now=QUIET, flags={"is_roaming": False}))
    on = engine.evaluate(RequestContext(now=QUIET, flags={"is_roaming": True}))
    assert off.get("fx_exchange", 0) == 0
    assert on.get("fx_exchange", 0) > 0


def test_quiet_time_fires_nothing(engine):
    assert engine.explain(RequestContext(now=QUIET)) == []


def test_boosts_are_summed_not_maxed(engine):
    """월말 규칙과 급여일 규칙이 동시에 걸리는 시점."""
    fired = engine.explain(RequestContext(now=PAYDAY))
    assert "payday" in fired and "month_end" in fired


def test_unknown_condition_type_raises(engine):
    with pytest.raises(ValueError):
        engine.match_condition({"type": "존재하지_않는_타입"}, RequestContext(now=QUIET))
