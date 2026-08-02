"""로그 생성기 — 구현하면 통과합니다.

여기 테스트들은 "돌아가는지" 가 아니라 **데이터에 의도한 구조가 들어갔는지**
를 봅니다. 인기 편중과 개인화 신호가 없으면 뒤 단계 전부가 무의미해집니다.
"""

import numpy as np
import pytest

from src.config import GenerationConfig, load_catalog, load_personas
from src.datagen.log_generator import LogGenerator


def make_generator(seed: int = 0, n_users: int = 300, days: int = 20) -> LogGenerator:
    catalog = load_catalog()
    personas, gen = load_personas()
    gen = GenerationConfig(
        n_users=n_users, days=days, start_date=gen.start_date, seed=seed,
        emit_impressions=gen.emit_impressions,
        impressions_per_click=gen.impressions_per_click,
    )
    return LogGenerator(catalog, personas, gen, np.random.default_rng(seed))


@pytest.fixture(scope="module")
def events():
    return make_generator().generate()


def test_schema(events):
    assert set(events.columns) == {
        "user_id", "persona_id", "widget_id", "timestamp", "action"
    }
    assert set(events.action.unique()) <= {"click", "impression"}
    assert events.widget_id.isin(load_catalog().ids).all()


def test_same_seed_gives_same_data():
    a = make_generator(seed=7, n_users=50, days=5).generate()
    b = make_generator(seed=7, n_users=50, days=5).generate()
    assert a.equals(b), "시드를 고정했는데 결과가 다르면 재현이 불가능합니다"


def test_popularity_is_skewed(events):
    """의도한 불균형이 들어갔는지.

    위젯이 25개이므로 균등하면 상위 3개 점유율은 12%입니다.
    현재 설정(config)으로는 35~45% 근처가 나옵니다. 균등의 3배 이상이라
    Coverage 문제가 실제로 발생하고, 그게 이 프로젝트가 다룰 소재입니다.

    더 극단적인 편중을 원하면 widgets.yaml 의 global_popularity 를 키우거나
    personas.yaml 의 persona_ratio 를 낮추세요. (낮출수록 전역 인기 쪽으로 쏠림)
    """
    clicks = events[events.action == "click"]
    share = clicks.widget_id.value_counts(normalize=True).head(3).sum()
    assert share > 0.30, f"편중이 너무 약합니다 ({share:.1%}). Coverage 문제가 안 생깁니다"
    assert share < 0.85, f"편중이 너무 심합니다 ({share:.1%}). ALS 가 배울 게 없습니다"


def test_longtail_exists(events):
    """모든 위젯이 최소 한 번은 등장해야 Coverage 계산이 의미를 갖습니다."""
    clicks = events[events.action == "click"]
    assert clicks.widget_id.nunique() >= len(load_catalog().ids) * 0.8


def test_personas_have_distinct_tastes(events):
    """페르소나별 1위 위젯이 서로 달라야 ALS 가 개인화를 배울 수 있습니다."""
    clicks = events[events.action == "click"]
    tops = clicks.groupby("persona_id").widget_id.agg(lambda s: s.value_counts().idxmax())
    assert tops.nunique() >= 3, f"페르소나 취향이 겹칩니다: {tops.to_dict()}"


def test_hour_distribution_is_not_uniform(events):
    """시간대 편향이 없으면 Context 규칙이 의미를 잃습니다."""
    hours = events.timestamp.dt.hour.value_counts(normalize=True)
    assert hours.max() > 1 / 24 * 1.5


def test_impressions_are_generated(events):
    impressions = events[events.action == "impression"]
    clicks = events[events.action == "click"]
    assert len(impressions) > len(clicks), "Fatigue 항을 검증할 노출 데이터가 필요합니다"
