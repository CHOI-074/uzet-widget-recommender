"""설정 무결성 — 지금 바로 통과해야 하는 테스트.

YAML 오타는 런타임에 "규칙이 왜 안 걸리지?" 로만 드러나서 잡기 어렵습니다.
여기서 미리 걸러냅니다.
"""

from src.config import load_catalog, load_personas, load_scoring


def test_widget_ids_unique():
    ids = load_catalog().ids
    assert len(ids) == len(set(ids))


def test_popularity_covers_all_widgets():
    catalog = load_catalog()
    assert set(catalog.global_popularity) == set(catalog.ids)


def test_persona_references_are_valid():
    catalog_ids = set(load_catalog().ids)
    personas, _ = load_personas()
    assert personas, "페르소나가 하나도 없습니다"
    for p in personas:
        assert len(p.hour_weights) == 24, f"{p.id}: hour_weights 길이가 24가 아님"
        assert set(p.affinity) <= catalog_ids, f"{p.id}: affinity 에 없는 위젯"
        assert set(p.fallback) <= catalog_ids, f"{p.id}: fallback 에 없는 위젯"
        assert 0.0 <= p.persona_ratio <= 1.0


def test_context_rule_targets_are_valid():
    catalog_ids = set(load_catalog().ids)
    known_types = {
        "day_of_month_near", "day_of_month_gte", "days_before_day",
        "time_range", "weekday", "flag",
    }
    for rule in load_scoring().rules:
        assert set(rule.boost) <= catalog_ids, f"{rule.name}: 없는 위젯을 부스팅"
        for cond in rule.conditions:
            assert cond["type"] in known_types, f"{rule.name}: 미지원 조건 {cond['type']}"
