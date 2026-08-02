"""평가 지표 — 구현하면 통과합니다.

지표는 손으로 계산할 수 있는 작은 예제로 검증하세요. 큰 데이터로만 돌리면
공식이 틀려도 "그럴듯한 숫자" 가 나와서 못 잡습니다.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.model.evaluator import (
    coverage_at_k,
    leave_one_out_split,
    ndcg_at_k,
    recall_at_k,
)

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_hit_and_miss():
    assert recall_at_k(RANKED, "c", 3) == 1.0
    assert recall_at_k(RANKED, "d", 3) == 0.0
    assert recall_at_k(RANKED, "zzz", 5) == 0.0


def test_ndcg_matches_hand_calculation():
    # 1위면 1/log2(2) = 1.0
    assert ndcg_at_k(RANKED, "a", 5) == pytest.approx(1.0)
    # 3위면 1/log2(4) = 0.5
    assert ndcg_at_k(RANKED, "c", 5) == pytest.approx(0.5)
    # K 밖이면 0
    assert ndcg_at_k(RANKED, "e", 3) == 0.0


def test_ndcg_is_never_greater_than_recall():
    for truth in RANKED:
        assert ndcg_at_k(RANKED, truth, 5) <= recall_at_k(RANKED, truth, 5)


def test_coverage():
    # 두 유저에게 각각 상위 2개씩 → 고유 3개 / 카탈로그 10개
    assert coverage_at_k([["a", "b"], ["a", "c"]], 10, 2) == pytest.approx(0.3)


def _events():
    base = datetime(2025, 10, 1, 9, 0)
    rows = [
        ("u1", "w_a", base),
        ("u1", "w_b", base + timedelta(days=1)),
        ("u1", "w_c", base + timedelta(days=2)),   # u1 의 정답
        ("u2", "w_a", base),
        ("u2", "w_d", base + timedelta(days=5)),   # u2 의 정답
        ("u3", "w_a", base),                       # 클릭 1건 → 평가 제외
    ]
    return pd.DataFrame(
        [{"user_id": u, "widget_id": w, "timestamp": t, "action": "click"}
         for u, w, t in rows]
    )


def test_leave_one_out_takes_the_last_click():
    train, test = leave_one_out_split(_events())
    truth = dict(zip(test.user_id, test.widget_id))
    assert truth["u1"] == "w_c"
    assert truth["u2"] == "w_d"
    assert "u3" not in truth, "클릭 1건짜리 유저는 테스트에서 빠져야 합니다"
    assert len(test) == 2
    assert len(train) == len(_events()) - 2


def test_leave_one_out_has_no_leakage():
    """테스트 이벤트가 학습에 남아 있으면 지표가 부풀려집니다."""
    train, test = leave_one_out_split(_events())
    for row in test.itertuples():
        overlap = train[(train.user_id == row.user_id)
                        & (train.widget_id == row.widget_id)
                        & (train.timestamp == row.timestamp)]
        assert overlap.empty
