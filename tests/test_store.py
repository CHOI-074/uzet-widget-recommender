"""저장소 — FileCandidateStore 는 참조 구현이라 지금 바로 통과합니다.

RedisCandidateStore 를 구현하면 test_redis_roundtrip 도 같은 계약을 통과해야
합니다. 같은 테스트를 두 구현에 돌리는 게 인터페이스 분리의 실익입니다.
"""

from datetime import datetime

import pytest

from src.scoring.hybrid_scorer import UserState
from src.store.candidate_store import (
    FileCandidateStore,
    decode_state,
    encode_state,
)


def sample_state() -> UserState:
    return UserState(
        als_scores={"acct_balance": 0.9, "savings": 0.4},
        last_used={"acct_balance": datetime(2025, 10, 20, 9, 30)},
        unclicked_impressions={"savings": 3},
    )


def test_encode_decode_roundtrip():
    state, persona = decode_state(encode_state(sample_state(), "rookie"))
    assert persona == "rookie"
    assert state.als_scores == sample_state().als_scores
    assert state.last_used == sample_state().last_used
    assert state.unclicked_impressions == sample_state().unclicked_impressions


def test_file_store_roundtrip(tmp_path):
    store = FileCandidateStore(tmp_path / "cand.pkl")
    store.put("u1", sample_state(), "rookie")
    got = store.get("u1")
    assert got is not None
    assert got[1] == "rookie"
    assert got[0].als_scores == sample_state().als_scores


def test_file_store_misses_return_none(tmp_path):
    """콜드 스타트 폴백은 이 None 을 신호로 씁니다."""
    store = FileCandidateStore(tmp_path / "cand.pkl")
    assert store.get("없는유저") is None


def test_file_store_persists_across_instances(tmp_path):
    path = tmp_path / "cand.pkl"
    FileCandidateStore(path).bulk_put({"u1": (sample_state(), "rookie")})
    assert FileCandidateStore(path).get("u1") is not None


@pytest.mark.redis
def test_redis_roundtrip():
    """로컬에 Redis 가 떠 있을 때만: pytest -m redis"""
    redis = pytest.importorskip("redis")
    from src.store.candidate_store import RedisCandidateStore

    try:
        store = RedisCandidateStore()
        store.client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis 미기동")

    store.put("test_u1", sample_state(), "rookie")
    got = store.get("test_u1")
    assert got is not None and got[1] == "rookie"
    assert store.get("존재하지_않는_유저") is None
    store.client.delete(store._key("test_u1"))
    store.close()
