"""Step 4 — 후보 저장소.

10ms 를 만드는 구조의 핵심은 "무거운 건 오프라인, 온라인은 조회 + 가벼운 재랭킹"
입니다. 그 경계에 있는 게 이 저장소입니다.

    오프라인(배치, 일 1회)   ALS 학습 → 유저별 Top-K 후보 → put()
    온라인(요청, ~10ms)      get() → Context 평가 → 재랭킹 → 응답

인터페이스로 추상화한 이유
  - Redis 없이도 개발/테스트가 됩니다 (FileCandidateStore)
  - 면접에서 "왜 Redis 죠?" 에 "조회가 O(1)이고 TTL 이 공짜라서" 라고 답하되,
    "다만 우리 규모에선 파일 캐시로도 충분했고, 인터페이스를 분리해서 언제든
    바꿀 수 있게 했다" 까지 말할 수 있으면 훨씬 좋습니다.
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from src.scoring.hybrid_scorer import UserState


# ----------------------------------------------------------------------
# 직렬화 — 완성된 배관 (두 구현이 공유합니다)
# ----------------------------------------------------------------------


def encode_state(state: UserState, persona_id: str) -> str:
    return json.dumps(
        {
            "persona_id": persona_id,
            "als_scores": state.als_scores,
            "last_used": {k: v.isoformat() for k, v in state.last_used.items()},
            "unclicked": state.unclicked_impressions,
        },
        ensure_ascii=False,
    )


def decode_state(blob: str) -> tuple[UserState, str]:
    raw: dict[str, Any] = json.loads(blob)
    state = UserState(
        als_scores={k: float(v) for k, v in raw["als_scores"].items()},
        last_used={k: datetime.fromisoformat(v) for k, v in raw["last_used"].items()},
        unclicked_impressions={k: int(v) for k, v in raw["unclicked"].items()},
    )
    return state, raw["persona_id"]


class CandidateStore(ABC):
    """구현체는 이 3개만 채우면 됩니다."""

    @abstractmethod
    def put(self, user_id: str, state: UserState, persona_id: str) -> None: ...

    @abstractmethod
    def get(self, user_id: str) -> tuple[UserState, str] | None: ...

    @abstractmethod
    def bulk_put(self, items: dict[str, tuple[UserState, str]]) -> None: ...

    def close(self) -> None:  # 선택
        pass


# ----------------------------------------------------------------------
# 참조 구현 — 완성해 뒀습니다. Redis 구현의 본보기로 보세요.
# ----------------------------------------------------------------------


class FileCandidateStore(CandidateStore):
    """단일 pickle 파일 + 메모리 dict.

    프로세스 시작 시 한 번 읽어 메모리에 올리므로 조회는 dict 접근입니다.
    (즉 이 구현으로도 10ms 는 나옵니다. 다만 여러 API 프로세스가 각자
     사본을 들고 있어 배치 갱신이 즉시 반영되지 않는다는 한계가 있고,
     그게 바로 Redis 를 쓰는 이유입니다 — 이 문장을 README 에 쓰세요)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            with path.open("rb") as f:
                self._data = pickle.load(f)

    def put(self, user_id: str, state: UserState, persona_id: str) -> None:
        self._data[user_id] = encode_state(state, persona_id)
        self._flush()

    def bulk_put(self, items: dict[str, tuple[UserState, str]]) -> None:
        for uid, (state, persona_id) in items.items():
            self._data[uid] = encode_state(state, persona_id)
        self._flush()

    def get(self, user_id: str) -> tuple[UserState, str] | None:
        blob = self._data.get(user_id)
        return decode_state(blob) if blob else None

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as f:
            pickle.dump(self._data, f)


# ----------------------------------------------------------------------
# 여기서부터 직접 구현
# ----------------------------------------------------------------------


class RedisCandidateStore(CandidateStore):
    """Redis 구현.

    키 설계
        cand:{user_id}  →  encode_state() 결과 문자열
        TTL 은 배치 주기(1일)의 2~3배로 잡습니다. 배치가 한 번 실패해도
        캐시가 통째로 비지 않게 하는 안전장치입니다.

    구현 시 확인할 것
      - bulk_put 은 반드시 **파이프라인**으로 보내세요. 2000명을 SET 2000번
        왕복하면 배치가 느려집니다. `with self.client.pipeline() as pipe:`
      - decode_bytes: redis-py 는 기본이 bytes 반환입니다.
        생성자에서 decode_responses=True 를 주면 str 로 받습니다.
      - get() 은 키가 없으면 None 을 반환해야 합니다 (콜드 스타트 폴백 신호).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 60 * 60 * 72,
        prefix: str = "cand:",
    ) -> None:
        import redis  # 지연 임포트 — Redis 안 쓸 땐 패키지가 없어도 됩니다

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.ttl = ttl_seconds
        self.prefix = prefix

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}{user_id}"

    def put(self, user_id: str, state: UserState, persona_id: str) -> None:
        """TODO: SETEX 한 방. (self.client.set(key, value, ex=self.ttl))"""
        raise NotImplementedError

    def bulk_put(self, items: dict[str, tuple[UserState, str]]) -> None:
        """TODO: 파이프라인으로 일괄 SET."""
        raise NotImplementedError

    def get(self, user_id: str) -> tuple[UserState, str] | None:
        """TODO: GET 후 decode_state. 없으면 None."""
        raise NotImplementedError

    def close(self) -> None:
        self.client.close()


# ----------------------------------------------------------------------


def build_store(backend: str, path: Path | None = None, url: str | None = None) -> CandidateStore:
    """설정 문자열 하나로 백엔드를 바꿉니다. API 와 배치가 같이 씁니다."""
    if backend == "file":
        from src.config import DATA_DIR

        return FileCandidateStore(path or DATA_DIR / "candidates.pkl")
    if backend == "redis":
        return RedisCandidateStore(url or "redis://localhost:6379/0")
    raise ValueError(f"알 수 없는 backend: {backend}")
