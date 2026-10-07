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


class FileCandidateStore(CandidateStore):

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

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 60 * 60 * 72,
        prefix: str = "cand:",
    ) -> None:
        import redis 

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

    if backend == "file":
        from src.config import DATA_DIR

        return FileCandidateStore(path or DATA_DIR / "candidates.pkl")
    if backend == "redis":
        return RedisCandidateStore(url or "redis://localhost:6379/0")
    raise ValueError(f"알 수 없는 backend: {backend}")
