"""Step 4 — FastAPI 추천 서빙.

온라인 경로는 이 4단계가 전부여야 합니다 (설계서 5장).
    1. 저장소 GET                  ~1ms
    2. Context 평가 (순수 연산)    ~1ms
    3. 재랭킹 + 정렬 (K=50)        ~1ms
    4. 상위 N 반환

여기서 절대 하면 안 되는 것
    - ALS 행렬 곱 / 모델 로드
    - DB 조회, 외부 API 호출
    - 위젯 메타데이터를 파일에서 매번 읽기 (캐시된 catalog 를 쓰세요)

실행
    uvicorn src.api.main:app --reload
    curl "http://localhost:8000/recommend/u000001?n=6"
    curl "http://localhost:8000/recommend/u000001?n=6&is_roaming=true&now=2025-10-25T10:00:00"
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Query, Request
from pydantic import BaseModel

from src.config import DATA_DIR, load_catalog, load_scoring
from src.scoring.context_rules import RequestContext
from src.scoring.hybrid_scorer import HybridScorer, ScoredWidget, UserState
from src.store.candidate_store import CandidateStore, build_store

BACKEND = os.getenv("CANDIDATE_BACKEND", "file")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
STORE_PATH = Path(os.getenv("CANDIDATE_PATH", str(DATA_DIR / "candidates.pkl")))

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 무거운 초기화는 전부 여기서. 요청 경로에서는 아무것도 로드하지 않습니다.
    state["store"] = build_store(BACKEND, path=STORE_PATH, url=REDIS_URL)
    state["scorer"] = HybridScorer()
    state["catalog"] = load_catalog()
    state["scoring"] = load_scoring()
    yield
    store = state.get("store")
    if isinstance(store, CandidateStore):
        store.close()


app = FastAPI(title="UZET Widget Recommender", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """개별 요청 처리 시간. 부하 테스트 결과나 성능 보장은 아닙니다."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.3f}"
    return response


class WidgetOut(BaseModel):
    widget_id: str
    name: str
    category: str
    score: float
    parts: dict[str, float] = {}


class RecommendResponse(BaseModel):
    user_id: str
    generated_at: datetime
    source: str                 # "personalized" | "fallback"
    fired_rules: list[str]
    widgets: list[WidgetOut]


def _to_out(scored: list[ScoredWidget]) -> list[WidgetOut]:
    catalog = load_catalog()
    out = []
    for sw in scored:
        w = catalog.by_id(sw.widget_id)
        out.append(
            WidgetOut(
                widget_id=w.id, name=w.name, category=w.category,
                score=round(sw.score, 5), parts=sw.parts,
            )
        )
    return out


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "backend": BACKEND}


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    user_id: str,
    n: int | None = Query(default=None, ge=1, le=20, description="반환 개수 (미지정 시 설정값)"),
    now: datetime | None = Query(default=None, description="테스트용 시각 주입"),
    is_roaming: bool = Query(default=False),
) -> RecommendResponse:
    store: CandidateStore = state["store"]      # type: ignore[assignment]
    scorer: HybridScorer = state["scorer"]      # type: ignore[assignment]
    scoring = state["scoring"]
    n = n or scoring.default_n                  # type: ignore[union-attr]

    request_time = now or datetime.now(ZoneInfo("Asia/Seoul"))
    if request_time.tzinfo is not None:
        request_time = request_time.astimezone(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    ctx = RequestContext(now=request_time, flags={"is_roaming": is_roaming})
    cached = store.get(user_id)

    if cached is None:
        # No persona was supplied, so do not infer one. Use the documented
        # catalog prior and apply the same context rules to all new users.
        user_state = UserState(als_scores=dict(scorer.catalog.global_popularity))
    else:
        user_state, _persona_id = cached
    scored = scorer.recommend(user_state, ctx, n=n)
    if cached is None:
        for widget in scored:
            widget.parts["prior"] = widget.parts.pop("als")

    return RecommendResponse(
        user_id=user_id,
        generated_at=ctx.now,
        source="fallback" if cached is None else "personalized",
        fired_rules=scorer.engine.explain(ctx),
        widgets=_to_out(scored),
    )
