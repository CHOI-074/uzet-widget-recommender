"""Step 4 — 배치 파이프라인: 로그 → ALS 학습 → 유저별 Top-K 후보 → 저장소.

하루 한 번 도는 오프라인 경로 전부가 여기 있습니다. 이 스크립트가 하는 일이
곧 "온라인에서 하지 않는 일" 이고, 그래서 API 가 10ms 안에 끝납니다.

실행
    python -m src.batch.precompute --backend file
    python -m src.batch.precompute --backend redis --url redis://localhost:6379/0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, load_catalog, load_scoring
from src.model.als_trainer import ALSArtifacts, build_interactions, train
from src.scoring.hybrid_scorer import UserState
from src.store.candidate_store import build_store


# ----------------------------------------------------------------------
# 여기서부터 직접 구현
# ----------------------------------------------------------------------


def build_user_states(
    events: pd.DataFrame,
    artifacts: ALSArtifacts,
    top_k: int,
) -> dict[str, tuple[UserState, str]]:
    """유저별로 캐시에 넣을 상태를 만듭니다.

    유저 1명에 대해
      1. als_scores : artifacts.score_all(user_id) 중 **상위 top_k 개만**
         전부 저장하면 캐시가 커지고, 어차피 하위는 재랭킹으로도 안 올라옵니다.
         (top_k 를 얼마로 잡을지가 '정확도 vs 메모리' 트레이드오프입니다.
          config/scoring.yaml 의 candidate_top_k. 근거를 문서에 남기세요)
      2. last_used  : 클릭 이벤트 기준 위젯별 마지막 timestamp
      3. unclicked_impressions : action=="impression" 인 건수를 위젯별로 집계
         (같은 세션에서 클릭도 있었다면 빼는 게 정확하지만, 우선 단순 집계로
          시작하고 개선 여지를 문서에 남깁니다)
      4. persona_id : 로그의 persona_id (실서비스라면 온보딩 설문 결과)

    반환: {user_id: (UserState, persona_id)}

    성능 주의: 유저 2000명 x 위젯 25개면 groupby 한 번으로 끝납니다.
    유저마다 events 를 필터링하는 for 문을 돌리면 O(n_users * n_events) 라
    배치가 몇 분씩 걸립니다. **groupby 로 한 번에** 만드세요.

    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    result = {}
    trained = set(artifacts.user_ids)
    for user_id, group in events.groupby("user_id", sort=False):
        if user_id not in trained:
            continue
        scores = artifacts.score_all(user_id)
        top = dict(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k])
        clicks = group.loc[group.action == "click"]
        last = clicks.groupby("widget_id").timestamp.max().to_dict()
        impressions = group.loc[group.action == "impression"].groupby("widget_id").size().to_dict()
        result[user_id] = (UserState(top, last, impressions), str(group.persona_id.iloc[0]))
    return result


# ----------------------------------------------------------------------
# 아래는 완성된 배관
# ----------------------------------------------------------------------


def main() -> None:
    scoring = load_scoring()
    ap = argparse.ArgumentParser(description="배치: 학습 → 후보 → 저장소")
    ap.add_argument("--events", type=Path, default=DATA_DIR / "events.parquet")
    ap.add_argument("--backend", choices=["file", "redis"], default="file")
    ap.add_argument("--path", type=Path, default=DATA_DIR / "candidates.pkl")
    ap.add_argument("--url", type=str, default="redis://localhost:6379/0")
    ap.add_argument("--top-k", type=int, default=scoring.candidate_top_k)
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--save-model", type=Path, default=DATA_DIR / "als.pkl")
    args = ap.parse_args()

    t0 = time.perf_counter()
    events = pd.read_parquet(args.events)
    catalog = load_catalog()
    print(f"[1/4] 로그 로드 {len(events):,}행  ({time.perf_counter() - t0:.1f}s)")

    t = time.perf_counter()
    inter = build_interactions(events, catalog, alpha=args.alpha)
    art = train(inter, factors=args.factors)
    art.save(args.save_model)
    print(f"[2/4] ALS 학습 완료  ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    states = build_user_states(events, art, args.top_k)
    print(f"[3/4] 후보 생성 {len(states):,}명  ({time.perf_counter() - t:.1f}s)")

    t = time.perf_counter()
    store = build_store(args.backend, path=args.path, url=args.url)
    store.bulk_put(states)
    store.close()
    print(f"[4/4] {args.backend} 저장 완료  ({time.perf_counter() - t:.1f}s)")
    print(f"총 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
