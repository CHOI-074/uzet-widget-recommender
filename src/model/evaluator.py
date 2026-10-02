"""Step 2 — 오프라인 평가: Recall@K, NDCG@K, Coverage.

평가가 없으면 "돌아가는 걸 만들었다"에서 끝납니다. 특히 Coverage 를 같이 봐야
"인기 위젯만 계속 추천하는" 실패 모드를 잡아낼 수 있습니다.

지표 정의 (이대로 구현하고, README 에도 같은 정의를 적어두세요)
---------------------------------------------------------------
Recall@K   : 유저별로 정답(홀드아웃) 아이템이 상위 K 안에 있으면 1, 없으면 0.
             leave-one-out 이라 정답이 1개이므로 Hit Rate 와 같습니다.
             전 유저 평균.

NDCG@K     : 정답이 몇 번째에 있었는지까지 반영.
             정답이 rank r (1-based, r<=K) 에 있으면 DCG = 1 / log2(r + 1),
             없으면 0. 정답이 1개이므로 IDCG = 1 → NDCG = DCG.

Coverage@K : 전체 유저에게 추천된 **고유 아이템 수 / 전체 카탈로그 수**.
             이 값이 낮으면 롱테일을 전혀 못 건드리고 있다는 뜻입니다.

평가 프로토콜
-------------
leave-one-out: 유저별로 **시간상 가장 마지막 클릭 1건**을 테스트로 빼고
나머지로 학습합니다. 랜덤으로 빼면 미래 정보가 학습에 새어 들어가
(data leakage) 지표가 부풀려집니다. 면접에서 물어보면 이걸 말하면 됩니다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, load_catalog


@dataclass
class EvalResult:
    k: int
    recall: float
    ndcg: float
    coverage: float
    n_users: int

    def __str__(self) -> str:
        return (
            f"K={self.k}  Recall@K={self.recall:.4f}  "
            f"NDCG@K={self.ndcg:.4f}  Coverage@K={self.coverage:.4f}  "
            f"(users={self.n_users:,})"
        )


# ----------------------------------------------------------------------
# 여기서부터 직접 구현
# ----------------------------------------------------------------------


def leave_one_out_split(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """유저별 마지막 클릭 1건을 테스트로 분리.

    반환: (train_df, test_df)
      - test_df 는 유저당 정확히 1행
      - 클릭이 2건 미만인 유저는 학습에만 남기고 테스트에서 제외
        (정답을 빼면 학습 데이터가 0이 되는 유저는 평가 대상이 될 수 없습니다)

    힌트: action=="click" 필터 → sort_values("timestamp") → groupby("user_id").tail(1)
    """
    # Work with positional indices so duplicate input labels cannot remove extra rows.
    data = events.reset_index(drop=True)
    clicks = data.loc[data.action == "click"].sort_values("timestamp", kind="stable")
    eligible = clicks.groupby("user_id").filter(lambda group: len(group) >= 2)
    held = eligible.groupby("user_id", sort=False).tail(1)
    # Exclude later impressions too: they would leak future fatigue information.
    cutoff = held.set_index("user_id").timestamp
    limits = data.user_id.map(cutoff)
    keep = limits.isna() | (data.timestamp < limits)
    # Clicks tied with the held event are excluded; users without earlier clicks
    # cannot be evaluated and stay entirely in training.
    prior_users = set(data.loc[keep & (data.action == "click"), "user_id"])
    held = held[held.user_id.isin(prior_users)]
    limits = data.user_id.map(held.set_index("user_id").timestamp)
    keep = limits.isna() | (data.timestamp < limits)
    return data.loc[keep].copy(), held.copy()


def recall_at_k(ranked: list[str], truth: str, k: int) -> float:
    """정답이 상위 k 안에 있으면 1.0, 아니면 0.0.

    """
    return float(k > 0 and truth in ranked[:k])


def ndcg_at_k(ranked: list[str], truth: str, k: int) -> float:
    """정답의 순위를 반영한 점수. 정답이 1개이므로 1/log2(rank+1) (1-based).

    """
    if k <= 0 or truth not in ranked[:k]:
        return 0.0
    return float(1.0 / np.log2(ranked.index(truth) + 2))


def coverage_at_k(all_ranked: list[list[str]], catalog_size: int, k: int) -> float:
    """전 유저의 상위 k 추천에 등장한 고유 아이템 수 / 카탈로그 크기.

    """
    if catalog_size <= 0 or k <= 0:
        return 0.0
    return len({w for ranked in all_ranked for w in ranked[:k]}) / catalog_size


# ----------------------------------------------------------------------
# 아래는 완성된 배관
# ----------------------------------------------------------------------


def evaluate(
    recommend_fn,
    test_df: pd.DataFrame,
    catalog_size: int,
    k: int = 10,
) -> EvalResult:
    """recommend_fn(user_id) -> 랭킹된 widget_id 리스트 를 받아 지표를 계산합니다.

    이 시그니처 덕분에 ALS 단독 / 하이브리드 스코어러 / 인기순 베이스라인을
    같은 함수로 비교할 수 있습니다. **베이스라인 비교를 꼭 넣으세요.**
    "인기순보다 얼마나 나은가"가 없으면 숫자가 의미를 갖지 못합니다.
    """
    recalls, ndcgs, all_ranked = [], [], []
    for row in test_df.itertuples():
        ranked = recommend_fn(row.user_id)
        all_ranked.append(ranked[:k])
        recalls.append(recall_at_k(ranked, row.widget_id, k))
        ndcgs.append(ndcg_at_k(ranked, row.widget_id, k))

    return EvalResult(
        k=k,
        recall=float(np.mean(recalls)) if recalls else 0.0,
        ndcg=float(np.mean(ndcgs)) if ndcgs else 0.0,
        coverage=coverage_at_k(all_ranked, catalog_size, k),
        n_users=len(recalls),
    )


def popularity_baseline(events: pd.DataFrame) -> list[str]:
    """전역 인기순 랭킹 — 비교 기준선."""
    clicks = events[events.action == "click"]
    return clicks.widget_id.value_counts().index.tolist()


def main() -> None:
    from src.model.als_trainer import ALSArtifacts, build_interactions, train

    ap = argparse.ArgumentParser(description="ALS 오프라인 평가")
    ap.add_argument("--events", type=Path, default=DATA_DIR / "events.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--factors", type=int, default=64)
    args = ap.parse_args()

    events = pd.read_parquet(args.events)
    catalog = load_catalog()
    train_df, test_df = leave_one_out_split(events)

    inter = build_interactions(train_df, catalog)
    art: ALSArtifacts = train(inter, factors=args.factors)

    def als_rank(user_id: str) -> list[str]:
        scores = art.score_all(user_id)
        return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])]

    pop = popularity_baseline(train_df)

    print("ALS       ", evaluate(als_rank, test_df, len(catalog.ids), args.k))
    print("인기순 기준선", evaluate(lambda _u: pop, test_df, len(catalog.ids), args.k))


if __name__ == "__main__":
    main()
