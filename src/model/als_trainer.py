"""Step 2 — Implicit ALS 학습.

왜 implicit ALS 인가 (면접 단골)
--------------------------------
사용자는 위젯에 별점을 주지 않습니다. 클릭했거나 안 했거나뿐입니다.
Hu-Koren-Volinsky (2008) 는 이걸 두 가지로 분해합니다.

    선호   p_ui = 1 if r_ui > 0 else 0
    신뢰도 c_ui = 1 + alpha * r_ui

즉 "클릭 0 → 선호 0이지만 확신은 약함", "클릭 50 → 선호 1이고 확신이 강함".
손실은 sum_{u,i} c_ui * (p_ui - x_u^T y_i)^2 + reg 이고, **관측되지 않은 0까지
전부 합에 들어갑니다.** explicit MF 와 결정적으로 다른 지점입니다.

ALS 를 쓰는 이유는 x 를 고정하면 y 에 대해, y 를 고정하면 x 에 대해 각각
닫힌 해를 갖는 최소제곱 문제가 되어 병렬화가 쉽기 때문입니다. (SGD 는 전체 0을
훑어야 해서 이 세팅에서 불리)

주의: implicit 라이브러리는 0.5 이후로 fit() 에 **user × item** 행렬을 받습니다.
(예전 버전은 item × user 였습니다. 블로그 예제 보고 따라 하다 뒤집히는 사고가 잦습니다)
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.config import DATA_DIR, WidgetCatalog, load_catalog


@dataclass
class Interactions:
    """학습에 넣기 직전의 상태."""

    matrix: csr_matrix          # shape = (n_users, n_items), 값 = confidence
    user_ids: list[str]         # 행 인덱스 → user_id
    item_ids: list[str]         # 열 인덱스 → widget_id

    @property
    def user_pos(self) -> dict[str, int]:
        return {u: i for i, u in enumerate(self.user_ids)}

    @property
    def item_pos(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.item_ids)}


# ----------------------------------------------------------------------
# 여기서부터 직접 구현
# ----------------------------------------------------------------------


def to_confidence(counts: np.ndarray, alpha: float, log_scaling: bool) -> np.ndarray:
    """클릭 횟수 → 신뢰도 c_ui.

        log_scaling = False : c = 1 + alpha * count
        log_scaling = True  : c = 1 + alpha * log(1 + count)

    log 스케일링은 헤비 유저 편중을 눌러줍니다(설계서 3장 불균형 완화).
    둘 다 구현해 놓고 어느 쪽이 Recall/Coverage 에 유리했는지
    docs/evaluation.md 에 남기세요. 그 비교가 곧 면접 답변입니다.

    """
    counts = np.asarray(counts, dtype=float)
    if alpha < 0 or not np.isfinite(alpha) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts and alpha must be finite and non-negative")
    return 1.0 + alpha * (np.log1p(counts) if log_scaling else counts)


def build_interactions(
    events: pd.DataFrame,
    catalog: WidgetCatalog,
    alpha: float = 40.0,
    log_scaling: bool = True,
) -> Interactions:
    """이벤트 로그 → (user × item) confidence 희소 행렬.

    순서
        1. action == "click" 인 행만 남긴다 (impression 은 Fatigue 용이라 제외)
        2. (user_id, widget_id) 로 groupby → 클릭 횟수 r_ui
        3. user_id / widget_id 를 정수 인덱스로 매핑
           - item_ids 는 catalog.ids 순서를 그대로 쓰세요. 로그에 한 번도 안 뜬
             위젯이 있어도 열이 유지돼야 Coverage 계산이 정직해집니다.
        4. to_confidence() 로 값 변환
        5. csr_matrix((values, (rows, cols)), shape=(n_users, n_items))

    """
    clicks = events.loc[events.action == "click"]
    if not clicks.widget_id.isin(catalog.ids).all():
        raise ValueError("click log contains unknown widgets")
    user_ids = sorted(clicks.user_id.unique().tolist())
    item_ids = catalog.ids
    user_pos = {u: i for i, u in enumerate(user_ids)}
    item_pos = {w: i for i, w in enumerate(item_ids)}
    counts = clicks.groupby(["user_id", "widget_id"]).size()
    rows = [user_pos[u] for u, _ in counts.index]
    cols = [item_pos[w] for _, w in counts.index]
    values = to_confidence(counts.to_numpy(), alpha, log_scaling)
    return Interactions(csr_matrix((values, (rows, cols)),
                                  shape=(len(user_ids), len(item_ids)), dtype=np.float32),
                        user_ids, item_ids)


# ----------------------------------------------------------------------
# 아래는 라이브러리 호출 배관 — 완성해 뒀습니다
# ----------------------------------------------------------------------


@dataclass
class ALSArtifacts:
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_ids: list[str]
    item_ids: list[str]
    params: dict

    def score_all(self, user_id: str) -> dict[str, float]:
        """한 유저의 전 위젯 raw ALS 점수. 배치 단계에서 씁니다."""
        idx = self.user_ids.index(user_id)
        scores = self.item_factors @ self.user_factors[idx]
        return dict(zip(self.item_ids, scores.tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "ALSArtifacts":
        with path.open("rb") as f:
            return pickle.load(f)


def train(
    interactions: Interactions,
    factors: int = 64,
    regularization: float = 0.05,
    iterations: int = 20,
    random_state: int = 42,
) -> ALSArtifacts:
    from implicit.als import AlternatingLeastSquares

    if interactions.matrix.shape[0] == 0 or interactions.matrix.nnz == 0:
        raise ValueError("at least one click is required to train ALS")

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state,
        # alpha 는 이미 to_confidence() 에서 행렬 값에 반영했으므로 1.0 으로 둡니다.
        # 여기서 또 곱하면 alpha 가 이중 적용됩니다. (흔한 실수)
        alpha=1.0,
        use_gpu=False,
        num_threads=1,
    )
    model.fit(interactions.matrix)

    return ALSArtifacts(
        user_factors=np.asarray(model.user_factors),
        item_factors=np.asarray(model.item_factors),
        user_ids=interactions.user_ids,
        item_ids=interactions.item_ids,
        params={
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="ALS 학습")
    ap.add_argument("--events", type=Path, default=DATA_DIR / "events.parquet")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "als.pkl")
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--regularization", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--no-log-scaling", action="store_true")
    args = ap.parse_args()

    events = pd.read_parquet(args.events)
    catalog = load_catalog()
    inter = build_interactions(
        events, catalog, alpha=args.alpha, log_scaling=not args.no_log_scaling
    )
    print(f"행렬: {inter.matrix.shape}, 비영요소 {inter.matrix.nnz:,} "
          f"(밀도 {inter.matrix.nnz / np.prod(inter.matrix.shape):.4%})")

    art = train(
        inter,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
    )
    art.save(args.out)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
