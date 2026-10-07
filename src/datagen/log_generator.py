from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_DIR,
    GenerationConfig,
    Persona,
    WidgetCatalog,
    load_catalog,
    load_personas,
)


@dataclass
class Event:
    user_id: str
    persona_id: str
    widget_id: str
    timestamp: datetime
    action: str


def normalize(weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """{키: 가중치} → (키 리스트, 합이 1인 확률 배열). 배관이라 완성해 뒀습니다."""
    keys = list(weights.keys())
    probs = np.asarray([weights[k] for k in keys], dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("가중치 합이 0입니다")
    return keys, probs / total


class LogGenerator:
    def __init__(
        self,
        catalog: WidgetCatalog,
        personas: list[Persona],
        gen: GenerationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.catalog = catalog
        self.personas = personas
        self.gen = gen
        self.rng = rng
        self.start = datetime.fromisoformat(gen.start_date)

        # 자주 쓰는 분포는 미리 정규화해 둡니다.
        self._pop_keys, self._pop_probs = normalize(catalog.global_popularity)
        self._persona_keys, self._persona_probs = normalize(
            {p.id: p.weight for p in personas}
        )
        self._affinity_cache = {
            p.id: normalize(p.affinity) for p in personas
        }
        self._hour_cache = {
            p.id: np.asarray(p.hour_weights, dtype=float) / sum(p.hour_weights)
            for p in personas
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def assign_persona(self) -> Persona:
        """유저 1명에게 페르소나를 배정합니다.

        personas.yaml 의 weight 비율대로 뽑습니다.
        (rookie 0.35 / investor 0.25 / owner 0.2 / senior 0.2)

        이 함수가 유저를 4가지 유형으로 갈라놓기 때문에 개인화 신호가 생깁니다.
        모두가 같은 유형이면 ALS 가 배울 게 없어집니다.
        """

        persona_id = self.rng.choice(self._persona_keys, p=self._persona_probs)
        for persona in self.personas:
            if persona.id == persona_id:
                return persona
        raise KeyError(f"알 수 없는 페르소나: {persona_id}")

    def sample_widget(self, persona: Persona) -> str:
        """이 유저가 이번에 클릭할 위젯 1개를 고릅니다.

        혼합 분포입니다.
            확률 persona.persona_ratio      → affinity 분포에서 뽑기
            확률 1 - persona.persona_ratio  → global_popularity 분포에서 뽑기

        이 혼합이 곧 "개인 취향 + 전체 인기"의 데이터 생성 과정(generative process)
        이고, ALS 가 나중에 복원해야 할 구조입니다.
        """
        if self.rng.random() < persona.persona_ratio:
            keys, probs = self._affinity_cache[persona.id]   # 개인 취향
        else:
            keys, probs = self._pop_keys, self._pop_probs    # 전체 인기
        return str(self.rng.choice(keys, p=probs))

    def sample_hour(self, persona: Persona) -> int:
        """접속 시각(0~23)을 hour_weights 분포에서 뽑습니다.

        self._hour_cache[persona.id] 를 확률로 써서 0~23 중 하나 반환.
        """
        return int(self.rng.choice(24, p=self._hour_cache[persona.id]))

    def generate_user_logs(self, user_id: str, persona: Persona) -> list[Event]:

        events: list[Event] = []

        for day in range(self.gen.days):
            n_sessions = int(self.rng.poisson(persona.daily_sessions))

            for _ in range(n_sessions):
                hour = self.sample_hour(persona)
                # Poisson 이 0 을 뱉으면 빈 세션이 되므로 최소 1회는 보장합니다.
                n_actions = max(1, int(self.rng.poisson(persona.actions_per_session)))
                clicked: set[str] = set()

                for _ in range(n_actions):
                    timestamp = self.start + timedelta(
                        days=day,
                        hours=hour,
                        minutes=int(self.rng.integers(0, 60)),
                        seconds=int(self.rng.integers(0, 60)),
                    )
                    widget = self.sample_widget(persona)
                    events.append(
                        Event(user_id, persona.id, widget, timestamp, "click")
                    )
                    clicked.add(widget)

                    if not self.gen.emit_impressions:
                        continue

                    # 같은 화면에 같이 떠 있었지만 눌리지 않은 위젯들
                    for _ in range(self.gen.impressions_per_click):
                        shown = self.sample_widget(persona)
                        if shown in clicked:
                            continue
                        events.append(
                            Event(user_id, persona.id, shown, timestamp, "impression")
                        )

        return events


    # ------------------------------------------------------------------
    # 아래는 완성된 배관
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        rows: list[Event] = []
        for i in range(self.gen.n_users):
            user_id = f"u{i:06d}"
            persona = self.assign_persona()
            rows.extend(self.generate_user_logs(user_id, persona))

        df = pd.DataFrame([e.__dict__ for e in rows])
        return df.sort_values("timestamp").reset_index(drop=True)


def build_generator(seed: int | None = None) -> LogGenerator:
    catalog = load_catalog()
    personas, gen = load_personas()
    return LogGenerator(catalog, personas, gen, np.random.default_rng(seed or gen.seed))


def main() -> None:
    _, defaults = load_personas()
    ap = argparse.ArgumentParser(description="페르소나 기반 합성 로그 생성")
    ap.add_argument("--n-users", type=int, default=defaults.n_users)
    ap.add_argument("--days", type=int, default=defaults.days)
    ap.add_argument("--seed", type=int, default=defaults.seed)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "events.parquet")
    args = ap.parse_args()

    catalog = load_catalog()
    personas, gen = load_personas()
    gen = GenerationConfig(
        n_users=args.n_users,
        days=args.days,
        start_date=gen.start_date,
        seed=args.seed,
        emit_impressions=gen.emit_impressions,
        impressions_per_click=gen.impressions_per_click,
    )
    generator = LogGenerator(catalog, personas, gen, np.random.default_rng(args.seed))
    df = generator.generate()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".csv":
        df.to_csv(args.out, index=False)
    else:
        df.to_parquet(args.out, index=False)

    clicks = df[df.action == "click"]
    top = clicks.widget_id.value_counts(normalize=True)
    print(f"저장: {args.out}  (총 {len(df):,}행 / 클릭 {len(clicks):,}행)")
    print(f"상위 3개 위젯 클릭 점유율: {top.head(3).sum():.1%}")
    print(f"고유 위젯 수: {clicks.widget_id.nunique()} / {len(catalog.ids)}")


if __name__ == "__main__":
    main()
