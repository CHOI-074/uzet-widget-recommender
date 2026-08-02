"""설정 로딩 — 이 파일은 배관(plumbing)이라 완성해 두었습니다.

모든 튜닝 값은 config/*.yaml 에만 존재해야 합니다.
코드에 숫자를 하드코딩하면 "왜 이 값이죠?" 에 답할 근거가 사라집니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Widget:
    id: str
    name: str
    category: str


@dataclass(frozen=True)
class WidgetCatalog:
    widgets: list[Widget]
    global_popularity: dict[str, float]

    @property
    def ids(self) -> list[str]:
        return [w.id for w in self.widgets]

    def by_id(self, widget_id: str) -> Widget:
        for w in self.widgets:
            if w.id == widget_id:
                return w
        raise KeyError(f"unknown widget: {widget_id}")

    def category_of(self, widget_id: str) -> str:
        return self.by_id(widget_id).category


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    weight: float
    daily_sessions: float
    actions_per_session: float
    persona_ratio: float
    affinity: dict[str, float]
    hour_weights: list[float]
    fallback: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationConfig:
    n_users: int
    days: int
    start_date: str
    seed: int
    emit_impressions: bool
    impressions_per_click: int


@dataclass(frozen=True)
class ContextRule:
    name: str
    description: str
    conditions: list[dict[str, Any]]
    boost: dict[str, float]


@dataclass(frozen=True)
class ScoringConfig:
    alpha: float
    beta: float
    gamma: float
    delta: float
    half_life_hours: float
    fatigue_saturation: int
    candidate_top_k: int
    default_n: int
    max_per_category: int
    rules: list[ContextRule]


@lru_cache(maxsize=1)
def load_catalog(path: Path | None = None) -> WidgetCatalog:
    raw = _load_yaml(path or CONFIG_DIR / "widgets.yaml")
    return WidgetCatalog(
        widgets=[Widget(**w) for w in raw["widgets"]],
        global_popularity=dict(raw["global_popularity"]),
    )


@lru_cache(maxsize=1)
def load_personas(path: Path | None = None) -> tuple[list[Persona], GenerationConfig]:
    raw = _load_yaml(path or CONFIG_DIR / "personas.yaml")
    personas = [Persona(**p) for p in raw["personas"]]
    gen = GenerationConfig(**raw["generation"])
    return personas, gen


@lru_cache(maxsize=1)
def load_scoring(path: Path | None = None) -> ScoringConfig:
    raw = _load_yaml(path or CONFIG_DIR / "scoring.yaml")
    w, r, f, s = raw["weights"], raw["recency"], raw["fatigue"], raw["serving"]
    rules = [
        ContextRule(
            name=rule["name"],
            description=rule.get("description", ""),
            conditions=rule["all"],
            boost=dict(rule["boost"]),
        )
        for rule in raw["context_rules"]
    ]
    return ScoringConfig(
        alpha=w["alpha"],
        beta=w["beta"],
        gamma=w["gamma"],
        delta=w["delta"],
        half_life_hours=r["half_life_hours"],
        fatigue_saturation=f["saturation"],
        candidate_top_k=s["candidate_top_k"],
        default_n=s["default_n"],
        max_per_category=s["max_per_category"],
        rules=rules,
    )


def persona_by_id(persona_id: str) -> Persona:
    personas, _ = load_personas()
    for p in personas:
        if p.id == persona_id:
            return p
    raise KeyError(f"unknown persona: {persona_id}")
