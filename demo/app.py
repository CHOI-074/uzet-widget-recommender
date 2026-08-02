"""Step 5 (선택) — Streamlit 데모.

이번 범위(Step 1~4) 밖이라 뼈대만 둡니다. 실행:  streamlit run demo/app.py

추천 시스템은 눈에 보여야 설득력이 생깁니다. 시각 슬라이더를 09시 → 10시로
옮겼을 때 주식 위젯이 올라오는 장면을 GIF 로 찍어 README 최상단에 넣으세요.
코드 500줄보다 강합니다.

만들 것
  - 사이드바: user_id 선택, 날짜/시각 슬라이더, is_roaming 토글
  - 본문: 추천 6개를 카드로. 각 카드에 parts(als/context/recency/fatigue)를
          가로 막대로 표시 → "왜 이게 1위인지" 가 한눈에 보입니다
  - 하단: 현재 발동한 규칙 이름 (scorer.engine.explain)
"""

from datetime import datetime

import streamlit as st

from src.config import DATA_DIR, load_catalog
from src.scoring.context_rules import RequestContext
from src.scoring.hybrid_scorer import HybridScorer
from src.store.candidate_store import FileCandidateStore


@st.cache_resource
def load():
    return FileCandidateStore(DATA_DIR / "candidates.pkl"), HybridScorer(), load_catalog()


def main() -> None:
    st.title("UZET — 위젯 추천 데모")
    store, scorer, catalog = load()

    user_id = st.sidebar.text_input("user_id", "u000001")
    date = st.sidebar.date_input("날짜", datetime(2025, 10, 15))
    hour = st.sidebar.slider("시각", 0, 23, 10)
    roaming = st.sidebar.checkbox("로밍 중")

    ctx = RequestContext(
        now=datetime(date.year, date.month, date.day, hour, 30),
        flags={"is_roaming": roaming},
    )

    cached = store.get(user_id)
    if cached is None:
        st.warning("캐시에 후보가 없습니다. 먼저 python -m src.batch.precompute 를 실행하세요.")
        return

    # TODO: scorer.recommend(...) 결과를 카드로 렌더링하고 parts 를 막대로 표시
    st.info("TODO: 추천 결과 렌더링")
    st.caption(f"발동 규칙: {scorer.engine.explain(ctx) or '없음'}")


if __name__ == "__main__":
    main()
