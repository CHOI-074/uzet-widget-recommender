# 노트북

Step 1~2 에서 만들 실험 노트입니다. 결론은 반드시 `docs/evaluation.md` 로 옮기세요.
노트북만 남기면 채용 담당자가 읽지 않습니다.

- `01_eda.ipynb` — 위젯별 클릭 분포, 시간대 분포, 롱테일 확인, 페르소나별 취향 비교
- `02_als_tuning.ipynb` — factors / regularization / alpha 스윕, Recall-Coverage 트레이드오프 곡선

EDA에서 꼭 그려볼 그래프 두 개:

1. 위젯 클릭 수 내림차순 막대 → 롱테일이 눈에 보여야 합니다
2. 페르소나 × 위젯 히트맵 → 취향이 실제로 갈리는지 확인
