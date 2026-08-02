"""Step 3 — Context 규칙 엔진.

"지금" 이라는 신호를 위젯 부스팅으로 바꾸는 부분입니다. ALS 는 하루 한 번
학습되는 느린 신호라 급여일·장 운영시간 같은 빠른 신호를 절대 못 잡습니다.
그 간극을 규칙으로 메웁니다.

규칙을 코드가 아니라 YAML 에 둔 이유
  - 새 규칙 추가에 배포가 필요 없음
  - 기획자와 같은 파일을 보고 이야기할 수 있음
  - "이 부스팅 왜 넣었어요?" 에 description 필드로 답이 됨

성능 메모: 이 평가는 요청마다 도는 온라인 경로입니다. 규칙 수 x 조건 수가
수십 개 수준이라 순수 파이썬 연산으로 1ms 안쪽입니다. 여기서 DB 나 외부
호출을 하면 10ms 목표가 바로 깨집니다.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime, time

from src.config import ContextRule, ScoringConfig, load_scoring


@dataclass
class RequestContext:
    """요청 시점의 상태. API 가 이 객체를 만들어 스코어러에 넘깁니다."""

    now: datetime
    flags: dict[str, bool] = field(default_factory=dict)

    @property
    def hour_minute(self) -> time:
        return self.now.time()

    @property
    def day(self) -> int:
        return self.now.day

    @property
    def weekday(self) -> int:
        """0=월 ... 6=일"""
        return self.now.weekday()

    @property
    def days_in_month(self) -> int:
        return calendar.monthrange(self.now.year, self.now.month)[1]


def parse_hhmm(value: str) -> time:
    h, m = value.split(":")
    return time(int(h), int(m))


class ContextEngine:
    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or load_scoring()

    # ------------------------------------------------------------------
    # 여기서부터 직접 구현
    # ------------------------------------------------------------------

    def match_condition(self, cond: dict, ctx: RequestContext) -> bool:
        """조건 1개를 평가합니다.

        지원해야 할 type (config/scoring.yaml 주석과 동일)

        | type                | 필드            | 참이 되는 조건                          |
        |---------------------|-----------------|-----------------------------------------|
        | day_of_month_near   | day, tolerance  | abs(오늘 일자 - day) <= tolerance       |
        | day_of_month_gte    | day             | 오늘 일자 >= day                        |
        | days_before_day     | day, lead       | 0 < (day - 오늘 일자) <= lead           |
        | time_range          | start, end      | start <= 현재시각 <= end (HH:MM)        |
        | weekday             | days            | ctx.weekday() 가 days 안에 있음         |
        | flag                | name            | ctx.flags.get(name) 이 참               |

        모르는 type 이 오면 ValueError 를 던지세요. 조용히 False 를 반환하면
        YAML 오타가 "규칙이 안 걸리네?" 로만 보여서 디버깅이 지옥이 됩니다.

        경계 케이스 하나: day_of_month_near 에서 25일 ±1 이면 26일, 24일이
        걸립니다. 그런데 급여일이 1일이면 전달 말일도 잡아야 할까요?
        (지금은 안 잡아도 됩니다. 다만 이 한계를 README 에 적어두면
        "엣지 케이스를 인지하고 범위를 정했다"가 됩니다)

        TODO: 구현.
        """
        raise NotImplementedError

    def matches(self, rule: ContextRule, ctx: RequestContext) -> bool:
        """규칙의 모든 조건이 참이어야 규칙이 발동합니다 (AND).

        TODO: 구현. (match_condition 을 전부 통과하는지)
        """
        raise NotImplementedError

    def evaluate(self, ctx: RequestContext) -> dict[str, float]:
        """발동한 규칙들의 boost 를 위젯별로 합산해 반환합니다.

        여러 규칙이 같은 위젯을 부스팅하면 **더합니다** (max 가 아니라 sum).
        예: 월말 + 급여일이 겹치면 salary_manage 가 두 번 밀려 올라갑니다.
        이게 의도한 동작인지는 직접 판단해서 README 에 근거를 남기세요.

        반환 예: {"salary_manage": 1.0, "savings": 0.7}

        TODO: 구현.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------

    def explain(self, ctx: RequestContext) -> list[str]:
        """발동한 규칙 이름 목록. API 응답에 실어 보내면 데모 설득력이 큽니다."""
        return [r.name for r in self.config.rules if self.matches(r, ctx)]
