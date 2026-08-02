"""UZET 아키텍처 다이어그램 SVG 생성 (라이트/다크 2종)."""

from pathlib import Path

FONT = ("-apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', "
        "'Pretendard', 'Malgun Gothic', 'Noto Sans KR', sans-serif")
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"

LIGHT = dict(
    bg="#FAFAF8", panel="#FFFFFF", panel_stroke="#E1E4E8",
    box="#F6F8FA", box_stroke="#D8DEE4",
    text="#1F2328", muted="#6E7781", arrow="#9AA4AE",
    offline="#3D6B8C", online="#A65D3A", store="#2F6F5E",
    store_fill="#EAF5F1",
)

DARK = dict(
    bg="#0D1117", panel="#161B22", panel_stroke="#30363D",
    box="#1C2128", box_stroke="#3D444D",
    text="#E6EDF3", muted="#9198A1", arrow="#6E7681",
    offline="#6CB0E8", online="#E8A06C", store="#56C9A5",
    store_fill="#14251F",
)

W, H = 900, 528

OFFLINE_STEPS = [
    ("이벤트 로그", "click · impression"),
    ("상호작용 행렬", "user × item · CSR"),
    ("ALS 학습", "implicit · factors=64"),
    ("Top-K 후보", "유저당 50개"),
]

ONLINE_STEPS = [
    ("① Store GET", "~1ms"),
    ("② Context 평가", "~1ms"),
    ("③ 재랭킹 · 정렬", "~1ms"),
    ("④ 상위 N 반환", "N = 6"),
]

BOX_W, GAP = 166, 42
X0 = 54
XS = [X0 + i * (BOX_W + GAP) for i in range(4)]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def sub_font(s: str) -> str:
    """한글이 섞인 문자열은 mono 로 쓰지 않습니다.

    모노스페이스 폰트는 한글 글리프가 없는 경우가 많고, 렌더러에 따라
    두부(□)로 깨집니다. 코드성 ASCII 문자열에만 mono 를 씁니다.
    """
    has_hangul = any(0xAC00 <= ord(ch) <= 0xD7A3 or 0x3130 <= ord(ch) <= 0x318F
                     for ch in s)
    return FONT if has_hangul else MONO


def text(x, y, s, size=14, weight=400, fill="text", anchor="middle",
         font=FONT, ls=0, c=None):
    col = c[fill] if fill in c else fill
    extra = f' letter-spacing="{ls}"' if ls else ""
    return (f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
            f'font-weight="{weight}" fill="{col}" text-anchor="{anchor}"'
            f'{extra}>{esc(s)}</text>')


def rect(x, y, w, h, rx, fill, stroke, c, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{c.get(fill, fill)}" stroke="{c.get(stroke, stroke)}" '
            f'stroke-width="{sw}"{d}/>')


def build(c: dict) -> str:
    p = []
    a = p.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
      f'width="{W}" height="{H}" role="img" '
      f'aria-label="UZET 추천 엔진 아키텍처">')
    a('<title>UZET — 오프라인 학습 / 온라인 서빙 분리 아키텍처</title>')

    # 화살촉
    a('<defs>')
    for name in ("offline", "online", "arrow", "store"):
        a(f'<marker id="mk-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
          f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
          f'<path d="M0,1 L9,5 L0,9 z" fill="{c[name]}"/></marker>')
    a('</defs>')

    a(f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>')

    # ── 헤더 ───────────────────────────────────────────────
    a(text(40, 40, "10ms를 만드는 구조", 19, 700, "text", "start", c=c))
    a(text(40, 62, "무거운 연산은 오프라인으로, 실시간 요소만 실시간으로",
           13, 400, "muted", "start", c=c))

    # ── 오프라인 패널 ──────────────────────────────────────
    oy = 84
    a(rect(30, oy, 840, 136, 14, "panel", "panel_stroke", c))
    a(f'<rect x="30" y="{oy}" width="5" height="136" rx="2.5" '
      f'fill="{c["offline"]}"/>')
    a(text(54, oy + 28, "오프라인 — 배치 (일 1회)", 13, 700, "offline",
           "start", c=c, ls=0.3))

    by = oy + 46
    for i, (t1, t2) in enumerate(OFFLINE_STEPS):
        x = XS[i]
        a(rect(x, by, BOX_W, 64, 10, "box", "box_stroke", c, sw=1.2))
        a(text(x + BOX_W / 2, by + 27, t1, 14, 600, "text", c=c))
        a(text(x + BOX_W / 2, by + 46, t2, 11, 400, "muted", font=sub_font(t2), c=c))
        if i < 3:
            x1, x2 = x + BOX_W + 8, x + BOX_W + GAP - 8
            a(f'<line x1="{x1}" y1="{by + 32}" x2="{x2}" y2="{by + 32}" '
              f'stroke="{c["offline"]}" stroke-width="2" '
              f'marker-end="url(#mk-offline)"/>')

    # ── 저장 화살표 ────────────────────────────────────────
    a(f'<line x1="450" y1="{oy + 136}" x2="450" y2="248" '
      f'stroke="{c["store"]}" stroke-width="2" marker-end="url(#mk-store)"/>')
    a(text(462, 243, "저장", 12, 600, "store", "start", c=c))

    # ── CandidateStore ────────────────────────────────────
    sy = 250
    a(rect(180, sy, 540, 66, 12, "store_fill", "store", c, sw=1.8))
    a(text(450, sy + 28, "CandidateStore", 15, 700, "store", c=c))
    a(text(414, sy + 49, "key: cand:{user_id}", 11.5, 400, "muted",
           font=MONO, anchor="end", c=c))
    a(text(424, sy + 49, "·", 11.5, 400, "muted", anchor="middle", c=c))
    a(text(434, sy + 49, "Redis / File 교체 가능", 11.5, 400, "muted",
           anchor="start", c=c))

    # ── 조회 화살표 ────────────────────────────────────────
    a(f'<line x1="450" y1="{sy + 66}" x2="450" y2="346" '
      f'stroke="{c["store"]}" stroke-width="2" marker-end="url(#mk-store)"/>')
    a(text(462, 341, "조회", 12, 600, "store", "start", c=c))

    # ── 온라인 패널 ────────────────────────────────────────
    ny = 348
    a(rect(30, ny, 840, 136, 14, "panel", "panel_stroke", c))
    a(f'<rect x="30" y="{ny}" width="5" height="136" rx="2.5" '
      f'fill="{c["online"]}"/>')
    a(text(54, ny + 28, "온라인 — 요청당 ~10ms", 13, 700, "online",
           "start", c=c, ls=0.3))

    by2 = ny + 46
    for i, (t1, t2) in enumerate(ONLINE_STEPS):
        x = XS[i]
        a(rect(x, by2, BOX_W, 64, 10, "box", "box_stroke", c, sw=1.2))
        a(text(x + BOX_W / 2, by2 + 27, t1, 14, 600, "text", c=c))
        a(text(x + BOX_W / 2, by2 + 46, t2, 11, 400, "muted", font=sub_font(t2), c=c))
        if i < 3:
            x1, x2 = x + BOX_W + 8, x + BOX_W + GAP - 8
            a(f'<line x1="{x1}" y1="{by2 + 32}" x2="{x2}" y2="{by2 + 32}" '
              f'stroke="{c["online"]}" stroke-width="2" '
              f'marker-end="url(#mk-online)"/>')

    # ── 각주 ───────────────────────────────────────────────
    a(text(450, 508, "ALS 행렬 곱은 오프라인에서 끝냅니다. "
                     "온라인은 캐시된 후보에 대한 가벼운 재랭킹만 수행합니다.",
           12, 400, "muted", c=c))

    a('</svg>')
    return "\n".join(p)


def main() -> None:
    # 다이어그램을 고칠 일이 생기면 이 파일을 수정하고 다시 실행하세요.
    #   python docs/images/_generate.py
    out = Path(__file__).resolve().parent
    for name, palette in (("light", LIGHT), ("dark", DARK)):
        (out / f"architecture-{name}.svg").write_text(build(palette), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
