# -*- coding: utf-8 -*-
"""17페이지(증강) 뒤에 '데이터 양 → 성능' 슬라이드를 삽입.

원본: 01-model/experiments/results/phase1.d*.jsonl (MVTec 나사, n=10..100, seed5).
각 n의 map50 seed 평균을 실제 라인차트(SVG)로 그린다. 합성/가공 없음(평균만).
"""
import json
from pathlib import Path
from collections import defaultdict

DOCS = Path(__file__).resolve().parent
DECK = DOCS / "lesson1-theory-vision.html"
RES = DOCS.parent / "outputs" / "experiments" / "results"

# --- 원본 집계 ---
agg = defaultdict(list)
for f in sorted(RES.glob("phase1.d*.jsonl")):
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("phase") == 1 and "n" in o and "map50" in o:
            agg[o["n"]].append(o["map50"])
pts = [(n, sum(v) / len(v) * 100) for n, v in sorted(agg.items())]  # (n, mAP50%)
NSEED = len(next(iter(agg.values())))

# --- SVG 좌표 ---
X0, X1, Y0, Y1 = 78, 612, 36, 286          # plot box (Y0 top=high)
NMIN, NMAX, YMAX = 10, 100, 35              # 축 범위
def sx(n): return X0 + (n - NMIN) / (NMAX - NMIN) * (X1 - X0)
def sy(v): return Y1 - v / YMAX * (Y1 - Y0)

# y 그리드 (0,10,20,30 %)
grid = ""
for gv in range(0, YMAX + 1, 10):
    y = sy(gv)
    grid += f'<line x1="{X0}" y1="{y:.1f}" x2="{X1}" y2="{y:.1f}" stroke="#e2e8f2" stroke-width="1"/>'
    grid += f'<text x="{X0-10}" y="{y+4:.1f}" text-anchor="end" font-size="14" fill="#5a6478" font-weight="700">{gv}%</text>'
# x 라벨
xlab = ""
for n, _ in pts:
    xlab += f'<text x="{sx(n):.1f}" y="{Y1+24:.0f}" text-anchor="middle" font-size="13" fill="#5a6478" font-weight="700">{n}</text>'

poly = " ".join(f"{sx(n):.1f},{sy(v):.1f}" for n, v in pts)
area = f"{X0},{Y1} " + poly + f" {X1},{Y1}"
dots = ""
for n, v in pts:
    dots += f'<circle cx="{sx(n):.1f}" cy="{sy(v):.1f}" r="4.5" fill="#fff" stroke="#1746c4" stroke-width="2.5"/>'
# 핵심 지점 값 라벨 (10, 40, 100)
vlab = ""
for n, v in pts:
    if n in (10, 40, 100):
        vlab += f'<text x="{sx(n):.1f}" y="{sy(v)-12:.1f}" text-anchor="middle" font-size="14" fill="#0f3597" font-weight="900">{v:.0f}%</text>'

svg = f'''<svg viewBox="0 0 660 350" style="width:100%;height:auto">
 {grid}
 <polygon points="{area}" fill="rgba(23,70,196,.08)"/>
 <polyline points="{poly}" fill="none" stroke="#1746c4" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/>
 {dots}{vlab}
 <line x1="{X0}" y1="{Y1}" x2="{X1}" y2="{Y1}" stroke="#c3ccdb" stroke-width="1.5"/>
 {xlab}
 <text x="{(X0+X1)/2:.0f}" y="{Y1+46:.0f}" text-anchor="middle" font-size="14.5" fill="#0e1622" font-weight="800">학습 데이터 수 (장)</text>
 <text transform="translate(22,{(Y0+Y1)/2:.0f}) rotate(-90)" text-anchor="middle" font-size="14.5" fill="#0e1622" font-weight="800">성능 (mAP50)</text>
 <text x="{sx(27):.0f}" y="{sy(9):.0f}" font-size="14" fill="var(--warn)" font-weight="800">↑ 초반 급상승</text>
 <text x="{sx(64):.0f}" y="{sy(33.5):.0f}" font-size="14" fill="#5a6478" font-weight="800">완만해짐 (포화·체감)</text>
</svg>'''

section = f'''<!-- 14b DATA SIZE -> PERFORMANCE (real phase1 sweep) -->
<section class="slide">
 <span class="tag">우리 실험 근거 · 데이터 양</span>
 <h2>AI의 핵심 — <span class="em">데이터를 늘릴수록</span> 성능이 오른다</h2>
 <p class="lead" style="margin:6px 0 8px">앞의 증강이 데이터 '<b>종류</b>'를 늘린 거라면, 이건 데이터 '<b>양</b>' 이야기입니다. 룰 기반은 사람이 규칙을 못 늘리면 그대로지만, AI는 <b>예시를 더 주는 것만으로 스스로 좋아집니다.</b></p>
 <div class="row fill" style="align-items:stretch;gap:18px">
  <div class="col" style="flex:0 0 700px"><div class="demo" style="height:100%">
   <div class="label">학습 데이터 수 → 성능 (MVTec 나사 · 시드 {NSEED}개 평균 · 실측 mAP50)</div>
   {svg}
  </div></div>
  <div class="col"><div class="demo" style="height:100%">
   <div class="label">읽는 법</div>
   <div style="display:flex;flex-direction:column;gap:10px;margin-top:10px">
    <div class="takeaway" style="padding:11px 15px"><div class="n" style="min-width:34px;height:34px;font-size:18px">1</div><div><p style="font-size:17px;line-height:1.4"><b>적게 모아도 급상승</b> — 10→40장 구간에서 성능 대부분이 올라옵니다.</p></div></div>
    <div class="takeaway" style="padding:11px 15px"><div class="n" style="min-width:34px;height:34px;font-size:18px;background:var(--good)">2</div><div><p style="font-size:17px;line-height:1.4"><b>어느 순간 포화</b> — 무한정이 아니라 <b>'충분량'</b>이 있습니다.</p></div></div>
    <div class="takeaway" style="padding:11px 15px"><div class="n" style="min-width:34px;height:34px;font-size:18px;background:var(--warn)">3</div><div><p style="font-size:17px;line-height:1.4"><b>증강도 '양 늘리기'</b> — 새로 못 찍을 땐 변형으로 불립니다(앞장).</p></div></div>
   </div>
   <div class="term" style="margin-top:12px;font-size:15.5px"><b>그래서 "모델보다 데이터"</b> — 더 큰 모델을 사는 것보다, <b>내 현장 데이터를 조금 더 모으고 잘 다양화</b>하는 게 대개 먼저입니다.</div>
  </div></div>
 </div>
</section>

'''

html = DECK.read_text(encoding="utf-8")
import re
# 재실행 대비: 기존 삽입분 제거 후 재삽입
html = re.sub(r"<!-- 14b DATA SIZE.*?</section>\n\n", "", html, flags=re.S)
anchor = "<!-- 15 RECAP -->"
assert anchor in html, "RECAP 마커 없음"
html = html.replace(anchor, section + anchor, 1)
DECK.write_text(html, encoding="utf-8")
print("inserted data-size slide. points:", [(n, round(v, 1)) for n, v in pts])
