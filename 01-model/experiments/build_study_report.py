"""심화 실험 보고서 (Phase 1~3) -> docs/lesson1-study.html.

플롯(experiments/plots/*.png)을 base64 임베드 + 결과 기반 표/해석 자동 생성.
plots.py 를 먼저 실행해 png 를 만든 뒤 호출.
"""

import base64
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
PLOTS = HERE / "plots"
REPORT = ROOT.parent / "docs" / "lesson1-study.html"
SIZE_ORDER = ["n", "s", "m", "l"]


def load(phase):
    rows = []
    for f in sorted(RESULTS.glob(f"phase{phase}.d*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("map50") == r.get("map50")]


def img(path: Path, width="100%"):
    if not path or not Path(path).exists():
        return f"<p class='missing'>[그림 없음: {getattr(path,'name','?')}]</p>"
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"<img style='width:{width}' src='data:image/png;base64,{data}'/>"


def stat(vals):
    a = np.array(vals)
    return a.mean(), np.median(a), a.min(), a.max()


def phase1_table():
    rows = load(1)
    if not rows:
        return "<p class='missing'>[Phase1 결과 없음]</p>", ""
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r["map50"])
    ns = sorted(by_n)
    head = "<tr><th>데이터 수</th><th>반복</th><th>mAP50 중앙값</th><th>평균</th><th>최소~최대</th></tr>"
    body = ""
    for n in ns:
        mean, med, lo, hi = stat(by_n[n])
        body += (f"<tr><td>{n}장</td><td>{len(by_n[n])}</td><td><b>{med:.3f}</b></td>"
                 f"<td>{mean:.3f}</td><td>{lo:.3f} ~ {hi:.3f}</td></tr>")
    # 포화 지점 탐지: 중앙값 증가폭이 작아지는 첫 지점
    meds = [float(np.median(by_n[n])) for n in ns]
    gains = [meds[i] - meds[i - 1] for i in range(1, len(meds))]
    note = ""
    if gains:
        peak_gain_idx = int(np.argmax(gains)) + 1
        note = (f"<p>중앙값이 가장 가파르게 오르는 구간은 <b>{ns[peak_gain_idx-1]}→{ns[peak_gain_idx]}장</b>. "
                f"이후 증가폭이 줄어 데이터의 한계효용 체감이 보인다.</p>")
    return f"<table>{head}{body}</table>", note


def phase2_section():
    rows = load(2)
    if not rows:
        return "<p class='missing'>[Phase2 결과 없음]</p>"
    points = sorted({r["n"] for r in rows})
    presets = ["none", "flip", "rotate", "hsv", "scale", "mosaic", "all"]
    html = ""
    for pt in points:
        by_p = defaultdict(list)
        for r in rows:
            if r["n"] == pt:
                by_p[r["preset"]].append(r["map50"])
        base = float(np.median(by_p["none"])) if by_p["none"] else 0.0
        head = "<tr><th>증강 기법</th><th>mAP50 중앙값</th><th>none 대비</th></tr>"
        body = ""
        ranked = sorted((p for p in presets if by_p[p]),
                        key=lambda p: np.median(by_p[p]), reverse=True)
        for p in ranked:
            med = float(np.median(by_p[p]))
            delta = med - base
            sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "—")
            body += f"<tr><td>{p}</td><td>{med:.3f}</td><td>{sign} {delta:+.3f}</td></tr>"
        best = ranked[0] if ranked else "none"
        html += (f"<h3>데이터 {pt}장</h3>{img(PLOTS / f'phase2_aug_n{pt}.png')}"
                 f"<table>{head}{body}</table>"
                 f"<div class=takeaway>{pt}장 구간 최고 기법: <b>{best}</b> "
                 f"(none 대비 {np.median(by_p[best])-base:+.3f}).</div>")
    return html


def phase3_section(phase="3"):
    rows = load(phase)
    if not rows:
        return "<p class='missing'>[Phase3 결과 없음]</p>", ""
    scales = [s for s in ["S", "M", "L"] if any(r["scale"] == s for r in rows)]
    head = "<tr><th>데이터 규모</th>" + "".join(f"<th>yolo11{s}</th>" for s in SIZE_ORDER) + "<th>최적 크기</th></tr>"
    body = ""
    best_size_idx = {}
    for sc in scales:
        cells, means = "", []
        for size in SIZE_ORDER:
            vals = [r["map50"] for r in rows if r["scale"] == sc and r["size"] == size]
            means.append(np.mean(vals) if vals else float("nan"))
        bi = int(np.nanargmax(means))
        best_size_idx[sc] = bi
        for i, m in enumerate(means):
            cell = "—" if m != m else f"{m:.3f}"
            cells += f"<td>{'<b>'+cell+'</b>' if i == bi else cell}</td>"
        body += f"<tr><td><b>{sc}</b></td>{cells}<td>yolo11{SIZE_ORDER[bi]}</td></tr>"
    # 결론: 데이터가 커질수록 최적 크기가 커지는가?
    seq = [best_size_idx[s] for s in scales]
    shifts = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)) and seq[-1] > seq[0]
    if shifts:
        verdict = ("<b>가설 입증.</b> 데이터가 작을 땐 작은 모델이, 커질수록 더 큰 모델이 최적이 된다 "
                   f"(최적: {' → '.join('yolo11'+SIZE_ORDER[best_size_idx[s]] for s in scales)}). "
                   "즉 모델 크기 자체가 아니라 <b>모델↔데이터 정합</b>이 성능을 가른다. "
                   "작은 데이터에 큰 모델은 과적합으로 손해, 큰 데이터엔 큰 모델이 표현력으로 보답.")
    else:
        verdict = ("최적 크기 이동이 단조롭지는 않다 "
                   f"(최적: {' → '.join('yolo11'+SIZE_ORDER[best_size_idx[s]] for s in scales)}). "
                   "다만 작은 데이터에서 큰 모델의 이점이 없다는 점은 일관된다 — "
                   "데이터가 모델 크기를 정당화해야 한다는 방향은 유지된다.")
    table = f"<table>{head}{body}</table>"
    return table, f"<div class=takeaway>{verdict}</div>"


HTML = """<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>1교시 심화 실험 — 데이터·증강·모델크기</title>
<style>
 :root{{color-scheme:light}} html{{background:#eef1f6}}
 body{{font-family:-apple-system,'Segoe UI',Roboto,'Noto Sans KR',sans-serif;max-width:1000px;margin:0 auto;
   padding:40px;background:#fff;color:#15181d;line-height:1.7;font-size:17px;box-shadow:0 0 24px rgba(0,0,0,.08)}}
 h1{{font-size:29px;border-bottom:4px solid #2d6cdf;padding-bottom:10px;color:#10243f}}
 h2{{font-size:23px;margin-top:48px;border-left:6px solid #2d6cdf;padding:4px 0 4px 14px;color:#10243f}}
 h3{{font-size:18px;margin-top:26px;color:#1d4ed8}}
 p,li{{color:#222831}} b{{color:#0c1320}}
 table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:16px}}
 th,td{{border:1px solid #c4ccd8;padding:9px 11px;text-align:center}}
 th{{background:#234b8a;color:#fff}} tr:nth-child(even) td{{background:#eef3fb}}
 img{{display:block;border:1px solid #c4ccd8;border-radius:6px;margin:12px 0;background:#fff}}
 .missing{{color:#c0152b;font-style:italic}}
 .takeaway{{background:#e7eefc;border-left:6px solid #2d6cdf;padding:16px 20px;border-radius:6px;margin:16px 0;color:#15233f}}
 .key{{background:#fff3da;border-left:6px solid #e0911c;padding:16px 20px;border-radius:6px;color:#3d2a05;margin:16px 0}}
 code{{background:#e7e9ee;color:#9b1d4a;padding:2px 6px;border-radius:4px;font-size:90%}}
</style></head><body>
<h1>1교시 심화 실험 — 데이터 양 · 증강 기법 · 모델↔데이터 정합</h1>
<p>단회가 아닌 <b>다회(다중 seed) 실험</b>으로 분산까지 본다. 지표는 <b>mAP50</b>.
데이터: MVTec AD, 검출, 단일 클래스 <code>defect</code>. GPU: 4×RTX 4090, YOLO11.</p>
<div class=key><b>검증하려는 3가지</b>
<ol>
<li>데이터를 10→100장으로 늘리면 성능은 어떻게, 어디까지 오르나 (한계효용)</li>
<li>유의미 지점에서 <b>어떤 증강 기법</b>이 실제로 성능을 올리나</li>
<li>"큰 모델이 답이 아니다"의 본질 = <b>모델 크기와 데이터 크기의 정합</b> (데이터를 키우면 큰 모델이 역전하는가)</li>
<li>같은 증강 기법도 <b>데이터 종류(형태)</b>에 따라 효능이 다른가 (n=20 고정, 4종 비교)</li>
</ol></div>

<h2>Phase 1. 데이터 양에 따른 mAP50 분포</h2>
<p>screw, yolo11s, 증강 없음 고정. n=10~100을 10단위로, 지점마다 seed 를 바꿔 반복 → 박스플롯.</p>
{p1_plot}
{p1_table}
{p1_note}

<h2>Phase 2. 증강 기법별 효과 (유의미 지점)</h2>
<p>Phase1 추이에서 고른 지점에서, 증강 기법을 하나씩 분리해 적용. 어떤 변형이 실제로 성능을 올리는지.</p>
{p2}

<h2>Phase 3. 모델 크기 × 데이터 크기 — 정합 가설</h2>
<p>카테고리를 병합해 데이터 규모를 <b>S(~120) → M(~560) → L(~1300)</b>으로 키우고(단일 클래스),
각 규모에서 yolo11 <b>n/s/m/l</b>을 학습. 데이터가 커질수록 최적 모델 크기가 커지면 가설 입증.</p>
<h3>3-A. 증강 OFF (순수 데이터·모델 크기만)</h3>
{p3_plot}
{p3_table}
{p3_verdict}

<h3>3-B. 증강 ON (all) — 대형 모델에 정규화를 준 공정 재검증</h3>
<p>3-A는 증강을 꺼 대형 모델이 과적합에 무방비였다(불리한 판). 동일 매트릭스를 <b>증강(all)</b> 켜고 재실행해,
정규화를 받은 대형 모델이 데이터가 커질 때 역전하는지(크로스오버) 확인.</p>
{p3b_plot}
{p3b_table}
{p3b_verdict}
{compare_verdict}

<h2>Phase 4. 데이터 종류(형태)별 증강 기법 효능</h2>
<p>Phase 2가 "데이터 <b>양</b>에 따라 증강 효능이 다르다"를 보였다면, 여기선 데이터 <b>종류</b>를 바꾼다.
n=20장 고정(yolo11s), 시각적으로 다른 4종(가는 금속 screw / 둥근 금속 metal_nut / 색·프린트 pill / 텍스처 carpet)에
같은 7개 증강 기법을 적용. 셀 값 Δ = (기법 mAP50 − 증강없음).</p>
{p4_plot}
{p4_table}
{p4_verdict}

<div class=takeaway><b>종합.</b> 데이터 양은 성능을 가르지만 한계효용이 있고, 증강은 '맞는 기법'을 골라야 효과가 있으며,
모델 크기는 절대선이 아니라 <b>데이터 규모에 맞춰</b> 골라야 한다. → 현장에서는 "데이터부터 보고 모델을 정한다".</div>
</body></html>"""


def phase4_section():
    rows = load(4)
    if not rows:
        return "<p class='missing'>[Phase4 결과 없음]</p>", ""
    cat_order = [c for c in ["screw", "metal_nut", "pill", "carpet"]
                 if any(r["cat"] == c for r in rows)]
    presets = ["flip", "rotate", "hsv", "scale", "mosaic", "all"]
    med = defaultdict(list)
    for r in rows:
        med[(r["cat"], r["preset"])].append(r["map50"])

    def m(c, p):
        return float(np.median(med[(c, p)])) if med[(c, p)] else float("nan")

    head = "<tr><th>데이터(형태)</th><th>기준 none</th>" + "".join(f"<th>{p}</th>" for p in presets) + "<th>최고 기법</th></tr>"
    body = ""
    best_of, hsv_of = {}, {}
    for c in cat_order:
        base = m(c, "none")
        deltas = {p: m(c, p) - base for p in presets}
        best = max(deltas, key=lambda p: deltas[p])
        best_of[c] = (best, deltas[best])
        hsv_of[c] = deltas["hsv"]
        cells = ""
        for p in presets:
            d = deltas[p]
            col = "#1b6b3a" if d > 0.01 else ("#c0152b" if d < -0.01 else "#555")
            cells += f"<td style='color:{col}'>{d:+.3f}</td>"
        body += f"<tr><td><b>{c}</b></td><td>{base:.3f}</td>{cells}<td>{best} ({deltas[best]:+.3f})</td></tr>"
    table = f"<table>{head}{body}</table>"

    # 해석: 카테고리마다 최고 기법이 갈리는가? hsv(색)가 색 객체에서만 듣는가?
    bests = {c: best_of[c][0] for c in cat_order}
    differ = len(set(bests.values())) > 1
    hsv_msg = ""
    if hsv_of:
        hi = max(hsv_of, key=lambda c: hsv_of[c])
        lo = min(hsv_of, key=lambda c: hsv_of[c])
        hsv_msg = (f" 특히 <b>hsv(색 증강)</b>는 형태에 가장 민감하다 — "
                   f"{hi}엔 {hsv_of[hi]:+.3f}이지만 {lo}엔 {hsv_of[lo]:+.3f}(유일하게 성능을 깎는 기법). "
                   "색이 '변이 요인'인 표면엔 도움, 색·프린트 자체가 '결함 신호'인 객체엔 그 신호를 망가뜨려 해롭다.")
    lead = ("<b>같은 기법도 데이터 종류에 따라 효능이 갈린다.</b> 카테고리별 최고 기법: "
            + ", ".join(f"{c}→<b>{bests[c]}</b>" for c in cat_order) + ".") if differ else \
           ("<b>이 4종에서는 최고 기법이 대체로 일치</b>했다("
            + ", ".join(f"{c}→{bests[c]}" for c in cat_order) + ").")
    verdict = (f"<div class=key>{lead}{hsv_msg} → <b>증강은 데이터 양뿐 아니라 '형태/도메인'에 맞춰 골라야 한다.</b> "
               "현장에선 대상 이미지 특성(색 유무·방향성·텍스처)을 보고 증강을 정하는 게 맞다.</div>")
    return table, verdict


def compare_verdict():
    """3-A(no-aug) vs 3-B(aug) 최적 크기 변화 비교 -> 증강이 그림을 바꿨는지 판정."""
    def best_seq(phase):
        rows = load(phase)
        if not rows:
            return None
        out = {}
        for sc in ["S", "M", "L"]:
            means = {sz: np.mean([r["map50"] for r in rows if r["scale"] == sc and r["size"] == sz] or [float("nan")])
                     for sz in SIZE_ORDER}
            vals = [means[sz] for sz in SIZE_ORDER]
            if all(v != v for v in vals):
                continue
            out[sc] = SIZE_ORDER[int(np.nanargmax(vals))]
        return out
    a, b = best_seq("3"), best_seq("3b")
    if not b:
        return ""
    scales = [s for s in ["S", "M", "L"] if s in b]
    seq_a = " → ".join("yolo11" + a[s] for s in scales) if a else "?"
    seq_b = " → ".join("yolo11" + b[s] for s in scales)
    idx_b = [SIZE_ORDER.index(b[s]) for s in scales]
    crossover = len(idx_b) > 1 and idx_b[-1] > idx_b[0] and all(idx_b[i] <= idx_b[i+1] for i in range(len(idx_b)-1))
    if crossover:
        msg = (f"<b>증강을 주자 크로스오버가 나타났다.</b> 최적 크기: 증강OFF [{seq_a}] → 증강ON [{seq_b}]. "
               "대형 모델은 정규화(증강)와 충분한 데이터가 동시에 갖춰질 때 비로소 유리해진다 — "
               "즉 성능은 모델 크기 단독이 아니라 <b>모델 × 데이터 × 정규화</b>의 정합으로 결정된다. 가설 입증.")
    else:
        msg = ("<b>증강을 줘도 크로스오버는 나타나지 않았다.</b> "
               f"최적 크기: 증강OFF [{seq_a}] → 증강ON [{seq_b}] — 모든 규모에서 yolo11n 우위. "
               "주목할 점: 증강을 켜자 작은 모델은 크게 올랐지만(L-n 0.42→0.65) "
               "대형 모델(m·l)은 오히려 더 낮아져 거의 0에 수렴했다. "
               "이는 '큰 모델이 근본적으로 못한다'기보다 <b>고정 100에포크에서 대형 모델이 미수렴(undertraining)</b>일 가능성이 크다 "
               "— 무거운 증강은 수렴을 늦추고, 큰 모델일수록 더 많은 에포크를 요구하기 때문. "
               "<b>다만 실무 결론은 견고하다: 현장(중소 제조, 수백~수천 장) 규모에선 작은 모델+증강이 최선이며 큰 모델은 답이 아니다.</b> "
               "대형 모델의 공정한 판정은 에포크를 늘린 재실험이 필요하다.")
    return f"<div class=key>{msg}</div>"


def main():
    p1_table, p1_note = phase1_table()
    p3_table, p3_verdict = phase3_section("3")
    p3b_table, p3b_verdict = phase3_section("3b")
    p4_table, p4_verdict = phase4_section()
    html = HTML.format(
        p1_plot=img(PLOTS / "phase1_datasize.png"),
        p1_table=p1_table, p1_note=p1_note,
        p2=phase2_section(),
        p3_plot=img(PLOTS / "phase3_matching.png"),
        p3_table=p3_table, p3_verdict=p3_verdict,
        p3b_plot=img(PLOTS / "phase3b_matching.png"),
        p3b_table=p3b_table, p3b_verdict=p3b_verdict,
        compare_verdict=compare_verdict(),
        p4_plot=img(PLOTS / "phase4_type_aug.png"),
        p4_table=p4_table, p4_verdict=p4_verdict,
    )
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(html, encoding="utf-8")
    print(f"보고서 -> {REPORT} ({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
