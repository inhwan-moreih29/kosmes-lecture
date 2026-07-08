"""실험 결과 -> 그림 (박스플롯 / 크로스오버).

입력: experiments/results/phase<p>.d*.jsonl  (run.py 가 append)
출력: experiments/plots/*.png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 (없으면 □□□ 깨짐). 사용 가능한 첫 폰트 선택 + 마이너스 기호 정상화.
from matplotlib import font_manager as _fm
_avail = {f.name for f in _fm.fontManager.ttflist}
for _f in ("NanumGothic", "Noto Sans CJK KR", "NanumBarunGothic"):
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent.parent / "outputs" / "experiments" / "results"
PLOTS = HERE.parent.parent / "outputs" / "experiments" / "plots"
SIZE_ORDER = ["n", "s", "m", "l"]
SIZE_PARAMS = {"n": 2.6, "s": 9.4, "m": 20.0, "l": 25.3}  # M params (참고)


def load(phase) -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob(f"phase{phase}.d*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("map50") == r.get("map50")]  # drop nan


def plot_phase1():
    rows = load(1)
    if not rows:
        return None
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r["map50"])
    ns = sorted(by_n)
    data = [by_n[n] for n in ns]
    medians = [float(np.median(by_n[n])) for n in ns]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot(data, positions=ns, widths=6, patch_artist=True,
               boxprops=dict(facecolor="#bcd3f7", color="#234b8a"),
               medianprops=dict(color="#c0152b", linewidth=2),
               whiskerprops=dict(color="#234b8a"), capprops=dict(color="#234b8a"),
               flierprops=dict(marker="o", markersize=4, markerfacecolor="#888"))
    ax.plot(ns, medians, "-o", color="#1d4ed8", linewidth=2, label="중앙값 추이", zorder=3)
    ax.set_xlabel("학습 데이터 수 (장)")
    ax.set_ylabel("mAP50")
    ax.set_title(f"데이터 양에 따른 mAP50 분포 (screw, yolo11s, 증강X, 지점당 {len(data[0])}회)")
    ax.set_xticks(ns)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = PLOTS / "phase1_datasize.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_phase2():
    rows = load(2)
    if not rows:
        return None
    points = sorted({r["n"] for r in rows})
    presets = ["none", "flip", "rotate", "hsv", "scale", "mosaic", "all"]
    outs = []
    for pt in points:
        by_p = defaultdict(list)
        for r in rows:
            if r["n"] == pt:
                by_p[r["preset"]].append(r["map50"])
        labels = [p for p in presets if by_p[p]]
        data = [by_p[p] for p in labels]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(data, tick_labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="#cfe8d4", color="#1b6b3a"),
                   medianprops=dict(color="#c0152b", linewidth=2))
        base = float(np.median(by_p["none"])) if by_p["none"] else None
        if base is not None:
            ax.axhline(base, ls="--", color="#888", label="증강 없음 중앙값")
            ax.legend()
        ax.set_ylabel("mAP50")
        ax.set_title(f"증강 기법별 mAP50 (screw {pt}장, yolo11s)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = PLOTS / f"phase2_aug_n{pt}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_crossover(phase, out_name, title):
    rows = load(phase)
    if not rows:
        return None
    scales = [s for s in ["S", "M", "L"] if any(r["scale"] == s for r in rows)]
    colors = {"S": "#c0152b", "M": "#e0911c", "L": "#1d4ed8"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for sc in scales:
        means, lo, hi, xs = [], [], [], []
        for i, size in enumerate(SIZE_ORDER):
            vals = [r["map50"] for r in rows if r["scale"] == sc and r["size"] == size]
            if not vals:
                continue
            xs.append(i)
            means.append(float(np.mean(vals)))
            lo.append(float(np.min(vals)))
            hi.append(float(np.max(vals)))
        if not xs:
            continue
        err = [np.array(means) - np.array(lo), np.array(hi) - np.array(means)]
        ax.errorbar(xs, means, yerr=err, marker="o", capsize=4, linewidth=2,
                    color=colors.get(sc, None), label=f"데이터 {sc}")
    ax.set_xticks(range(len(SIZE_ORDER)))
    ax.set_xticklabels([f"yolo11{s}\n({SIZE_PARAMS[s]}M)" for s in SIZE_ORDER])
    ax.set_xlabel("모델 크기")
    ax.set_ylabel("mAP50 (평균, 막대=최소~최대)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="데이터 규모")
    fig.tight_layout()
    out = PLOTS / out_name
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


CAT_LABEL = {
    "screw": "screw\n(가는 금속)", "metal_nut": "metal_nut\n(둥근 금속)",
    "pill": "pill\n(색·프린트)", "carpet": "carpet\n(텍스처)",
}


def plot_phase4():
    """데이터 종류 × 증강기법 효능 히트맵 (Δ = 중앙값 - none 중앙값)."""
    rows = load(4)
    if not rows:
        return None
    cat_order = [c for c in ["screw", "metal_nut", "pill", "carpet"]
                 if any(r["cat"] == c for r in rows)]
    presets = ["flip", "rotate", "hsv", "scale", "mosaic", "all"]  # none=기준(생략)
    med = defaultdict(list)
    for r in rows:
        med[(r["cat"], r["preset"])].append(r["map50"])
    base = {c: float(np.median(med[(c, "none")])) for c in cat_order if med[(c, "none")]}
    M = np.full((len(cat_order), len(presets)), np.nan)
    for i, c in enumerate(cat_order):
        for j, p in enumerate(presets):
            if med[(c, p)] and c in base:
                M[i, j] = float(np.median(med[(c, p)])) - base[c]

    vmax = np.nanmax(np.abs(M)) or 0.1
    fig, ax = plt.subplots(figsize=(8.5, 1.0 + 1.1 * len(cat_order)))
    im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels(presets)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([f"{CAT_LABEL.get(c,c)}\nbase {base.get(c,float('nan')):.2f}" for c in cat_order])
    for i in range(len(cat_order)):
        for j in range(len(presets)):
            v = M[i, j]
            if v == v:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > vmax * 0.55 else "black", fontsize=11)
    ax.set_title("데이터 종류 × 증강기법 효능 (Δ mAP50 = 기법 − 증강없음)\n파랑=향상, 빨강=악화 / screw 20장 등 모두 n=20")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ mAP50")
    fig.tight_layout()
    out = PLOTS / "phase4_type_aug.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_phase5():
    """MSD 실데이터: 데이터량(log) × 모델크기 mAP50 곡선 + 최적크기 이동(크로스오버)."""
    rows = load(5)
    if not rows:
        return None
    from collections import defaultdict
    tab = defaultdict(dict)
    for r in rows:
        tab[r["n"]][r["size"]] = r["map50"]
    ns = sorted(tab)
    colors = {"n": "#c0152b", "s": "#e0911c", "m": "#1d9e55", "l": "#1d4ed8"}
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for size in SIZE_ORDER:
        xs = [n for n in ns if size in tab[n]]
        ys = [tab[n][size] for n in xs]
        if xs:
            ax.plot(xs, ys, "-o", color=colors[size], linewidth=2,
                    label=f"yolo11{size} ({SIZE_PARAMS[size]}M)")
    # 각 데이터량의 최적 모델 크기 표시
    for n in ns:
        best = max(tab[n], key=lambda s: tab[n][s])
        ax.annotate(best, (n, tab[n][best]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color=colors[best], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("학습 데이터 수 (장, log scale)")
    ax.set_ylabel("mAP50")
    ax.set_title("실데이터(MSD 휴대폰결함 3클래스): 데이터량 × 모델크기\n"
                 "극소데이터=큰 모델 유리, 데이터↑=작은 모델 유리 (라벨=각 지점 최적크기, 단일런)")
    ax.grid(alpha=0.3)
    ax.legend(title="모델 (파라미터)")
    fig.tight_layout()
    out = PLOTS / "phase5_msd_matching.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_phase4_trafficlight():
    """데이터 종류 × 증강 효율을 신호등(도움/미미/주의)으로 — 비전문가용."""
    rows = load(4)
    if not rows:
        return None
    cat_order = [c for c in ["screw", "metal_nut", "pill", "carpet"]
                 if any(r["cat"] == c for r in rows)]
    cat_ko = {"screw": "가는 금속\n(나사)", "metal_nut": "둥근 금속\n(너트)",
              "pill": "색·프린트\n(알약)", "carpet": "텍스처\n(원단)"}
    presets = ["flip", "rotate", "hsv", "scale", "mosaic", "all"]
    preset_ko = {"flip": "반전", "rotate": "회전", "hsv": "색(hsv)",
                 "scale": "크기", "mosaic": "모자이크", "all": "종합"}
    med = defaultdict(list)
    for r in rows:
        med[(r["cat"], r["preset"])].append(r["map50"])
    base = {c: float(np.median(med[(c, "none")])) for c in cat_order if med[(c, "none")]}

    GREEN, GREY, RED = "#1f9e54", "#c2c8d2", "#d23b3b"
    fig, ax = plt.subplots(figsize=(9, 1.2 + 1.05 * len(cat_order)))
    for i, c in enumerate(cat_order):
        for j, p in enumerate(presets):
            if not med[(c, p)] or c not in base:
                continue
            d = float(np.median(med[(c, p)])) - base[c]
            if d >= 0.05:
                col, sym = GREEN, "+"
            elif d <= -0.02:
                col, sym = RED, "—"
            else:
                col, sym = GREY, ""
            ax.scatter(j, i, s=820, color=col, zorder=2)
            if sym:
                ax.text(j, i, sym, ha="center", va="center", color="white",
                        fontsize=15, fontweight="bold", zorder=3)
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels([preset_ko[p] for p in presets], fontsize=12)
    ax.set_yticks(range(len(cat_order)))
    ax.set_yticklabels([cat_ko[c] for c in cat_order], fontsize=12)
    ax.set_xlim(-0.6, len(presets) - 0.4)
    ax.set_ylim(-0.6, len(cat_order) - 0.4)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(color="#eef0f4", zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markersize=14, label="도움"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markersize=14, label="미미"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=14, label="주의(악화)")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=12)
    ax.set_title("내 제품 유형별, 켜면 도움 되는 증강 / 끄는 게 나은 증강\n"
                 "규칙: 색·무늬가 '결함 신호'인 제품(알약 프린트)엔 색(hsv) 증강 끄기 · 표면 흠집엔 켜면 도움",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    out = PLOTS / "phase4_trafficlight.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_phase5_simple(n_point=600):
    """모델크기: 실용 데이터량 1지점에서 작은 vs 큰 막대 — '크게 키워도 그대로'."""
    rows = [r for r in load(5) if r["n"] == n_point]
    if not rows:
        return None
    by = {r["size"]: r["map50"] for r in rows}
    pairs = [("n", "작은 모델\nyolo11n · 2.6M"), ("l", "큰 모델\nyolo11l · 25.3M (약 10배)")]
    pairs = [(s, lab) for s, lab in pairs if s in by]
    vals = [by[s] for s, _ in pairs]
    labels = [lab for _, lab in pairs]
    colors = ["#1f9e54", "#1d4ed8"]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    bars = ax.bar(range(len(vals)), vals, width=0.55, color=colors[:len(vals)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", fontsize=15, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mAP50 (높을수록 좋음)")
    ax.set_title(f"같은 데이터({n_point}장)로 학습 — 모델만 10배 키우면?\n"
                 "성능 거의 같음 → 큰 모델·큰 GPU에 돈 쓸 필요 없다",
                 fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS / "phase5_simple.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    PLOTS.mkdir(parents=True, exist_ok=True)
    print("phase4_trafficlight:", plot_phase4_trafficlight())
    print("phase5_simple:", plot_phase5_simple())
    print("phase5:", plot_phase5())
    print("phase1:", plot_phase1())
    print("phase2:", plot_phase2())
    print("phase3:", plot_crossover(3, "phase3_matching.png",
          "데이터 규모별 '모델 크기 vs 성능' — 증강 OFF"))
    print("phase3b:", plot_crossover("3b", "phase3b_matching.png",
          "데이터 규모별 '모델 크기 vs 성능' — 증강 ON (all)"))
    print("phase4:", plot_phase4())
