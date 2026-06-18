# -*- coding: utf-8 -*-
"""1교시 시나리오용 시각자료 생성 (허블 화면 대체).

생성(eval/out/lesson/):
  task3.png    : 같은 이미지 -> 분류 / 검출 / 분할 출력 비교 (3종 감 잡기)
  labeling.png : 라벨링 임시 화면 느낌 (사람이 박스 그리고 클래스 단다)
한글 깨짐 방지 위해 matplotlib(한글폰트)로 렌더.
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

_avail = {f.name for f in fm.fontManager.ttflist}
for _f in ("NanumGothic", "Noto Sans CJK KR", "NanumBarunGothic"):
    if _f in _avail:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MSD = ROOT.parent / "data" / "MSD"
OUT = HERE / "out" / "lesson"
VAL2NAME = {38: "oil", 113: "scratch", 75: "stain"}
COLORS = {"oil": "#1d4ed8", "scratch": "#c0152b", "stain": "#1d9e55"}


def boxes_of(mask, val, min_area=60):
    binary = ((mask == val).astype("uint8")) * 255
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        out.append((x, y, w, h))
    return out


def load(cls_folder, gt_folder, stem):
    img = cv2.cvtColor(cv2.imread(str(MSD / cls_folder / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(MSD / gt_folder / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    return img, mask


def task3(cls_folder, gt_folder, stem, cls_name):
    img, mask = load(cls_folder, gt_folder, stem)
    val = [k for k, v in VAL2NAME.items() if v == cls_name][0]
    boxes = boxes_of(mask, val)
    color = COLORS[cls_name]

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax in axes:
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])

    axes[0].set_title("① 분류 (Classification)\n이미지 1장 = 라벨 1개", fontsize=12)
    axes[0].text(0.5, -0.13, f"예측: 불량 · {cls_name}", transform=axes[0].transAxes,
                 ha="center", fontsize=13, color=color, fontweight="bold")

    axes[1].set_title("② 검출 (Object Detection)\n어디에 = 박스 좌표", fontsize=12)
    for (x, y, w, h) in boxes:
        axes[1].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
    axes[1].text(0.5, -0.13, f"{cls_name} × {len(boxes)}개 위치", transform=axes[1].transAxes,
                 ha="center", fontsize=13, color=color, fontweight="bold")

    over = img.copy()
    cmap = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)])
    sel = mask == val
    over[sel] = (0.45 * over[sel] + 0.55 * cmap).astype(np.uint8)
    axes[2].imshow(over)
    axes[2].set_title("③ 분할 (Instance Segmentation)\n픽셀 경계까지", fontsize=12)
    axes[2].text(0.5, -0.13, "결함 영역 = 픽셀 마스크", transform=axes[2].transAxes,
                 ha="center", fontsize=13, color=color, fontweight="bold")

    fig.suptitle("같은 이미지, 세 가지 출력 — 정밀할수록 라벨링 비용 ↑ (분류 < 검출 < 분할)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "task3.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def labeling(cls_folder, gt_folder, stem, cls_name):
    img, mask = load(cls_folder, gt_folder, stem)
    val = [k for k, v in VAL2NAME.items() if v == cls_name][0]
    boxes = boxes_of(mask, val)
    color = COLORS[cls_name]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for (x, y, w, h) in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2.2))
        ax.add_patch(Rectangle((x, y - 26), 96, 24, facecolor=color, edgecolor="none"))
        ax.text(x + 5, y - 9, cls_name, color="white", fontsize=10, fontweight="bold")
    ax.set_title("라벨링 화면(예시) — 사람이 결함에 박스를 그리고 클래스를 단다\n"
                 "이 '정답'의 일관성이 곧 데이터 품질", fontsize=12)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "labeling.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    # task3: oil(면적 커서 분할 패널이 잘 보임) / labeling: scratch(선형이라 박스 라벨 직관적)
    print("task3:", task3("oil", "ground_truth_2", "Oil_0005", "oil"))
    print("labeling:", labeling("scratch", "ground_truth_1", "Scr_0010", "scratch"))
