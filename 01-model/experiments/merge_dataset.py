"""Phase3 용: 여러 MVTec 카테고리를 단일 클래스 'defect' YOLO 데이터셋으로 병합.

목적: 데이터 '양'을 독립 변수로 키운다(클래스 수는 1로 고정).
스케일:
  S = screw                              (~119 결함)
  M = screw,metal_nut,capsule,hazelnut,bottle,tile  (~538 결함)
  L = 전 15 카테고리                       (~1258 결함)
각 스케일 내에서 모델 n/s/m/l 을 학습 -> '데이터가 커질수록 큰 모델이 유리해지는가' 검증.

출력: experiments/datasets/merged_<scale>/{images,labels}/{train,val} + data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_ROOT = ROOT.parent / "data"
OUT_ROOT = HERE / "datasets"
MIN_AREA = 80

SCALES = {
    "S": ["screw"],
    "M": ["screw", "metal_nut", "capsule", "hazelnut", "bottle", "tile"],
    "L": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
          "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
}


def mask_to_boxes(mask):
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        out.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def label_text(mask_path: Path | None) -> str:
    if mask_path is None or not mask_path.exists():
        return ""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return "\n".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                     for cx, cy, bw, bh in mask_to_boxes(mask))


def collect(category):
    src = DATA_ROOT / category
    test_dir, gt_dir = src / "test", src / "ground_truth"
    defects, goods = [], []
    for ddir in sorted(test_dir.iterdir()):
        if not ddir.is_dir():
            continue
        for img in sorted(ddir.glob("*.png")):
            if ddir.name == "good":
                goods.append((category, img, None))
            else:
                defects.append((category, img, gt_dir / ddir.name / f"{img.stem}_mask.png"))
    return defects, goods


def write(items, split, out):
    img_out, lbl_out = out / "images" / split, out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for cat, img_path, mask_path in items:
        stem = f"{cat}_{img_path.parent.name}_{img_path.stem}"
        shutil.copy(img_path, img_out / f"{stem}.png")
        (lbl_out / f"{stem}.txt").write_text(label_text(mask_path))
    return len(items)


def build(scale: str, val_ratio=0.2, seed=0, good_ratio=0.3) -> Path:
    cats = SCALES[scale]
    defects, goods = [], []
    for c in cats:
        d, g = collect(c)
        defects += d
        goods += g
    rng = random.Random(seed)
    rng.shuffle(defects)
    rng.shuffle(goods)
    goods = goods[: int(len(defects) * good_ratio)]

    def sp(lst):
        nv = max(1, int(len(lst) * val_ratio)) if lst else 0
        return lst[nv:], lst[:nv]

    d_tr, d_va = sp(defects)
    g_tr, g_va = sp(goods)
    out = OUT_ROOT / f"merged_{scale}"
    if out.exists():
        shutil.rmtree(out)
    nt = write(d_tr + g_tr, "train", out)
    nv = write(d_va + g_va, "val", out)
    data = {"path": str(out.resolve()), "train": "images/train",
            "val": "images/val", "names": {0: "defect"}}
    (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    print(f"[{scale}] cats={len(cats)} train={nt} val={nv} "
          f"(defect {len(defects)} + good {len(goods)}) -> {out}")
    return out / "data.yaml"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=list(SCALES) + ["all"], default="all")
    args = ap.parse_args()
    targets = list(SCALES) if args.scale == "all" else [args.scale]
    for s in targets:
        build(s)
